"""cell_MERGE training: cell-level MLP + GATNet refinement.

Reads the shared cell-level data interface from
/home/sb202604/cell-benchmark/processed_cell/, trains one (CellMLP, GATNet)
per (slice1, slice2, seed), evaluates per-cell PCC on the held-out slice, and
aggregates 3 seeds into a summary.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --dataset hSkin_Melanoma
    python train.py --config config.yaml --dataset hSkin_Melanoma --seeds 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cell-benchmark" / "scripts"))
import cell_data as cd

from model import CellMLP, GATNet
from utils import build_cell_graph


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_torch(a) -> torch.Tensor:
    import scipy.sparse as sp
    if isinstance(a, np.ndarray):
        return torch.from_numpy(a.astype(np.float32, copy=False))
    if sp.issparse(a):
        return torch.from_numpy(a.toarray().astype(np.float32))
    return torch.from_numpy(np.asarray(a, dtype=np.float32))


def pearson_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """1 - mean per-gene Pearson correlation."""
    pred_c = pred - pred.mean(0, keepdim=True)
    true_c = true - true.mean(0, keepdim=True)
    num = (pred_c * true_c).sum(0)
    den = pred_c.norm(dim=0) * true_c.norm(dim=0) + 1e-8
    return (1 - num / den).mean()


# ---------------------------------------------------------------------------
# MLP stage
# ---------------------------------------------------------------------------

def train_mlp(src_x, src_y, cfg, device):
    """Train CellMLP on source slice."""
    epochs = cfg.get("mlp_epochs", 15)
    lr = cfg.get("mlp_lr", 5e-5)
    batch_size = cfg.get("mlp_batch_size", 8)
    dropout = cfg.get("mlp_dropout", 0.2)

    model = CellMLP(
        in_dim=src_x.shape[1],
        num_genes=src_y.shape[1],
        hidden_dim=256,
        dropout=dropout,
    ).to(device)

    ds = TensorDataset(src_x, src_y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.get("mlp_step_size", 2),
        gamma=cfg.get("mlp_gamma", 0.5),
    )

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f"  [MLP] epoch {epoch}/{epochs}  loss={total_loss / n:.4f}  "
                  f"({time.time()-t0:.0f}s)")
    print(f"  [MLP] training done in {time.time()-t0:.1f}s")
    return model


# ---------------------------------------------------------------------------
# GNN stage (node-level mini-batches via NeighborLoader)
# ---------------------------------------------------------------------------

def _make_gnn_loader(data, cfg, shuffle):
    """Build a PyG NeighborLoader for GATNet."""
    from torch_geometric.loader import NeighborLoader
    batch_size = cfg.get("gnn_batch_size", 4096)
    num_neighbors = cfg.get("gnn_num_neighbors", [-1, -1])
    return NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def gnn_train_epoch(gnn, loader, optimizer, alpha=0.0):
    gnn.train()
    train_mse, train_corr = [], []
    for batch in loader:
        batch = batch.to(next(gnn.parameters()).device)
        # Only compute loss on seed nodes (first batch_size nodes)
        out = gnn(batch.x, batch.edge_index)[:batch.batch_size]
        y = batch.y[:batch.batch_size]

        mse = F.mse_loss(out, y)
        corr = 1 - pearson_loss(out, y)
        loss = mse + alpha * (1 - corr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_mse.append(mse.item())
        train_corr.append(corr.item())
    return np.mean(train_mse), np.mean(train_corr)


def gnn_predict(gnn, data, cfg):
    """Return full-node predictions for a graph using mini-batches."""
    gnn.eval()
    loader = _make_gnn_loader(data, cfg, shuffle=False)
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(next(gnn.parameters()).device)
            out = gnn(batch.x, batch.edge_index)[:batch.batch_size]
            all_preds.append(out.cpu())
            all_targets.append(batch.y[:batch.batch_size].cpu())
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    return preds, targets


def train_gnn(graph_data, num_genes, cfg, device):
    """Train GATNet on a cell-level graph with node mini-batches."""
    epochs = cfg.get("gnn_epochs", 400)
    lr = cfg.get("gnn_lr", 1e-3)
    num_heads = cfg.get("gnn_attn_heads", 8)
    drop_edge = cfg.get("gnn_drop_edge", 0.2)
    alpha = cfg.get("gnn_alpha", 0.0)
    warmup_steps = cfg.get("gnn_warmup_steps", 10)

    gnn = GATNet(num_genes=num_genes, num_heads=num_heads,
                 drop_edge=drop_edge).to(device)
    optimizer = optim.Adam(gnn.parameters(), lr=lr, weight_decay=0.0)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_loader = _make_gnn_loader(graph_data, cfg, shuffle=True)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_mse, train_corr = gnn_train_epoch(gnn, train_loader, optimizer,
                                                alpha=alpha)
        scheduler.step()

        if epoch % 40 == 0 or epoch == epochs:
            print(f"  [GNN] epoch {epoch}/{epochs}  "
                  f"train_mse={train_mse:.4f}  train_corr={train_corr:.4f}  "
                  f"({time.time()-t0:.0f}s)")
    return gnn


# ---------------------------------------------------------------------------
# Cross-section training
# ---------------------------------------------------------------------------

def train_one_direction(src_adata, tgt_adata, cfg, seed, log_prefix=""):
    """Train on src slice and evaluate on tgt slice."""
    set_seed(seed)
    device = cfg["device"]

    src_x = to_torch(src_adata.obsm["he"]).to(device)
    src_y = to_torch(src_adata.X).to(device)
    tgt_x = to_torch(tgt_adata.obsm["he"]).to(device)
    tgt_y = to_torch(tgt_adata.X).to(device)

    # ---- Stage 1: MLP ----
    print(f"  {log_prefix} [Stage 1] training CellMLP")
    mlp = train_mlp(src_x, src_y, cfg, device)

    with torch.no_grad():
        src_feat = mlp.extract_features(src_x)
        tgt_feat = mlp.extract_features(tgt_x)
    del mlp
    torch.cuda.empty_cache()

    # ---- Stage 2: GNN ----
    print(f"  {log_prefix} [Stage 2] training GATNet")
    src_graph = build_cell_graph(src_adata, src_feat, src_y, cfg, device)
    gnn = train_gnn(src_graph, src_y.shape[1], cfg, device)

    # ---- Inference on target ----
    tgt_graph = build_cell_graph(tgt_adata, tgt_feat, tgt_y, cfg, device)
    pred, true = gnn_predict(gnn, tgt_graph, cfg)
    pred_np = pred.numpy()
    true_np = true.numpy()
    metrics = cd.compute_pcc_metrics(true_np, pred_np)

    del gnn, pred, true
    torch.cuda.empty_cache()
    return {
        "mean_pcc": metrics["mean_pcc"],
        "median_pcc": metrics["median_pcc"],
        "pred": pred_np,
        "per_gene_pcc": metrics["per_gene"],
    }


def run_seed(a1, a2, cfg, seed: int, dataset: str, splits, seed_dir: Path) -> dict:
    seed_dir.mkdir(parents=True, exist_ok=True)
    if (seed_dir / "result.json").exists():
        print(f"  seed {seed}: result.json exists, skipping")
        with open(seed_dir / "result.json") as f:
            return json.load(f)

    print(f"\n  [SEED {seed}] train={splits[1]} -> test={splits[0]}")
    r21 = train_one_direction(a2, a1, cfg, seed=seed,
                              log_prefix=f"[t={splits[1]}/e={splits[0]}]")

    print(f"\n  [SEED {seed}] train={splits[0]} -> test={splits[1]}")
    r12 = train_one_direction(a1, a2, cfg, seed=seed,
                              log_prefix=f"[t={splits[0]}/e={splits[1]}]")

    np.save(seed_dir / f"pred_{splits[0]}_from_{splits[1]}.npy", r21["pred"])
    np.save(seed_dir / f"pred_{splits[1]}_from_{splits[0]}.npy", r12["pred"])
    np.save(seed_dir / f"per_gene_pcc_{splits[0]}.npy", r21["per_gene_pcc"])
    np.save(seed_dir / f"per_gene_pcc_{splits[1]}.npy", r12["per_gene_pcc"])

    dir1_key = f"mean_pcc_train_{splits[1]}_test_{splits[0]}"
    dir2_key = f"mean_pcc_train_{splits[0]}_test_{splits[1]}"
    result = {
        "dataset": dataset,
        "seed": seed,
        "n_genes": int(a1.n_vars),
        "n_cells": {splits[0]: int(a1.n_obs), splits[1]: int(a2.n_obs)},
        dir1_key: r21["mean_pcc"],
        dir2_key: r12["mean_pcc"],
        "median_pcc_slice1": r21["median_pcc"],
        "median_pcc_slice2": r12["median_pcc"],
    }
    with open(seed_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  [SEED {seed}] PCC train={splits[1]}->test={splits[0]}: "
          f"{r21['mean_pcc']:.4f}  med={r21['median_pcc']:.4f}")
    print(f"  [SEED {seed}] PCC train={splits[0]}->test={splits[1]}: "
          f"{r12['mean_pcc']:.4f}  med={r12['median_pcc']:.4f}")
    return result


def run_dataset(dataset: str, cfg: dict) -> None:
    a1, a2, meta = cd.load_pair(dataset)
    splits = cd.DATASETS[dataset]["splits"]
    out_dir = Path(cfg["output_root"]) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"DATASET: {dataset}")
    print(f"  splits: {splits}  mode: {meta['mode']}")
    print(f"  slice1: {a1.shape}  slice2: {a2.shape}  genes: {meta['n_genes']}")
    print(f"{'='*60}")

    results = []
    for seed in cfg["seeds"]:
        r = run_seed(a1, a2, cfg, seed=seed, dataset=dataset,
                     splits=splits, seed_dir=out_dir / f"seed_{seed}")
        results.append(r)

    summary = cd.aggregate_seeds(results)
    summary["dataset"] = dataset
    summary["splits"] = list(splits)
    summary["n_cells"] = meta["n_cells"]
    summary["n_genes"] = meta["n_genes"]
    summary["config"] = {
        k: cfg[k] for k in (
            "num_neighbors", "mlp_epochs", "mlp_lr", "mlp_batch_size",
            "gnn_epochs", "gnn_lr", "gnn_batch_size", "gnn_num_neighbors",
            "gnn_attn_heads", "gnn_drop_edge", "gnn_alpha",
            "gnn_hierarchical", "gnn_spatial_clusters",
            "gnn_feature_clusters", "epochs", "lr")
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SUMMARY {dataset} (n={len(results)} seeds)")
    print(f"{'='*60}")
    o = summary["overall_mean_pcc"]
    print(f"  overall_mean_pcc:  {o['mean']:.4f} ± {o['std']:.4f}  (n={o['n']})")
    for k in summary:
        if k.startswith("mean_pcc_train") or k.startswith("median_pcc"):
            v = summary[k]
            print(f"  {k:40s}  {v['mean']:.4f} ± {v['std']:.4f}  (n={v['n']})")
    print(f"  saved -> {out_dir / 'summary.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.seeds is not None:
        cfg["seeds"] = args.seeds
    if args.device is not None:
        cfg["device"] = args.device

    datasets = [args.dataset] if args.dataset else cfg["datasets"]
    for ds in datasets:
        run_dataset(ds, cfg)


if __name__ == "__main__":
    main()
