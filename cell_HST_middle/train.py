"""cell_HST_middle training: cell-level HeteroST + MGM.

Reads /home/sb202604/cell-benchmark/processed_cell/, trains one CellHST per
(slice1, slice2, seed), evaluates per-cell PCC on the held-out slice, and
aggregates seeds into a Table-2-style summary.

Layout produced (per dataset):
    runs/<dataset>/
    ├── seed_42/result.json
    ├── seed_43/result.json
    ├── seed_44/result.json
    └── summary.json

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
import torch.nn as nn
import yaml

# Shared cell-level data interface.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cell-benchmark" / "scripts"))
import cell_data as cd

from model import CellHST
from utils import build_cell_hetero_data, get_gene_names, sample_mgm_mask


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


def prepare_slice_tensors(adata, device, num_neighbors: int,
                          morph_top_k: int, morph_sim_thresh: float):
    """Build the per-slice tensors + HeteroData for the model."""
    x_feat = to_torch(adata.obsm["he"]).to(device)
    y_expr = to_torch(adata.X).to(device)

    H = cd.build_cell_hypergraph(adata, num_neighbors=num_neighbors,
                                 return_type="crs")
    spatial_ei = cd.hypergraph_to_pyg_edges(H).to(device)

    hdata = build_cell_hetero_data(x_feat, spatial_ei,
                                   morph_top_k=morph_top_k,
                                   morph_sim_thresh=morph_sim_thresh)
    return {"x_feat": x_feat, "y_expr": y_expr, "hdata": hdata}


def compute_mask_ratio(epoch: int, total_epochs: int, mgm_cfg: dict) -> float:
    """Linear ramp from min_ratio to max_ratio over the first
    ramp_epochs_frac of training."""
    if not mgm_cfg.get("enabled", False):
        return 0.0
    ramp_end = int(mgm_cfg.get("ramp_epochs_frac", 0.5) * total_epochs)
    if epoch >= ramp_end:
        return mgm_cfg["max_ratio"]
    if ramp_end == 0:
        return mgm_cfg["max_ratio"]
    frac = epoch / ramp_end
    return mgm_cfg["min_ratio"] + frac * (mgm_cfg["max_ratio"] - mgm_cfg["min_ratio"])


def train_one_direction(src_data, tgt_data, cfg, seed, log_prefix=""):
    set_seed(seed)
    device = src_data["x_feat"].device

    model = CellHST(
        in_dim=src_data["x_feat"].shape[1],
        n_genes=src_data["y_expr"].shape[1],
        hidden_dim=cfg["hidden_dim"],
        heads=cfg["heads"],
        dropout=cfg["dropout"],
        n_hgt_layers=cfg["n_hgt_layers"],
        pred_head_depth=cfg["pred_head_depth"],
    ).to(device)

    opt = torch.optim.Adam(model.parameters(),
                           lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"])

    loss_fn = nn.MSELoss() if cfg["loss"] == "mse" else None
    if loss_fn is None:
        def pearson_loss(pred, true):
            pred_c = pred - pred.mean(0, keepdim=True)
            true_c = true - true.mean(0, keepdim=True)
            num = (pred_c * true_c).sum(0)
            den = pred_c.norm(dim=0) * true_c.norm(dim=0) + 1e-8
            return (1 - num / den).mean()
        loss_fn = pearson_loss

    print(f"  {log_prefix} model: hidden={cfg['hidden_dim']} "
          f"hgt_layers={cfg['n_hgt_layers']} heads={cfg['heads']}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    N_src = src_data["y_expr"].shape[0]
    mgm_cfg = cfg.get("mgm", {"enabled": False})

    t0 = time.time()
    for epoch in range(cfg["epochs"]):
        model.train()
        opt.zero_grad()

        # Build MGM mask for this epoch
        mask_ratio = compute_mask_ratio(epoch, cfg["epochs"], mgm_cfg)
        if mgm_cfg.get("enabled", False):
            mask = sample_mgm_mask(N_src, mask_ratio, device)
        else:
            mask = None

        # Forward: provide both gene_expr (true) and mask. Masked cells fall
        # back to img_to_gene; unmasked cells condition on real gene_expr.
        pred = model(src_data["x_feat"], src_data["hdata"],
                     gene_expr=src_data["y_expr"], mask=mask)

        # Loss only on masked cells in MGM mode; otherwise on all cells.
        if mgm_cfg.get("enabled", False) and mask is not None and mask.any():
            loss = loss_fn(pred[mask], src_data["y_expr"][mask])
        else:
            loss = loss_fn(pred, src_data["y_expr"])

        loss.backward()
        opt.step()

        if (epoch + 1) % max(1, cfg["epochs"] // 10) == 0 or epoch == 0:
            print(f"  {log_prefix} epoch {epoch+1:>4d}/{cfg['epochs']}  "
                  f"loss={loss.item():.4f}  mask={mask_ratio:.2f}  "
                  f"({time.time()-t0:.0f}s)")

    print(f"  {log_prefix} training done in {time.time()-t0:.1f}s")

    # ---- inference on target slice ----
    # At inference, gene_expr is unknown for the target -> all cells go via
    # img_to_gene (mask=all-True equivalent). Pass gene_expr=None to use that
    # path directly (clean test-time semantics).
    model.eval()
    with torch.no_grad():
        pred_tgt = model(tgt_data["x_feat"], tgt_data["hdata"],
                         gene_expr=None, mask=None)
    pred_np = pred_tgt.cpu().numpy()
    true_np = tgt_data["y_expr"].cpu().numpy()
    metrics = cd.compute_pcc_metrics(true_np, pred_np)

    del model, opt, pred_tgt
    torch.cuda.empty_cache()
    return {
        "mean_pcc": metrics["mean_pcc"],
        "median_pcc": metrics["median_pcc"],
        "pred": pred_np,
        "per_gene_pcc": metrics["per_gene"],
    }


def run_seed(a1, a2, cfg, seed: int, dataset: str, splits, seed_dir: Path,
             gene_names: list[str]) -> dict:
    seed_dir.mkdir(parents=True, exist_ok=True)
    if (seed_dir / "result.json").exists():
        print(f"  seed {seed}: result.json exists, skipping")
        with open(seed_dir / "result.json") as f:
            return json.load(f)

    device = cfg["device"]
    print(f"\n  [build tensors] {splits[0]}  ({a1.shape})")
    t = time.time()
    d1 = prepare_slice_tensors(a1, device,
                               num_neighbors=cfg["num_neighbors"],
                               morph_top_k=cfg["morph_top_k"],
                               morph_sim_thresh=cfg["morph_sim_thresh"])
    print(f"    done in {time.time()-t:.1f}s")
    print(f"  [build tensors] {splits[1]}  ({a2.shape})")
    t = time.time()
    d2 = prepare_slice_tensors(a2, device,
                               num_neighbors=cfg["num_neighbors"],
                               morph_top_k=cfg["morph_top_k"],
                               morph_sim_thresh=cfg["morph_sim_thresh"])
    print(f"    done in {time.time()-t:.1f}s")

    print(f"\n  [SEED {seed}] train={splits[1]} -> test={splits[0]}")
    r21 = train_one_direction(d2, d1, cfg, seed=seed,
                              log_prefix=f"[t={splits[1]}/e={splits[0]}]")

    print(f"\n  [SEED {seed}] train={splits[0]} -> test={splits[1]}")
    r12 = train_one_direction(d1, d2, cfg, seed=seed,
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
        "gene_names": gene_names,
        "n_genes": int(a1.n_vars),
        "n_cells": {splits[0]: int(a1.n_obs), splits[1]: int(a2.n_obs)},
        "epochs": cfg["epochs"],
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

    # Subset to canonical HVG panel if requested.
    n_top_hvgs = cfg.get("n_top_hvgs")
    if n_top_hvgs is not None:
        gene_names = get_gene_names(
            dataset, n_top_hvgs,
            Path(cfg.get("processed_root", "/home/sb202604/cell-benchmark/processed"))
        )
        a1 = a1[:, gene_names].copy()
        a2 = a2[:, gene_names].copy()
        assert a1.n_vars == len(gene_names) and a2.n_vars == len(gene_names), \
            "HVG gene subset mismatch"
    else:
        gene_names = list(a1.var_names)

    print(f"\n{'='*60}")
    print(f"DATASET: {dataset}")
    print(f"  splits: {splits}  mode: {meta['mode']}")
    print(f"  slice1: {a1.shape}  slice2: {a2.shape}  genes: {a1.n_vars}")
    print(f"{'='*60}")

    results = []
    for seed in cfg["seeds"]:
        r = run_seed(a1, a2, cfg, seed=seed, dataset=dataset,
                     splits=splits, seed_dir=out_dir / f"seed_{seed}",
                     gene_names=gene_names)
        results.append(r)

    summary = cd.aggregate_seeds(results)
    summary["dataset"] = dataset
    summary["splits"] = list(splits)
    summary["n_cells"] = meta["n_cells"]
    summary["n_genes"] = int(a1.n_vars)
    summary["gene_names"] = gene_names
    summary["n_top_hvgs"] = n_top_hvgs
    summary["config"] = {
        k: cfg[k] for k in (
            "hidden_dim", "n_hgt_layers", "heads", "dropout",
            "num_neighbors", "morph_top_k", "morph_sim_thresh",
            "epochs", "lr", "loss", "n_top_hvgs")
    }
    summary["mgm"] = cfg.get("mgm", {})
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
