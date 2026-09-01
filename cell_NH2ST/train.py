"""cell_NH2ST training: cell-level NGHist2ST.

Reads the shared cell-level data interface from
/home/sb202604/cell-benchmark/processed_cell/, trains one CellNGHist2ST per
(slice1, slice2, seed), evaluates per-cell PCC on the held-out slice, and
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
import yaml
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cell-benchmark" / "scripts"))
import cell_data as cd

from model import CellNGHist2ST
from dataset import CellNH2STDataset, collate_fn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_samples = 0
    for x, exp, x_neighbor, neighbor_exp, _ in dataloader:
        x = x.to(device)
        exp = exp.to(device)
        x_neighbor = x_neighbor.to(device)
        neighbor_exp = neighbor_exp.to(device)

        optimizer.zero_grad()
        outputs = model(x, exp, x_neighbor, neighbor_exp)
        loss = model.compute_loss(outputs, exp)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        n_samples += x.size(0)
    return total_loss / n_samples if n_samples > 0 else 0.0


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, exp, x_neighbor, neighbor_exp, _ in dataloader:
            x = x.to(device)
            exp = exp.to(device)
            x_neighbor = x_neighbor.to(device)
            neighbor_exp = neighbor_exp.to(device)
            outputs = model(x, exp, x_neighbor, neighbor_exp)
            all_preds.append(outputs[5].cpu())
            all_targets.append(exp.cpu())
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    return model.compute_metrics(preds, targets)


def train_one_direction(src_adata, tgt_adata, cfg, seed, log_prefix=""):
    """Train on src slice and evaluate on tgt slice."""
    set_seed(seed)
    device = cfg["device"]

    src_ds = CellNH2STDataset(
        src_adata,
        neighbor_k=cfg["neighbor_k"],
        num_neighbors=cfg["num_neighbors"],
    )
    tgt_ds = CellNH2STDataset(
        tgt_adata,
        neighbor_k=cfg["neighbor_k"],
        num_neighbors=cfg["num_neighbors"],
    )

    train_loader = DataLoader(
        src_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    tgt_loader = DataLoader(
        tgt_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = CellNGHist2ST(
        num_genes=src_adata.n_vars,
        emb_dim=cfg["emb_dim"],
        depth1=cfg["depth1"],
        num_heads1=cfg["num_heads1"],
        mlp_ratio1=cfg["mlp_ratio1"],
        dropout1=cfg["dropout1"],
        temperature1=cfg["temperature1"],
        temperature2=cfg["temperature2"],
        loss_ratio1=cfg["loss_ratio1"],
        loss_ratio2=cfg["loss_ratio2"],
    ).to(device)

    print(f"  {log_prefix} model: emb_dim={cfg['emb_dim']} "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["wd"])

    epochs = cfg["epochs"]

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"  {log_prefix} epoch {epoch}/{epochs}  "
              f"loss={train_loss:.4f}  ({time.time()-t0:.0f}s)")

    # ---- inference on target ----
    tgt_metrics = evaluate(model, tgt_loader, device)
    pred_np = None
    true_np = None
    with torch.no_grad():
        all_preds = []
        all_true = []
        for x, exp, x_neighbor, neighbor_exp, _ in tgt_loader:
            x = x.to(device)
            exp = exp.to(device)
            x_neighbor = x_neighbor.to(device)
            neighbor_exp = neighbor_exp.to(device)
            outputs = model(x, exp, x_neighbor, neighbor_exp)
            all_preds.append(outputs[5].cpu())
            all_true.append(exp.cpu())
        pred_np = torch.cat(all_preds, dim=0).numpy()
        true_np = torch.cat(all_true, dim=0).numpy()
    metrics = cd.compute_pcc_metrics(true_np, pred_np)

    del model, optimizer
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
            "emb_dim", "depth1", "num_heads1", "mlp_ratio1", "dropout1",
            "temperature1", "temperature2", "loss_ratio1", "loss_ratio2",
            "neighbor_k", "num_neighbors", "batch_size", "epochs", "lr",
            "wd")
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
