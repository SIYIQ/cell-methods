"""cell_STFlow training: cell-level STFlow flow-matching denoiser.

Reads the shared cell-level data interface from
/home/sb202604/cell-benchmark/processed_cell/, trains one CellSTFlow per
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cell-benchmark" / "scripts"))
import cell_data as cd

from model import CellSTFlow


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_torch(a) -> torch.Tensor:
    """AnnData X / obsm -> dense float32 torch tensor."""
    import scipy.sparse as sp
    if isinstance(a, np.ndarray):
        return torch.from_numpy(a.astype(np.float32, copy=False))
    if sp.issparse(a):
        return torch.from_numpy(a.toarray().astype(np.float32))
    return torch.from_numpy(np.asarray(a, dtype=np.float32))


def prepare_slice_tensors(adata, device):
    """Build the per-slice tensors that CellSTFlow consumes."""
    x_feat = to_torch(adata.obsm["he"]).to(device)
    y_expr = to_torch(adata.X).to(device)
    coords = to_torch(adata.obsm["spatial"]).to(device)
    return {"x_feat": x_feat, "y_expr": y_expr, "coords": coords}


def train_one_direction(src_data, tgt_data, cfg, seed, log_prefix=""):
    """Train one CellSTFlow on ``src_data`` and evaluate on ``tgt_data``."""
    set_seed(seed)
    device = src_data["x_feat"].device

    model = CellSTFlow(
        n_genes=src_data["y_expr"].shape[1],
        feature_dim=src_data["x_feat"].shape[1],
        hidden_dim=cfg["hidden_dim"],
        pairwise_hidden_dim=cfg["pairwise_hidden_dim"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        dropout=cfg["dropout"],
        attn_dropout=cfg["attn_dropout"],
        n_neighbors=cfg["n_neighbors"],
        activation=cfg["activation"],
        n_sample_steps=cfg["n_sample_steps"],
        prior_sampler=cfg["prior_sampler"],
        zinb_logits=cfg["zinb_logits"],
        zinb_total_count=cfg["zinb_total_count"],
        zinb_zi_logits=cfg["zinb_zi_logits"],
    ).to(device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    print(f"  {log_prefix} model: hidden={cfg['hidden_dim']} "
          f"layers={cfg['n_layers']} heads={cfg['n_heads']} "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    t0 = time.time()
    for epoch in range(cfg["epochs"]):
        model.train()
        opt.zero_grad()
        _, loss = model.train_step(
            src_data["x_feat"], src_data["y_expr"], src_data["coords"]
        )
        loss.backward()
        opt.step()
        if (epoch + 1) % max(1, cfg["epochs"] // 10) == 0 or epoch == 0:
            print(f"  {log_prefix} epoch {epoch+1:>4d}/{cfg['epochs']}  "
                  f"loss={loss.item():.4f}  ({time.time()-t0:.0f}s)")
    print(f"  {log_prefix} training done in {time.time()-t0:.1f}s")

    # ---- inference on target slice ----
    model.eval()
    with torch.no_grad():
        pred_tgt = model.predict(tgt_data["x_feat"], tgt_data["coords"])
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


def run_seed(a1, a2, cfg, seed: int, dataset: str, splits, seed_dir: Path) -> dict:
    """Run both cross-section directions for one seed."""
    seed_dir.mkdir(parents=True, exist_ok=True)
    if (seed_dir / "result.json").exists():
        print(f"  seed {seed}: result.json exists, skipping")
        with open(seed_dir / "result.json") as f:
            return json.load(f)

    device = cfg["device"]
    print(f"\n  [build tensors] {splits[0]}  ({a1.shape})")
    t = time.time()
    d1 = prepare_slice_tensors(a1, device)
    print(f"    done in {time.time()-t:.1f}s")
    print(f"  [build tensors] {splits[1]}  ({a2.shape})")
    t = time.time()
    d2 = prepare_slice_tensors(a2, device)
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
            "hidden_dim", "pairwise_hidden_dim", "n_layers", "n_heads",
            "dropout", "attn_dropout", "n_neighbors", "n_sample_steps",
            "prior_sampler", "epochs", "lr", "loss")
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
