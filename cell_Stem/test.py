"""cell_Stem test script: compute per-task/dataset MSE and MAE.

Reads saved predictions ``runs/<dataset>/v<N>/<dataset>_fold<k>_seed42/pred.npy``
(produced by ``predict.py``) and the corresponding ground-truth test-half
expression, then computes per-gene and mean MSE/MAE.

Stem trains in ``log2(count+1)`` space, so predictions and targets are both on
that scale.

Usage:
    python test.py --config config.yaml
    python test.py --config config.yaml --dataset hSkin_Melanoma
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import yaml

# Match predict.py's DATASETS registry.
DATASETS = {
    "hSkin_Melanoma": {"mode": "single", "dir": "hSkin_Melanoma", "cell_file": "adata_S1.h5ad"},
    "hColon_Non_diseased": {"mode": "single", "dir": "hColon_Non_diseased", "cell_file": "adata_S1.h5ad"},
    "mouse_Colon": {"mode": "single", "dir": "mouse_Colon", "cell_file": "adata_S1.h5ad"},
    "Human_Breast_Cancer": {
        "mode": "pair",
        "members": [
            {"label": "Rep1", "dir": "Human_Breast_Cancer_Rep1", "cell_file": "adata_Rep1.h5ad"},
            {"label": "Rep2", "dir": "Human_Breast_Cancer_Rep2", "cell_file": "adata_Rep2.h5ad"},
        ],
    },
}


def extract_gene_expr_log2(adata, gene_names):
    """Return ``log2(count+1)`` matrix [N, len(gene_names)]."""
    common = [g for g in gene_names if g in adata.var_names]
    if len(common) == len(gene_names):
        expr = adata[:, gene_names].X
        if hasattr(expr, "toarray"):
            expr = expr.toarray()
        expr = np.asarray(expr, dtype=np.float32)
    else:
        expr_mat = adata[:, common].X
        if hasattr(expr_mat, "toarray"):
            expr_mat = expr_mat.toarray()
        df = pd.DataFrame(expr_mat, index=adata.obs_names, columns=common)
        df = df.reindex(columns=gene_names, fill_value=0.0)
        expr = df.values.astype(np.float32)
    return np.log2(expr + 1.0).astype(np.float32)


def compute_mse(true: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, float]:
    per_gene = np.mean((true - pred) ** 2, axis=0)
    per_gene = np.where(np.isfinite(per_gene), per_gene, 0.0)
    return per_gene, float(np.mean(per_gene))


def compute_mae(true: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, float]:
    per_gene = np.mean(np.abs(true - pred), axis=0)
    per_gene = np.where(np.isfinite(per_gene), per_gene, 0.0)
    return per_gene, float(np.mean(per_gene))


def load_test_truth(ds_name: str, test_half: str, gene_names: list[str], cfg: dict):
    """Load the test-half ground-truth log2(count+1) expression matrix."""
    use_pc = cfg.get("use_processed_cell", False)
    min_counts = int(cfg.get("min_counts", 10))

    if use_pc:
        pc_root = Path(cfg.get("processed_cell_root", "/home/sb202604/cell-benchmark/processed_cell"))
        adata_path = pc_root / ds_name / f"adata_{test_half}.h5ad"
        adata = ad.read_h5ad(adata_path)
        if "raw" in adata.layers:
            adata.X = adata.layers["raw"].copy()
    else:
        proc_root = Path(cfg["processed_root"])
        spec = DATASETS[ds_name]
        if spec["mode"] == "single":
            ds_dir = proc_root / spec["dir"]
            adata = ad.read_h5ad(ds_dir / "cells.h5ad")
            splits = json.load(open(ds_dir / "splits.json"))
            test_idx = np.asarray(splits["spatial_ood"][test_half], dtype=np.int64)
            adata = adata[test_idx].copy()
        else:
            member = next(m for m in spec["members"] if m["label"] == test_half)
            adata = ad.read_h5ad(proc_root / member["dir"] / "cells.h5ad")

    # Apply the same cell filtering as predict.py's preprocess_half.
    if "is_gene" in adata.var.columns:
        adata = adata[:, adata.var["is_gene"].astype(bool).values].copy()
    sc.pp.filter_cells(adata, min_counts=min_counts)
    return extract_gene_expr_log2(adata, gene_names)


def find_latest_version(ds_dir: Path) -> Path | None:
    """Find the latest vN version folder under ds_dir."""
    if not ds_dir.exists():
        return None
    versions = [
        d for d in ds_dir.iterdir()
        if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit()
    ]
    if not versions:
        return None
    latest = max(versions, key=lambda d: int(d.name[1:]))
    return latest


def evaluate_dataset(ds_name: str, output_root: Path, cfg: dict):
    """Evaluate all folds of the latest version for one dataset."""
    ds_dir = output_root / ds_name
    version_dir = find_latest_version(ds_dir)
    if version_dir is None:
        print(f"  No version folder found for {ds_name}, skipping")
        return None
    print(f"  using latest version {version_dir.name}")

    # Load the config that was used for this version.
    version_cfg_path = version_dir / "config.yaml"
    if version_cfg_path.exists():
        with open(version_cfg_path) as f:
            version_cfg = yaml.safe_load(f)
    else:
        version_cfg = cfg

    fold_dirs = sorted([d for d in version_dir.iterdir() if d.is_dir()])
    all_runs = []
    for fold_dir in fold_dirs:
        result_path = fold_dir / "result.json"
        pred_path = fold_dir / "pred.npy"
        if not (result_path.exists() and pred_path.exists()):
            continue

        with open(result_path) as f:
            result = json.load(f)

        pred = np.load(pred_path).astype(np.float32)
        gene_names = result["gene_names"]
        test_half = result["test_half"]
        train_half = result["train_half"]

        true = load_test_truth(ds_name, test_half, gene_names, version_cfg)
        if pred.shape != true.shape:
            raise ValueError(
                f"Shape mismatch in {fold_dir}: pred {pred.shape} vs true {true.shape}"
            )

        per_gene_mse, mean_mse = compute_mse(true, pred)
        per_gene_mae, mean_mae = compute_mae(true, pred)

        all_runs.append({
            "dataset": ds_name,
            "fold": result.get("fold"),
            "seed": result.get("seed"),
            "train_half": train_half,
            "test_half": test_half,
            "n_cells": int(true.shape[0]),
            "n_genes": int(true.shape[1]),
            "mean_mse": mean_mse,
            "mean_mae": mean_mae,
            "per_gene_mse": per_gene_mse.tolist(),
            "per_gene_mae": per_gene_mae.tolist(),
            "gene_names": gene_names,
        })

    if not all_runs:
        return None

    def agg(values):
        arr = np.asarray(values)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "n": len(arr),
            "values": values,
        }

    summary = {
        "dataset": ds_name,
        "version": version_dir.name,
        "n_runs": len(all_runs),
        "overall_mean_mse": agg([r["mean_mse"] for r in all_runs]),
        "overall_mean_mae": agg([r["mean_mae"] for r in all_runs]),
        "directions": {},
        "runs": all_runs,
    }
    for train_half, test_half in set((r["train_half"], r["test_half"]) for r in all_runs):
        dir_runs = [r for r in all_runs
                    if r["train_half"] == train_half and r["test_half"] == test_half]
        key = f"train_{train_half}_test_{test_half}"
        summary["directions"][key] = {
            "mean_mse": agg([r["mean_mse"] for r in dir_runs]),
            "mean_mae": agg([r["mean_mae"] for r in dir_runs]),
        }

    return summary


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate cell_Stem predictions and compute MSE/MAE."
    )
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--output", default="test_results.json")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_root = Path(cfg.get("output_root", "./runs"))
    datasets = [args.dataset] if args.dataset else cfg.get("datasets", [])

    dataset_summaries = {}
    for ds_name in datasets:
        print(f"\n{'='*60}")
        print(f"DATASET: {ds_name}")
        print(f"{'='*60}")
        try:
            summary = evaluate_dataset(ds_name, output_root, cfg)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
        if summary is None:
            print(f"  No results found for {ds_name}")
            continue

        dataset_summaries[ds_name] = summary
        print(f"  overall MSE: {summary['overall_mean_mse']['mean']:.4f} "
              f"± {summary['overall_mean_mse']['std']:.4f}")
        print(f"  overall MAE: {summary['overall_mean_mae']['mean']:.4f} "
              f"± {summary['overall_mean_mae']['std']:.4f}")
        for key, vals in summary["directions"].items():
            print(f"  {key}: "
                  f"MSE={vals['mean_mse']['mean']:.4f} ± {vals['mean_mse']['std']:.4f}, "
                  f"MAE={vals['mean_mae']['mean']:.4f} ± {vals['mean_mae']['std']:.4f}")

    final_output = {
        "config": args.config,
        "output_root": str(output_root),
        "n_datasets": len(dataset_summaries),
        "dataset_summaries": dataset_summaries,
    }
    with open(args.output, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
