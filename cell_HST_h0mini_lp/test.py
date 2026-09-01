"""cell_HST_h0mini_lp test script: compute per-direction and aggregated MSE/MAE.

Loads saved predictions from ``runs/<dataset>/v<N>/<dataset>_fold<k>_seed42/pred.npy``
(produced by ``predict.py``) and the corresponding ground-truth test-half
expression (after the same ``preprocess_half`` filtering/normalization/log1p),
then computes mean ± std of per-gene MSE and MAE across folds.

Usage:
    python test.py --config config.yaml
    python test.py --config config.yaml --dataset Human_Breast_Cancer
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cell-benchmark" / "scripts"))
import cell_data as cd

# Reuse predict.py's data loading and utils.preprocess_half logic.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import DATASETS, load_dataset
from utils import extract_gene_expr


def compute_mse(true: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-gene MSE and mean across genes."""
    per_gene = np.mean((true - pred) ** 2, axis=0)
    per_gene = np.where(np.isfinite(per_gene), per_gene, 0.0)
    return per_gene, float(np.mean(per_gene))


def compute_mae(true: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-gene MAE and mean across genes."""
    per_gene = np.mean(np.abs(true - pred), axis=0)
    per_gene = np.where(np.isfinite(per_gene), per_gene, 0.0)
    return per_gene, float(np.mean(per_gene))


def preprocess_truth(adata_raw: ad.AnnData, gene_names: list[str], min_counts: int = 10):
    """Apply the same filtering/normalization/log1p as predict.py's preprocess_half.

    Returns only the expression matrix; image-feature alignment is ignored
    because the saved ``pred.npy`` is already aligned with the surviving cells.
    """
    if "is_gene" in adata_raw.var.columns:
        sub = adata_raw[:, adata_raw.var["is_gene"].astype(bool).values].copy()
    else:
        sub = adata_raw.copy()

    sc.pp.filter_cells(sub, min_counts=min_counts)
    if hasattr(sub.X, "toarray"):
        sub.X = sub.X.toarray()
    sc.pp.normalize_total(sub, inplace=True)
    sc.pp.log1p(sub)
    return extract_gene_expr(sub, gene_names)


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
    return max(versions, key=lambda d: int(d.name[1:]))


def evaluate_dataset(dataset: str, output_root: Path, cfg: dict):
    """Compute MSE/MAE for all folds of the latest version for one dataset."""
    ds_dir = output_root / dataset
    version_dir = find_latest_version(ds_dir)
    if version_dir is None:
        print(f"  No version folder found for {dataset}, skipping")
        return None
    print(f"  using latest version {version_dir.name}")

    # Load the config that was actually used for this version.
    version_cfg_path = version_dir / "config.yaml"
    if version_cfg_path.exists():
        with open(version_cfg_path) as f:
            version_cfg = yaml.safe_load(f)
    else:
        version_cfg = cfg

    # Load raw-count halves the same way predict.py does.
    ds_info = load_dataset(dataset, version_cfg)
    halves = {h["label"]: h["adata"] for h in ds_info["halves"]}
    splits = cd.DATASETS[dataset]["splits"]

    fold_dirs = sorted([d for d in version_dir.iterdir() if d.is_dir()])
    all_runs = []
    for fold_dir in fold_dirs:
        result_path = fold_dir / "result.json"
        pred_path = fold_dir / "pred.npy"
        if not (result_path.exists() and pred_path.exists()):
            continue

        with open(result_path) as f:
            result = json.load(f)

        gene_names = result.get("gene_names")
        train_half = result["train_half"]
        test_half = result["test_half"]
        min_counts = int(version_cfg.get("min_counts", 10))

        pred = np.load(pred_path).astype(np.float32)
        true = preprocess_truth(halves[test_half], gene_names, min_counts=min_counts)

        if pred.shape != true.shape:
            raise ValueError(
                f"Shape mismatch in {fold_dir}: pred {pred.shape} vs true {true.shape}"
            )

        per_gene_mse, mean_mse = compute_mse(true, pred)
        per_gene_mae, mean_mae = compute_mae(true, pred)

        all_runs.append({
            "dataset": dataset,
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
            "gene_names": list(gene_names) if gene_names is not None else None,
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

    meta = cd.load_pair(dataset)[2]
    summary = {
        "dataset": dataset,
        "version": version_dir.name,
        "splits": list(splits),
        "n_cells": meta["n_cells"],
        "n_genes": int(all_runs[0]["n_genes"]),
        "seeds": version_cfg.get("seeds", cfg.get("seeds", [42])),
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
        description="Aggregate cell_HST_h0mini_lp predictions and compute MSE/MAE."
    )
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--output", default="test_results.json")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_root = Path(cfg.get("output_root", "./runs"))
    datasets = [args.dataset] if args.dataset else cfg.get("datasets", [])

    dataset_summaries = {}
    all_runs = []

    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"DATASET: {ds}")
        print(f"{'='*60}")
        try:
            summary = evaluate_dataset(ds, output_root, cfg)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
        if summary is None:
            print(f"  No results found for {ds}")
            continue

        dataset_summaries[ds] = summary
        all_runs.extend(summary["runs"])

        print(f"  overall_mean_mse: {summary['overall_mean_mse']['mean']:.4f} "
              f"± {summary['overall_mean_mse']['std']:.4f}")
        print(f"  overall_mean_mae: {summary['overall_mean_mae']['mean']:.4f} "
              f"± {summary['overall_mean_mae']['std']:.4f}")
        for key, vals in summary["directions"].items():
            print(f"  {key}: "
                  f"MSE={vals['mean_mse']['mean']:.4f} ± {vals['mean_mse']['std']:.4f}, "
                  f"MAE={vals['mean_mae']['mean']:.4f} ± {vals['mean_mae']['std']:.4f}")

    final_output = {
        "config": args.config,
        "output_root": str(output_root),
        "n_runs": len(all_runs),
        "dataset_summaries": dataset_summaries,
    }
    with open(args.output, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
