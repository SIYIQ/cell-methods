"""cell_NH2ST test script: compute per-direction and aggregated MSE/MAE.

Loads saved predictions from ``runs/<dataset>/seed_*/pred_*.npy`` and the
corresponding ground-truth cell-level expression matrices, then computes
mean ± std of per-gene MSE and MAE across seeds and cross-section directions.

Usage:
    python test.py --config config.yaml
    python test.py --config config.yaml --dataset Human_Breast_Cancer
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cell-benchmark" / "scripts"))
import cell_data as cd


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


def to_dense(a) -> np.ndarray:
    """Convert AnnData X to dense numpy array."""
    if hasattr(a, "toarray"):
        return a.toarray().astype(np.float32)
    return np.asarray(a, dtype=np.float32)


def evaluate_dataset(dataset: str, output_root: Path, seeds: list[int]):
    """Compute MSE/MAE for all seeds and both directions of one dataset."""
    a1, a2, meta = cd.load_pair(dataset)
    splits = cd.DATASETS[dataset]["splits"]
    y1 = to_dense(a1.X)
    y2 = to_dense(a2.X)
    y_by_split = {splits[0]: y1, splits[1]: y2}

    ds_dir = output_root / dataset
    all_dir_results = []

    for seed in seeds:
        seed_dir = ds_dir / f"seed_{seed}"
        if not seed_dir.exists():
            print(f"  WARNING: {seed_dir} not found, skipping seed {seed}")
            continue

        # Two cross-section directions.
        directions = [
            (splits[0], splits[1]),  # train on split1, predict split2
            (splits[1], splits[0]),  # train on split2, predict split1
        ]
        for src, tgt in directions:
            pred_path = seed_dir / f"pred_{tgt}_from_{src}.npy"
            if not pred_path.exists():
                print(f"  WARNING: {pred_path} not found, skipping")
                continue

            pred = np.load(pred_path).astype(np.float32)
            true = y_by_split[tgt]

            if pred.shape != true.shape:
                raise ValueError(
                    f"Shape mismatch in {dataset} seed {seed} {pred_path.name}: "
                    f"pred {pred.shape} vs true {true.shape}"
                )

            per_gene_mse, mean_mse = compute_mse(true, pred)
            per_gene_mae, mean_mae = compute_mae(true, pred)

            all_dir_results.append({
                "dataset": dataset,
                "seed": seed,
                "train": src,
                "test": tgt,
                "n_cells": int(true.shape[0]),
                "n_genes": int(true.shape[1]),
                "mean_mse": mean_mse,
                "mean_mae": mean_mae,
                "per_gene_mse": per_gene_mse.tolist(),
                "per_gene_mae": per_gene_mae.tolist(),
            })

    if not all_dir_results:
        return None

    # Aggregate per direction and overall.
    def agg(values):
        arr = np.asarray(values)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "n": len(arr),
            "values": values,
        }

    overall_mse = [r["mean_mse"] for r in all_dir_results]
    overall_mae = [r["mean_mae"] for r in all_dir_results]

    summary = {
        "dataset": dataset,
        "splits": list(splits),
        "n_cells": meta["n_cells"],
        "n_genes": meta["n_genes"],
        "seeds": seeds,
        "overall_mean_mse": agg(overall_mse),
        "overall_mean_mae": agg(overall_mae),
        "directions": {},
        "runs": all_dir_results,
    }

    for src, tgt in [(splits[0], splits[1]), (splits[1], splits[0])]:
        dir_results = [r for r in all_dir_results
                       if r["train"] == src and r["test"] == tgt]
        if dir_results:
            key = f"train_{src}_test_{tgt}"
            summary["directions"][key] = {
                "mean_mse": agg([r["mean_mse"] for r in dir_results]),
                "mean_mae": agg([r["mean_mae"] for r in dir_results]),
            }

    return summary


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate cell_NH2ST predictions and compute MSE/MAE."
    )
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--output", default="test_results.json")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seeds = args.seeds if args.seeds is not None else cfg.get("seeds", [42, 43, 44])
    output_root = Path(cfg.get("output_root", "./runs"))
    datasets = [args.dataset] if args.dataset else cfg.get("datasets", [])

    dataset_summaries = {}
    all_runs = []

    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"DATASET: {ds}")
        print(f"{'='*60}")
        summary = evaluate_dataset(ds, output_root, seeds)
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
