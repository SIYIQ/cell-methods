"""cell_SpatialEx test script: compute per-task/dataset MSE and MAE.

Reads saved predictions (``pred_S1_from_S2.npy``, ``pred_S2_from_S1.npy``)
from ``runs/<dataset>/`` or ``runs_native/<dataset>/`` outputs and computes
per-gene + mean MSE/MAE for each cross-section direction. Aggregates multi-seed
runs when ``seed_*/`` subdirectories exist.

Supports three result layouts produced by this repo:
  1. ``runs/<dataset>/``                -> ``run_pair.py`` single-seed output
  2. ``runs_native/<dataset>_<mode>/``  -> ``run_native.py`` multi-seed output
  3. ``runs_native/<slice1>__<slice2>/``-> ``run_pair_native.py`` multi-seed output

Usage:
    python test.py
    python test.py --runs-root runs --dataset hSkin_Melanoma
    python test.py --runs-root runs_native --dataset hSkin_Melanoma_chessboard
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import anndata as ad
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cell-benchmark" / "scripts"))

PROCESSED_ROOT = Path("/home/sb202604/cell-benchmark/processed")


def to_dense(a) -> np.ndarray:
    """Convert AnnData X to dense numpy array."""
    if hasattr(a, "toarray"):
        return a.toarray().astype(np.float32)
    return np.asarray(a, dtype=np.float32)


def compute_mse(true: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, float]:
    per_gene = np.mean((true - pred) ** 2, axis=0)
    per_gene = np.where(np.isfinite(per_gene), per_gene, 0.0)
    return per_gene, float(np.mean(per_gene))


def compute_mae(true: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, float]:
    per_gene = np.mean(np.abs(true - pred), axis=0)
    per_gene = np.where(np.isfinite(per_gene), per_gene, 0.0)
    return per_gene, float(np.mean(per_gene))


def split_S1_S2_by_chessboard(adata, block_um: float = 200.0):
    """Chessboard split matching run_native.py."""
    x = adata.obs["x_centroid"].to_numpy(np.float64)
    y = adata.obs["y_centroid"].to_numpy(np.float64)
    xb = (x - x.min()) // block_um
    yb = (y - y.min()) // block_um
    parity = (xb.astype(np.int64) + yb.astype(np.int64)) % 2
    return adata[parity == 0].copy(), adata[parity != 0].copy()


def split_S1_S2_by_x_median(adata):
    """Median-x split matching run_native.py."""
    x = adata.obs["x_centroid"].to_numpy(np.float64)
    med = float(np.median(x))
    return adata[x < med].copy(), adata[x >= med].copy()


def _match_split(adata_full, target_n1: int, target_n2: int):
    """Try x_median / chessboard splits and return the one matching target shapes."""
    candidates = [
        ("x_median", split_S1_S2_by_x_median(adata_full)),
        ("chessboard_200", split_S1_S2_by_chessboard(adata_full, block_um=200.0)),
    ]
    for name, (a1, a2) in candidates:
        if a1.n_obs == target_n1 and a2.n_obs == target_n2:
            print(f"    matched ground-truth split: {name} "
                  f"(S1={a1.n_obs}, S2={a2.n_obs})")
            return a1, a2
    # Fallback: return x_median and let caller raise if shapes still mismatch.
    return candidates[0][1]


def load_ground_truth(result_dir: Path, result: dict, pred_n1: int, pred_n2: int):
    """Resolve S1/S2 ground-truth expression matrices for a result directory."""
    dataset = result.get("dataset") or result.get("slice1", "").split("_")[0]

    # Layout 1: processed paired adata (run_pair.py output under runs/)
    proc_dir = PROCESSED_ROOT / dataset / "spatialex"
    s1_path = proc_dir / "S1.h5ad"
    s2_path = proc_dir / "S2.h5ad"
    if s1_path.exists() and s2_path.exists():
        a1 = ad.read_h5ad(s1_path)
        a2 = ad.read_h5ad(s2_path)
        if a1.n_obs == pred_n1 and a2.n_obs == pred_n2:
            return to_dense(a1.X), to_dense(a2.X), list(a1.var_names)
        print(f"  processed S1/S2 shape ({a1.n_obs}, {a2.n_obs}) does not match "
              f"predictions ({pred_n1}, {pred_n2}); falling back to cache")

    # Layout 2/3: cache in the result directory
    cache_dir = result_dir / "cache"
    if cache_dir.exists():
        # Pair of full slices
        pair_files = sorted(cache_dir.glob("*_adata.h5ad"))
        if len(pair_files) >= 2:
            a1 = ad.read_h5ad(pair_files[0])
            a2 = ad.read_h5ad(pair_files[1])
            return to_dense(a1.X), to_dense(a2.X), list(a1.var_names)

        # Single section that needs re-splitting
        full_adata_path = cache_dir / "full_adata.h5ad"
        if full_adata_path.exists():
            adata_full = ad.read_h5ad(full_adata_path)
            a1, a2 = _match_split(adata_full, pred_n1, pred_n2)
            return to_dense(a1.X), to_dense(a2.X), list(a1.var_names)

    raise FileNotFoundError(
        f"Could not find ground truth for {result_dir}. "
        f"Tried {proc_dir}/S1.h5ad and {cache_dir}."
    )


def evaluate_one_dir(result_dir: Path):
    """Evaluate a single run directory (may contain seed_* subdirs)."""
    result_path = result_dir / "result.json"
    seed_dirs = sorted([d for d in result_dir.iterdir()
                        if d.is_dir() and d.name.startswith("seed_")])

    if seed_dirs:
        items = []
        for sd in seed_dirs:
            rp = sd / "result.json"
            if rp.exists():
                items.append((sd, json.load(open(rp))))
    elif result_path.exists():
        items = [(result_dir, json.load(open(result_path)))]
    else:
        return None

    all_dir_runs = []
    for run_dir, result in items:
        # Load predictions first so we can match the right ground-truth split.
        pred_s1 = np.load(run_dir / "pred_S1_from_S2.npy").astype(np.float32)
        pred_s2 = np.load(run_dir / "pred_S2_from_S1.npy").astype(np.float32)

        y1, y2, gene_names = load_ground_truth(
            result_dir, result, pred_n1=pred_s1.shape[0], pred_n2=pred_s2.shape[0]
        )

        if pred_s1.shape != y1.shape:
            raise ValueError(
                f"Shape mismatch in {run_dir}: pred_S1 {pred_s1.shape} vs true {y1.shape}"
            )
        if pred_s2.shape != y2.shape:
            raise ValueError(
                f"Shape mismatch in {run_dir}: pred_S2 {pred_s2.shape} vs true {y2.shape}"
            )

        # Determine direction names.
        # result may use S1/S2 or slice1/slice2 keys.
        splits = ("S1", "S2")
        if "slice1" in result and "slice2" in result:
            splits = (result["slice1"], result["slice2"])

        for src, tgt, true, pred in [
            (splits[1], splits[0], y1, pred_s1),  # train S2 -> test S1
            (splits[0], splits[1], y2, pred_s2),  # train S1 -> test S2
        ]:
            per_gene_mse, mean_mse = compute_mse(true, pred)
            per_gene_mae, mean_mae = compute_mae(true, pred)
            all_dir_runs.append({
                "run_dir": str(run_dir),
                "seed": result.get("seed"),
                "train": src,
                "test": tgt,
                "n_cells": int(true.shape[0]),
                "n_genes": int(true.shape[1]),
                "mean_mse": mean_mse,
                "mean_mae": mean_mae,
                "per_gene_mse": per_gene_mse.tolist(),
                "per_gene_mae": per_gene_mae.tolist(),
                "gene_names": gene_names,
            })

    def agg(values):
        arr = np.asarray(values)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "n": len(arr),
            "values": values,
        }

    summary = {
        "result_dir": str(result_dir),
        "name": result_dir.name,
        "n_runs": len(all_dir_runs),
        "overall_mean_mse": agg([r["mean_mse"] for r in all_dir_runs]),
        "overall_mean_mae": agg([r["mean_mae"] for r in all_dir_runs]),
        "directions": {},
        "runs": all_dir_runs,
    }
    for src, tgt in [(splits[1], splits[0]), (splits[0], splits[1])]:
        dir_runs = [r for r in all_dir_runs if r["train"] == src and r["test"] == tgt]
        if dir_runs:
            key = f"train_{src}_test_{tgt}"
            summary["directions"][key] = {
                "mean_mse": agg([r["mean_mse"] for r in dir_runs]),
                "mean_mae": agg([r["mean_mae"] for r in dir_runs]),
            }

    return summary


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate cell_SpatialEx predictions and compute MSE/MAE."
    )
    ap.add_argument("--runs-root", default="runs",
                    help="Root directory containing result subdirectories")
    ap.add_argument("--dataset", default=None,
                    help="Evaluate a single dataset/subdirectory only")
    ap.add_argument("--output", default="test_results.json")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    if args.dataset:
        candidates = [runs_root / args.dataset]
    else:
        candidates = [d for d in runs_root.iterdir() if d.is_dir()]

    dataset_summaries = {}
    for cand in sorted(candidates):
        if not (cand / "result.json").exists() and not any(
            (cand / sd).exists() for sd in ("seed_42", "seed_0")
        ):
            continue
        print(f"\n{'='*60}")
        print(f"EVALUATING: {cand.name}")
        print(f"{'='*60}")
        try:
            summary = evaluate_one_dir(cand)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
        if summary is None:
            print(f"  No results found in {cand}")
            continue

        dataset_summaries[cand.name] = summary
        print(f"  overall MSE: {summary['overall_mean_mse']['mean']:.4f} "
              f"± {summary['overall_mean_mse']['std']:.4f}")
        print(f"  overall MAE: {summary['overall_mean_mae']['mean']:.4f} "
              f"± {summary['overall_mean_mae']['std']:.4f}")
        for key, vals in summary["directions"].items():
            print(f"  {key}: "
                  f"MSE={vals['mean_mse']['mean']:.4f} ± {vals['mean_mse']['std']:.4f}, "
                  f"MAE={vals['mean_mae']['mean']:.4f} ± {vals['mean_mae']['std']:.4f}")

    final_output = {
        "runs_root": str(runs_root),
        "n_datasets": len(dataset_summaries),
        "dataset_summaries": dataset_summaries,
    }
    with open(args.output, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
