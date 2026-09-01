"""UNI single-cell Xenium linear-probe evaluation.

Data layout expected (per dataset):
  /home/sb202604/cell-benchmark/processed/<dataset>/
    cells.h5ad   raw-count AnnData with var['is_gene']
    patches.npy  uint8 H&E patches, shape (N, 3, 224, 224)
    splits.json  {'spatial_ood': {'S1': [...], 'S2': [...]}}

For Human_Breast_Cancer_Rep1/Rep2, use mode='pair' in config.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
import yaml

from utils import (
    load_uni,
    load_or_encode_features,
    load_cached_he_features,
    preprocess_half,
    select_top_hvgs_official,
    evaluate_fold,
    _filter_control_probes,
)

warnings.filterwarnings("ignore", message="Received a view of an AnnData")


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
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


def load_half(processed_root: Path, ds_spec: dict, member: dict) -> dict:
    """Load one half (single split or pair member) and its H&E features."""
    half_dir = processed_root / member["dir"]
    print(f"  load {member['label']} from {half_dir}")
    adata = sc.read_h5ad(half_dir / "cells.h5ad")
    return {"label": member["label"], "dir": member["dir"], "adata": adata}


def load_half_processed_cell(processed_cell_root: Path, ds_name: str, member: dict) -> dict:
    """Load one half from the shared cell-level interface."""
    half_file = processed_cell_root / ds_name / member["cell_file"]
    print(f"  load {member['label']} from {half_file}")
    adata = sc.read_h5ad(half_file)
    # select_top_hvgs_official expects raw counts in adata.X.
    if "raw" in adata.layers:
        adata.X = adata.layers["raw"].copy()
    return {"label": member["label"], "dir": member["dir"], "adata": adata, "cell_file": member["cell_file"]}


def load_dataset(ds_name: str, cfg: dict):
    """Return a dict with two halves: {label, adata_raw}."""
    spec = DATASETS[ds_name]

    if cfg.get("use_processed_cell", False):
        processed_cell_root = Path(cfg.get("processed_cell_root", "/home/sb202604/cell-benchmark/processed_cell"))
        if spec["mode"] == "single":
            ds_dir = processed_cell_root / spec["dir"]
            adata_s1 = sc.read_h5ad(ds_dir / "adata_S1.h5ad")
            adata_s2 = sc.read_h5ad(ds_dir / "adata_S2.h5ad")
            for adata in (adata_s1, adata_s2):
                if "raw" in adata.layers:
                    adata.X = adata.layers["raw"].copy()
            return {
                "mode": "single",
                "name": ds_name,
                "halves": [
                    {"label": "S1", "dir": spec["dir"], "adata": adata_s1, "cell_file": "adata_S1.h5ad"},
                    {"label": "S2", "dir": spec["dir"], "adata": adata_s2, "cell_file": "adata_S2.h5ad"},
                ],
            }
        else:
            halves = [load_half_processed_cell(processed_cell_root, ds_name, m) for m in spec["members"]]
            return {"mode": "pair", "name": ds_name, "halves": halves}

    processed_root = Path(cfg["processed_root"])
    if spec["mode"] == "single":
        ds_dir = processed_root / spec["dir"]
        adata = sc.read_h5ad(ds_dir / "cells.h5ad")
        splits = json.load(open(ds_dir / "splits.json"))
        s1_idx = np.asarray(splits["spatial_ood"]["S1"], dtype=np.int64)
        s2_idx = np.asarray(splits["spatial_ood"]["S2"], dtype=np.int64)
        return {
            "mode": "single",
            "name": ds_name,
            "halves": [
                {"label": "S1", "dir": spec["dir"], "adata": adata[s1_idx].copy(), "cell_idx": s1_idx},
                {"label": "S2", "dir": spec["dir"], "adata": adata[s2_idx].copy(), "cell_idx": s2_idx},
            ],
        }
    else:
        halves = [load_half(processed_root, spec, m) for m in spec["members"]]
        for h in halves:
            h["cell_idx"] = None
        return {"mode": "pair", "name": ds_name, "halves": halves}


# ---------------------------------------------------------------------------
# Per-fold runner
# ---------------------------------------------------------------------------
def run_one_fold(
    ds_name: str,
    fold_idx: int,
    half1: dict,
    half2: dict,
    gene_names: list,
    cfg: dict,
    model,
    preprocess,
    device: str,
    seed: int,
    output_root: Path,
    config_path: Path,
):
    train_half, test_half = (half1, half2) if fold_idx == 0 else (half2, half1)
    run_name = f"{ds_name}_fold{fold_idx}_seed{seed}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    if config_path is not None and config_path.exists():
        shutil.copy(config_path, run_dir / "config.yaml")

    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name} | Fold: {fold_idx} | train={train_half['label']} test={test_half['label']}")
    print(f"{'='*60}")

    # Load / encode features
    use_pc = cfg.get("use_processed_cell", False)
    cache_root = Path(cfg.get("cache_root", "./cache"))
    feature_dim = int(cfg.get("feature_dim", 1024))

    if use_pc:
        # Read precomputed UNI features from the shared cell-level interface.
        train_cache = cache_root / ds_name / f"{train_half['label']}_he.npy"
        test_cache = cache_root / ds_name / f"{test_half['label']}_he.npy"
        train_he = load_cached_he_features(train_half["adata"], cache_path=train_cache, feature_key="he")
        test_he = load_cached_he_features(test_half["adata"], cache_path=test_cache, feature_key="he")
        cfg["feature_dim"] = train_he.shape[1]
    else:
        img_batch_size = int(cfg.get("img_batch_size", 48))
        feature_dim = int(cfg.get("feature_dim", 1024))
        train_cache = cache_root / ds_name / f"{train_half['label']}_uni_cls.npy"
        test_cache = cache_root / ds_name / f"{test_half['label']}_uni_cls.npy"

        train_he = load_or_encode_features(
            Path(cfg["processed_root"]) / train_half["dir"],
            train_cache,
            model,
            preprocess,
            device,
            img_batch_size,
            feature_dim,
            encoder_name="uni",
            cell_idx=train_half.get("cell_idx"),
        )
        test_he = load_or_encode_features(
            Path(cfg["processed_root"]) / test_half["dir"],
            test_cache,
            model,
            preprocess,
            device,
            img_batch_size,
            feature_dim,
            encoder_name="uni",
            cell_idx=test_half.get("cell_idx"),
        )

    # Preprocess expression
    train_expr, train_he = preprocess_half(
        train_half["adata"], train_he, gene_names, min_counts=int(cfg.get("min_counts", 10))
    )
    test_expr, test_he = preprocess_half(
        test_half["adata"], test_he, gene_names, min_counts=int(cfg.get("min_counts", 10))
    )

    print(f"  train cells: {train_expr.shape[0]}, test cells: {test_expr.shape[0]}, genes: {train_expr.shape[1]}")

    y_pred, per_gene_pcc, mean_pcc, info = evaluate_fold(
        train_expr, train_he, test_expr, test_he, cfg, seed
    )

    print(f"  Linear probe: D_pca={info['pca_components']}, alpha={info['ridge_alpha']:.3e}")
    print(f"  Test PCC: {mean_pcc:.4f}")

    np.save(run_dir / "pred.npy", y_pred)
    np.save(run_dir / "pcc_per_gene.npy", np.asarray(per_gene_pcc))

    result = {
        "dataset": ds_name,
        "fold": fold_idx,
        "seed": seed,
        "train_half": train_half["label"],
        "test_half": test_half["label"],
        "n_genes": len(gene_names),
        "gene_names": gene_names,
        "test_pcc": float(mean_pcc),
        "linear_probe_info": info,
        "status": "completed",
    }
    with open(run_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--fold", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    output_root = Path(cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("seeds", [42])[0])

    use_pc = cfg.get("use_processed_cell", False)
    model, preprocess = None, None
    if not use_pc:
        # Load encoder once
        local_path = cfg["uni_local_path"]
        print(f"Loading UNI from: {local_path}")
        model, preprocess, feature_dim = load_uni(local_path, device)
        cfg["feature_dim"] = feature_dim
    else:
        print("Using precomputed UNI features from processed_cell/")

    config_path = Path(args.config)
    all_results = []

    for ds_name in cfg["datasets"]:
        if args.dataset is not None and ds_name != args.dataset:
            continue

        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        ds_info = load_dataset(ds_name, cfg)
        half1, half2 = ds_info["halves"]

        # Select genes: either all real genes or a top-HVG subset.
        n_top_hvgs = cfg.get("n_top_hvgs")
        if n_top_hvgs is None or (isinstance(n_top_hvgs, str) and n_top_hvgs.lower() == "all"):
            # Use the full real gene panel without min-cell filtering.
            genes_sets = []
            for half in (half1, half2):
                adata = half["adata"]
                if "is_gene" in adata.var.columns:
                    genes = adata.var_names[adata.var["is_gene"].astype(bool).values].tolist()
                else:
                    genes = adata.var_names.tolist()
                genes_sets.append(set(_filter_control_probes(genes)))
            gene_names = sorted(genes_sets[0] & genes_sets[1])
            print(f"  using full real panel: {len(gene_names)} genes")
        else:
            gene_names = select_top_hvgs_official(
                [half1["adata"], half2["adata"]],
                n_top=int(n_top_hvgs),
                min_cells_pct=float(cfg.get("hvg_min_cells_pct", 0.10)),
            )
            print(f"  selected {len(gene_names)} HVGs")

        # Versioned output dir
        ds_dir = output_root / ds_name
        ds_dir.mkdir(parents=True, exist_ok=True)
        existing = [d for d in ds_dir.iterdir() if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit()]
        version = f"v{max(int(d.name[1:]) for d in existing) + 1}" if existing else "v1"
        version_dir = ds_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(config_path, version_dir / "config.yaml")
        print(f"  saving runs to {version_dir}/")

        for fold in range(2):
            if args.fold is not None and fold != args.fold:
                continue
            try:
                result = run_one_fold(
                    ds_name,
                    fold,
                    half1,
                    half2,
                    gene_names,
                    cfg,
                    model,
                    preprocess,
                    device,
                    seed,
                    version_dir,
                    config_path,
                )
                all_results.append(result)
            except Exception as e:
                print(f"ERROR in {ds_name} fold={fold}: {e}")
                import traceback

                traceback.print_exc()
                all_results.append(
                    {"dataset": ds_name, "fold": fold, "seed": seed, "error": str(e)}
                )

        # Dataset summary
        fold_pccs = [r["test_pcc"] for r in all_results if r.get("dataset") == ds_name and "error" not in r]
        if fold_pccs:
            summary = {
                "dataset": ds_name,
                "version": version,
                "mean": float(np.mean(fold_pccs)),
                "std": float(np.std(fold_pccs, ddof=1)) if len(fold_pccs) > 1 else 0.0,
                "n": len(fold_pccs),
                "values": fold_pccs,
            }
            with open(version_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            print(f"  dataset summary: {summary['mean']:.4f} ± {summary['std']:.4f}")

    # Global summary
    by_ds = {}
    for r in all_results:
        if "error" in r:
            continue
        by_ds.setdefault(r["dataset"], []).append(r["test_pcc"])

    global_summary = {
        "all_results": all_results,
        "dataset_summary": {
            ds: {
                "mean": float(np.mean(pccs)),
                "std": float(np.std(pccs, ddof=1)) if len(pccs) > 1 else 0.0,
                "n": len(pccs),
                "values": pccs,
            }
            for ds, pccs in by_ds.items()
        },
    }
    with open(output_root / "summary.json", "w") as f:
        json.dump(global_summary, f, indent=2)
    print(f"\nGlobal summary saved to {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
