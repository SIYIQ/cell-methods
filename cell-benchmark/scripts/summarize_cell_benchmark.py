"""Summarize cell-level benchmark runs with unified 50-HVG and all-gene metrics.

This script produces an apples-to-apples comparison table across all `cell_*`
projects in this repository (`cell_level_methods/`). It uses the 50 HVGs
selected by `cell_HST_h0mini_lp` as the common gene subset, and recomputes each
graph model's mean PCC on exactly those 50 genes from the saved per-gene PCC
arrays.

Outputs:
    - stdout: markdown table
    - <repo>/cell-benchmark/runs_summary.csv: CSV with the same numbers

Usage:
    python summarize_cell_benchmark.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

# Repo root (cell_level_methods/); run outputs are read from each project's
# in-repo runs/ directory.
REPO = Path(__file__).resolve().parents[2]
# Benchmark data (processed_cell) lives outside the repo; only gene panels are
# read from there.
BENCH = Path("/home/sb202604/cell-benchmark")
PROCESSED_CELL = BENCH / "processed_cell"
OUT_CSV = REPO / "cell-benchmark" / "runs_summary.csv"

DATASETS = ["hSkin_Melanoma", "hColon_Non_diseased", "mouse_Colon", "Human_Breast_Cancer"]

# Which projects report only 50-HVG results, and which report all genes.
LINEAR_PROBE_PROJECTS = {
    "HST_h0mini_lp": REPO / "cell_HST_h0mini_lp",
    "HST_uni_lp": REPO / "cell_HST_uni_lp",
    "Stem": REPO / "cell_Stem",
}
GRAPH_PROJECTS = {
    "HST_middle": REPO / "cell_HST_middle",
    "STFlow": REPO / "cell_STFlow",
    "MERGE": REPO / "cell_MERGE",
    "NH2ST": REPO / "cell_NH2ST",
    "SpatialEx_native": REPO / "cell_SpatialEx" / "runs_native",
}


def get_gene_order(dataset: str) -> list[str]:
    if dataset == "Human_Breast_Cancer":
        adata = sc.read_h5ad(PROCESSED_CELL / "Human_Breast_Cancer" / "adata_Rep1.h5ad")
    else:
        adata = sc.read_h5ad(PROCESSED_CELL / dataset / "adata_S1.h5ad")
    return list(adata.var_names)


def find_files(root: Path, filename: str) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob(filename) if p.is_file())


def get_h0mini_hvgs(dataset: str) -> list[str] | None:
    """Use h0mini's 50 HVG list as the canonical subset."""
    root = REPO / "cell_HST_h0mini_lp" / "runs" / dataset
    for p in find_files(root, "result.json"):
        with open(p) as f:
            r = json.load(f)
        return r["gene_names"]
    return None


def collect_linear_probe(project_name: str, dataset: str) -> dict:
    root = LINEAR_PROBE_PROJECTS[project_name] / "runs" / dataset
    all_values = []
    pc_values = []
    for p in find_files(root, "result.json"):
        with open(p) as f:
            r = json.load(f)
        # Skip Stem smoke-test runs (train_steps == 500).
        cfg_path = p.with_name("config.yaml")
        use_pc = False
        if cfg_path.exists():
            import yaml
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            if project_name == "Stem" and cfg.get("train_steps", 30000) < 1000:
                continue
            use_pc = cfg.get("use_processed_cell", False)
        if use_pc:
            pc_values.append(r["test_pcc"])
        else:
            all_values.append(r["test_pcc"])
    # Prefer processed-cell runs if any exist.
    values = pc_values if pc_values else all_values
    if not values:
        return {"pcc_50hvg_vals": [], "pcc_all_vals": [], "n": 0}
    return {"pcc_50hvg_vals": values, "pcc_all_vals": [], "n": len(values)}


def collect_graph(project_name: str, dataset: str, hvg_genes: list[str] | None, n_genes: int) -> dict:
    root = GRAPH_PROJECTS[project_name]

    # Resolve dataset directory (SpatialEx uses _chessboard suffixes).
    if project_name == "SpatialEx_native":
        if not root.exists():
            return {"pcc_50hvg_vals": [], "pcc_all_vals": [], "n": 0}
        candidates = [d for d in root.iterdir() if d.is_dir() and dataset in d.name]
        if not candidates:
            return {"pcc_50hvg_vals": [], "pcc_all_vals": [], "n": 0}
        ds_root = candidates[0]
    else:
        ds_root = root / "runs" / dataset

    if not ds_root.exists():
        return {"pcc_50hvg_vals": [], "pcc_all_vals": [], "n": 0}

    # Try to read canonical gene list from a result.json if present.
    result_jsons = find_files(ds_root, "result.json")
    run_gene_names = None
    run_n_genes = None
    if result_jsons:
        with open(result_jsons[0]) as f:
            first_res = json.load(f)
        run_gene_names = first_res.get("gene_names")
        run_n_genes = first_res.get("n_genes")

    all_values = []
    sub_values = []

    # Per-gene PCC files have varying names across projects.
    per_gene_files = []
    per_gene_files.extend(find_files(ds_root, "per_gene_pcc.npy"))          # GHIST
    per_gene_files.extend(find_files(ds_root, "per_gene_pcc_*.npy"))        # middle/novae/STFlow/MERGE/NH2ST
    per_gene_files.extend(find_files(ds_root, "pcc_*_per_gene.npy"))        # SpatialEx native

    for pcc_path in per_gene_files:
        pcc = np.load(pcc_path)
        if pcc.ndim != 1:
            pcc = pcc.reshape(-1)

        # Case 1: this run already predicts only the canonical 50 HVGs.
        if pcc.shape[0] == 50 and run_gene_names is not None and hvg_genes is not None:
            if set(run_gene_names) == set(hvg_genes):
                sub_values.append(float(np.mean(pcc)))
                # No all-gene metric available for this run.
                continue

        # Case 2: full gene panel.
        if pcc.shape[0] == n_genes and hvg_genes is not None:
            gene_order = get_gene_order(dataset)
            hvg_idx = [gene_order.index(g) for g in hvg_genes]
            all_values.append(float(np.mean(pcc)))
            sub_values.append(float(np.mean(pcc[hvg_idx])))
        elif pcc.shape[0] == run_n_genes and run_gene_names is not None and hvg_genes is not None:
            # Partial/custom panel: map by gene name.
            sub_idx = [run_gene_names.index(g) for g in hvg_genes if g in run_gene_names]
            if len(sub_idx) == len(hvg_genes):
                all_values.append(float(np.mean(pcc)))
                sub_values.append(float(np.mean(pcc[sub_idx])))
        else:
            print(f"    [warn] {pcc_path}: shape {pcc.shape} not matching n_genes={n_genes} or 50-HVG set; skipping")

    # Fallback to result.json mean_pcc if no per-gene arrays exist.
    if not all_values and not sub_values:
        for p in result_jsons:
            with open(p) as f:
                r = json.load(f)
            # If result.json explicitly reports 50 HVGs, treat mean_pcc as 50-HVG metric.
            is_50hvg = (r.get("n_genes") == 50)
            for k, v in r.items():
                if k.startswith("mean_pcc"):
                    if is_50hvg:
                        sub_values.append(float(v))
                    else:
                        all_values.append(float(v))

    if not all_values and not sub_values:
        return {"pcc_50hvg_vals": [], "pcc_all_vals": [], "n": 0}

    return {
        "pcc_all_vals": all_values,
        "pcc_50hvg_vals": sub_values,
        "n": len(all_values) + len(sub_values),
    }


def main():
    rows = []
    for dataset in DATASETS:
        gene_order = get_gene_order(dataset)
        n_genes = len(gene_order)
        hvg_genes = get_h0mini_hvgs(dataset)

        # Linear probe methods (already 50 HVG).
        for proj in LINEAR_PROBE_PROJECTS:
            res = collect_linear_probe(proj, dataset)
            vals = res["pcc_50hvg_vals"]
            rows.append({
                "dataset": dataset,
                "n_genes": n_genes,
                "model": proj,
                "pcc_all_genes_mean": np.nan,
                "pcc_all_genes_std": np.nan,
                "pcc_50hvg_mean": float(np.mean(vals)) if vals else np.nan,
                "pcc_50hvg_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "n_directions": res["n"],
            })

        # Graph / slice methods (all genes, plus recomputed 50 HVG).
        for proj in GRAPH_PROJECTS:
            res = collect_graph(proj, dataset, hvg_genes, n_genes)
            all_vals = res["pcc_all_vals"]
            sub_vals = res["pcc_50hvg_vals"]
            rows.append({
                "dataset": dataset,
                "n_genes": n_genes,
                "model": proj,
                "pcc_all_genes_mean": float(np.mean(all_vals)) if all_vals else np.nan,
                "pcc_all_genes_std": float(np.std(all_vals, ddof=1)) if len(all_vals) > 1 else 0.0,
                "pcc_50hvg_mean": float(np.mean(sub_vals)) if sub_vals else np.nan,
                "pcc_50hvg_std": float(np.std(sub_vals, ddof=1)) if len(sub_vals) > 1 else 0.0,
                "n_directions": res["n"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, float_format="%.4f")

    def fmt_mean_std(row, col_mean, col_std):
        m = row[col_mean]
        s = row[col_std]
        if pd.isna(m):
            return "—"
        if pd.isna(s) or s == 0.0:
            return f"{m:.4f}"
        return f"{m:.4f}±{s:.4f}"

    # Build wide tables for display with mean±std strings.
    print("\n# Mean ± std PCC on the 50 HVG subset selected by HST_h0mini_lp\n")
    sub = df[["dataset", "model", "pcc_50hvg_mean", "pcc_50hvg_std"]].copy()
    sub["val"] = sub.apply(lambda r: fmt_mean_std(r, "pcc_50hvg_mean", "pcc_50hvg_std"), axis=1)
    pivot50 = sub.pivot_table(index="dataset", columns="model", values="val", aggfunc="first")
    print(pivot50.to_markdown())

    print("\n# Mean ± std PCC on all genes (linear probes only predict 50, so their column is empty)\n")
    sub_all = df[["dataset", "model", "pcc_all_genes_mean", "pcc_all_genes_std"]].copy()
    sub_all["val"] = sub_all.apply(lambda r: fmt_mean_std(r, "pcc_all_genes_mean", "pcc_all_genes_std"), axis=1)
    pivot_all = sub_all.pivot_table(index="dataset", columns="model", values="val", aggfunc="first")
    print(pivot_all.to_markdown())

    print(f"\nCSV saved to: {OUT_CSV}")


if __name__ == "__main__":
    main()
