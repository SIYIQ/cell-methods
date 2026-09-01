"""Shared helpers for cell-level benchmark methods.

This module is imported by/ cell_HST_middle / cell_SpatialEx training scripts. It exposes a stable cell-level data interface
backed by `/home/sb202604/cell-benchmark/processed_cell/`.

Typical usage in a training script::

    from cell_data import (
        load_pair, build_cell_hypergraph, hypergraph_to_pyg_edges,
        compute_pcc_metrics, DATASETS,
    )

    a1, a2, meta = load_pair("hSkin_Melanoma")
    # a1: AnnData with X (log1p), obsm['spatial'] (um), obsm['he'] (UNI 1024)
    # a2: same, for the other half / replicate
    # meta: dict with mode, n_cells, n_genes, ...

    # SpatialEx-style spatial hypergraph (cell -> KNN neighbors as hyperedges).
    H1 = build_cell_hypergraph(a1, num_neighbors=7, return_type="crs")

    # If you want PyG edge_index instead (for novae GAT or HGT):
    edge_index = hypergraph_to_pyg_edges(H1)

The interface is intentionally thin: it adapts the SpatialEx package APIs and
the shared processed_cell layout, but does NOT introduce any model-specific
logic. Each method's training loop owns its own loss + optimizer.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

# Stub cellpose so we can import SpatialEx without its optional segmentation deps.
# Only stub when the real package is NOT installed; otherwise a fake module would
# shadow the real cellpose for the whole process (sys.modules lookup wins).
try:
    import cellpose  # noqa: F401
except ImportError:
    for _m in ("cellpose", "cellpose.models"):
        sys.modules.setdefault(_m, types.ModuleType(_m))

# Make the cell_SpatialEx package importable. This is the canonical SpatialEx
# installation we use across the benchmark — its preprocess utilities (e.g.
# Build_hypergraph_spatial_and_HE) are reused so that every method sees the
# *exact same* cell hypergraph at training time.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cell_SpatialEx"))

import json
from typing import Iterable

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch

import SpatialEx as se


PROCESSED_ROOT = Path("/home/sb202604/cell-benchmark/processed_cell")


# Datasets registered in build_cell_dataset.py. Each entry describes which
# split files exist on disk and what the cross-section directions are called.
DATASETS = {
    "hSkin_Melanoma": {
        "mode": "chessboard",
        "splits": ("S1", "S2"),
        "files": ("adata_S1.h5ad", "adata_S2.h5ad"),
    },
    "hColon_Non_diseased": {
        "mode": "chessboard",
        "splits": ("S1", "S2"),
        "files": ("adata_S1.h5ad", "adata_S2.h5ad"),
    },
    "mouse_Colon": {
        "mode": "chessboard",
        "splits": ("S1", "S2"),
        "files": ("adata_S1.h5ad", "adata_S2.h5ad"),
    },
    "Human_Breast_Cancer": {
        "mode": "pair",
        "splits": ("Rep1", "Rep2"),
        "files": ("adata_Rep1.h5ad", "adata_Rep2.h5ad"),
    },
}


def list_datasets() -> list[str]:
    return list(DATASETS.keys())


def load_pair(dataset: str) -> tuple[ad.AnnData, ad.AnnData, dict]:
    """Load the two halves / replicates of a benchmark dataset.

    Returns (slice1_adata, slice2_adata, meta_dict). Each AnnData has
    log1p-normalized X (cells × genes, dense float32), obsm['spatial']
    (microns), obsm['he'] (UNI 1024-d features). Var contains gene names.
    """
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; "
                         f"choices: {list(DATASETS.keys())}")
    info = DATASETS[dataset]
    ds_dir = PROCESSED_ROOT / dataset
    meta_path = ds_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found. Run build_cell_dataset.py first.")
    with open(meta_path) as f:
        meta = json.load(f)

    a1_path = ds_dir / info["files"][0]
    a2_path = ds_dir / info["files"][1]
    a1 = ad.read_h5ad(a1_path)
    a2 = ad.read_h5ad(a2_path)
    # Sanity check that downstream obsm keys are present.
    for a, label in ((a1, info["splits"][0]), (a2, info["splits"][1])):
        if "spatial" not in a.obsm:
            raise RuntimeError(f"{label} missing obsm['spatial']")
        if "he" not in a.obsm:
            raise RuntimeError(f"{label} missing obsm['he']")
    return a1, a2, meta


def build_cell_hypergraph(adata: ad.AnnData, num_neighbors: int = 7,
                          return_type: str = "crs") -> sp.spmatrix:
    """SpatialEx-style spatial KNN hypergraph over cells. Reuses
    `Build_hypergraph_spatial_and_HE` so every method consumes an identical
    graph.

    Returns
    -------
    H : scipy sparse (n_cells x n_hyperedges)
        Each column is a hyperedge (one per cell) connecting the cell and
        its K nearest spatial neighbors.
    """
    return se.pp.Build_hypergraph_spatial_and_HE(
        adata, num_neighbors=num_neighbors, graph_kind="spatial",
        return_type=return_type)


def hypergraph_to_pyg_edges(H: sp.spmatrix) -> torch.Tensor:
    """Convert SpatialEx hypergraph H (n_cells x n_hyperedges) into a PyG
    edge_index tensor [2, E] where two cells are connected if they share a
    hyperedge.

    For a KNN-derived hypergraph, this is equivalent to the symmetrized KNN
    cell graph: A = H @ H.T (binary, with self-loops). We strip self-loops
    and return undirected edges.

    Use this when feeding a homogeneous-graph model (e.g. novae GATv2) on top
    of the same neighborhoods that SpatialEx's HGNN sees.
    """
    H = H.tocsr()
    # A = H H^T  -> nonzero where two cells share at least one hyperedge.
    A = (H @ H.T).tocoo()
    rows, cols = A.row, A.col
    mask = rows != cols
    rows, cols = rows[mask], cols[mask]
    edge_index = torch.from_numpy(
        np.stack([rows, cols], axis=0).astype(np.int64))
    return edge_index


def compute_pcc_metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    """Per-gene Pearson correlation between (n_cells x n_genes) matrices.

    Returns dict with keys:
        mean_pcc        scalar, mean over genes
        median_pcc      scalar, median over genes
        per_gene        ndarray (n_genes,)
    """
    per_gene, mean_pcc = se.utils.Compute_metrics(
        true.copy(), pred.copy(), metric="pcc", reduce="mean")
    per_gene = np.asarray(per_gene)
    return {
        "mean_pcc": float(mean_pcc),
        "median_pcc": float(np.median(per_gene)),
        "per_gene": per_gene,
    }


def aggregate_seeds(results: list[dict]) -> dict:
    """Aggregate a list of per-seed result dicts into a Table-2-style
    headline number (mean ± std of `mean_pcc` pooled across the two
    cross-section directions).

    Each input dict must contain keys:
        seed
        mean_pcc_train_S2_test_S1   (or mean_pcc_train_Rep2_test_Rep1)
        mean_pcc_train_S1_test_S2
        median_pcc_slice1
        median_pcc_slice2

    Returns a summary dict matching the format used by
    cell_SpatialEx/scripts/run_native.py's `summarize()`.
    """
    if not results:
        raise ValueError("aggregate_seeds called with empty list")

    def agg(key):
        vals = [r[key] for r in results]
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        return {"mean": m, "std": s, "n": len(vals), "values": vals}

    # Detect direction names from first result
    dir_keys = [k for k in results[0].keys() if k.startswith("mean_pcc_train")]
    if len(dir_keys) != 2:
        raise ValueError(f"expected 2 direction keys, got {dir_keys}")

    pooled = []
    for k in dir_keys:
        pooled.extend(r[k] for r in results)
    overall = {
        "mean": float(np.mean(pooled)),
        "std": float(np.std(pooled, ddof=1)) if len(pooled) > 1 else 0.0,
        "n": len(pooled),
        "values": pooled,
    }

    summary = {
        "seeds": [r["seed"] for r in results],
        "overall_mean_pcc": overall,
    }
    for k in dir_keys:
        summary[k] = agg(k)
    if "median_pcc_slice1" in results[0]:
        summary["median_pcc_slice1"] = agg("median_pcc_slice1")
    if "median_pcc_slice2" in results[0]:
        summary["median_pcc_slice2"] = agg("median_pcc_slice2")
    return summary


__all__ = [
    "DATASETS",
    "PROCESSED_ROOT",
    "list_datasets",
    "load_pair",
    "build_cell_hypergraph",
    "hypergraph_to_pyg_edges",
    "compute_pcc_metrics",
    "aggregate_seeds",
]
