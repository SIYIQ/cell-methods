"""Utilities for cell_HST_middle: build HeteroData with cell-level edges."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch_geometric.data import HeteroData


# =============================================================================
# HVG selection (matches cell_HST_h0mini_lp benchmark)
# =============================================================================
def select_top_hvgs_official(adata_list, n_top=50, min_cells_pct=0.10):
    """Select top HVGs across a list of raw-count AnnData objects.

    Byte-identical to cell_HST_h0mini_lp/utils.py::select_top_hvgs_official.
    Operates on raw counts, filters control probes, then log1p + HVG.
    """
    common_genes = None
    for adata in adata_list:
        my_adata = adata.copy()
        if min_cells_pct:
            sc.pp.filter_genes(
                my_adata, min_cells=np.ceil(min_cells_pct * len(my_adata.obs))
            )
        curr_genes = np.array(my_adata.to_df().columns)
        if common_genes is None:
            common_genes = curr_genes
        else:
            common_genes = np.intersect1d(common_genes, curr_genes)

    common_genes = [
        g
        for g in common_genes
        if "BLANK" not in g
        and "Control" not in g
        and not g.startswith("NegControlProbe_")
        and not g.startswith("UnassignedCodeword_")
    ]

    stacked = None
    for adata in adata_list:
        df = adata.to_df()[common_genes]
        stacked = df if stacked is None else pd.concat([stacked, df])

    stacked_adata = sc.AnnData(stacked.astype(np.float32))
    sc.pp.filter_genes(stacked_adata, min_cells=0)
    sc.pp.log1p(stacked_adata)
    sc.pp.highly_variable_genes(stacked_adata, n_top_genes=n_top)
    hvg_mask = stacked_adata.var["highly_variable"].values
    return stacked_adata.var_names[hvg_mask].tolist()[:n_top]


def _load_raw_chessboard(dataset: str, processed_root: Path) -> tuple[ad.AnnData, ad.AnnData]:
    """Load raw-count adata and split into S1/S2 using spatial_ood indices."""
    ds_dir = processed_root / dataset
    adata = ad.read_h5ad(ds_dir / "cells.h5ad")
    with open(ds_dir / "splits.json") as f:
        splits = json.load(f)
    s1_idx = np.asarray(splits["spatial_ood"]["S1"], dtype=np.int64)
    s2_idx = np.asarray(splits["spatial_ood"]["S2"], dtype=np.int64)
    return adata[s1_idx].copy(), adata[s2_idx].copy()


def _load_raw_pair(dataset: str, processed_root: Path) -> tuple[ad.AnnData, ad.AnnData]:
    """Load raw-count adata for Human_Breast_Cancer Rep1/Rep2 pair mode."""
    a1 = ad.read_h5ad(processed_root / f"{dataset}_Rep1" / "cells.h5ad")
    a2 = ad.read_h5ad(processed_root / f"{dataset}_Rep2" / "cells.h5ad")
    return a1, a2


def load_raw_pair_for_hvg(dataset: str, processed_root: Path) -> tuple[ad.AnnData, ad.AnnData]:
    """Return two raw-count AnnData halves matching the processed_cell layout."""
    processed_root = Path(processed_root)
    if dataset == "Human_Breast_Cancer":
        return _load_raw_pair(dataset, processed_root)
    return _load_raw_chessboard(dataset, processed_root)


def get_gene_names(dataset: str, n_top_hvgs: int | None,
                   processed_root: Path) -> list[str] | None:
    """Get the canonical HVG gene list for a dataset.

    Uses raw counts from processed_root to match cell_HST_h0mini_lp.
    Returns None when n_top_hvgs is None (train on full panel).
    """
    if n_top_hvgs is None:
        return None
    a1_raw, a2_raw = load_raw_pair_for_hvg(dataset, processed_root)
    return select_top_hvgs_official([a1_raw, a2_raw], n_top=n_top_hvgs)


def build_morph_edges(img_embeds: torch.Tensor, top_k: int = 5,
                      sim_thresh: float = 0.6) -> torch.Tensor:
    """Top-K cosine similarity edges over cells in UNI feature space.

    Mirrors HST-middle/utils.py::build_morph_edges (spot version). At cell
    scale (~10^5 cells) this is too big to do as a dense N x N matrix, so we
    chunk along the first axis.
    """
    if not torch.is_tensor(img_embeds):
        img_embeds = torch.from_numpy(np.asarray(img_embeds)).float()
    device = img_embeds.device
    N = img_embeds.shape[0]
    embeds_norm = torch.nn.functional.normalize(img_embeds, p=2, dim=1)

    chunk = 4096
    src_list = []
    dst_list = []
    k = min(top_k, N - 1)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        # [chunk, N] cosine similarities
        sims = embeds_norm[start:end] @ embeds_norm.t()
        # remove self
        idx_row = torch.arange(start, end, device=device)
        sims[torch.arange(end - start, device=device), idx_row] = -float('inf')
        topk_vals, topk_idx = torch.topk(sims, k=k, dim=1)
        mask = topk_vals > sim_thresh
        if mask.any():
            row_idx = idx_row.view(-1, 1).expand_as(mask)[mask]
            col_idx = topk_idx[mask]
            src_list.append(row_idx)
            dst_list.append(col_idx)
    if not src_list:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    edge_index = torch.stack([torch.cat(src_list), torch.cat(dst_list)], dim=0)
    return edge_index


def build_cell_hetero_data(x_feat: torch.Tensor, spatial_edge_index: torch.Tensor,
                           morph_top_k: int = 5,
                           morph_sim_thresh: float = 0.6) -> HeteroData:
    """Construct the HeteroData object for one slice.

    Note: we leave 'image' / 'gene' node features unset here — the model's
    forward() populates them via input_mlp + gene branches. This keeps the
    HeteroData light (just edges + N).
    """
    N = x_feat.shape[0]
    device = x_feat.device
    data = HeteroData()
    # Empty node features placeholders so PyG knows the node counts.
    data['image'].num_nodes = N
    data['gene'].num_nodes = N

    # Self-pairing: each cell <-> its own gene node, both directions.
    corr = torch.arange(N, device=device)
    data['image', 'corresponds_to', 'gene'].edge_index = torch.stack([corr, corr], dim=0)
    data['gene', 'corresponds_to', 'image'].edge_index = torch.stack([corr, corr], dim=0)

    # Spatial: KNN from SpatialEx-style hypergraph (caller provides edge_index).
    spa = spatial_edge_index.to(device)
    data['image', 'spatially_adjacent', 'image'].edge_index = spa
    data['gene', 'spatially_adjacent', 'gene'].edge_index = spa.clone()

    # Morphological similarity in UNI feature space.
    morph = build_morph_edges(x_feat, top_k=morph_top_k, sim_thresh=morph_sim_thresh)
    if morph.numel() > 0:
        data['image', 'morphologically_similar', 'image'].edge_index = morph
        data['image', 'morphologically_similar_rev', 'image'].edge_index = morph.flip(0)
    else:
        # Provide empty edge_index so HGTConv metadata stays consistent.
        empty = torch.zeros((2, 0), dtype=torch.long, device=device)
        data['image', 'morphologically_similar', 'image'].edge_index = empty
        data['image', 'morphologically_similar_rev', 'image'].edge_index = empty

    return data


def sample_mgm_mask(n_cells: int, mask_ratio: float, device) -> torch.Tensor:
    """Bernoulli mask over cells for MGM. True = use image->gene fallback."""
    return torch.bernoulli(torch.full((n_cells,), mask_ratio, device=device)).bool()
