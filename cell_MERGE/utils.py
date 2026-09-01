"""Utilities for cell_MERGE: build cell-level graph from SpatialEx hypergraph."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cell-benchmark" / "scripts"))
import cell_data as cd


def _add_cluster_edges(coords: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """For each cluster, connect its nearest cell to the centroid to all others."""
    edges = []
    for lab in np.unique(labels):
        mask = labels == lab
        idx = np.where(mask)[0]
        if len(idx) <= 1:
            continue
        cluster_coords = coords[idx]
        centroid = cluster_coords.mean(axis=0)
        nearest = idx[np.argmin(np.linalg.norm(cluster_coords - centroid, axis=1))]
        for j in idx:
            if j != nearest:
                edges.append([j, nearest])
                edges.append([nearest, j])
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    return np.asarray(edges).T.astype(np.int64)


def build_cell_graph(adata, x_feat: torch.Tensor, y_expr: torch.Tensor,
                     cfg: dict, device):
    """Build a PyG ``Data`` object for GATNet.

    Returns:
        ``torch_geometric.data.Data`` with ``x``, ``edge_index``, ``y`` on
        ``device``.
    """
    num_neighbors = cfg.get("num_neighbors", 7)
    H = cd.build_cell_hypergraph(adata, num_neighbors=num_neighbors,
                                 return_type="crs")
    edge_index = cd.hypergraph_to_pyg_edges(H).cpu()
    edge_index, _ = add_self_loops(edge_index, num_nodes=x_feat.shape[0])

    if cfg.get("gnn_hierarchical", False):
        coords = np.asarray(adata.obsm["spatial"])
        feats = x_feat.detach().cpu().numpy()
        spatial_clusters = min(cfg.get("gnn_spatial_clusters", 5), len(coords))
        feature_clusters = min(cfg.get("gnn_feature_clusters", 5), len(coords))

        if spatial_clusters > 1:
            clusterer = KMeans(n_clusters=spatial_clusters, max_iter=1000,
                               n_init=10, random_state=42)
            spatial_labels = clusterer.fit_predict(coords)
            spatial_edges = _add_cluster_edges(coords, spatial_labels)
            if spatial_edges.size > 0:
                edge_index = torch.cat([edge_index,
                                        torch.from_numpy(spatial_edges)], dim=1)

        if feature_clusters > 1:
            clusterer = KMeans(n_clusters=feature_clusters, max_iter=1000,
                               n_init=10, random_state=42)
            feature_labels = clusterer.fit_predict(feats)
            feature_edges = _add_cluster_edges(coords, feature_labels)
            if feature_edges.size > 0:
                edge_index = torch.cat([edge_index,
                                        torch.from_numpy(feature_edges)], dim=1)

        edge_index = torch.unique(edge_index, dim=1)

    data = Data(
        x=x_feat.to(device),
        edge_index=edge_index.to(device),
        y=y_expr.to(device),
    )
    return data
