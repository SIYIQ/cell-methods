import os
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp

from tqdm import tqdm
from pathlib import Path
from sklearn.cluster import KMeans
from torch.utils.data import Dataset
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import from_scipy_sparse_matrix


### GRAPH DATASET ###
class GraphDataset(Dataset):
    """Dataset for slide-level graph data.

    Each item is one slide: (edge_index, labels, patch_embeddings).
    """
    def __init__(self, adj, slide_data_list, device='cpu'):
        """Args:
            adj: list of adjacency matrices (one per slide)
            slide_data_list: list of dicts with keys 'labels', 'patch_embeddings'
            device: torch device
        """
        self.slide_indices = []
        self.edge_indices = []
        self.labels = []
        self.patch_embeddings = []

        for idx, data in enumerate(slide_data_list):
            edge_idx = from_scipy_sparse_matrix(sp.coo_matrix(adj[idx]))[0].to(device)
            labels = data['labels'].to(device)
            embeddings = data['patch_embeddings'].to(device)

            self.slide_indices.append(idx)
            self.edge_indices.append(edge_idx)
            self.labels.append(labels)
            self.patch_embeddings.append(embeddings)

    def __len__(self):
        return len(self.edge_indices)

    def __getitem__(self, idx):
        return (self.slide_indices[idx], self.edge_indices[idx],
                self.labels[idx], self.patch_embeddings[idx])


### GRAPH CONSTRUCTION ###
def update_adj(adj, cluster_labels, patch_embeddings, old_labels=None):
    """Update adjacency matrix with cluster centroid connections."""
    unique_cluster_labels = np.unique(cluster_labels)
    centroid_spots = []

    for cluster_label in unique_cluster_labels:
        cluster_spots = np.where(cluster_labels == cluster_label)[0]
        cluster_centroid = patch_embeddings[cluster_spots].mean(axis=0)
        nearest_spot_idx = np.argmin(
            np.linalg.norm(patch_embeddings[cluster_spots] - cluster_centroid, axis=1)
        )
        nearest_spot = cluster_spots[nearest_spot_idx]

        # Connect nearest spot to all other spots in cluster
        for j in range(len(cluster_spots)):
            if cluster_spots[j] != nearest_spot:
                adj[cluster_spots[j], nearest_spot] = 1
                adj[nearest_spot, cluster_spots[j]] = 1

        centroid_spots.append(nearest_spot_idx)
        cluster_labels[cluster_spots[nearest_spot_idx]] *= -1
        if cluster_labels[cluster_spots[nearest_spot_idx]] == 0:
            cluster_labels[cluster_spots[nearest_spot_idx]] = -(len(unique_cluster_labels))

    if old_labels is not None:
        for j, old_label in enumerate(old_labels):
            if old_label < 0:
                centroid_spots.append(j)

    centroid_spots = list(set(centroid_spots))
    for j in range(len(centroid_spots)):
        for k in range(j + 1, len(centroid_spots)):
            adj[centroid_spots[j], centroid_spots[k]] = 1
            adj[centroid_spots[k], centroid_spots[j]] = 1

    return adj, cluster_labels


def build_one_hop_graph(coords_list):
    """Build 8-neighbor spatial graph for each slide.

    Args:
        coords_list: list of [N, 2] coordinate tensors (from hest-bench adata.obsm['spatial'])

    Returns:
        list of adjacency matrices (torch tensors, N x N)
    """
    adj = []
    for coords in coords_list:
        n = coords.shape[0]
        adj.append(torch.zeros(n, n))

    for slide_idx, coords in enumerate(coords_list):
        coords_np = coords.cpu().numpy() if torch.is_tensor(coords) else np.asarray(coords)
        # Build 8-neighbor graph using spatial coordinates
        tmp_adj = kneighbors_graph(
            coords_np, mode='connectivity', n_neighbors=min(8, n - 1)
        ).toarray()
        tmp_adj = (tmp_adj + tmp_adj.T) > 0
        tmp_adj = torch.tensor(tmp_adj, dtype=torch.float)
        adj[slide_idx] = tmp_adj
        adj[slide_idx].fill_diagonal_(1)

    return adj


def build_hierarchical_graph(coords_list, patch_embeddings_list, config):
    """Build hierarchical graph with spatial + feature clustering.

    Args:
        coords_list: list of [N, 2] coordinate tensors
        patch_embeddings_list: list of [N, D] feature tensors
        config: dict with keys 'spatial_clusters', 'feature_clusters'

    Returns:
        list of updated adjacency matrices
    """
    adj = build_one_hop_graph(coords_list)

    spatial_clusters = config.get('spatial_clusters', 5)
    feature_clusters = config.get('feature_clusters', 5)

    for i in tqdm(range(len(coords_list)), desc='Building hierarchical graph'):
        coords = coords_list[i]
        embeddings = patch_embeddings_list[i]
        coords_np = coords.cpu().numpy() if torch.is_tensor(coords) else np.asarray(coords)
        embeddings_np = embeddings.cpu().numpy() if torch.is_tensor(embeddings) else np.asarray(embeddings)

        # Spatial clustering
        n_clusters = min(spatial_clusters, len(coords_np))
        if n_clusters > 1:
            clusterer = KMeans(n_clusters=n_clusters, max_iter=1000, n_init=10)
            clusterer.fit(coords_np)
            cluster_labels = clusterer.predict(coords_np)
            adj[i] = adj[i].numpy() if torch.is_tensor(adj[i]) else adj[i]
            adj[i], cluster_labels = update_adj(adj[i], cluster_labels, embeddings_np, None)
            spatial_cluster_labels = cluster_labels.copy()

            # Feature clustering
            n_clusters = min(feature_clusters, len(coords_np))
            if n_clusters > 1:
                clusterer = KMeans(n_clusters=n_clusters, max_iter=1000, n_init=10)
                clusterer.fit(embeddings_np)
                cluster_labels = clusterer.predict(embeddings_np)
                adj[i], cluster_labels = update_adj(
                    adj[i], cluster_labels, embeddings_np, spatial_cluster_labels
                )

        adj[i] = torch.tensor(adj[i], dtype=torch.float)

    return adj


def graph_construction(coords_list, patch_embeddings_list, labels_list, config, device='cpu'):
    """Construct graph datasets for training and validation.

    Args:
        coords_list: list of coordinate tensors for all slides
        patch_embeddings_list: list of patch embedding tensors
        labels_list: list of label tensors (gene expression)
        config: dict with hierarchical graph settings
        device: torch device

    Returns:
        GraphDataset instance
    """
    print('Building the spatial graph...')
    adj = build_one_hop_graph(coords_list)
    print('Building the spatial graph done.')

    if config.get('hierarchical', False):
        print('Building the hierarchical graph...')
        adj = build_hierarchical_graph(coords_list, patch_embeddings_list, config)
        print('Building the hierarchical graph done.')

    slide_data_list = []
    for labels, embeddings in zip(labels_list, patch_embeddings_list):
        slide_data_list.append({
            'labels': labels if torch.is_tensor(labels) else torch.tensor(labels),
            'patch_embeddings': embeddings if torch.is_tensor(embeddings) else torch.tensor(embeddings),
        })

    dataset = GraphDataset(adj, slide_data_list, device=device)
    return dataset
