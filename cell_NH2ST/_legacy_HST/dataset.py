import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.neighbors import kneighbors_graph

from utils import load_slide, extract_gene_expr


class HSTNH2STDataset(Dataset):
    """Spot-level dataset for NH2ST on hest-bench data.

    Loads slides from hest-bench format (patches.h5 + adata.h5ad),
    builds per-spot neighbor patches based on spatial coordinates,
    and returns spot-level samples compatible with NGHist2ST.forward().
    """

    def __init__(self, slide_paths, gene_names, neighbor_k=8,
                 dist_thresh_um=150.0, pixel_size_um=0.46,
                 mode='train', aug_cfg=None):
        """Args:
            slide_paths: list of (patches_h5, st_h5ad) tuples
            gene_names: list of gene names to predict
            neighbor_k: number of neighbors per spot
            dist_thresh_um: neighbor search radius in microns
            pixel_size_um: pixel size in microns
            mode: 'train' or 'test'
            aug_cfg: augmentation config dict (optional)
        """
        self.slide_paths = slide_paths
        self.gene_names = gene_names
        self.neighbor_k = neighbor_k
        self.dist_thresh_um = dist_thresh_um
        self.pixel_size_um = pixel_size_um
        self.mode = mode
        self.aug_cfg = aug_cfg

        # Load all slides
        self.all_patches = []
        self.all_expr = []
        self.all_coords = []
        self.cumlen = []
        total = 0

        for pf, st_path in slide_paths:
            imgs, adata, coords = load_slide(pf, st_path)
            gene_expr = extract_gene_expr(adata, gene_names)

            self.all_patches.append(imgs)
            self.all_expr.append(gene_expr)
            self.all_coords.append(coords)

            total += imgs.shape[0]
            self.cumlen.append(total)

        # Pre-compute neighbor indices for all slides
        self.neighbor_indices = []
        for coords in self.all_coords:
            indices = self._build_neighbors(coords)
            self.neighbor_indices.append(indices)

    def _build_neighbors(self, coords):
        """Build neighbor indices for each spot based on spatial coords."""
        coords_np = coords.cpu().numpy() if torch.is_tensor(coords) else np.asarray(coords)
        n = len(coords_np)
        coords_um = coords_np * self.pixel_size_um

        # Use kneighbors_graph for spatial neighbors
        k = min(self.neighbor_k, n - 1)
        adj = kneighbors_graph(coords_um, n_neighbors=k, mode='connectivity').toarray()

        neighbor_idx = []
        for i in range(n):
            neighbors = np.where(adj[i] > 0)[0]
            # Always include self
            if i not in neighbors:
                neighbors = np.append(neighbors, i)
            # Pad or truncate to neighbor_k
            if len(neighbors) < self.neighbor_k:
                # Pad with self
                pad = np.full(self.neighbor_k - len(neighbors), i, dtype=np.int64)
                neighbors = np.concatenate([neighbors, pad])
            else:
                neighbors = neighbors[:self.neighbor_k]
            neighbor_idx.append(neighbors)

        return np.array(neighbor_idx, dtype=np.int64)

    def __len__(self):
        return self.cumlen[-1] if self.cumlen else 0

    def __getitem__(self, index):
        # Find which slide this index belongs to
        slide_idx = 0
        while index >= self.cumlen[slide_idx]:
            slide_idx += 1

        spot_idx = index
        if slide_idx > 0:
            spot_idx = index - self.cumlen[slide_idx - 1]

        patches = self.all_patches[slide_idx]
        expr = self.all_expr[slide_idx]
        neighbor_idx = self.neighbor_indices[slide_idx]

        # Target patch and expression
        patch = patches[spot_idx]  # [3, 224, 224]
        exp = expr[spot_idx]       # [n_genes]

        # Neighbor patches and expressions
        n_idx = neighbor_idx[spot_idx]
        neighbor_patches = patches[n_idx]      # [k, 3, 224, 224]
        neighbor_exp = expr[n_idx]             # [k, n_genes]

        # Position (spatial coords)
        position = self.all_coords[slide_idx][spot_idx]  # [2]

        return patch, exp, neighbor_patches, neighbor_exp, position


def collate_fn(batch):
    """Collate function for DataLoader."""
    patches = torch.stack([item[0] for item in batch])
    exps = torch.stack([item[1] for item in batch])
    neighbor_patches = torch.stack([item[2] for item in batch])
    neighbor_exps = torch.stack([item[3] for item in batch])
    positions = torch.stack([item[4] for item in batch])
    return patches, exps, neighbor_patches, neighbor_exps, positions
