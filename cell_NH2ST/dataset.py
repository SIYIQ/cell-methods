"""cell_NH2ST dataset: cell-level neighbor sampling."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cell-benchmark" / "scripts"))
import cell_data as cd


def _to_dense_float(a):
    import scipy.sparse as sp
    if sp.issparse(a):
        return a.toarray().astype(np.float32)
    return np.asarray(a, dtype=np.float32)


class CellNH2STDataset(Dataset):
    """Cell-level dataset for NH2ST.

    For each cell, returns the target cell's UNI feature/expression and the
    UNI features/expressions of its K nearest spatial neighbors (self included).
    """

    def __init__(self, adata, neighbor_k=8, num_neighbors=7):
        self.adata = adata
        self.neighbor_k = neighbor_k

        self.x_feat = torch.from_numpy(_to_dense_float(adata.obsm["he"]))
        self.y_expr = torch.from_numpy(_to_dense_float(adata.X))
        self.coords = torch.from_numpy(np.asarray(adata.obsm["spatial"],
                                                   dtype=np.float32))

        H = cd.build_cell_hypergraph(adata, num_neighbors=num_neighbors,
                                     return_type="crs")
        H = H.tocsr()
        self.neighbor_indices = self._build_neighbors(H)

    def _build_neighbors(self, H):
        """For each cell, collect the hyperedge members and pad to neighbor_k."""
        n = H.shape[0]
        neighbor_idx = []
        for i in range(n):
            # column i contains cell i and its spatial neighbors
            members = H[:, i].nonzero()[0]
            # ensure self is first
            members = [i] + [m for m in members if m != i]
            members = np.array(members, dtype=np.int64)
            if len(members) < self.neighbor_k:
                pad = np.full(self.neighbor_k - len(members), i,
                              dtype=np.int64)
                members = np.concatenate([members, pad])
            else:
                members = members[:self.neighbor_k]
            neighbor_idx.append(members)
        return np.array(neighbor_idx, dtype=np.int64)

    def __len__(self):
        return self.x_feat.shape[0]

    def __getitem__(self, idx):
        n_idx = self.neighbor_indices[idx]
        return (self.x_feat[idx],
                self.y_expr[idx],
                self.x_feat[n_idx],
                self.y_expr[n_idx],
                self.coords[idx])


def collate_fn(batch):
    x = torch.stack([item[0] for item in batch])
    exp = torch.stack([item[1] for item in batch])
    x_neighbor = torch.stack([item[2] for item in batch])
    neighbor_exp = torch.stack([item[3] for item in batch])
    coords = torch.stack([item[4] for item in batch])
    return x, exp, x_neighbor, neighbor_exp, coords
