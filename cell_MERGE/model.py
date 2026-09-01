"""cell_MERGE model: cell-level MLP + GATNet refinement.

Replaces the CNN image encoder in HST_MERGE with an MLP on pre-computed
UNI features (adata.obsm['he'], 1024-d). The GATNet graph refinement stage
is kept unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge
from torch_geometric.nn import GATConv, LayerNorm


class CellMLP(nn.Module):
    """Cell-level feature projector: UNI (1024) -> hidden (256) -> num_genes."""

    def __init__(self, in_dim=1024, num_genes=200, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.input_mlp = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_dim, num_genes)

    def forward(self, x):
        h = F.relu(self.input_mlp(x))
        h = self.dropout(h)
        return self.fc(h)

    def extract_features(self, x):
        """Return the 256-dim embedding used by the downstream GATNet."""
        return F.relu(self.input_mlp(x))


class GATNet(torch.nn.Module):
    """4-layer Graph Attention Network for gene expression refinement.

    Input: 256-dim patch embeddings (from CellMLP.extract_features).
    Output: per-node gene expression predictions.

    Architecture (from MERGE):
      - 4 GATConv layers with 8 attention heads
      - Hidden dims: 256 -> 448 -> 384 -> 256 -> num_genes
      - LayerNorm after each layer
      - Edge dropout (p=0.2) during training
    """

    def __init__(self, num_genes, num_heads=8, drop_edge=0.2):
        super(GATNet, self).__init__()
        dim1 = 448
        dim2 = 384
        dim3 = 256
        headn = num_heads

        self.drop_edge = drop_edge

        self.nn1 = GATConv(256, dim1, headn)
        self.layer_norm1 = LayerNorm(dim1 * headn)

        self.nn2 = GATConv(dim1 * headn, dim2, headn)
        self.layer_norm2 = LayerNorm(dim2 * headn)

        self.nn3 = GATConv(dim2 * headn, dim3, headn)
        self.layer_norm3 = LayerNorm(dim3 * headn)

        self.nn4 = GATConv(dim3 * headn, num_genes)

    def forward(self, x, edge_index):
        edge_index, _ = dropout_edge(edge_index, p=self.drop_edge, training=self.training)

        x = F.relu(self.nn1(x, edge_index))
        x = self.layer_norm1(x)

        x = F.relu(self.nn2(x, edge_index))
        x = self.layer_norm2(x)

        x = F.relu(self.nn3(x, edge_index))
        x = self.layer_norm3(x)

        x = self.nn4(x, edge_index)
        return x
