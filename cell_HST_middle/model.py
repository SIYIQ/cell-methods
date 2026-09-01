"""cell_HST_middle model: cell-level adaptation of HST-middle's HeteroST.

HeteroST is a Heterogeneous Graph Transformer over a two-view graph:
each cell appears twice — once as an 'image' node (UNI feature) and once
as a 'gene' node (encoded gene expression, or its image-derived fallback
during inference / MGM-masked spots). Edge types:

    image --corresponds_to-->   gene     (cell <-> own gene representation)
    gene  --corresponds_to-->   image
    image --spatially_adjacent--> image  (cell spatial KNN)
    gene  --spatially_adjacent--> gene   (same KNN, mirrored on the gene branch)
    image --morphologically_similar--> image  (cell UNI top-K similarity)
    image --morphologically_similar_rev--> image  (flip of the above)

This module is parametric over cells (not spots) — the architecture is
identical to the original HST-middle HeteroST. We only:
  - drop the `FlexibleImageEncoder` (UNI features are pre-computed, see
    /home/sb202604/cell-benchmark/processed_cell/),
  - simplify the forward signature: x_img is the raw UNI feature [N, 1024],
    projected into hidden_dim by an input MLP before the HGT layers.

MGM (Masked Gene Modeling) is preserved as in the original: if `mask` is
given, masked cells use the image->gene fallback while unmasked cells use
the real gene encoder, providing a self-supervised auxiliary signal.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv


class ImageToGene(nn.Module):
    """Projects image-node features into the gene-node feature space.

    Used as the fallback encoder when a cell's gene_expr is unavailable
    (test time, or training-time masked cells under MGM).
    """

    def __init__(self, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class GeneEncoder(nn.Module):
    """Encodes raw gene_expr [N, n_genes] into the gene-node feature space."""

    def __init__(self, n_genes=200, hidden_dim=512, dropout=0.1, pca_components=None):
        super().__init__()
        if pca_components is not None:
            self.pca_proj = nn.Linear(n_genes, hidden_dim, bias=False)
            self.pca_proj.weight = nn.Parameter(
                torch.from_numpy(pca_components).float(), requires_grad=False
            )
            first_dim = hidden_dim
        else:
            self.pca_proj = None
            first_dim = n_genes

        self.mlp = nn.Sequential(
            nn.Linear(first_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        if self.pca_proj is not None:
            x = self.pca_proj(x)
        return self.mlp(x)


class CellHST(nn.Module):
    """HST-middle HeteroST adapted for cell-level inputs (UNI features)."""

    def __init__(
        self,
        in_dim: int = 1024,
        n_genes: int = 200,
        hidden_dim: int = 256,
        heads: int = 4,
        dropout: float = 0.1,
        n_hgt_layers: int = 3,
        pred_head_depth: int = 3,
        pca_components=None,
    ):
        super().__init__()
        self.n_hgt_layers = n_hgt_layers
        self.hidden_dim = hidden_dim
        self.n_genes = n_genes

        # Project pre-computed UNI features (1024) into hidden_dim. Matches
        # SpatialEx Predictor_spot and cell_novae's input_mlp so the only
        # variable across methods is the graph backbone.
        self.input_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim),
        )

        self.img_to_gene = ImageToGene(hidden_dim=hidden_dim, dropout=dropout)
        self.gene_encoder = GeneEncoder(
            n_genes=n_genes, hidden_dim=hidden_dim, dropout=dropout,
            pca_components=pca_components,
        )

        metadata = (
            ['image', 'gene'],
            [
                ('image', 'corresponds_to', 'gene'),
                ('gene', 'corresponds_to', 'image'),
                ('image', 'spatially_adjacent', 'image'),
                ('gene', 'spatially_adjacent', 'gene'),
                ('image', 'morphologically_similar', 'image'),
                ('image', 'morphologically_similar_rev', 'image'),
            ]
        )

        self.convs = nn.ModuleList(
            [HGTConv(hidden_dim, hidden_dim, metadata, heads=heads)
             for _ in range(n_hgt_layers)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(n_hgt_layers)])

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Predictor head: outputs 2*n_genes (mean + log_var).
        pred_layers = []
        dims = [hidden_dim] + [256, 128][: pred_head_depth - 1] + [n_genes * 2]
        for i in range(len(dims) - 1):
            pred_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                pred_layers.append(nn.ReLU())
                pred_layers.append(nn.Dropout(dropout))
        self.pred_head = nn.Sequential(*pred_layers)

    def forward(self, x_feat, data, gene_expr=None, mask=None, return_log_var=False):
        """Args:
            x_feat: [N, in_dim] pre-computed UNI features
            data: torch_geometric.data.HeteroData with edge_index_dict
                  populated for the 6 edge types.
            gene_expr: [N, n_genes] optional log1p expression for MGM.
            mask: [N] bool tensor, True = use image->gene fallback. If
                  gene_expr or mask is None, all cells go through img_to_gene.
        """
        x_img = self.input_mlp(x_feat)

        if gene_expr is None or mask is None:
            x_gene = self.img_to_gene(x_img)
        else:
            mask_b = mask.to(dtype=torch.bool, device=x_img.device)
            x_gene_real = self.gene_encoder(gene_expr)
            x_gene_img = self.img_to_gene(x_img)
            x_gene = torch.where(mask_b.unsqueeze(-1), x_gene_img, x_gene_real)

        x_dict = {'image': x_img, 'gene': x_gene}
        for i in range(self.n_hgt_layers):
            out_dict = self.convs[i](x_dict, data.edge_index_dict)
            for k in out_dict.keys():
                out_dict[k] = self.norms[i](x_dict[k] + out_dict[k])
                out_dict[k] = self.dropout(self.act(out_dict[k]))
            x_dict = out_dict

        pred = self.pred_head(x_dict['gene'])
        if return_log_var:
            mean, log_var = pred.chunk(2, dim=-1)
            return mean, log_var
        return pred[..., :self.n_genes]
