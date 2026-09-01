import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge
from torch_geometric.nn import GATConv, LayerNorm


class H0MiniEncoder(nn.Module):
    """H0-mini image encoder (ViT, 768-dim CLS token).

    Loaded from local path /home/sb202604/H0-mini.
    Output is [B, 768] after taking the CLS token.
    Backbone is frozen by default (no gradient).
    """

    LOCAL_PATH = '/home/sb202604/H0-mini'
    FEAT_DIM = 768

    def __init__(self):
        super().__init__()
        import timm

        config_path = os.path.join(self.LOCAL_PATH, 'config.json')
        with open(config_path, 'r') as f:
            model_cfg = json.load(f)

        model_args = model_cfg.get('model_args', {})
        model_args['mlp_layer'] = timm.layers.SwiGLUPacked
        model_args['act_layer'] = torch.nn.SiLU

        self.backbone = timm.create_model(
            model_cfg['architecture'],
            pretrained=False,
            **model_args,
        )

        bin_path = os.path.join(self.LOCAL_PATH, 'pytorch_model.bin')
        sft_path = os.path.join(self.LOCAL_PATH, 'model.safetensors')
        if os.path.isfile(bin_path):
            state_dict = torch.load(bin_path, map_location='cpu')
        elif os.path.isfile(sft_path):
            from safetensors.torch import load_file
            state_dict = load_file(sft_path, device='cpu')
        else:
            raise FileNotFoundError(
                f"No weights found in {self.LOCAL_PATH}"
            )
        self.backbone.load_state_dict(state_dict, strict=True)
        print(f"[H0MiniEncoder] Loaded from {self.LOCAL_PATH}")

        # Freeze backbone (frozen by design, same as HST-middle default)
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            h = self.backbone(x)  # [B, N, 768]
            if h.dim() == 3:
                h = h[:, 0]       # CLS token -> [B, 768]
        return h


class CNN_Predictor(nn.Module):
    """CNN gene expression predictor with H0-mini encoder.

    H0-mini (768-dim) -> projector (256-dim) -> ReLU -> Dropout -> FC(num_genes)
    """
    def __init__(self, num_genes, device='cpu', dropout=0.2, pretrained_path=None):
        super(CNN_Predictor, self).__init__()
        self.encoder = H0MiniEncoder().to(device)
        self.projector = nn.Linear(H0MiniEncoder.FEAT_DIM, 256).to(device)

        if pretrained_path is not None:
            self.load_state_dict(
                torch.load(pretrained_path, map_location=device),
                strict=False,
            )

        self.dropout = nn.Dropout(p=dropout).to(device)
        self.fc = nn.Linear(256, num_genes).to(device)

    def forward(self, x):
        h = self.encoder(x)          # [B, 768]
        x = self.projector(h)        # [B, 256]
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class GATNet(torch.nn.Module):
    """4-layer Graph Attention Network for gene expression refinement.

    Input: 256-dim patch embeddings (from CNN_Predictor projector).
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

        # First input dim is 256 (H0-mini -> projector output)
        self.nn1 = GATConv(256, dim1, headn)
        self.layer_norm1 = LayerNorm(dim1 * headn)

        self.nn2 = GATConv(dim1 * headn, dim2, headn)
        self.layer_norm2 = LayerNorm(dim2 * headn)

        self.nn3 = GATConv(dim2 * headn, dim3, headn)
        self.layer_norm3 = LayerNorm(dim3 * headn)

        # Output dim is the number of genes
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
