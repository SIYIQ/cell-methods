import os
import json
import numpy as np
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.module import (
    HGNN,
    EXPNN,
    TWOFusionEncoder,
    Decoder
)


class H0MiniEncoder(nn.Module):
    """H0-mini image encoder (ViT, 768-dim CLS token).

    Loaded from local path /home/sb202604/H0-mini.
    Output is [B, 768] after taking the CLS token.
    Backbone is frozen by default.
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

        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            h = self.backbone(x)  # [B, N, 768]
            if h.dim() == 3:
                h = h[:, 0]       # CLS token -> [B, 768]
        return h


class NGHist2ST(nn.Module):
    """NGHist2ST model with H0-mini encoder.

    Original design preserved:
      - HGNN neighbor encoder + EXPNN expression encoder
      - TWOFusionEncoder cross-attention fusion
      - Decoder prediction
      - Contrastive learning loss + MSE reconstruction loss

    Encoder changed from TenPercent ResNet18 to H0-mini (ViT, 768-dim).
    """

    def __init__(self, num_genes=50, emb_dim=512, depth1=2, num_heads1=8,
                 mlp_ratio1=2.0, dropout1=0.1, res_neighbor=(5, 5),
                 learning_rate=0.0001, temperature1=0.05, temperature2=0.05,
                 loss_ratio1=1.0, loss_ratio2=0.5):
        super().__init__()
        self.learning_rate = learning_rate
        self.best_loss = np.inf
        self.best_cor = -1
        self.num_genes = num_genes
        self.alpha = 0.3
        self.num_n = res_neighbor[0]
        self.ratio1 = loss_ratio1
        self.ratio2 = loss_ratio2
        self.temperature1 = temperature1
        self.temperature2 = temperature2

        # Target encoder: H0-mini (768-dim CLS token)
        self.target_encoder = H0MiniEncoder()
        self.fc_target = nn.Linear(emb_dim, num_genes)

        # Expression encoder
        self.exp_encoder = nn.Sequential(
            nn.Linear(num_genes, emb_dim),
            nn.Linear(emb_dim, emb_dim)
        )

        # Projection from H0-mini (768) to HGNN input dim (25088)
        # to preserve downstream architecture unchanged
        self.h0_to_hyper = nn.Linear(H0MiniEncoder.FEAT_DIM, 25088)

        # Neighbor encoders
        self.neighbor_encoder = HGNN(25088, 1024, 512)
        self.neighbor_exp_encoder = EXPNN(512, 1024, 512)
        self.fc_neighbor = nn.Linear(emb_dim, num_genes)
        self.fc_nuclei = nn.Linear(emb_dim, num_genes)

        self.fc_global = nn.Linear(emb_dim, num_genes)
        self.cross_encoder = TWOFusionEncoder(
            emb_dim, depth1, num_heads1, int(emb_dim * mlp_ratio1), dropout1
        )
        # H0-mini outputs 768-dim vectors; project to emb_dim
        self.fc = nn.Linear(H0MiniEncoder.FEAT_DIM, emb_dim)
        self.decoder = Decoder(input_dim=emb_dim, output_dim=num_genes)

    def contrastive_loss(self, features1, features2, temperature, negative_weight=0.1):
        if features1.dim() == 1:
            features1 = features1.unsqueeze(0)
        if features2.dim() == 1:
            features2 = features2.unsqueeze(0)
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)
        similarity_matrix = torch.mm(features1, features2.t()) / temperature
        batch_size = features1.size(0)
        mask = torch.eye(batch_size, device=features1.device)
        similarity_matrix = similarity_matrix * mask + similarity_matrix * (1 - mask) * negative_weight
        labels = torch.arange(batch_size, device=features1.device)
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss

    def build_graph(self, neighbor_nodes, neighbor_exp, k=3):
        num_nodes = neighbor_nodes.size(0)
        x = self.target_encoder(neighbor_nodes)   # [N, 768]
        x = self.h0_to_hyper(x)                   # [N, 25088]
        x_exp = self.exp_encoder(neighbor_exp)    # [N, emb_dim]
        x_norm = F.normalize(x, p=2, dim=1)       # [N, 25088]
        sim_matrix = torch.mm(x_norm, x_norm.T)
        mask = torch.eye(num_nodes, dtype=torch.bool, device=sim_matrix.device)
        sim_matrix = sim_matrix.masked_fill(mask, float("-inf"))
        _, topk_indices = torch.topk(sim_matrix, k=min(k, num_nodes - 1), dim=1)
        edge_index = []
        for i in range(num_nodes):
            for j in topk_indices[i]:
                if j >= 0:
                    edge_index.append([i, j.item()])
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        else:
            edge_index = torch.tensor([[], []], dtype=torch.long)
        return x, x_exp, edge_index

    def build_hypergraph(self, neighbor_nodes, neighbor_exp, k=3):
        num_nodes = neighbor_nodes.size(0)
        x = self.target_encoder(neighbor_nodes)   # [N, 768]
        x = self.h0_to_hyper(x)                   # [N, 25088]
        x_exp = self.exp_encoder(neighbor_exp)    # [N, emb_dim]
        x_norm = F.normalize(x, p=2, dim=1)       # [N, 25088]
        sim_matrix = torch.mm(x_norm, x_norm.T)
        mask = torch.eye(num_nodes, dtype=torch.bool, device=sim_matrix.device)
        sim_matrix = sim_matrix.masked_fill(mask, -1)
        _, topk_indices = torch.topk(sim_matrix, k=min(k, num_nodes - 1), dim=1)
        topk_indices = topk_indices.long()
        hyperedge_indices = []
        for i in range(num_nodes):
            hyperedge_indices.extend([(node_idx, i) for node_idx in topk_indices[i]])
            hyperedge_indices.append((i, i))
        if hyperedge_indices:
            rows, cols = zip(*hyperedge_indices)
            hyperedge_index = torch.tensor([rows, cols], dtype=torch.long)
        else:
            hyperedge_index = torch.tensor([[[], []]], dtype=torch.long)
        hyperedge_index = hyperedge_index.long()
        return x, x_exp, hyperedge_index

    def forward(self, x, exp, x_neighbor, x_neighbor_exp):
        x = x.squeeze()
        if x.dim() != 4:
            x = x.unsqueeze(0)
        # H0-mini encoder: [B, 3, 224, 224] -> [B, 768]
        x = self.target_encoder(x)
        # Project to emb_dim: [B, 768] -> [B, 1, emb_dim]
        x = x.unsqueeze(1)
        x = self.fc(x)

        x_neighbor = x_neighbor.to(torch.float32)
        x_neighbor_exp = x_neighbor_exp.to(torch.float32)

        if x_neighbor.dim() == 4:
            neighbor, neighbor_exp_tensor, hyperedge = self.build_hypergraph(x_neighbor, x_neighbor_exp)
        elif x_neighbor.dim() == 5:
            batch_size, num_patches, channels, height, width = x_neighbor.size()
            neighbor = []
            neighbor_exp_tensor = []
            hyperedge = []
            for i in range(batch_size):
                n, n_exp, h = self.build_hypergraph(x_neighbor[i].squeeze(0), x_neighbor_exp[i].squeeze(0))
                neighbor.append(n)
                neighbor_exp_tensor.append(n_exp)
                hyperedge.append(h)
            neighbor = torch.stack(neighbor).view(batch_size, -1, 25088).to(x_neighbor.device)
            neighbor_exp_tensor = torch.stack(neighbor_exp_tensor).view(batch_size, -1, 512).to(x_neighbor.device)
            hyperedge = torch.stack(hyperedge).to(x_neighbor.device)

        neighbor = neighbor.view(x.shape[0], -1, 25088).to(x.device)
        neighbor_exp_tensor = neighbor_exp_tensor.view(x.shape[0], -1, 512).to(x.device)
        if hyperedge.dim() == 2:
            hyperedge = hyperedge.unsqueeze(0)
        hyperedge = hyperedge.to(x.device)

        all_neighbors = []
        all_neighbor_exps = []
        for i in range(x.shape[0]):
            neighbor_i = neighbor[i]
            neighbor_exp_i = neighbor_exp_tensor[i]
            hyperedge_i = hyperedge[i]
            neighbors_i = self.neighbor_encoder(neighbor_i, hyperedge_i).view(1, -1).to(x.device)
            neighbor_exps_i = self.neighbor_exp_encoder(neighbor_exp_i, hyperedge_i).view(1, -1).to(x.device)
            all_neighbors.append(neighbors_i)
            all_neighbor_exps.append(neighbor_exps_i)
        neighbors = torch.stack(all_neighbors, dim=0)
        neighbor_exps = torch.stack(all_neighbor_exps, dim=0)

        patch_fusion = x.reshape(x.shape[0], -1).to(x.device)
        patch_exp = self.exp_encoder(exp).reshape(x.shape[0], -1).to(x.device)
        neighbors = neighbors.reshape(x.shape[0], -1).to(x.device)
        neighbor_exps = neighbor_exps.reshape(x.shape[0], -1).to(x.device)
        pred_exp = self.decoder(patch_fusion)
        decoded_exp = self.decoder(patch_fusion)

        patch_fusion = self.cross_encoder(patch_exp, patch_fusion)
        patch_exp = self.cross_encoder(patch_fusion, patch_exp)
        neighbors = self.cross_encoder(neighbor_exps, neighbors)
        neighbor_exps = self.cross_encoder(neighbors, neighbor_exps)

        return patch_fusion, patch_exp, neighbors, neighbor_exps, decoded_exp, pred_exp

    def compute_loss(self, outputs, exp):
        """Compute NH2ST loss: contrastive + reconstruction."""
        patch_fusion, patch_exp, neighbors, neighbor_exps, decoded_exp, pred_exp = outputs

        loss_patch = self.contrastive_loss(
            patch_fusion.squeeze(), patch_exp.squeeze(), self.temperature1
        )
        loss_neighbor = self.contrastive_loss(
            neighbors.squeeze(), neighbor_exps.squeeze(), self.temperature2
        )
        reconstruction_loss = F.mse_loss(decoded_exp, exp)

        loss = self.ratio1 * loss_patch + self.ratio2 * loss_neighbor + reconstruction_loss
        return loss

    def compute_metrics(self, pred, exp):
        """Compute per-gene MSE, MAE, and PCC."""
        pred_np = pred.detach().cpu().numpy()
        exp_np = exp.detach().cpu().numpy()

        mse = F.mse_loss(pred, exp).item()
        mae = F.l1_loss(pred, exp).item()

        pcc_list = []
        for g in range(exp_np.shape[1]):
            std_p = np.std(pred_np[:, g])
            std_e = np.std(exp_np[:, g])
            if std_p == 0 or std_e == 0:
                pcc_list.append(0.0)
                continue
            r = pearsonr(pred_np[:, g], exp_np[:, g])[0]
            pcc_list.append(0.0 if np.isnan(r) else float(r))

        mean_pcc = float(np.nanmean(pcc_list))

        return {
            'mse': mse,
            'mae': mae,
            'pcc_list': pcc_list,
            'mean_pcc': mean_pcc,
        }
