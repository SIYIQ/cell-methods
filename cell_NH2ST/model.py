"""cell_NH2ST model: cell-level NGHist2ST.

Replaces the H0-mini image encoder in HST_NH2ST with projections on
pre-computed UNI features (adata.obsm['he'], 1024-d). The neighbor hypergraph,
contrastive losses, and reconstruction loss are preserved by reusing the
original NH2ST module building blocks from ``models.module``.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr

from models.module import HGNN, EXPNN, TWOFusionEncoder, Decoder


class CellNGHist2ST(nn.Module):
    """NGHist2ST adapted for cell-level UNI features."""

    def __init__(self, num_genes=200, emb_dim=512, depth1=2, num_heads1=8,
                 mlp_ratio1=2.0, dropout1=0.1, temperature1=0.05,
                 temperature2=0.05, loss_ratio1=1.0, loss_ratio2=0.5):
        super().__init__()
        self.num_genes = num_genes
        self.ratio1 = loss_ratio1
        self.ratio2 = loss_ratio2
        self.temperature1 = temperature1
        self.temperature2 = temperature2

        # Target cell: UNI 1024 -> emb_dim
        self.target_fc = nn.Linear(1024, emb_dim)

        # Expression encoder: num_genes -> emb_dim -> emb_dim
        self.exp_encoder = nn.Sequential(
            nn.Linear(num_genes, emb_dim),
            nn.Linear(emb_dim, emb_dim)
        )

        # Neighbor cells: UNI 1024 -> 25088 to keep the original HGNN dims
        self.neighbor_proj = nn.Linear(1024, 25088)

        self.neighbor_encoder = HGNN(25088, 1024, 512)
        self.neighbor_exp_encoder = EXPNN(512, 1024, 512)

        self.cross_encoder = TWOFusionEncoder(
            emb_dim, depth1, num_heads1, int(emb_dim * mlp_ratio1), dropout1
        )
        self.decoder = Decoder(input_dim=emb_dim, output_dim=num_genes)

    def contrastive_loss(self, features1, features2, temperature):
        if features1.dim() == 1:
            features1 = features1.unsqueeze(0)
        if features2.dim() == 1:
            features2 = features2.unsqueeze(0)
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)
        similarity_matrix = torch.mm(features1, features2.t()) / temperature
        batch_size = features1.size(0)
        mask = torch.eye(batch_size, device=features1.device)
        negative_weight = 0.1
        similarity_matrix = (similarity_matrix * mask
                             + similarity_matrix * (1 - mask) * negative_weight)
        labels = torch.arange(batch_size, device=features1.device)
        return F.cross_entropy(similarity_matrix, labels)

    def build_hypergraph(self, neighbor_features, neighbor_exp, k=3):
        """Build a similarity hypergraph among neighbor cells.

        Args:
            neighbor_features: [N, 1024] UNI features of neighbors.
            neighbor_exp: [N, n_genes] expression of neighbors.
            k: top-k similarity edges per node.

        Returns:
            x [N, 25088], x_exp [N, emb_dim], hyperedge_index [2, E]
        """
        num_nodes = neighbor_features.size(0)
        x = self.neighbor_proj(neighbor_features)  # [N, 25088]
        x_exp = self.exp_encoder(neighbor_exp)     # [N, emb_dim]
        x_norm = F.normalize(x, p=2, dim=1)
        sim_matrix = torch.mm(x_norm, x_norm.T)
        mask = torch.eye(num_nodes, dtype=torch.bool, device=sim_matrix.device)
        sim_matrix = sim_matrix.masked_fill(mask, -1)
        _, topk_indices = torch.topk(sim_matrix, k=min(k, num_nodes - 1), dim=1)
        topk_indices = topk_indices.long()

        hyperedge_indices = []
        for i in range(num_nodes):
            hyperedge_indices.extend([(node_idx.item(), i)
                                      for node_idx in topk_indices[i]])
            hyperedge_indices.append((i, i))
        if hyperedge_indices:
            rows, cols = zip(*hyperedge_indices)
            hyperedge_index = torch.tensor([rows, cols], dtype=torch.long,
                                           device=neighbor_features.device)
        else:
            hyperedge_index = torch.zeros((2, 0), dtype=torch.long,
                                          device=neighbor_features.device)
        return x, x_exp, hyperedge_index

    def forward(self, x, exp, x_neighbor, x_neighbor_exp):
        """Forward for cell-level features.

        Args:
            x: [B, 1024] target cell UNI features.
            exp: [B, n_genes] target expression.
            x_neighbor: [B, k, 1024] neighbor UNI features.
            x_neighbor_exp: [B, k, n_genes] neighbor expression.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # Target branch
        x = self.target_fc(x).unsqueeze(1)  # [B, 1, emb_dim]

        x_neighbor = x_neighbor.to(torch.float32)
        x_neighbor_exp = x_neighbor_exp.to(torch.float32)

        if x_neighbor.dim() == 2:
            # Single sample: [k, 1024]
            n, n_exp, h = self.build_hypergraph(x_neighbor, x_neighbor_exp)
            neighbor = n.unsqueeze(0)
            neighbor_exp_tensor = n_exp.unsqueeze(0)
            hyperedge = h.unsqueeze(0)
        else:
            # Batch: [B, k, 1024]
            neighbor_list, nexp_list, hedge_list = [], [], []
            for i in range(x_neighbor.size(0)):
                n, n_exp, h = self.build_hypergraph(x_neighbor[i],
                                                    x_neighbor_exp[i])
                neighbor_list.append(n)
                nexp_list.append(n_exp)
                hedge_list.append(h)
            neighbor = torch.stack(neighbor_list).to(x_neighbor.device)
            neighbor_exp_tensor = torch.stack(nexp_list).to(x_neighbor.device)
            hyperedge = torch.stack(hedge_list).to(x_neighbor.device)

        neighbor = neighbor.view(x.size(0), -1, 25088).to(x.device)
        neighbor_exp_tensor = neighbor_exp_tensor.view(x.size(0), -1,
                                                       512).to(x.device)
        if hyperedge.dim() == 2:
            hyperedge = hyperedge.unsqueeze(0)
        hyperedge = hyperedge.to(x.device)

        all_neighbors = []
        all_neighbor_exps = []
        for i in range(x.size(0)):
            neighbor_i = neighbor[i]
            neighbor_exp_i = neighbor_exp_tensor[i]
            hyperedge_i = hyperedge[i]
            neighbors_i = self.neighbor_encoder(neighbor_i, hyperedge_i).view(
                1, -1).to(x.device)
            neighbor_exps_i = self.neighbor_exp_encoder(
                neighbor_exp_i, hyperedge_i).view(1, -1).to(x.device)
            all_neighbors.append(neighbors_i)
            all_neighbor_exps.append(neighbor_exps_i)
        neighbors = torch.stack(all_neighbors, dim=0)
        neighbor_exps = torch.stack(all_neighbor_exps, dim=0)

        patch_fusion = x.reshape(x.size(0), -1).to(x.device)
        patch_exp = self.exp_encoder(exp).reshape(x.size(0), -1).to(x.device)
        neighbors = neighbors.reshape(x.size(0), -1).to(x.device)
        neighbor_exps = neighbor_exps.reshape(x.size(0), -1).to(x.device)

        pred_exp = self.decoder(patch_fusion)
        decoded_exp = self.decoder(patch_fusion)

        patch_fusion = self.cross_encoder(patch_exp, patch_fusion)
        patch_exp = self.cross_encoder(patch_fusion, patch_exp)
        neighbors = self.cross_encoder(neighbor_exps, neighbors)
        neighbor_exps = self.cross_encoder(neighbors, neighbor_exps)

        return patch_fusion, patch_exp, neighbors, neighbor_exps, decoded_exp, pred_exp

    def compute_loss(self, outputs, exp):
        """NH2ST loss: contrastive + reconstruction."""
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
        """Per-gene MSE, MAE, and PCC."""
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
        return {'mse': mse, 'mae': mae, 'pcc_list': pcc_list,
                'mean_pcc': mean_pcc}
