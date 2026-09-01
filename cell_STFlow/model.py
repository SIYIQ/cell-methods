"""cell_STFlow model: cell-level adaptation of HST_STFlow.

Drops the online image encoder and consumes pre-computed UNI features from
``adata.obsm['he']`` (1024-d). The STFlow mathematical core (Denoiser +
Interpolant) is imported from /home/sb202604/STFlow without source edits.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

# Pull in STFlow without modifying the original repo.
_STFLOW_ROOT = os.environ.get('STFLOW_ROOT', '/home/sb202604/STFlow')
if _STFLOW_ROOT not in sys.path:
    sys.path.insert(0, _STFLOW_ROOT)

from stflow.model.denoiser import Denoiser  # noqa: E402
from stflow.flow.interpolant import Interpolant  # noqa: E402
from stflow.model import transformer as _stflow_transformer  # noqa: E402


def _patch_stflow_gene_update():
    """Patch GeneUpdate.__init__ to accept the upstream ``non_negative`` kwarg.

    This is the same runtime patch used by HST_STFlow; it lets us import
    STFlow verbatim while avoiding a TypeError on Denoiser instantiation.
    """
    _GeneUpdate = _stflow_transformer.GeneUpdate
    _orig_init = _GeneUpdate.__init__
    import inspect
    sig = inspect.signature(_orig_init)
    if 'non_negative' in sig.parameters:
        return

    def _patched_init(self, d_model, n_genes, proj_drop=0., non_negative=True, **kwargs):
        _orig_init(self, d_model=d_model, n_genes=n_genes, proj_drop=proj_drop)
    _GeneUpdate.__init__ = _patched_init


_patch_stflow_gene_update()


def _patch_stflow_build_graph():
    """Replace STFlow's O(N^2) distance-matrix KNN with cKDTree.

    The original ``SpatialTransformer._build_graph`` materialises an [N, N]
    distance matrix, which blows up GPU memory on cell-level slices with
    50k-150k cells.  This patch builds a per-batch cKDTree on CPU and only
    keeps the top-k neighbor indices, cutting memory from O(N^2) to O(N).
    """
    from scipy.spatial import cKDTree

    _SpatialTransformer = _stflow_transformer.SpatialTransformer
    _orig_build_graph = _SpatialTransformer._build_graph

    # Avoid double-patching if the module is reloaded.
    if getattr(_orig_build_graph, '_cellstflow_patched', False):
        return

    def _patched_build_graph(self, coords, batch_idx, n_neighbors, exclude_self=True):
        coords_np = coords.detach().cpu().numpy()
        batch_idx_np = batch_idx.detach().cpu().numpy()
        n = coords_np.shape[0]
        n_neighbors = min(n_neighbors, n - 1) if exclude_self else min(n_neighbors, n)
        out = torch.empty((n, n_neighbors), dtype=torch.long, device=coords.device)

        unique_batches = np.unique(batch_idx_np)
        for b in unique_batches:
            mask = batch_idx_np == b
            idx = np.nonzero(mask)[0]
            n_b = idx.shape[0]
            if n_b == 0:
                continue
            # Query enough neighbors so that after dropping self we still have n_neighbors.
            k = min(n_neighbors + 1, n_b) if exclude_self else min(n_neighbors, n_b)
            tree = cKDTree(coords_np[idx])
            _, nn = tree.query(coords_np[idx], k=k)
            nn = np.atleast_2d(nn)  # [n_b, k]

            if exclude_self and nn.shape[1] > 1:
                # The closest point to each query is itself; drop it.
                nn_valid = nn[:, 1:]
            else:
                nn_valid = nn

            # Pad with the last column if we don't have enough neighbors.
            if nn_valid.shape[1] < n_neighbors:
                pad = np.repeat(nn_valid[:, -1:], n_neighbors - nn_valid.shape[1], axis=1)
                nn_valid = np.concatenate([nn_valid, pad], axis=1)

            # Clamp to exactly n_neighbors columns and write back as global indices.
            nn_valid = nn_valid[:, :n_neighbors]
            out[mask] = torch.from_numpy(idx[nn_valid]).to(coords.device)
        return out

    _patched_build_graph._cellstflow_patched = True
    _SpatialTransformer._build_graph = _patched_build_graph


_patch_stflow_build_graph()


def _make_denoiser_cfg(n_genes, feature_dim, hidden_dim, pairwise_hidden_dim,
                       n_layers, n_heads, dropout, attn_dropout, n_neighbors,
                       activation):
    """Argparse-style namespace expected by ``stflow.model.denoiser.Denoiser``."""
    return SimpleNamespace(
        n_genes=n_genes,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        pairwise_hidden_dim=pairwise_hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        attn_dropout=attn_dropout,
        n_neighbors=n_neighbors,
        activation=activation,
    )


class CellSTFlow(nn.Module):
    """UNI features + STFlow flow-matching denoiser for cell-level inputs.

    Forward path:
      1. ``x_feat`` [N, 1024] is passed straight into the Denoiser; its
         ``image_transform`` projects the UNI features into STFlow's d_model.
      2. ``coords`` [N, 2] are consumed by SpatialTransformer to build an
         internal KNN graph.
      3. Training corrupts gene expression with the Interpolant and runs the
         native MSE flow-matching loss.
      4. Inference samples from the ZINB prior and Euler-steps through
         ``n_sample_steps`` denoising iterations.
    """

    def __init__(
        self,
        n_genes=200,
        feature_dim=1024,
        hidden_dim=128,
        pairwise_hidden_dim=128,
        n_layers=4,
        n_heads=4,
        dropout=0.2,
        attn_dropout=0.2,
        n_neighbors=7,
        activation='swiglu',
        n_sample_steps=5,
        prior_sampler='zinb',
        zinb_logits=0.1,
        zinb_total_count=1.0,
        zinb_zi_logits=0.0,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.n_sample_steps = int(n_sample_steps)
        self.prior_sampler_type = prior_sampler

        cfg = _make_denoiser_cfg(
            n_genes=n_genes,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            pairwise_hidden_dim=pairwise_hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            attn_dropout=attn_dropout,
            n_neighbors=n_neighbors,
            activation=activation,
        )
        self.denoiser = Denoiser(cfg)

        self.interpolant = Interpolant(
            prior_sampler,
            total_count=torch.tensor([zinb_total_count]),
            logits=torch.tensor([zinb_logits]),
            zi_logits=zinb_zi_logits,
            normalize=(prior_sampler != 'gaussian'),
        )

    # ---------- training ----------

    def _sync_interpolant_device(self, device):
        """STFlow's Interpolant defaults to cuda at init; align it with inputs."""
        self.interpolant.device = str(device)

    def train_step(self, x_feat, gene_expr, coords):
        """One flow-matching loss step on a single cell slice.

        Args:
            x_feat:      [N, feature_dim] UNI features.
            gene_expr:   [N, n_genes] log1p target expression.
            coords:      [N, 2] spatial coordinates (microns).

        Returns:
            (pred [N, n_genes], loss scalar)
        """
        self._sync_interpolant_device(x_feat.device)
        # STFlow expects [B, N_cells, ...]; we always use B=1.
        x_feat_b = x_feat.unsqueeze(0)
        gene_b = gene_expr.unsqueeze(0)
        coords_b = coords.unsqueeze(0)

        noisy_exp, t_steps = self.interpolant.corrupt_exp(gene_b)
        pred, loss = self.denoiser(
            exp=noisy_exp,
            img_features=x_feat_b,
            coords=coords_b,
            labels=gene_b,
            t_steps=t_steps,
        )
        return pred.squeeze(0), loss

    # ---------- inference ----------

    @torch.no_grad()
    def predict(self, x_feat, coords):
        """Euler-step sampling to predict gene expression.

        Args:
            x_feat: [N, feature_dim] UNI features.
            coords: [N, 2] spatial coordinates.

        Returns:
            pred [N, n_genes].
        """
        device = x_feat.device
        self._sync_interpolant_device(device)
        x_feat_b = x_feat.unsqueeze(0)
        coords_b = coords.unsqueeze(0)

        shape = (1, x_feat.shape[0], self.n_genes)
        exp_t1 = self.interpolant.sample_from_prior(shape).to(device)

        ts = torch.linspace(0.01, 1.0, self.n_sample_steps)[:, None].expand(
            self.n_sample_steps, exp_t1.shape[0]
        ).to(device)

        pred = None
        for step, (t1, t2) in enumerate(zip(ts[:-1], ts[1:])):
            pred = self.denoiser.inference(
                exp_t1, x_feat_b, coords_b, t1, predict=True
            )
            d_t = t2 - t1
            if step == self.n_sample_steps - 2:
                break
            exp_t1 = self.interpolant.denoise(pred, exp_t1, t1, d_t)

        assert pred is not None, "n_sample_steps must be >= 2"
        return pred.squeeze(0)
