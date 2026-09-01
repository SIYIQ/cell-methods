"""HST_STFlow model: H&E image encoder + STFlow flow-matching denoiser.

This is the "STFlow" comparison variant of the HST-middle pipeline. It keeps
the data pipeline (offline cache, ImageNet-normalized patches, log1p adata,
HVG selection, k-fold CV, grid+halo training, PCC eval) identical to
HST-middle / HST_novae, so the only difference between the runs is the
gene-prediction architecture.

The STFlow side is imported from /home/sb202604/STFlow without modification:
  - stflow.model.denoiser.Denoiser:    SpatialTransformer + Fourier time emb
  - stflow.flow.interpolant.Interpolant: ZINB prior + linear interpolant

The image encoder is the same FlexibleImageEncoder used in HST-middle /
HST_novae (so h0_mini partial unfreeze, ImageNet normalization, gradient
checkpointing, etc. all behave identically). Its output (a hidden_dim vector
per spot) is projected by Denoiser.image_transform into STFlow's d_model
space exactly as it would be for STFlow's official UNI features — except
here the encoder is trainable on h0_mini patches instead of reading
pre-extracted UNI embeddings from disk.
"""

import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn
from torchvision import models

# Pull in STFlow without modifying the original repo.
_STFLOW_ROOT = os.environ.get('STFLOW_ROOT', '/home/sb202604/STFlow')
if _STFLOW_ROOT not in sys.path:
    sys.path.insert(0, _STFLOW_ROOT)

from stflow.model.denoiser import Denoiser  # noqa: E402
from stflow.flow.interpolant import Interpolant  # noqa: E402
from stflow.model import transformer as _stflow_transformer  # noqa: E402


def _patch_stflow_gene_update():
    """Make stflow.model.transformer.GeneUpdate accept (and ignore) the
    `non_negative=` kwarg that TransformerBlock.__init__ passes in but
    GeneUpdate.__init__ never declared.

    This is a known upstream bug in /home/sb202604/STFlow: TransformerBlock
    has a `gene_exp_non_negative` flag (default True) that it forwards as
    `non_negative=` to GeneUpdate, but the latter's signature drops it,
    so any Denoiser instantiation crashes with `TypeError: GeneUpdate
    got an unexpected keyword argument 'non_negative'`. We honor the
    instruction to leave STFlow's native source untouched by patching the
    class in-place from this wrapper at import time. The kwarg is ignored
    because GeneUpdate has no implementation path that uses it (the
    final nn.Linear has no non-negativity constraint applied to its
    output in the current STFlow source).
    """
    _GeneUpdate = _stflow_transformer.GeneUpdate
    _orig_init = _GeneUpdate.__init__
    import inspect
    sig = inspect.signature(_orig_init)
    if 'non_negative' in sig.parameters:
        return  # upstream already patched

    def _patched_init(self, d_model, n_genes, proj_drop=0., non_negative=True, **kwargs):
        _orig_init(self, d_model=d_model, n_genes=n_genes, proj_drop=proj_drop)
    _GeneUpdate.__init__ = _patched_init


_patch_stflow_gene_update()


TORCHVISION_ENCODERS = {
    'resnet50': 'ResNet50_Weights',
    'resnet101': 'ResNet101_Weights',
    'resnext50_32x4d': 'ResNeXt50_32X4D_Weights',
    'convnext_tiny': 'ConvNeXt_Tiny_Weights',
    'convnext_small': 'ConvNeXt_Small_Weights',
    'efficientnet_b0': 'EfficientNet_B0_Weights',
    'efficientnet_b3': 'EfficientNet_B3_Weights',
    'regnet_y_8gf': 'RegNet_Y_8GF_Weights',
    'swin_t': 'Swin_T_Weights',
}

TIMM_ENCODERS = {
    'uni': {'model_name': 'hf_hub:MahmoodLab/UNI', 'feat_dim': 1024},
    'h0_mini': {'model_name': 'hf_hub:bioptimus/H0-mini', 'feat_dim': 768},
}


class FlexibleImageEncoder(nn.Module):
    """Image encoder supporting torchvision and timm backbones (unchanged from HST-middle)."""

    def __init__(self, encoder_name='resnet50', out_dim=512, unfreeze_backbone=False,
                 local_path=None, unfreeze_last_n_blocks=None):
        super().__init__()
        self.encoder_name = encoder_name
        self.unfreeze_backbone = unfreeze_backbone
        self.unfreeze_last_n_blocks = unfreeze_last_n_blocks

        if encoder_name in TORCHVISION_ENCODERS:
            feat_dim = self._build_torchvision_encoder(encoder_name)
        elif encoder_name in TIMM_ENCODERS:
            feat_dim = self._build_timm_encoder(
                encoder_name, local_path=local_path,
                unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            )
        else:
            raise ValueError(
                f"Unsupported encoder: {encoder_name}. "
                f"Supported torchvision: {list(TORCHVISION_ENCODERS.keys())}, "
                f"Supported timm/HF: {list(TIMM_ENCODERS.keys())}"
            )

        self.projector = nn.Linear(feat_dim, out_dim)

    def _build_torchvision_encoder(self, name):
        model = self._load_torchvision_model(name)
        if 'resnet' in name or 'resnext' in name or 'wide_resnet' in name:
            feat_dim = model.fc.in_features
            self.backbone = nn.Sequential(*list(model.children())[:-1])
            self.norm = None
            self.is_swin = False
            self.is_vit = False
        elif 'convnext' in name:
            feat_dim = model.classifier[2].in_features
            self.backbone = model.features
            self.norm = model.classifier[0]
            self.is_swin = False
            self.is_vit = False
        elif 'efficientnet' in name:
            feat_dim = model.classifier[1].in_features
            self.backbone = model.features
            self.norm = None
            self.is_swin = False
            self.is_vit = False
        elif 'regnet' in name:
            feat_dim = model.fc.in_features
            self.backbone = nn.Sequential(model.stem, model.trunk_output)
            self.norm = None
            self.is_swin = False
            self.is_vit = False
        elif 'swin' in name:
            feat_dim = model.head.in_features
            self.backbone = model.features
            self.norm = model.norm
            self.is_swin = True
            self.is_vit = False
        else:
            raise ValueError(f"Unsupported torchvision encoder: {name}")

        self.pool = nn.AdaptiveAvgPool2d(1)

        if not self.unfreeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        return feat_dim

    def _build_timm_encoder(self, name, local_path=None, unfreeze_last_n_blocks=None):
        import timm
        import json

        cfg = TIMM_ENCODERS[name]

        if local_path is not None and os.path.isdir(local_path):
            config_path = os.path.join(local_path, 'config.json')
            with open(config_path, 'r') as f:
                model_cfg = json.load(f)
            model_args = model_cfg.get('model_args', {})
            if name == 'h0_mini':
                model_args['mlp_layer'] = timm.layers.SwiGLUPacked
                model_args['act_layer'] = torch.nn.SiLU
            self.backbone = timm.create_model(
                model_cfg['architecture'],
                pretrained=False,
                **model_args,
            )
            bin_path = os.path.join(local_path, 'pytorch_model.bin')
            sft_path = os.path.join(local_path, 'model.safetensors')
            if os.path.isfile(bin_path):
                state_dict = torch.load(bin_path, map_location='cpu')
            elif os.path.isfile(sft_path):
                from safetensors.torch import load_file
                state_dict = load_file(sft_path, device='cpu')
            else:
                raise FileNotFoundError(
                    f"No weights found in {local_path}: expected "
                    f"'pytorch_model.bin' or 'model.safetensors'."
                )
            self.backbone.load_state_dict(state_dict, strict=True)
            print(f"Loaded {name} from local path: {local_path}")
        else:
            self.backbone = timm.create_model(
                cfg['model_name'], pretrained=True, num_classes=0, global_pool='',
            )
            print(f"Loaded {name} from timm/HuggingFace.")

        self.norm = None
        self.is_swin = False
        self.is_vit = True
        self.pool = None

        if not self.unfreeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif unfreeze_last_n_blocks is not None and unfreeze_last_n_blocks > 0:
            for param in self.backbone.parameters():
                param.requires_grad = False
            n = int(unfreeze_last_n_blocks)
            if not hasattr(self.backbone, 'blocks') or len(self.backbone.blocks) < n:
                raise ValueError(
                    f"Cannot unfreeze last {n} blocks: backbone exposes "
                    f"{len(getattr(self.backbone, 'blocks', []))} blocks."
                )
            for block in self.backbone.blocks[-n:]:
                for param in block.parameters():
                    param.requires_grad = True
            if hasattr(self.backbone, 'norm') and self.backbone.norm is not None:
                for param in self.backbone.norm.parameters():
                    param.requires_grad = True
            n_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
            print(f"Partial unfreeze: last {n} blocks + final norm trainable "
                  f"({n_trainable:,} params); earlier layers frozen.")

        return cfg['feat_dim']

    def _load_torchvision_model(self, name):
        model_fn = getattr(models, name)
        weights_enum_name = TORCHVISION_ENCODERS[name]
        try:
            weights = getattr(models, weights_enum_name).DEFAULT
            return model_fn(weights=weights)
        except Exception as e:
            print(f"Warning: Failed to load pretrained weights for {name}: {e}")
            print("Falling back to random initialization.")
            return model_fn(weights=None)

    def forward(self, x):
        if not self.unfreeze_backbone:
            with torch.no_grad():
                h = self._backbone_forward(x)
        elif (self.training
              and self.unfreeze_last_n_blocks is not None
              and self.unfreeze_last_n_blocks > 0):
            from torch.utils.checkpoint import checkpoint
            h = checkpoint(self._backbone_forward, x, use_reentrant=False)
        else:
            h = self._backbone_forward(x)
        return self.projector(h)

    def _backbone_forward(self, x):
        if self.is_vit:
            h = self.backbone(x)
            if h.dim() == 3:
                h = h[:, 0]
        else:
            h = self.backbone(x)
            if self.is_swin:
                h = self.norm(h)
                h = h.permute(0, 3, 1, 2)
            elif self.norm is not None:
                h = self.norm(h)
            h = self.pool(h)
            h = h.flatten(1)
        return h


def _make_denoiser_cfg(n_genes, encoder_out_dim, hidden_dim, pairwise_hidden_dim,
                       n_layers, n_heads, dropout, attn_dropout, n_neighbors,
                       activation):
    """Build the argparse-style namespace that stflow.model.denoiser.Denoiser expects.

    The fields below match the names that Denoiser reads off `config` in
    stflow.model.denoiser:53. `feature_dim` is the dimensionality of the
    per-spot features we feed in — i.e. the FlexibleImageEncoder output
    width — because Denoiser.image_transform = nn.Linear(feature_dim, hidden_dim).
    """
    return SimpleNamespace(
        n_genes=n_genes,
        feature_dim=encoder_out_dim,
        hidden_dim=hidden_dim,
        pairwise_hidden_dim=pairwise_hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        attn_dropout=attn_dropout,
        n_neighbors=n_neighbors,
        activation=activation,
    )


class STFlowModel(nn.Module):
    """H&E image encoder + STFlow flow-matching denoiser.

    Forward path:
      1. ImageNet-normalized [N, 3, 224, 224] patches → FlexibleImageEncoder
         → [N, encoder_out_dim] per-spot features.
      2. Those features + a noisy gene-expression draw + a time-step are
         passed through stflow.model.denoiser.Denoiser.inference(...) which
         internally projects features → d_model, adds a Fourier time
         embedding, then runs a SpatialTransformer that builds its own
         KNN graph from `coords`.
      3. At training time, we use the Interpolant to corrupt the target
         gene expression and compute STFlow's native MSE flow-matching loss
         (Denoiser.forward).
      4. At inference time, we sample from the ZINB prior and Euler-step
         through n_sample_steps with Interpolant.denoise. The final
         prediction is returned as the per-spot, per-gene expression.

    The encoder + denoiser are wired up here so the training script
    (`train.py`) does not have to know about Interpolant / flow-matching
    details — it just calls `model.train_step` and `model.predict`.
    """

    def __init__(
        self,
        n_genes=200,
        encoder_out_dim=512,
        hidden_dim=128,
        pairwise_hidden_dim=128,
        n_layers=4,
        n_heads=4,
        dropout=0.2,
        attn_dropout=0.2,
        n_neighbors=8,
        activation='swiglu',
        n_sample_steps=5,
        prior_sampler='zinb',
        zinb_logits=0.1,
        zinb_total_count=1.0,
        zinb_zi_logits=0.0,
        encoder_name='resnet50',
        encoder_unfreeze=False,
        encoder_local_path=None,
        encoder_unfreeze_last_n_blocks=None,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.n_sample_steps = int(n_sample_steps)
        self.prior_sampler_type = prior_sampler

        self.image_encoder = FlexibleImageEncoder(
            encoder_name=encoder_name,
            out_dim=encoder_out_dim,
            unfreeze_backbone=encoder_unfreeze,
            local_path=encoder_local_path,
            unfreeze_last_n_blocks=encoder_unfreeze_last_n_blocks,
        )

        cfg = _make_denoiser_cfg(
            n_genes=n_genes,
            encoder_out_dim=encoder_out_dim,
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

    # ---------- shared helpers ----------

    def _encode_patches(self, imgs, batch_size=128, training=None):
        """Run FlexibleImageEncoder over [N, 3, H, W] in chunks.

        Returns [N, encoder_out_dim] on imgs.device. When training is True
        we keep grads on; otherwise we wrap in no_grad to save memory at
        eval time."""
        if training is None:
            training = self.image_encoder.training

        outs = []
        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for i in range(0, imgs.shape[0], batch_size):
                outs.append(self.image_encoder(imgs[i:i + batch_size]))
        return torch.cat(outs, dim=0)

    # ---------- training ----------

    def train_step(self, imgs, gene_expr, coords, img_batch_size=128):
        """One flow-matching loss step on a single slide / grid patch.

        Args:
            imgs:      [N, 3, 224, 224] ImageNet-normalized patches.
            gene_expr: [N, n_genes] log1p ST target (matched to imgs).
            coords:    [N, 2] spatial coordinates (raw pixel units).

        Returns:
            (pred [N, n_genes], loss scalar)
        """
        img_feats = self._encode_patches(imgs, batch_size=img_batch_size, training=True)

        # STFlow Denoiser expects [B, N_cells, ...]. We always use B=1 here
        # (one slide / grid at a time) and rely on the SpatialTransformer's
        # internal padding logic by adding the batch dim ourselves.
        img_feats_b = img_feats.unsqueeze(0)
        gene_b = gene_expr.unsqueeze(0)
        coords_b = coords.unsqueeze(0)

        noisy_exp, t_steps = self.interpolant.corrupt_exp(gene_b)
        pred, loss = self.denoiser(
            exp=noisy_exp,
            img_features=img_feats_b,
            coords=coords_b,
            labels=gene_b,
            t_steps=t_steps,
        )
        return pred.squeeze(0), loss

    # ---------- inference ----------

    @torch.no_grad()
    def predict(self, imgs, coords, img_batch_size=128):
        """Run STFlow's Euler-step sampling to predict gene expression.

        Mirrors stflow.app.flow.test.test: sample exp_t1 from the prior,
        then take (n_sample_steps - 1) denoise steps; the last call returns
        the final pred without a further denoise step.

        Args:
            imgs:   [N, 3, 224, 224] patches.
            coords: [N, 2] spatial coordinates.

        Returns:
            pred [N, n_genes].
        """
        device = imgs.device
        img_feats = self._encode_patches(imgs, batch_size=img_batch_size, training=False)
        img_feats_b = img_feats.unsqueeze(0)
        coords_b = coords.unsqueeze(0)

        shape = (1, img_feats.shape[0], self.n_genes)
        exp_t1 = self.interpolant.sample_from_prior(shape).to(device)

        ts = torch.linspace(0.01, 1.0, self.n_sample_steps)[:, None].expand(
            self.n_sample_steps, exp_t1.shape[0]
        ).to(device)

        pred = None
        for step, (t1, t2) in enumerate(zip(ts[:-1], ts[1:])):
            pred = self.denoiser.inference(
                exp_t1, img_feats_b, coords_b, t1, predict=True
            )
            d_t = t2 - t1
            if step == self.n_sample_steps - 2:
                break
            exp_t1 = self.interpolant.denoise(pred, exp_t1, t1, d_t)

        assert pred is not None, "n_sample_steps must be >= 2"
        return pred.squeeze(0)
