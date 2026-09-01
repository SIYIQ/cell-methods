"""Shared utilities for cell_Stem.

Adapts the Stem diffusion model (Zhu et al., ICLR 2025) to the single-cell
Xenium benchmark used by cell_HST_uni_lp. Conditioning is concat(UNI[1024],
CONCH[512]) -> diffusion in log2(count+1) space -> ensemble K samples at test
time and take the mean.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


# =============================================================================
# HVG selection -- byte-identical copy of HST-middle's official selector.
# Operates on raw-count AnnData objects.
# =============================================================================
def select_top_hvgs_official(adata_list, n_top=50, min_cells_pct=0.10):
    """Select top HVGs across a list of raw-count AnnData objects."""
    common_genes = None
    for adata in adata_list:
        my_adata = adata.copy()
        if min_cells_pct:
            sc.pp.filter_genes(
                my_adata, min_cells=np.ceil(min_cells_pct * len(my_adata.obs))
            )
        curr_genes = np.array(my_adata.to_df().columns)
        if common_genes is None:
            common_genes = curr_genes
        else:
            common_genes = np.intersect1d(common_genes, curr_genes)

    common_genes = [
        g
        for g in common_genes
        if "BLANK" not in g
        and "Control" not in g
        and not g.startswith("NegControlProbe_")
        and not g.startswith("UnassignedCodeword_")
    ]

    stacked = None
    for adata in adata_list:
        df = adata.to_df()[common_genes]
        stacked = df if stacked is None else pd.concat([stacked, df])

    stacked_adata = sc.AnnData(stacked.astype(np.float32))
    sc.pp.filter_genes(stacked_adata, min_cells=0)
    sc.pp.log1p(stacked_adata)
    sc.pp.highly_variable_genes(stacked_adata, n_top_genes=n_top)
    hvg_mask = stacked_adata.var["highly_variable"].values
    return stacked_adata.var_names[hvg_mask].tolist()[:n_top]


def extract_gene_expr_log2(adata, gene_names):
    """Return ``log2(count+1)`` matrix [N, len(gene_names)] for Stem."""
    common = [g for g in gene_names if g in adata.var_names]
    if len(common) == len(gene_names):
        expr = adata[:, gene_names].X
        if hasattr(expr, "toarray"):
            expr = expr.toarray()
        expr = np.asarray(expr, dtype=np.float32)
    else:
        expr_mat = adata[:, common].X
        if hasattr(expr_mat, "toarray"):
            expr_mat = expr_mat.toarray()
        df = pd.DataFrame(expr_mat, index=adata.obs_names, columns=common)
        df = df.reindex(columns=gene_names, fill_value=0.0)
        expr = df.values.astype(np.float32)
    return np.log2(expr + 1.0).astype(np.float32)


def compute_pcc(pred, target):
    """Per-gene Pearson r and mean."""
    pred_np = pred if isinstance(pred, np.ndarray) else np.asarray(pred)
    target_np = target if isinstance(target, np.ndarray) else np.asarray(target)
    pcc_list = []
    for g in range(pred_np.shape[1]):
        std_pred = np.std(pred_np[:, g])
        std_target = np.std(target_np[:, g])
        if std_pred == 0 or std_target == 0:
            pcc_list.append(0.0)
            continue
        r = np.corrcoef(pred_np[:, g], target_np[:, g])[0, 1]
        pcc_list.append(0.0 if np.isnan(r) else float(r))
    return pcc_list, float(np.mean(pcc_list))


# =============================================================================
# UNI / CONCH encoders with on-disk caching.
# =============================================================================
def _device(dev):
    return dev if isinstance(dev, torch.device) else torch.device(dev)


def load_uni(local_path, device):
    """Load UNI ViT-L/16 from local snapshot. Returns (model, preprocess)."""
    import timm

    config_path = os.path.join(local_path, "config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    arg_keys = (
        "patch_size",
        "img_size",
        "init_values",
        "num_classes",
        "dynamic_img_size",
        "global_pool",
        "mlp_ratio",
        "reg_tokens",
    )
    model_args = {k: cfg[k] for k in arg_keys if k in cfg}

    backbone = timm.create_model(cfg["architecture"], pretrained=False, **model_args)

    bin_path = os.path.join(local_path, "pytorch_model.bin")
    sft_path = os.path.join(local_path, "model.safetensors")
    if os.path.isfile(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
    elif os.path.isfile(sft_path):
        from safetensors.torch import load_file

        state_dict = load_file(sft_path, device="cpu")
    else:
        raise FileNotFoundError(local_path)
    backbone.load_state_dict(state_dict, strict=True)
    backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    pcfg = cfg.get("pretrained_cfg", {})
    mean = pcfg.get("mean", (0.485, 0.456, 0.406))
    std = pcfg.get("std", (0.229, 0.224, 0.225))
    img_size = pcfg.get("input_size", [3, 224, 224])[-1]
    interp = {
        "bicubic": transforms.InterpolationMode.BICUBIC,
        "bilinear": transforms.InterpolationMode.BILINEAR,
    }.get(pcfg.get("interpolation", "bicubic"), transforms.InterpolationMode.BICUBIC)
    preprocess = transforms.Compose(
        [
            transforms.Resize(img_size, interpolation=interp),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return backbone, preprocess


def load_conch(local_path, device):
    """Load CONCH ViT-B/16 from local pytorch_model.bin.

    Returns (model, preprocess). CONCH's official preprocess upscales the
    patch to 448x448 before the ViT.
    """
    from conch.open_clip_custom import create_model_from_pretrained

    bin_path = os.path.join(local_path, "pytorch_model.bin")
    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16", bin_path, device=device
    )
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, preprocess


def _encode_patches_from_memmap(
    patches: np.ndarray,
    model,
    preprocess,
    device: str,
    batch_size: int,
    feature_dim: int,
    encoder_name: str,
) -> np.ndarray:
    """Encode patches of shape [N, 3, 224, 224] uint8 -> [N, feature_dim].

    The input ``patches`` can be a memmap. Channel order is CHW; each patch
    is converted to HWC PIL.Image before preprocessing.
    """
    n_cells = patches.shape[0]
    features = np.empty((n_cells, feature_dim), dtype=np.float32)
    n_batches = (n_cells + batch_size - 1) // batch_size
    report_every = max(1, n_batches // 10)
    t0 = time.time()

    with torch.inference_mode():
        for bi in range(n_batches):
            lo = bi * batch_size
            hi = min((bi + 1) * batch_size, n_cells)
            arr = np.ascontiguousarray(patches[lo:hi].transpose(0, 2, 3, 1))
            batch = torch.stack(
                [preprocess(Image.fromarray(a)) for a in arr], dim=0
            ).to(device)
            h = model(batch)
            if h.dim() == 3:
                h = h[:, 0]
            features[lo:hi] = h.float().cpu().numpy()
            if (bi + 1) % report_every == 0 or hi == n_cells:
                rate = hi / (time.time() - t0 + 1e-6)
                eta = (n_cells - hi) / rate
                print(
                    f"    {encoder_name} {hi:,}/{n_cells:,} ({100*hi/n_cells:.0f}%) "
                    f"rate={rate:.0f}/s ETA={eta:.0f}s"
                )
    return features


def _encode_patches_conch(
    model, preprocess, patches: np.ndarray, device: str, batch_size: int
) -> np.ndarray:
    """Encode patches with CONCH -> np.ndarray [N, 512]."""
    n_cells = patches.shape[0]
    features = np.empty((n_cells, 512), dtype=np.float32)
    n_batches = (n_cells + batch_size - 1) // batch_size
    report_every = max(1, n_batches // 10)
    t0 = time.time()

    with torch.inference_mode():
        for bi in range(n_batches):
            lo = bi * batch_size
            hi = min((bi + 1) * batch_size, n_cells)
            arr = np.ascontiguousarray(patches[lo:hi].transpose(0, 2, 3, 1))
            batch = torch.stack(
                [preprocess(Image.fromarray(a)) for a in arr], dim=0
            ).to(device)
            h = model.encode_image(batch, proj_contrast=False, normalize=False)
            features[lo:hi] = h.float().cpu().numpy()
            if (bi + 1) % report_every == 0 or hi == n_cells:
                rate = hi / (time.time() - t0 + 1e-6)
                eta = (n_cells - hi) / rate
                print(
                    f"    CONCH {hi:,}/{n_cells:,} ({100*hi/n_cells:.0f}%) "
                    f"rate={rate:.0f}/s ETA={eta:.0f}s"
                )
    return features


def _file_checksum(path: Path, sample_mb: int = 10) -> str:
    """Fast hash of a file: size + head + tail samples."""
    size = path.stat().st_size
    h = hashlib.md5()
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(sample_mb * 1024 * 1024))
        if size > sample_mb * 2 * 1024 * 1024:
            f.seek(-sample_mb * 1024 * 1024, 2)
            h.update(f.read(sample_mb * 1024 * 1024))
    return h.hexdigest()


def load_or_encode_features(
    half_dir: Path,
    cache_path: Path,
    uni_model,
    uni_preprocess,
    conch_model,
    conch_preprocess,
    device: str,
    uni_batch: int,
    conch_batch: int,
    cell_idx: np.ndarray | None = None,
) -> torch.FloatTensor:
    """Load cached UNI+CONCH features or encode patches.npy and cache them.

    Returns
    -------
    torch.FloatTensor of shape (N_cells, 1536)
    """
    cache_path = Path(cache_path)
    manifest_path = cache_path.with_suffix(".manifest.json")
    patches_path = half_dir / "patches.npy"

    if cache_path.exists() and manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        current_checksum = _file_checksum(patches_path)
        idx_match = (
            manifest.get("cell_idx") is None
            if cell_idx is None
            else (
                manifest.get("cell_idx") is not None
                and np.array_equal(np.asarray(manifest["cell_idx"]), cell_idx)
            )
        )
        if (
            manifest.get("patches_checksum") == current_checksum
            and idx_match
        ):
            print(f"  [cache hit] {cache_path}")
            feats = np.load(cache_path, mmap_mode="r")
            return torch.from_numpy(np.array(feats, copy=True, dtype=np.float32))
        print(f"  [cache stale] re-encoding {half_dir.name}")

    cells = ad.read_h5ad(half_dir / "cells.h5ad")
    n_total = cells.n_obs
    patch_size = 224
    patches = np.memmap(
        patches_path, dtype=np.uint8, mode="r", shape=(n_total, 3, patch_size, patch_size)
    )

    if cell_idx is not None:
        patches = patches[cell_idx]
        n_cells = patches.shape[0]
    else:
        n_cells = n_total

    print(f"  [encode] UNI for {n_cells:,} cells")
    uni_feats = _encode_patches_from_memmap(
        patches,
        uni_model,
        uni_preprocess,
        device,
        uni_batch,
        feature_dim=1024,
        encoder_name="UNI",
    )

    print(f"  [encode] CONCH for {n_cells:,} cells")
    conch_feats = _encode_patches_conch(
        conch_model,
        conch_preprocess,
        patches,
        device,
        conch_batch,
    )

    feats = np.concatenate([uni_feats, conch_feats], axis=1).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, feats)
    manifest = {
        "patches_checksum": _file_checksum(patches_path),
        "n_total": int(n_total),
        "n_cells": int(n_cells),
        "feature_dim": 1536,
    }
    if cell_idx is not None:
        manifest["cell_idx"] = cell_idx.tolist()
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return torch.from_numpy(feats)


def load_cached_he_features(
    adata: ad.AnnData,
    cache_path: Path | str | None = None,
    feature_key: str = "he",
) -> torch.FloatTensor:
    """Load precomputed UNI features from ``adata.obsm[feature_key]``.

    If ``cache_path`` is provided, write a small on-disk cache so that
    reruns avoid reloading the full AnnData just for the features.
    """
    cache_path = Path(cache_path) if cache_path is not None else None
    if cache_path is not None and cache_path.exists():
        print(f"  [cache hit] {cache_path}")
        return torch.from_numpy(np.load(cache_path).astype(np.float32))

    if feature_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{feature_key}'] not found; keys={list(adata.obsm.keys())}")
    features = np.asarray(adata.obsm[feature_key], dtype=np.float32)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, features)
    return torch.from_numpy(features)


# =============================================================================
# Preprocessing for single-cell halves
# =============================================================================
def preprocess_half(adata_raw: ad.AnnData, cond: torch.FloatTensor, gene_names, min_counts=10):
    """Filter to genes, filter cells, extract log2(count+1) expression.

    Returns
    -------
    expr : torch.FloatTensor, shape (n_surviving_cells, n_genes)
    cond : torch.FloatTensor, shape (n_surviving_cells, 1536)
    """
    if "is_gene" in adata_raw.var.columns:
        sub = adata_raw[:, adata_raw.var["is_gene"].astype(bool).values].copy()
    else:
        sub = adata_raw.copy()

    pre_ids = sub.obs_names.values.copy()
    sc.pp.filter_cells(sub, min_counts=min_counts)
    survivor_pos = pd.Index(pre_ids).get_indexer(sub.obs_names.values)
    assert (survivor_pos >= 0).all(), "filter_cells dropped unknown cells"
    cond_kept = cond[survivor_pos]

    expr = extract_gene_expr_log2(sub, gene_names)
    return torch.from_numpy(expr), cond_kept


# =============================================================================
# EMA helper (matches Stem/train_helper.py:update_ema).
# =============================================================================
@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = 0.9999):
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def freeze(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False
