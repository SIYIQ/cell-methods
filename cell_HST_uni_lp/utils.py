"""Shared utilities for cell_HST_uni_lp.

Adapts the frozen UNI + PCA + Ridge linear-probe protocol from
HST_uni_lp to the single-cell Xenium benchmark in
/home/sb202604/cell-benchmark/processed.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from PIL import Image
from torchvision import transforms


def _filter_control_probes(genes):
    """Drop Xenium / panel control probes from a gene list."""
    keep = []
    for g in genes:
        if "BLANK" in g or "Control" in g:
            continue
        if g.startswith("NegControlProbe_") or g.startswith("UnassignedCodeword_"):
            continue
        keep.append(g)
    return keep


# =============================================================================
# HVG selection -- byte-identical copy of HST_h0mini_lp's official selector.
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


def extract_gene_expr(adata, gene_names):
    """Extract expression matrix for gene_names from a log1p AnnData."""
    common = [g for g in gene_names if g in adata.var_names]
    if len(common) == len(gene_names):
        expr = adata[:, gene_names].X
        if hasattr(expr, "toarray"):
            expr = expr.toarray()
        return np.asarray(expr, dtype=np.float32)
    expr_mat = adata[:, common].X
    if hasattr(expr_mat, "toarray"):
        expr_mat = expr_mat.toarray()
    df = pd.DataFrame(expr_mat, index=adata.obs_names, columns=common)
    df = df.reindex(columns=gene_names, fill_value=0.0)
    return df.values.astype(np.float32)


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
# UNI loading + encoding
# =============================================================================
_TIMM_INTERP = {
    "bicubic": transforms.InterpolationMode.BICUBIC,
    "bilinear": transforms.InterpolationMode.BILINEAR,
    "nearest": transforms.InterpolationMode.NEAREST,
}


def load_uni(local_path, device):
    """Load UNI ViT-L/16 from a local HuggingFace snapshot directory.

    Returns ``(model, preprocess, embedding_dim)``. Normalization uses
    UNI's own pretrained_cfg (which happens to be ImageNet for UNI).
    """
    import timm

    config_path = os.path.join(local_path, "config.json")
    with open(config_path, "r") as f:
        model_cfg = json.load(f)

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
    model_args = {k: model_cfg[k] for k in arg_keys if k in model_cfg}

    backbone = timm.create_model(
        model_cfg["architecture"],
        pretrained=False,
        **model_args,
    )

    bin_path = os.path.join(local_path, "pytorch_model.bin")
    sft_path = os.path.join(local_path, "model.safetensors")
    if os.path.isfile(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
    elif os.path.isfile(sft_path):
        from safetensors.torch import load_file

        state_dict = load_file(sft_path, device="cpu")
    else:
        raise FileNotFoundError(
            f"No weights found in {local_path}: expected "
            f"'pytorch_model.bin' or 'model.safetensors'."
        )
    backbone.load_state_dict(state_dict, strict=True)
    backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    pcfg = model_cfg.get("pretrained_cfg", {})
    mean = pcfg.get("mean", (0.485, 0.456, 0.406))
    std = pcfg.get("std", (0.229, 0.224, 0.225))
    input_size = pcfg.get("input_size", [3, 224, 224])
    img_size = input_size[-1]
    interp = _TIMM_INTERP.get(
        pcfg.get("interpolation", "bicubic"), transforms.InterpolationMode.BICUBIC
    )
    crop_pct = pcfg.get("crop_pct", 1.0)
    resize_size = int(round(img_size / crop_pct)) if crop_pct < 1.0 else img_size

    preprocess = transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=interp),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    embedding_dim = int(model_cfg.get("num_features", 1024))
    print(
        f"Loaded UNI from {local_path}: dim={embedding_dim}, "
        f"mean={mean}, std={std}, img_size={img_size}"
    )
    return backbone, preprocess, embedding_dim


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


def encode_patches_from_memmap(
    patches: np.ndarray,
    model: torch.nn.Module,
    preprocess,
    device: str,
    batch_size: int = 64,
    feature_dim: int = 768,
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

    with torch.no_grad():
        for bi in range(n_batches):
            lo = bi * batch_size
            hi = min((bi + 1) * batch_size, n_cells)
            # uint8 CHW -> float HWC for PIL
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
                    f"  {hi:,}/{n_cells:,} ({100*hi/n_cells:.0f}%) "
                    f"rate={rate:.0f}/s ETA={eta:.0f}s"
                )
    return features


def load_or_encode_features(
    half_dir: Path,
    cache_path: Path,
    model,
    preprocess,
    device: str,
    batch_size: int,
    feature_dim: int,
    encoder_name: str,
    cell_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Load cached image features or encode patches.npy and cache them.

    Parameters
    ----------
    half_dir
        Directory containing ``cells.h5ad`` and ``patches.npy``.
    cache_path
        Project-local cache file where features will be written.
    cell_idx
        Optional integer indices into ``patches.npy`` to encode. If None,
        encodes all cells in ``cells.h5ad``.
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
            manifest.get("encoder") == encoder_name
            and manifest.get("patches_checksum") == current_checksum
            and manifest.get("feature_dim") == feature_dim
            and idx_match
        ):
            print(f"  [cache hit] {cache_path}")
            cached = np.load(cache_path, mmap_mode="r")
            if cached.shape[1] == feature_dim:
                return np.asarray(cached, dtype=np.float32)
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

    print(f"  [encode] {encoder_name} on {device} for {n_cells:,} cells")
    features = encode_patches_from_memmap(
        patches, model, preprocess, device, batch_size=batch_size, feature_dim=feature_dim
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, features)
    manifest = {
        "encoder": encoder_name,
        "patches_checksum": _file_checksum(patches_path),
        "n_total": int(n_total),
        "n_cells": int(n_cells),
        "feature_dim": int(feature_dim),
    }
    if cell_idx is not None:
        manifest["cell_idx"] = cell_idx.tolist()
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return features


def load_cached_he_features(
    adata: ad.AnnData,
    cache_path: Path | str | None = None,
    feature_key: str = "he",
) -> np.ndarray:
    """Load precomputed H&E image features from ``adata.obsm[feature_key]``.

    If ``cache_path`` is provided, write a small on-disk cache so that
    reruns avoid reloading the full AnnData just for the features.
    """
    cache_path = Path(cache_path) if cache_path is not None else None
    if cache_path is not None and cache_path.exists():
        print(f"  [cache hit] {cache_path}")
        return np.load(cache_path).astype(np.float32)

    if feature_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{feature_key}'] not found; keys={list(adata.obsm.keys())}")
    features = np.asarray(adata.obsm[feature_key], dtype=np.float32)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, features)
    return features


# =============================================================================
# Preprocessing and fold evaluation
# =============================================================================
def preprocess_half(adata_raw: ad.AnnData, he_features: np.ndarray, gene_names, min_counts=10):
    """Filter to genes, filter cells, normalize, log1p, subset HVGs.

    Returns
    -------
    expr : np.ndarray, shape (n_surviving_cells, n_genes)
    he : np.ndarray, shape (n_surviving_cells, feature_dim)
    """
    # Keep only real genes if the annotation exists
    if "is_gene" in adata_raw.var.columns:
        sub = adata_raw[:, adata_raw.var["is_gene"].astype(bool).values].copy()
    else:
        sub = adata_raw.copy()

    pre_ids = sub.obs_names.values.copy()
    sc.pp.filter_cells(sub, min_counts=min_counts)
    survivor_pos = pd.Index(pre_ids).get_indexer(sub.obs_names.values)
    assert (survivor_pos >= 0).all(), "filter_cells dropped unknown cells"
    he_kept = he_features[survivor_pos]

    if hasattr(sub.X, "toarray"):
        sub.X = sub.X.toarray()
    sc.pp.normalize_total(sub, inplace=True)
    sc.pp.log1p(sub)

    expr = extract_gene_expr(sub, gene_names)
    return expr, he_kept.astype(np.float32)


def hest_linear_probe(
    X_train, y_train, X_test, n_components=256, alpha=None, max_iter=1000, random_state=0, ridge_solver="lsqr"
):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    n_components = min(n_components, X_train.shape[1], X_train.shape[0])
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("PCA", PCA(n_components=n_components, random_state=random_state)),
        ]
    )
    Xtr = pipe.fit_transform(X_train)
    Xte = pipe.transform(X_test)

    D = Xtr.shape[1]
    G = y_train.shape[1]
    if alpha is None:
        alpha = 100.0 / (D * G)

    reg = Ridge(
        solver=ridge_solver,
        alpha=alpha,
        random_state=random_state,
        fit_intercept=False,
        max_iter=max_iter,
    )
    reg.fit(Xtr, y_train)
    y_pred = reg.predict(Xte)
    return y_pred, pipe, reg


def evaluate_fold(
    expr_train,
    he_train,
    expr_test,
    he_test,
    cfg,
    seed: int,
):
    """Run PCA+Ridge on train half and evaluate on test half."""
    y_pred, _, _ = hest_linear_probe(
        he_train,
        expr_train,
        he_test,
        n_components=int(cfg.get("pca_components", 256)),
        alpha=cfg.get("ridge_alpha", None),
        max_iter=int(cfg.get("ridge_max_iter", 1000)),
        random_state=seed,
        ridge_solver=cfg.get("ridge_solver", "lsqr"),
    )
    per_gene_pcc, mean_pcc = compute_pcc(y_pred, expr_test)
    info = {
        "n_train_cells": int(he_train.shape[0]),
        "n_test_cells": int(he_test.shape[0]),
        "embedding_dim": int(he_train.shape[1]),
        "pca_components": int(min(cfg.get("pca_components", 256), he_train.shape[1], he_train.shape[0])),
        "ridge_alpha": float(
            cfg.get("ridge_alpha")
            if cfg.get("ridge_alpha") is not None
            else 100.0 / (min(cfg.get("pca_components", 256), he_train.shape[1], he_train.shape[0]) * expr_train.shape[1])
        ),
    }
    return y_pred, per_gene_pcc, mean_pcc, info
