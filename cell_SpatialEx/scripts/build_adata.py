"""Build SpatialEx-compatible AnnData files from /home/sb202604/cell-benchmark/processed.

The result for each dataset is two AnnData files (one per S1/S2 half) saved to
/home/sb202604/cell-benchmark/processed/<dataset>/spatialex/{S1,S2}.h5ad

Each saved AnnData has:
  X            log1p-normalized expression on Gene-Expression panel only (dense float32)
  obs          x_centroid, y_centroid (in MICRONS — SpatialEx expects μm)
               image_col, image_row (H&E px, mainly for plotting)
  obsm
    spatial   [N, 2]  (x_um, y_um)  — required by SpatialEx
    image_coor [N, 2]  (col=x_he, row=y_he) int
    he        [N, 768] H0-mini CLS features
  uns
    dataset, half (S1 or S2)

Patches are read from processed/<dataset>/patches.npy (memmap), passed through
H0-mini with H&E ImageNet normalization, and stored in `obsm['he']`.
"""

from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader

PROCESSED_ROOT = Path("/home/sb202604/cell-benchmark/processed")
H0_MINI_DIR = "/home/sb202604/H0-mini"


def load_h0_mini(device: str) -> tuple[nn.Module, int]:
    """Load the H0-mini ViT backbone. Returns (model, feat_dim).

    Same loading logic as HST_novae/model.py::FlexibleImageEncoder._build_timm_encoder.
    """
    import timm

    with open(os.path.join(H0_MINI_DIR, "config.json")) as f:
        model_cfg = json.load(f)
    model_args = dict(model_cfg.get("model_args", {}))
    model_args["mlp_layer"] = timm.layers.SwiGLUPacked
    model_args["act_layer"] = torch.nn.SiLU

    model = timm.create_model(
        model_cfg["architecture"],
        pretrained=False,
        **model_args,
    )

    bin_path = os.path.join(H0_MINI_DIR, "pytorch_model.bin")
    sft_path = os.path.join(H0_MINI_DIR, "model.safetensors")
    if os.path.isfile(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
    elif os.path.isfile(sft_path):
        from safetensors.torch import load_file

        state_dict = load_file(sft_path, device="cpu")
    else:
        raise FileNotFoundError(f"no weights in {H0_MINI_DIR}")

    model.load_state_dict(state_dict, strict=True)
    model = model.eval().to(device)
    return model, 768


def encode_patches(
    patches_path: Path,
    n_cells: int,
    model: nn.Module,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    """Stream patches through H0-mini with ImageNet normalization, return [N, 768]."""
    P = 224
    patches = np.memmap(patches_path, dtype=np.uint8, mode="r",
                        shape=(n_cells, 3, P, P))

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    features = np.empty((n_cells, 768), dtype=np.float32)
    t0 = time.time()
    n_batches = (n_cells + batch_size - 1) // batch_size
    report_every = max(1, n_batches // 10)

    with torch.no_grad():
        for bi in range(n_batches):
            lo = bi * batch_size
            hi = min((bi + 1) * batch_size, n_cells)
            x = torch.from_numpy(np.array(patches[lo:hi])).to(device, dtype=torch.float32) / 255.0
            x = (x - mean) / std
            h = model(x)
            if h.dim() == 3:
                h = h[:, 0]  # CLS token
            features[lo:hi] = h.cpu().numpy()
            if (bi + 1) % report_every == 0:
                rate = (hi) / (time.time() - t0)
                eta = (n_cells - hi) / max(rate, 1e-6)
                print(f"    {hi:,}/{n_cells:,}  ({100*hi/n_cells:.0f}%)  "
                      f"rate={rate:.0f}/s  ETA={eta:.0f}s")
    print(f"  H0-mini encoding done in {time.time()-t0:.1f}s")
    return features


def make_adata_half(
    cells: ad.AnnData,
    he_feats: np.ndarray,
    half_idx: np.ndarray,
    half_label: str,
    dataset: str,
    min_counts: int = 10,
) -> ad.AnnData:
    """Filter to gene-expression features only, apply S1/S2 subset, log1p-normalize."""
    sub = cells[half_idx].copy()
    he_sub = he_feats[half_idx]

    # Restrict to real genes (drop control / codeword / unassigned features).
    sub = sub[:, sub.var["is_gene"].astype(bool).values].copy()

    # Mirror SpatialEx Preprocess_adata logic:
    #   sc.pp.filter_cells(min_counts=10), normalize_total, log1p
    pre_ids = sub.obs_names.values
    sc.pp.filter_cells(sub, min_counts=min_counts)
    survivor_pos = pd.Index(pre_ids).get_indexer(sub.obs_names.values)
    assert (survivor_pos >= 0).all(), "filter_cells dropped cells not in pre_ids"
    he_kept = he_sub[survivor_pos]

    sub.layers["raw"] = sub.X.copy()
    if hasattr(sub.X, "toarray"):
        sub.X = np.asarray(sub.X.todense())
    sc.pp.normalize_total(sub, inplace=True)
    sc.pp.log1p(sub)

    # Required by SpatialEx
    sub.obs["x_centroid"] = sub.obs["x_um"].astype(np.float64)
    sub.obs["y_centroid"] = sub.obs["y_um"].astype(np.float64)
    sub.obsm["spatial"] = sub.obs[["x_centroid", "y_centroid"]].to_numpy(np.float64)

    # image_coor: [col, row] = [x_he, y_he] in H&E px
    img_coor = np.stack([
        sub.obs["x_he"].to_numpy(np.float64).round().astype(int),
        sub.obs["y_he"].to_numpy(np.float64).round().astype(int),
    ], axis=1)
    sub.obsm["image_coor"] = img_coor
    sub.obs["image_col"] = img_coor[:, 0]
    sub.obs["image_row"] = img_coor[:, 1]

    sub.obsm["he"] = he_kept.astype(np.float32)

    sub.uns["dataset"] = dataset
    sub.uns["half"] = half_label
    return sub


def process_dataset(dataset: str, device: str, batch_size: int, force: bool) -> None:
    ds_dir = PROCESSED_ROOT / dataset
    out_dir = ds_dir / "spatialex"
    out_dir.mkdir(parents=True, exist_ok=True)

    s1_path = out_dir / "S1.h5ad"
    s2_path = out_dir / "S2.h5ad"
    if not force and s1_path.exists() and s2_path.exists():
        print(f"[skip] {dataset}: spatialex/S1.h5ad and S2.h5ad already exist")
        return

    print(f"\n===== {dataset} =====")
    cells = ad.read_h5ad(ds_dir / "cells.h5ad")
    print(f"  cells.h5ad shape: {cells.shape}")
    splits = json.load(open(ds_dir / "splits.json"))
    s1 = np.asarray(splits["spatial_ood"]["S1"], dtype=np.int64)
    s2 = np.asarray(splits["spatial_ood"]["S2"], dtype=np.int64)
    print(f"  S1: {len(s1):,} cells   S2: {len(s2):,} cells")

    # Encode H&E patches once per dataset (then subset by S1/S2)
    cached_he_path = ds_dir / "h0mini_features.npy"
    if cached_he_path.exists() and not force:
        he_feats = np.load(cached_he_path, mmap_mode="r")
        if he_feats.shape != (cells.n_obs, 768):
            raise RuntimeError(
                f"cached h0mini features shape mismatch: got {he_feats.shape}, "
                f"expected {(cells.n_obs, 768)}; rerun with --force")
        print(f"  using cached H0-mini features: {cached_he_path}")
    else:
        print(f"  encoding {cells.n_obs:,} cells with H0-mini on {device}...")
        model, _ = load_h0_mini(device)
        he_feats = encode_patches(ds_dir / "patches.npy", cells.n_obs, model,
                                  device=device, batch_size=batch_size)
        np.save(cached_he_path, he_feats)
        del model
        torch.cuda.empty_cache()

    print(f"  building S1 adata...")
    a1 = make_adata_half(cells, he_feats, s1, "S1", dataset)
    print(f"    S1 shape: {a1.shape}  he obsm: {a1.obsm['he'].shape}")
    a1.write_h5ad(s1_path, compression="gzip")

    print(f"  building S2 adata...")
    a2 = make_adata_half(cells, he_feats, s2, "S2", dataset)
    print(f"    S2 shape: {a2.shape}  he obsm: {a2.obsm['he'].shape}")
    a2.write_h5ad(s2_path, compression="gzip")

    print(f"  done: {s1_path}  +  {s2_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   help="dataset name (e.g. hSkin_Melanoma)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    process_dataset(args.dataset, args.device, args.batch_size, args.force)


if __name__ == "__main__":
    main()
