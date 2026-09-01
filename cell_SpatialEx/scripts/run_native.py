"""Native SpatialEx Tutorial 1 replica for SINGLE-section datasets (S1/S2 split).

For hSkin_Melanoma, hColon_Non_diseased, mouse_Colon — datasets that have only
one physical Xenium section. We split by median(x_centroid) into a left half
(S1) and a right half (S2), then mirror the Tutorial 1 setup: H&E → UNI →
hypergraph → SpatialEx → auto_inference → PCC.

The paper reports per-dataset PCC boxplots in Fig. 2 for these S1/S2
configurations; the exact split coordinates are not published. The
median(x_centroid) split is the simplest "manually split into two
non-overlapping slices" interpretation. Whether the resulting PCC matches the
paper depends on how the authors chose the split line.

Multi-seed evaluation (mirrors HST-middle/h0mini_official_offline/train.py
convention): default seeds=[42, 43, 44], 3 runs. Reports mean ± std across
seeds in `summary.json`, plus an `overall_mean_pcc` that pools the 2
cross-section directions into a single Table-2-style headline number.

UNI features and AnnData prep are deterministic, so cached on disk and
reused across seeds.

Layout produced:
    runs_native/<dataset>/
    ├── cache/
    │   ├── full_he.npz       UNI features for the whole slide
    │   └── full_adata.h5ad   AnnData with obs/obsm/X for the whole slide
    ├── seed_42/result.json
    ├── seed_43/result.json
    ├── seed_44/result.json
    └── summary.json

Usage:
    python scripts/run_native.py --dataset hSkin_Melanoma --device cuda:0
    python scripts/run_native.py --dataset hSkin_Melanoma --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
import zipfile
from pathlib import Path

# Stub cellpose so we can import SpatialEx (segmentation utils unused).
for _m in ("cellpose", "cellpose.models"):
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anndata as ad
import numpy as np
import pandas as pd

import SpatialEx as se


RAW_ROOT = Path("/home/sb202604/cell-benchmark/raw")
RUNS_ROOT = Path(__file__).resolve().parents[1] / "runs_native"

# Single-section datasets we split into S1/S2 here.
DATASETS = {
    "hSkin_Melanoma": (
        "Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE_outs.zip",
        "Xenium_V1_hSkin_Melanoma_Base_FFPE_he_image.ome.tif",
        "Xenium_V1_hSkin_Melanoma_Base_FFPE_he_imagealignment.csv",
    ),
    "hColon_Non_diseased": (
        "Xenium_V1_hColon_Non_diseased_Add_on_FFPE_outs.zip",
        "Xenium_V1_hColon_Non_diseased_Base_FFPE_he_image.ome.tif",
        "Xenium_V1_hColon_Non_diseased_Base_FFPE_he_imagealignment.csv",
    ),
    "mouse_Colon": (
        "Xenium_V1_mouse_Colon_FF_outs.zip",
        "Xenium_V1_mouse_Colon_FF_he_image.ome.tif",
        "Xenium_V1_mouse_Colon_FF_he_imagealignment.csv",
    ),
}

CORE_MEMBERS = ("cell_feature_matrix.h5", "cells.csv.gz")


# ---------------------------- IO helpers -----------------------------------
def ensure_outs(outs_dir: Path, zip_path: Path) -> None:
    """Extract core files. 10x zips ship them either at the top level
    (newer 2023+ bundles) or under an 'outs/' prefix (older 2022 bundles)."""
    outs_dir.mkdir(parents=True, exist_ok=True)
    needed = [m for m in CORE_MEMBERS if not (outs_dir / m).exists()]
    if not needed:
        return
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for m in needed:
            if m in names:
                src = m
            elif f"outs/{m}" in names:
                src = f"outs/{m}"
            else:
                raise FileNotFoundError(
                    f"neither '{m}' nor 'outs/{m}' in {zip_path.name}")
            print(f"  unzip {src}...")
            with zf.open(src) as f_in:
                with open(outs_dir / m, "wb") as f_out:
                    f_out.write(f_in.read())


# ---------------------- full-slice deterministic prep ----------------------
def prepare_full_slice(dataset: str, resolution: int, image_encoder: str,
                       device: str, cache_dir: Path):
    """Tutorial 1 cell[8] equivalent on the whole section (before S1/S2 split).
    Reuses cache_dir/full_adata.h5ad + full_he.npz across seeds.
    """
    adata_cache = cache_dir / "full_adata.h5ad"
    he_cache = cache_dir / "full_he.npz"

    if adata_cache.exists() and he_cache.exists():
        print(f"\n[cache hit] {dataset}: loading cached adata + he features")
        adata = ad.read_h5ad(adata_cache)
        he = np.load(he_cache)["he"]
        assert he.shape[0] == adata.n_obs, "cache shape mismatch; delete cache_dir"
        adata.obsm["he"] = he
        print(f"  {adata.shape}, he obsm: {he.shape}")
        return adata

    print(f"\n[prepare] {dataset}")
    zip_name, he_name, align_name = DATASETS[dataset]
    ds_dir = RAW_ROOT / dataset
    outs_dir = ds_dir / "outs"
    ensure_outs(outs_dir, ds_dir / zip_name)

    adata = se.pp.Read_Xenium(
        str(outs_dir / "cell_feature_matrix.h5"),
        str(outs_dir / "cells.csv.gz"))
    print(f"  raw: {adata.shape}")
    adata = se.pp.Preprocess_adata(adata)
    print(f"  after Preprocess_adata: {adata.shape}")

    img, scale = se.pp.Read_HE_image(str(ds_dir / he_name))
    print(f"  H&E shape={img.shape}  PhysicalSizeX={scale}")
    transform_mtx = pd.read_csv(str(ds_dir / align_name), header=None).values
    adata = se.pp.Register_physical_to_pixel(adata, transform_mtx, scale=scale)

    print(f"[tile] {resolution}px patches around each cell")
    he_patches, adata = se.pp.Tiling_HE_patches(resolution, adata, img)
    print(f"  he_patches: {tuple(he_patches.shape)}")
    del img

    print(f"[encode] {image_encoder} on {device}")
    t0 = time.time()
    adata = se.pp.Extract_HE_patches_representaion(
        he_patches, adata=adata, image_encoder=image_encoder,
        device=device, store_key="he")
    print(f"  encoding done in {time.time()-t0:.1f}s, he obsm: {adata.obsm['he'].shape}")
    del he_patches

    cache_dir.mkdir(parents=True, exist_ok=True)
    he = np.asarray(adata.obsm["he"], dtype=np.float32)
    np.savez_compressed(he_cache, he=he)
    adata.write_h5ad(adata_cache, compression="gzip")
    print(f"  cached: {adata_cache.name}  +  {he_cache.name}")
    return adata


def split_S1_S2_by_x_median(adata):
    """SpatialEx S1/S2 convention: split a single section by median(x_centroid)."""
    x = adata.obs["x_centroid"].to_numpy(np.float64)
    xmed = float(np.median(x))
    mask_s1 = x < xmed
    mask_s2 = ~mask_s1
    print(f"  median(x) = {xmed:.1f} um")
    print(f"  S1: {mask_s1.sum():,} cells (x<med)   S2: {mask_s2.sum():,} cells")
    return adata[mask_s1].copy(), adata[mask_s2].copy()


def split_S1_S2_by_chessboard(adata, block_um: float = 200.0):
    """Chessboard split: tile the section with `block_um x block_um` blocks
    and alternate parity to assign cells to S1 vs S2.

    This keeps each S1/S2 half spatially non-overlapping (no cell shared) while
    making the two halves' cell-type distributions virtually identical, which
    is what 'manually split into two non-overlapping slices' likely means in
    SpatialEx's paper for single-section datasets.
    """
    x = adata.obs["x_centroid"].to_numpy(np.float64)
    y = adata.obs["y_centroid"].to_numpy(np.float64)
    xb = (x - x.min()) // block_um
    yb = (y - y.min()) // block_um
    parity = (xb.astype(np.int64) + yb.astype(np.int64)) % 2
    mask_s1 = parity == 0
    mask_s2 = ~mask_s1
    print(f"  chessboard block = {block_um} um")
    print(f"  blocks: {int(xb.max())+1} x {int(yb.max())+1} "
          f"= {int((xb.max()+1)*(yb.max()+1))} cells")
    print(f"  S1 (even blocks): {mask_s1.sum():,} cells   "
          f"S2 (odd blocks): {mask_s2.sum():,} cells")
    return adata[mask_s1].copy(), adata[mask_s2].copy()


# ---------------------- per-seed train + eval ------------------------------
def train_one_seed(a1, a2, g1, g2, seed: int, epochs: int, prune: int,
                   device: str, dataset: str, out_dir: Path) -> dict:
    """Run one SpatialEx training + auto_inference + PCC with a given seed."""
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")

    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "result.json").exists():
        print(f"  result.json already exists, skipping seed {seed}")
        with open(out_dir / "result.json") as f:
            return json.load(f)

    t0 = time.time()
    trainer = se.SpatialEx(a1, a2, g1, g2,
                           epochs=epochs, device=device,
                           prune=prune, seed=seed)
    trainer.train()
    print(f"  training done in {time.time()-t0:.1f}s")

    panelB1, panelA2 = trainer.auto_inference()

    print(f"[eval] PCC")
    pcc_S1_per, pcc_S1 = se.utils.Compute_metrics(
        a1.X.copy(), panelB1.copy(), metric="pcc", reduce="mean")
    pcc_S2_per, pcc_S2 = se.utils.Compute_metrics(
        a2.X.copy(), panelA2.copy(), metric="pcc", reduce="mean")
    pcc_S1_per = np.asarray(pcc_S1_per); pcc_S2_per = np.asarray(pcc_S2_per)
    med_S1 = float(np.median(pcc_S1_per))
    med_S2 = float(np.median(pcc_S2_per))
    print(f"  seed={seed}  PCC train=S2/test=S1: {pcc_S1:.4f}  med={med_S1:.4f}")
    print(f"  seed={seed}  PCC train=S1/test=S2: {pcc_S2:.4f}  med={med_S2:.4f}")

    np.save(out_dir / "pred_S1_from_S2.npy", panelB1)
    np.save(out_dir / "pred_S2_from_S1.npy", panelA2)
    np.save(out_dir / "pcc_S1_from_S2_per_gene.npy", pcc_S1_per)
    np.save(out_dir / "pcc_S2_from_S1_per_gene.npy", pcc_S2_per)

    result = {
        "dataset": dataset,
        "seed": seed,
        "n_genes": int(a1.n_vars),
        "n_cells": {"S1": int(a1.n_obs), "S2": int(a2.n_obs)},
        "epochs": epochs,
        "prune": prune,
        "mean_pcc_train_S2_test_S1": float(pcc_S1),
        "mean_pcc_train_S1_test_S2": float(pcc_S2),
        "median_pcc_slice1": med_S1,
        "median_pcc_slice2": med_S2,
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    del trainer, panelB1, panelA2
    import torch
    torch.cuda.empty_cache()
    return result


def summarize(results: list[dict], pair_dir: Path, **kw) -> dict:
    """Mean ± std across seeds. `overall_mean_pcc` pools both cross-section
    directions into a single Table-2-style headline number."""
    if not results:
        raise SystemExit("no results to summarize")

    def agg(key):
        vals = [r[key] for r in results]
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        return {"mean": m, "std": s, "n": len(vals), "values": vals}

    pooled = ([r["mean_pcc_train_S2_test_S1"] for r in results] +
              [r["mean_pcc_train_S1_test_S2"] for r in results])
    overall = {
        "mean": float(np.mean(pooled)),
        "std": float(np.std(pooled, ddof=1)) if len(pooled) > 1 else 0.0,
        "n": len(pooled),
        "values": pooled,
    }

    summary = {
        "dataset": results[0]["dataset"],
        "n_genes": results[0]["n_genes"],
        "n_cells": results[0]["n_cells"],
        "seeds": [r["seed"] for r in results],
        "epochs": results[0]["epochs"],
        "prune": results[0]["prune"],
        "overall_mean_pcc": overall,
        "mean_pcc_train_S2_test_S1": agg("mean_pcc_train_S2_test_S1"),
        "mean_pcc_train_S1_test_S2": agg("mean_pcc_train_S1_test_S2"),
        "median_pcc_slice1": agg("median_pcc_slice1"),
        "median_pcc_slice2": agg("median_pcc_slice2"),
        **kw,
    }
    with open(pair_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY  ({results[0]['dataset']}, n={len(results)} seeds)")
    print(f"{'='*60}")
    print(f"  overall_mean_pcc                     "
          f"{overall['mean']:.4f} ± {overall['std']:.4f}  (n={overall['n']})")
    for k in ("mean_pcc_train_S2_test_S1", "mean_pcc_train_S1_test_S2",
              "median_pcc_slice1", "median_pcc_slice2"):
        v = summary[k]
        print(f"  {k:35s}  {v['mean']:.4f} ± {v['std']:.4f}  (n={v['n']})")
    print(f"  saved -> {pair_dir / 'summary.json'}")
    return summary


# ---------------------------- main -----------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--resolution", type=int, default=64)
    ap.add_argument("--num-neighbors", type=int, default=7)
    ap.add_argument("--prune", type=int, default=20000,
                    help="grid prune size in microns")
    ap.add_argument("--image-encoder", default="uni")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                    help="Random seeds to run (each gets its own subdir).")
    ap.add_argument("--split-mode", choices=["x_median", "chessboard"],
                    default="chessboard",
                    help="How to split the single section into S1/S2. "
                         "'x_median' = left/right halves (distribution shift). "
                         "'chessboard' = alternating blocks (matched distribution).")
    ap.add_argument("--block-um", type=float, default=200.0,
                    help="Chessboard block size in microns (only for "
                         "--split-mode=chessboard).")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else \
        RUNS_ROOT / f"{args.dataset}_{args.split_mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"

    # ---- deterministic prep (run once, reused for every seed) ----
    adata_full = prepare_full_slice(args.dataset, args.resolution,
                                    args.image_encoder, args.device, cache_dir)

    print(f"\n[split] {args.split_mode}")
    if args.split_mode == "x_median":
        a1, a2 = split_S1_S2_by_x_median(adata_full)
    else:
        a1, a2 = split_S1_S2_by_chessboard(adata_full, block_um=args.block_um)

    print(f"\n[graph] spatial KNN hypergraphs (k={args.num_neighbors})")
    t0 = time.time()
    g1 = se.pp.Build_hypergraph_spatial_and_HE(
        a1, args.num_neighbors, graph_kind="spatial", return_type="crs")
    g2 = se.pp.Build_hypergraph_spatial_and_HE(
        a2, args.num_neighbors, graph_kind="spatial", return_type="crs")
    print(f"  done in {time.time()-t0:.1f}s")

    # ---- per-seed training ----
    results = []
    for seed in args.seeds:
        seed_dir = out_dir / f"seed_{seed}"
        r = train_one_seed(a1, a2, g1, g2, seed=seed,
                           epochs=args.epochs, prune=args.prune,
                           device=args.device, dataset=args.dataset,
                           out_dir=seed_dir)
        results.append(r)

    summarize(results, out_dir,
              encoder=args.image_encoder,
              resolution=args.resolution,
              num_neighbors=args.num_neighbors,
              split_mode=args.split_mode,
              block_um=args.block_um if args.split_mode == "chessboard" else None)


if __name__ == "__main__":
    main()
