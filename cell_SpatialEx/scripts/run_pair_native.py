"""Native SpatialEx Tutorial 1 replication on TWO biological replicate slides.

This is the exact mirror of Tutorial 1 cells [7-10, 13, 19, 21]:
take two 10x Xenium outs bundles (Rep1 + Rep2 of the same panel), preprocess,
extract H&E patches per cell, encode with UNI, build hypergraphs, train
SpatialEx, run auto_inference, evaluate PCC.

Tutorial 1's own saved outputs (single seed=0):
  Slice 1 (Rep1) PCC = 0.273   SSIM = 0.381   CMD = 0.203
  Slice 2 (Rep2) PCC = 0.258   SSIM = 0.365   CMD = 0.215

Multi-seed evaluation (mirrors HST-middle/h0mini_official_offline/train.py
convention): default seeds=[42, 43, 44], 3 runs per pair. Reports
mean ± std across seeds in `summary.json`.

UNI features and AnnData prep are deterministic (no_grad, frozen backbone),
so they are computed once per slice and cached on disk (npz + pickle). Only
the SpatialEx training loop is re-run per seed (~5 min on a 5090).

Layout produced:
    runs_native/<slice1>__<slice2>/
    ├── cache/
    │   ├── <slice1>_he.npz       UNI features  [N, 1024]  fp32
    │   ├── <slice1>_adata.h5ad   AnnData with obs/obsm/X
    │   ├── <slice2>_he.npz
    │   └── <slice2>_adata.h5ad
    ├── seed_42/
    │   └── result.json
    ├── seed_43/result.json
    ├── seed_44/result.json
    └── summary.json              mean ± std across seeds

Usage:
    python scripts/run_pair_native.py \
        --slice1 Human_Breast_Cancer_Rep1 \
        --slice2 Human_Breast_Cancer_Rep2 \
        --device cuda:0
    # Override seeds:
    python scripts/run_pair_native.py ... --seeds 42 43 44
    # Run one seed only:
    python scripts/run_pair_native.py ... --seeds 42
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

DATASETS = {
    "hSkin_Melanoma": (
        "Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE_outs.zip",
        "Xenium_V1_hSkin_Melanoma_Base_FFPE_he_image.ome.tif",
        "Xenium_V1_hSkin_Melanoma_Base_FFPE_he_imagealignment.csv",
    ),
    "Human_Breast_Cancer_Rep1": (
        "Xenium_FFPE_Human_Breast_Cancer_Rep1_outs.zip",
        "Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.ome.tif",
        "Xenium_FFPE_Human_Breast_Cancer_Rep1_he_imagealignment.csv",
    ),
    "Human_Breast_Cancer_Rep2": (
        "Xenium_FFPE_Human_Breast_Cancer_Rep2_outs.zip",
        "Xenium_FFPE_Human_Breast_Cancer_Rep2_he_image.ome.tif",
        "Xenium_FFPE_Human_Breast_Cancer_Rep2_he_imagealignment.csv",
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


# ---------------------- per-slice deterministic prep -----------------------
def prepare_slice(dataset: str, resolution: int, image_encoder: str,
                  device: str, cache_dir: Path):
    """Tutorial 1 cell[8] equivalent. Returns AnnData with obsm['he'].

    Reuses cache_dir/<dataset>_adata.h5ad + <dataset>_he.npz across seeds.
    """
    adata_cache = cache_dir / f"{dataset}_adata.h5ad"
    he_cache = cache_dir / f"{dataset}_he.npz"

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
    # SpatialEx writes adata.X dense after Preprocess_adata, so direct write is fine.
    adata.write_h5ad(adata_cache, compression="gzip")
    print(f"  cached: {adata_cache.name}  +  {he_cache.name}")
    return adata


# ---------------------- per-seed train + eval ------------------------------
def train_one_seed(a1, a2, g1, g2, seed: int, epochs: int, prune: int,
                   device: str, slice1_name: str, slice2_name: str,
                   out_dir: Path) -> dict:
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
    print(f"  seed={seed}  PCC train={slice2_name}/test={slice1_name}: {pcc_S1:.4f}  med={med_S1:.4f}")
    print(f"  seed={seed}  PCC train={slice1_name}/test={slice2_name}: {pcc_S2:.4f}  med={med_S2:.4f}")

    np.save(out_dir / "pred_S1_from_S2.npy", panelB1)
    np.save(out_dir / "pred_S2_from_S1.npy", panelA2)
    np.save(out_dir / "pcc_S1_from_S2_per_gene.npy", pcc_S1_per)
    np.save(out_dir / "pcc_S2_from_S1_per_gene.npy", pcc_S2_per)

    result = {
        "slice1": slice1_name,
        "slice2": slice2_name,
        "seed": seed,
        "n_genes": int(a1.n_vars),
        "n_cells": {"slice1": int(a1.n_obs), "slice2": int(a2.n_obs)},
        "epochs": epochs,
        "prune": prune,
        "mean_pcc_train_S2_test_S1": float(pcc_S1),
        "mean_pcc_train_S1_test_S2": float(pcc_S2),
        "median_pcc_slice1": med_S1,
        "median_pcc_slice2": med_S2,
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    # Free GPU memory between seeds
    del trainer, panelB1, panelA2
    import torch
    torch.cuda.empty_cache()
    return result


def summarize(results: list[dict], pair_dir: Path, **kw) -> dict:
    """HST-middle style mean ± std (sample stddev, ddof=1) across seeds.

    `overall_mean_pcc` collapses both cross-section directions into a single
    headline number, formatted for Table-2-style benchmark tables:
    pool the 2 directions x N seeds = 2N values, then mean ± std.
    """
    if not results:
        raise SystemExit("no results to summarize")

    def agg(key):
        vals = [r[key] for r in results]
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        return {"mean": m, "std": s, "n": len(vals), "values": vals}

    # Pool both cross-section directions into one bag of 2*N_seeds values.
    pooled = ([r["mean_pcc_train_S2_test_S1"] for r in results] +
              [r["mean_pcc_train_S1_test_S2"] for r in results])
    overall = {
        "mean": float(np.mean(pooled)),
        "std": float(np.std(pooled, ddof=1)) if len(pooled) > 1 else 0.0,
        "n": len(pooled),
        "values": pooled,
    }

    summary = {
        "slice1": results[0]["slice1"],
        "slice2": results[0]["slice2"],
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
    print("FINAL SUMMARY  (n={} seeds)".format(len(results)))
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
    ap.add_argument("--slice1", required=True, choices=list(DATASETS))
    ap.add_argument("--slice2", required=True, choices=list(DATASETS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--resolution", type=int, default=64)
    ap.add_argument("--num-neighbors", type=int, default=7)
    ap.add_argument("--prune", type=int, default=20000)
    ap.add_argument("--image-encoder", default="uni")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                    help="Random seeds to run (each gets its own subdir).")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    if args.slice1 == args.slice2:
        raise SystemExit("slice1 must differ from slice2")

    pair_name = f"{args.slice1}__{args.slice2}"
    pair_dir = Path(args.output_dir) if args.output_dir else RUNS_ROOT / pair_name
    pair_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = pair_dir / "cache"

    # ---- deterministic prep (run once, then reused for every seed) ----
    a1 = prepare_slice(args.slice1, args.resolution, args.image_encoder,
                       args.device, cache_dir)
    a2 = prepare_slice(args.slice2, args.resolution, args.image_encoder,
                       args.device, cache_dir)

    if list(a1.var_names) != list(a2.var_names):
        print(f"\n[warn] gene panels differ; intersecting...")
        common = sorted(set(a1.var_names) & set(a2.var_names))
        a1 = a1[:, common].copy()
        a2 = a2[:, common].copy()
        print(f"  shared genes: {len(common)}")

    print(f"\n[shapes] S1 {a1.shape} | S2 {a2.shape}")

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
        seed_dir = pair_dir / f"seed_{seed}"
        r = train_one_seed(a1, a2, g1, g2, seed=seed,
                           epochs=args.epochs, prune=args.prune,
                           device=args.device,
                           slice1_name=args.slice1, slice2_name=args.slice2,
                           out_dir=seed_dir)
        results.append(r)

    summarize(results, pair_dir,
              encoder=args.image_encoder,
              resolution=args.resolution,
              num_neighbors=args.num_neighbors)


if __name__ == "__main__":
    main()
