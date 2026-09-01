"""Wrap SpatialEx Tutorial 1 (Section "Within the sequencing area") to run
training + cross-slice inference + PCC evaluation on a pair (S1, S2)
already preprocessed by `build_adata.py`.

Usage:
  python run_pair.py --dataset hSkin_Melanoma [--epochs 500] [--device cuda:0]
                     [--output-dir /home/sb202604/cell_SpatialEx/runs/hSkin_Melanoma]

This corresponds to Tutorial 1 cells [13] (train) + [19,21] (eval).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

# Stub out cellpose (used only by SpatialEx's optional H&E segmentation utils;
# we don't call those — our cells are already segmented by Xenium).
for _m in ("cellpose", "cellpose.models"):
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)

# Make the local SpatialEx package importable when running from this directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anndata as ad
import numpy as np
import scanpy as sc

import SpatialEx as se

PROCESSED_ROOT = Path("/home/sb202604/cell-benchmark/processed")
RUNS_ROOT = Path(__file__).resolve().parents[1] / "runs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--num-neighbors", type=int, default=7)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--prune", type=int, default=20000,
                    help="grid prune size in MICRONS for SpatialEx batching. "
                         "Set larger than slice extent to keep one ROI per "
                         "slice and avoid duplicate-prediction across grid overlaps.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else RUNS_ROOT / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    s1_path = PROCESSED_ROOT / args.dataset / "spatialex" / "S1.h5ad"
    s2_path = PROCESSED_ROOT / args.dataset / "spatialex" / "S2.h5ad"
    if not (s1_path.exists() and s2_path.exists()):
        raise SystemExit(
            f"missing S1/S2 adata for {args.dataset}; "
            f"run scripts/build_adata.py first."
        )

    print(f"[load] {s1_path}")
    a1 = ad.read_h5ad(s1_path)
    print(f"  S1: {a1.shape}  he obsm: {a1.obsm['he'].shape}")
    print(f"[load] {s2_path}")
    a2 = ad.read_h5ad(s2_path)
    print(f"  S2: {a2.shape}  he obsm: {a2.obsm['he'].shape}")

    print(f"\n[graph] building spatial KNN hypergraphs (k={args.num_neighbors})...")
    t0 = time.time()
    g1 = se.pp.Build_hypergraph_spatial_and_HE(
        a1, args.num_neighbors, graph_kind="spatial", return_type="crs")
    g2 = se.pp.Build_hypergraph_spatial_and_HE(
        a2, args.num_neighbors, graph_kind="spatial", return_type="crs")
    print(f"  done in {time.time()-t0:.1f}s")

    print(f"\n[train] SpatialEx  epochs={args.epochs}  device={args.device}  prune={args.prune}")
    t0 = time.time()
    trainer = se.SpatialEx(
        a1, a2, g1, g2,
        epochs=args.epochs,
        device=args.device,
        prune=args.prune,
    )
    trainer.train()
    print(f"  training done in {time.time()-t0:.1f}s")

    print(f"\n[infer] auto cross-slice prediction")
    # Tutorial returns (panel_1b, panel_2a). For us the two slices share the
    # same gene panel, so panelB1 = "what S2's model predicts on S1's H&E"
    # and panelA2 = "what S1's model predicts on S2's H&E". This is exactly
    # the cross-slice held-out setup.
    panelB1, panelA2 = trainer.auto_inference()
    print(f"  panelB1 (S1 cells via S2-trained model): {panelB1.shape}")
    print(f"  panelA2 (S2 cells via S1-trained model): {panelA2.shape}")

    print(f"\n[eval] PCC per gene")
    # SpatialEx's Compute_metrics(reduce='mean') returns (per_gene, mean).
    # We unpack and use those directly.
    pcc_S1_per, pcc_S1 = se.utils.Compute_metrics(
        a1.X.copy(), panelB1.copy(), metric="pcc", reduce="mean")
    pcc_S2_per, pcc_S2 = se.utils.Compute_metrics(
        a2.X.copy(), panelA2.copy(), metric="pcc", reduce="mean")
    print(f"  mean PCC (train=S2, test=S1): {pcc_S1:.4f}")
    print(f"  mean PCC (train=S1, test=S2): {pcc_S2:.4f}")

    np.save(out_dir / "pred_S1_from_S2.npy", panelB1)
    np.save(out_dir / "pred_S2_from_S1.npy", panelA2)
    np.save(out_dir / "pcc_S1_from_S2_per_gene.npy", np.asarray(pcc_S1_per))
    np.save(out_dir / "pcc_S2_from_S1_per_gene.npy", np.asarray(pcc_S2_per))

    results = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "num_neighbors": args.num_neighbors,
        "device": args.device,
        "prune": args.prune,
        "n_cells": {"S1": int(a1.n_obs), "S2": int(a2.n_obs)},
        "n_genes": int(a1.n_vars),
        "mean_pcc_train_S2_test_S1": float(pcc_S1),
        "mean_pcc_train_S1_test_S2": float(pcc_S2),
        "encoder": "H0-mini",
        "encoder_feat_dim": int(a1.obsm["he"].shape[1]),
        "gene_names": list(a1.var_names),
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[save] {out_dir}/")


if __name__ == "__main__":
    main()
