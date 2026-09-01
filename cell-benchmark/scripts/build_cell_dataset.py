"""Build the shared cell-level data interface used by cell_HST_middle /
cell_SpatialEx.

Reads the chessboard cache produced by cell_SpatialEx/scripts/run_native.py
(or the pair-mode cache for HBC Rep1/Rep2) and emits a unified per-dataset
directory that every method consumes.

Output layout (per dataset):
    cell-benchmark/processed_cell/<dataset>/
    ├── adata_S1.h5ad   for chessboard datasets: cells in the S1 half
    ├── adata_S2.h5ad   for chessboard datasets: cells in the S2 half
    ├── (or for the Rep1/Rep2 pair:)
    ├── adata_Rep1.h5ad
    ├── adata_Rep2.h5ad
    └── meta.json

Each AnnData contains:
    X         log1p-normalized counts (dense float32, SpatialEx convention)
    obs       x_centroid, y_centroid (microns), image_col, image_row, ...
    obsm
      spatial    [N, 2] microns
      he         [N, 1024] UNI features (float32)
    var       gene names
    uns
      dataset, split (S1/S2/Rep1/Rep2), parent_dataset

The cell-cell hypergraph is NOT stored; it is rebuilt at training time by
each method via SpatialEx's Build_hypergraph_spatial_and_HE (deterministic
from adata.obsm['spatial']).

This is the *only* preprocessing each downstream method should need. From
here, each fork consumes the .h5ad files directly with no rerun of UNI.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

# Stub cellpose for SpatialEx import. Only stub when the real package is NOT
# installed; otherwise a fake module would shadow the real cellpose for the
# whole process (sys.modules lookup wins).
try:
    import cellpose  # noqa: F401
except ImportError:
    for _m in ("cellpose", "cellpose.models"):
        sys.modules.setdefault(_m, types.ModuleType(_m))

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cell_SpatialEx"))

import anndata as ad
import numpy as np


RUNS_NATIVE = Path("/home/sb202604/cell_SpatialEx/runs_native")
OUT_ROOT = Path("/home/sb202604/cell-benchmark/processed_cell")


# Single-section datasets: chessboard split into S1/S2
CHESSBOARD_DATASETS = {
    "hSkin_Melanoma":      RUNS_NATIVE / "hSkin_Melanoma_chessboard",
    "hColon_Non_diseased": RUNS_NATIVE / "hColon_Non_diseased_chessboard",
    "mouse_Colon":         RUNS_NATIVE / "mouse_Colon_chessboard",
}

# Pair-mode (true biological replicates): no split needed
PAIR_DATASETS = {
    "Human_Breast_Cancer": (
        RUNS_NATIVE / "Human_Breast_Cancer_Rep1__Human_Breast_Cancer_Rep2",
        ("Human_Breast_Cancer_Rep1", "Human_Breast_Cancer_Rep2"),
    ),
}


def split_by_chessboard(adata, block_um: float = 200.0):
    """Chessboard split: tile section with `block_um` blocks, alternate parity."""
    x = adata.obs["x_centroid"].to_numpy(np.float64)
    y = adata.obs["y_centroid"].to_numpy(np.float64)
    xb = (x - x.min()) // block_um
    yb = (y - y.min()) // block_um
    parity = (xb.astype(np.int64) + yb.astype(np.int64)) % 2
    mask_s1 = parity == 0
    mask_s2 = ~mask_s1
    return mask_s1, mask_s2


def load_cached_full(cache_dir: Path) -> ad.AnnData:
    """Load SpatialEx cache: adata + UNI he features."""
    adata_path = cache_dir / "full_adata.h5ad"
    he_path = cache_dir / "full_he.npz"
    if not (adata_path.exists() and he_path.exists()):
        raise FileNotFoundError(f"missing cache files in {cache_dir}")
    adata = ad.read_h5ad(adata_path)
    he = np.load(he_path)["he"]
    assert he.shape[0] == adata.n_obs, \
        f"shape mismatch: adata={adata.n_obs} he={he.shape[0]}"
    adata.obsm["he"] = he
    return adata


def load_cached_pair(cache_dir: Path, names: tuple[str, str]) -> tuple[ad.AnnData, ad.AnnData]:
    """Load SpatialEx pair-mode cache: per-slice adata + UNI he."""
    n1, n2 = names
    a1_path = cache_dir / f"{n1}_adata.h5ad"
    a2_path = cache_dir / f"{n2}_adata.h5ad"
    h1_path = cache_dir / f"{n1}_he.npz"
    h2_path = cache_dir / f"{n2}_he.npz"
    for p in (a1_path, a2_path, h1_path, h2_path):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")
    a1 = ad.read_h5ad(a1_path)
    a2 = ad.read_h5ad(a2_path)
    a1.obsm["he"] = np.load(h1_path)["he"]
    a2.obsm["he"] = np.load(h2_path)["he"]
    return a1, a2


def subset_and_save(adata: ad.AnnData, mask: np.ndarray, out_path: Path,
                    dataset: str, split: str, parent_dataset: str) -> int:
    """Subset adata, drop other obsm we don't need, add provenance, save."""
    sub = adata[mask].copy()
    # Slim down: keep only spatial + he in obsm
    obsm_keep = {}
    for k in ("spatial", "he"):
        if k in sub.obsm:
            obsm_keep[k] = sub.obsm[k]
    sub.obsm = obsm_keep
    sub.uns["dataset"] = dataset
    sub.uns["split"] = split
    sub.uns["parent_dataset"] = parent_dataset
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.write_h5ad(out_path, compression="gzip")
    print(f"  {split}: {sub.shape}  he={sub.obsm['he'].shape}  -> {out_path.name}")
    return sub.n_obs


def process_chessboard(dataset: str, cache_root: Path,
                       block_um: float, force: bool) -> dict:
    out_dir = OUT_ROOT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    s1_path = out_dir / "adata_S1.h5ad"
    s2_path = out_dir / "adata_S2.h5ad"

    if (not force) and s1_path.exists() and s2_path.exists() and (out_dir / "meta.json").exists():
        print(f"[skip] {dataset}: already built (use --force to rebuild)")
        with open(out_dir / "meta.json") as f:
            return json.load(f)

    print(f"\n===== {dataset} (chessboard) =====")
    adata = load_cached_full(cache_root / "cache")
    print(f"  loaded full: {adata.shape}  he={adata.obsm['he'].shape}")

    mask_s1, mask_s2 = split_by_chessboard(adata, block_um=block_um)
    n1 = subset_and_save(adata, mask_s1, s1_path,
                         dataset=dataset, split="S1", parent_dataset=dataset)
    n2 = subset_and_save(adata, mask_s2, s2_path,
                         dataset=dataset, split="S2", parent_dataset=dataset)

    meta = {
        "dataset": dataset,
        "mode": "chessboard",
        "block_um": block_um,
        "n_cells": {"S1": n1, "S2": n2},
        "n_genes": int(adata.n_vars),
        "he_dim": int(adata.obsm["he"].shape[1]),
        "source_cache": str(cache_root / "cache"),
        "outputs": {"S1": s1_path.name, "S2": s2_path.name},
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def process_pair(dataset: str, cache_root: Path, names: tuple[str, str],
                 force: bool) -> dict:
    n1_name, n2_name = names
    out_dir = OUT_ROOT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    s1_path = out_dir / f"adata_Rep1.h5ad"
    s2_path = out_dir / f"adata_Rep2.h5ad"

    if (not force) and s1_path.exists() and s2_path.exists() and (out_dir / "meta.json").exists():
        print(f"[skip] {dataset}: already built (use --force to rebuild)")
        with open(out_dir / "meta.json") as f:
            return json.load(f)

    print(f"\n===== {dataset} (pair Rep1/Rep2) =====")
    a1, a2 = load_cached_pair(cache_root / "cache", names)
    print(f"  Rep1: {a1.shape}  he={a1.obsm['he'].shape}")
    print(f"  Rep2: {a2.shape}  he={a2.obsm['he'].shape}")

    # The pair caches were saved with separate var lists; the panel is the
    # same (313 genes) but enforce intersection for safety.
    if list(a1.var_names) != list(a2.var_names):
        common = sorted(set(a1.var_names) & set(a2.var_names))
        a1 = a1[:, common].copy(); a2 = a2[:, common].copy()
        print(f"  intersected genes: {len(common)}")

    n1 = subset_and_save(a1, np.ones(a1.n_obs, dtype=bool), s1_path,
                         dataset=n1_name, split="Rep1", parent_dataset=dataset)
    n2 = subset_and_save(a2, np.ones(a2.n_obs, dtype=bool), s2_path,
                         dataset=n2_name, split="Rep2", parent_dataset=dataset)

    meta = {
        "dataset": dataset,
        "mode": "pair",
        "n_cells": {"Rep1": n1, "Rep2": n2},
        "n_genes": int(a1.n_vars),
        "he_dim": int(a1.obsm["he"].shape[1]),
        "source_cache": str(cache_root / "cache"),
        "outputs": {"Rep1": s1_path.name, "Rep2": s2_path.name},
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", nargs="*", default=None,
                   help="subset of datasets to build (default: all)")
    p.add_argument("--block-um", type=float, default=200.0,
                   help="chessboard block size for single-section datasets")
    p.add_argument("--force", action="store_true",
                   help="rebuild even if outputs exist")
    args = p.parse_args()

    targets = set(args.dataset) if args.dataset else (set(CHESSBOARD_DATASETS) | set(PAIR_DATASETS))

    all_meta = {}
    for ds in CHESSBOARD_DATASETS:
        if ds in targets:
            all_meta[ds] = process_chessboard(ds, CHESSBOARD_DATASETS[ds],
                                              args.block_um, args.force)
    for ds in PAIR_DATASETS:
        if ds in targets:
            cache_root, names = PAIR_DATASETS[ds]
            all_meta[ds] = process_pair(ds, cache_root, names, args.force)

    print(f"\n===== SUMMARY =====")
    for ds, m in all_meta.items():
        if m["mode"] == "chessboard":
            print(f"  {ds:30s}  S1={m['n_cells']['S1']:>7,}  S2={m['n_cells']['S2']:>7,}  "
                  f"genes={m['n_genes']}")
        else:
            print(f"  {ds:30s}  Rep1={m['n_cells']['Rep1']:>7,}  Rep2={m['n_cells']['Rep2']:>7,}  "
                  f"genes={m['n_genes']}")


if __name__ == "__main__":
    main()
