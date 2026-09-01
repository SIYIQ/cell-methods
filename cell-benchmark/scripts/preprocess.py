"""Cell-level preprocessing for Xenium datasets.

Produces a unified `processed/<dataset>/` directory consumed by:
  - cell_SpatialEx  (via its own data loader wrapper)

Outputs per dataset:
  cells.h5ad       AnnData  N_cells x N_genes (sparse counts)
                     obs columns:
                       x_um, y_um             — Xenium um (cell centroid)
                       x_he, y_he             — H&E pixel coordinates after alignment
                       nucleus_area, cell_area — μm²
                       total_counts           — sum over genes (Xenium-reported)
                       pass_qc                — bool
                     var:
                       feature_type           — Gene Expression / Negative ... / Codeword
                       is_gene                — bool (True only for real genes)
  patches.npy      uint8 memmap [N_cells_qc, 3, 224, 224] H&E patches at level 0
                     in cell_id order matching cells.h5ad after filtering
  cell_graph.npz   keys: edge_index [2, E] int64 (knn k=8, undirected)
                          edge_dist [E] float32 (in μm)
  splits.json      train/val/test cell_id lists for multiple split modes:
                     - in_slide_random (80/10/10 random)
                     - spatial_ood     (left/right half by x_um median, SpatialEx S1/S2 style)
                   For Rep1/Rep2 datasets, only in_slide_random is emitted
                   here; cross-section splits are constructed across datasets
                   by the evaluator.
  meta.json        provenance: paths, pixel_size, n_cells_raw/qc, etc.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pyarrow.parquet as pq
import scipy.sparse as sp
import tifffile
from scipy.spatial import cKDTree


# ----------------------------- dataset registry -----------------------------
# Each entry: outs_zip_member_dir name, H&E .ome.tif filename, alignment csv,
# and the optional "section_id" used for SpatialEx-style S1/S2 split.
RAW_ROOT = Path("/home/sb202604/cell-benchmark/raw")
OUT_ROOT = Path("/home/sb202604/cell-benchmark/processed")

# Each entry maps dataset name -> dict with:
#   outs_zip:      the *_outs.zip filename (will be unpacked on demand)
#   he_image:      the H&E .ome.tif filename
#   align_csv:     the alignment matrix CSV
# All paths are relative to RAW_ROOT / dataset_name.
DATASETS = {
    "hSkin_Melanoma": {
        "outs_zip":  "Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE_outs.zip",
        "he_image":  "Xenium_V1_hSkin_Melanoma_Base_FFPE_he_image.ome.tif",
        "align_csv": "Xenium_V1_hSkin_Melanoma_Base_FFPE_he_imagealignment.csv",
    },
    "Human_Breast_Cancer_Rep1": {
        "outs_zip":  "Xenium_FFPE_Human_Breast_Cancer_Rep1_outs.zip",
        "he_image":  "Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.ome.tif",
        "align_csv": "Xenium_FFPE_Human_Breast_Cancer_Rep1_he_imagealignment.csv",
    },
    "Human_Breast_Cancer_Rep2": {
        "outs_zip":  "Xenium_FFPE_Human_Breast_Cancer_Rep2_outs.zip",
        "he_image":  "Xenium_FFPE_Human_Breast_Cancer_Rep2_he_image.ome.tif",
        "align_csv": "Xenium_FFPE_Human_Breast_Cancer_Rep2_he_imagealignment.csv",
    },
    "hColon_Non_diseased": {
        "outs_zip":  "Xenium_V1_hColon_Non_diseased_Add_on_FFPE_outs.zip",
        "he_image":  "Xenium_V1_hColon_Non_diseased_Base_FFPE_he_image.ome.tif",
        "align_csv": "Xenium_V1_hColon_Non_diseased_Base_FFPE_he_imagealignment.csv",
    },
    "mouse_Colon": {
        "outs_zip":  "Xenium_V1_mouse_Colon_FF_outs.zip",
        "he_image":  "Xenium_V1_mouse_Colon_FF_he_image.ome.tif",
        "align_csv": "Xenium_V1_mouse_Colon_FF_he_imagealignment.csv",
    },
}


# ----------------------------- IO helpers ----------------------------------
CORE_MEMBERS = [
    "cells.parquet",
    "cell_feature_matrix.h5",
    "gene_panel.json",
    "nucleus_boundaries.parquet",
    "experiment.xenium",
]


def ensure_outs_extracted(dataset_dir: Path, zip_name: str) -> Path:
    """Unzip only the core small files we need. Returns the outs/ directory."""
    outs_dir = dataset_dir / "outs"
    needs = [m for m in CORE_MEMBERS if not (outs_dir / m).exists()]
    if not needs:
        return outs_dir
    import zipfile
    zpath = dataset_dir / zip_name
    if not zpath.exists():
        raise FileNotFoundError(f"missing outs zip: {zpath}")
    outs_dir.mkdir(parents=True, exist_ok=True)
    print(f"  unzipping {len(needs)} core members from {zpath.name}...")
    with zipfile.ZipFile(zpath) as z:
        for m in needs:
            # Some Xenium zips have entries at top level; verify.
            try:
                z.extract(m, outs_dir)
            except KeyError:
                # Try with a 'outs/' prefix
                z.extract(f"outs/{m}", dataset_dir)
    return outs_dir


def load_xenium_metadata(outs_dir: Path) -> dict:
    with open(outs_dir / "experiment.xenium") as f:
        return json.load(f)


def load_alignment(csv_path: Path) -> np.ndarray:
    """Load the 3x3 affine. Convention: A @ [he_px,1]^T = [morpho_px,1]^T."""
    A = np.loadtxt(csv_path, delimiter=",")
    assert A.shape == (3, 3), f"unexpected alignment shape {A.shape}"
    return A


def morpho_um_to_he_px(um_xy: np.ndarray, A: np.ndarray, pixel_size_um: float) -> np.ndarray:
    """um (in Xenium morphology coords) -> H&E pixel coords.

    Steps: um -> morpho_px (divide by pixel_size) then morpho_px -> he_px via A^-1.
    """
    morpho_px = um_xy / pixel_size_um
    homo = np.hstack([morpho_px, np.ones((len(morpho_px), 1))])
    Ainv = np.linalg.inv(A)
    he_xy = (Ainv @ homo.T).T[:, :2]
    return he_xy


def load_cell_feature_matrix(h5_path: Path) -> tuple[sp.csr_matrix, list[str], list[str], list[str]]:
    """Returns (X[csr], barcodes, feature_names, feature_types).

    The matrix is stored as CSC in the 10x format (features along rows).
    We transpose to (cells, features).
    """
    with h5py.File(h5_path, "r") as f:
        m = f["matrix"]
        data = m["data"][...]
        indices = m["indices"][...]
        indptr = m["indptr"][...]
        shape = m["shape"][...]  # [n_features, n_cells]
        barcodes = [b.decode() for b in m["barcodes"][...]]
        feats = m["features"]
        names = [b.decode() for b in feats["name"][...]]
        ftypes = [b.decode() for b in feats["feature_type"][...]]
    X_feat_by_cell = sp.csc_matrix((data, indices, indptr), shape=tuple(shape))
    X_cell_by_feat = X_feat_by_cell.T.tocsr()
    return X_cell_by_feat, barcodes, names, ftypes


def build_knn_graph(coords_um: np.ndarray, k: int = 8) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(coords_um)
    # k+1 because self is included
    d, idx = tree.query(coords_um, k=k + 1)
    src = np.repeat(np.arange(len(coords_um)), k)
    dst = idx[:, 1:].reshape(-1)
    dist = d[:, 1:].reshape(-1).astype(np.float32)
    # Make undirected: stack with reverse
    edge_index = np.stack([
        np.concatenate([src, dst]),
        np.concatenate([dst, src]),
    ]).astype(np.int64)
    edge_dist = np.concatenate([dist, dist])
    # Deduplicate (u,v) pairs (KNN is asymmetric so duplicates may occur after symmetrising)
    keys = edge_index[0] * len(coords_um) + edge_index[1]
    _, keep = np.unique(keys, return_index=True)
    keep.sort()
    return edge_index[:, keep], edge_dist[keep]


def make_random_split(n: int, ratios=(0.8, 0.1, 0.1), seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr = int(n * ratios[0])
    n_va = int(n * ratios[1])
    return perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]


def make_spatial_ood_split(coords_um: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SpatialEx 'S1/S2' style: split by x-median into two halves."""
    x = coords_um[:, 0]
    xmed = float(np.median(x))
    s1 = np.where(x <  xmed)[0]
    s2 = np.where(x >= xmed)[0]
    return s1, s2


# ----------------------------- patch extraction ----------------------------
def extract_patches(
    he_path: Path,
    centers_he_px: np.ndarray,
    patch_size: int,
    out_path: Path,
) -> None:
    """Cut patch_size x patch_size patches around each centre.

    Reads the full level-0 H&E once into RAM (uint8, ~1-8 GB depending on
    slide), then slices each patch via numpy. This is dramatically faster
    than zarr cell-by-cell IO and well within our RAM budget (256+ GB).
    """
    P = patch_size
    H_pat = P // 2
    N = len(centers_he_px)

    print(f"  reading H&E level 0 into memory...")
    t0 = time.time()
    img = tifffile.imread(str(he_path), series=0, level=0)  # (H, W, 3) uint8
    H, W, _ = img.shape
    print(f"    H&E shape={img.shape}  loaded in {time.time()-t0:.1f}s  "
          f"({img.nbytes/1e9:.1f} GB RAM)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(out_path, dtype=np.uint8, mode="w+", shape=(N, 3, P, P))

    cx = centers_he_px[:, 0].round().astype(np.int64)
    cy = centers_he_px[:, 1].round().astype(np.int64)

    print(f"  extracting {N:,} patches of size {P}x{P}...")
    t0 = time.time()
    report_every = max(1, N // 10)
    for i in range(N):
        x0 = cx[i] - H_pat; x1 = x0 + P
        y0 = cy[i] - H_pat; y1 = y0 + P
        # All QC'd cells are guaranteed within (half, H-half) and (half, W-half)
        # by the pass_qc filter, so no bounds-checking needed here.
        patch = img[y0:y1, x0:x1, :]
        mm[i] = patch.transpose(2, 0, 1)
        if (i + 1) % report_every == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"    {i+1:,}/{N:,}  ({100*(i+1)/N:.0f}%)  rate={rate:.0f}/s")
    mm.flush()
    print(f"  patches done in {time.time()-t0:.1f}s")

    # Free RAM before returning
    del img


# ----------------------------- main per-dataset ----------------------------
def process(dataset: str, patch_size: int, knn_k: int, force: bool) -> None:
    print(f"\n===== {dataset} =====")
    info = DATASETS[dataset]
    ds_dir = RAW_ROOT / dataset
    out_dir = OUT_ROOT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    if not force and (out_dir / "meta.json").exists():
        print("  already processed (use --force to redo); skipping")
        return

    outs_dir = ensure_outs_extracted(ds_dir, info["outs_zip"])
    meta_xenium = load_xenium_metadata(outs_dir)
    pixel_size = float(meta_xenium["pixel_size"])
    print(f"  pixel_size = {pixel_size}  um/px  (DAPI/morpho)")

    # ---- cells.parquet ----
    cells_df = pq.read_table(outs_dir / "cells.parquet").to_pandas()
    print(f"  cells.parquet rows: {len(cells_df):,}")
    cell_ids = cells_df["cell_id"].astype(str).values
    um_xy = cells_df[["x_centroid", "y_centroid"]].to_numpy(np.float64)

    # ---- expression matrix ----
    X, barcodes, feat_names, feat_types = load_cell_feature_matrix(
        outs_dir / "cell_feature_matrix.h5")
    assert len(barcodes) == len(cell_ids), \
        f"cell count mismatch: cells.parquet={len(cell_ids)} vs h5={len(barcodes)}"
    # Reorder X to match cells.parquet barcode order (usually already aligned, but be safe)
    bc_to_idx = {b: i for i, b in enumerate(barcodes)}
    perm = np.fromiter((bc_to_idx[c] for c in cell_ids), dtype=np.int64, count=len(cell_ids))
    if not np.array_equal(perm, np.arange(len(cell_ids))):
        print("  reordering X to match cells.parquet order")
        X = X[perm]

    is_gene = np.array([t == "Gene Expression" for t in feat_types])
    print(f"  features: {len(feat_names)} total, {is_gene.sum()} genes")

    # ---- H&E alignment ----
    A = load_alignment(ds_dir / info["align_csv"])
    he_xy = morpho_um_to_he_px(um_xy, A, pixel_size)

    # ---- H&E image bounds (read just the shape) ----
    with tifffile.TiffFile(ds_dir / info["he_image"]) as tf:
        z_shape = tf.series[0].levels[0].shape  # (H, W, 3)
    HE_H, HE_W, _ = z_shape
    print(f"  H&E full-res shape: {z_shape}")

    # ---- QC ----
    half = patch_size // 2
    in_bounds = (
        (he_xy[:, 0] >= half) & (he_xy[:, 0] < HE_W - half) &
        (he_xy[:, 1] >= half) & (he_xy[:, 1] < HE_H - half)
    )
    nuc_area = cells_df["nucleus_area"].to_numpy(np.float32)
    total_counts = cells_df["total_counts"].to_numpy(np.int64)
    pass_qc = in_bounds & (nuc_area >= 10.0) & (total_counts > 0)
    print(f"  QC: in_bounds={in_bounds.sum():,}  "
          f"nuc>=10um2={(nuc_area>=10).sum():,}  "
          f"total>0={(total_counts>0).sum():,}  "
          f"pass={pass_qc.sum():,} / {len(pass_qc):,}")

    keep = np.where(pass_qc)[0]
    X_kept = X[keep]
    cell_ids_kept = cell_ids[keep]
    um_kept = um_xy[keep]
    he_kept = he_xy[keep]

    # ---- KNN graph on um coords (post-QC index space) ----
    print(f"  building KNN (k={knn_k}) on {len(keep):,} cells...")
    t0 = time.time()
    ei, ed = build_knn_graph(um_kept, k=knn_k)
    print(f"    edges: {ei.shape[1]:,}  ({time.time()-t0:.1f}s)")
    np.savez(out_dir / "cell_graph.npz", edge_index=ei, edge_dist=ed)

    # ---- splits ----
    n = len(keep)
    tr, va, te = make_random_split(n, seed=42)
    s1_idx, s2_idx = make_spatial_ood_split(um_kept)
    splits = {
        "in_slide_random": {"train": tr.tolist(), "val": va.tolist(), "test": te.tolist()},
        "spatial_ood": {"S1": s1_idx.tolist(), "S2": s2_idx.tolist()},
    }
    with open(out_dir / "splits.json", "w") as f:
        json.dump(splits, f)

    # ---- cells.h5ad ----
    obs = {
        "x_um": um_kept[:, 0].astype(np.float32),
        "y_um": um_kept[:, 1].astype(np.float32),
        "x_he": he_kept[:, 0].astype(np.float32),
        "y_he": he_kept[:, 1].astype(np.float32),
        "nucleus_area": nuc_area[keep],
        "cell_area":    cells_df["cell_area"].to_numpy(np.float32)[keep],
        "total_counts": total_counts[keep],
        "pass_qc": np.ones(n, dtype=bool),
    }
    import pandas as pd
    obs_df = pd.DataFrame(obs, index=cell_ids_kept)
    var_df = pd.DataFrame({
        "feature_type": feat_types,
        "is_gene": is_gene,
    }, index=feat_names)
    adata = ad.AnnData(X=X_kept.astype(np.float32), obs=obs_df, var=var_df)
    adata.uns["dataset"] = dataset
    adata.uns["pixel_size_morpho"] = pixel_size
    adata.write_h5ad(out_dir / "cells.h5ad", compression="gzip")
    print(f"  cells.h5ad written: {adata.shape}")

    # ---- patches ----
    patch_path = out_dir / "patches.npy"
    extract_patches(ds_dir / info["he_image"], he_kept, patch_size, patch_path)

    # ---- meta ----
    meta = {
        "dataset": dataset,
        "pixel_size_morpho_um": pixel_size,
        "patch_size": patch_size,
        "knn_k": knn_k,
        "n_cells_raw": int(len(cell_ids)),
        "n_cells_qc": int(n),
        "he_shape": [int(HE_H), int(HE_W), 3],
        "panel_name": meta_xenium.get("panel_name", ""),
        "panel_organism": meta_xenium.get("panel_organism", ""),
        "panel_tissue_type": meta_xenium.get("panel_tissue_type", ""),
        "n_genes": int(is_gene.sum()),
        "n_features": int(len(feat_names)),
        "preservation": meta_xenium.get("preservation_method", ""),
        "raw_dir": str(ds_dir),
        "files": {
            "outs_zip": info["outs_zip"],
            "he_image": info["he_image"],
            "align_csv": info["align_csv"],
        },
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  meta.json written")


# ----------------------------- CLI -----------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", nargs="*", default=list(DATASETS.keys()),
                   help="subset of dataset names to process")
    p.add_argument("--patch-size", type=int, default=224)
    p.add_argument("--knn-k", type=int, default=8)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    for d in args.dataset:
        if d not in DATASETS:
            raise SystemExit(f"unknown dataset {d}; choices={list(DATASETS)}")
        process(d, args.patch_size, args.knn_k, args.force)


if __name__ == "__main__":
    main()
