# Xenium Single-Cell Benchmark

Single-cell-resolution H&E + Xenium datasets for comparing against GHIST, sCellST,
DeepSpot2Cell, SpatialEx, and similar single-cell SGE prediction methods.

## Datasets

Following SpatialEx (Supplementary Table 1, H&E-to-Omics + Panel Integration sections).

| Dataset                     | Section(s)            | Cells (10x seg) | Panel genes | H&E size | outs.zip | Notes |
| --------------------------- | --------------------- | --------------- | ----------- | -------- | -------- | ----- |
| Human_Breast_Cancer_Rep1    | 1 section             | ~167k           | 313         | 1.33 GB  | 9.18 GB  | FFPE, Janesick 2023 |
| Human_Breast_Cancer_Rep2    | 1 section (serial)    | ~119k           | 313         | 1.16 GB  | 7.90 GB  | FFPE, serial to Rep1 |
| hColon_Non_diseased         | 1 section (S1/S2 split) | ~263k         | 325         | 7.15 GB  | 12.69 GB | FFPE, single physical slice |
| mouse_Colon                 | 1 section (S1/S2 split) | ~219k         | 379         | 1.71 GB  | 14.31 GB | FF, multimodal seg |
| hSkin_Melanoma              | 1 section (S1/S2 split) | ~105k         | 282         | 0.87 GB  | 3.52 GB  | FFPE, Avaden Biosciences |
| Human_Breast_IDC_Big_Rep1   | 1 section (whole chip)  | ~893k         | 140         | 8.23 GB  | 48.29 GB | FFPE, Entire Sample Area |
| Human_Breast_IDC_Big_Rep2   | 1 section (whole chip)  | ~886k         | 140         | 8.06 GB  | 46.76 GB | FFPE, Entire Sample Area |

`Rep` = serial biological replicate (cross-section split possible).
`S1/S2` = SpatialEx convention for one physical section artificially halved (only spatial-OOD split possible).

**Core group** (5 datasets, ~60 GB):  Human_Breast_Cancer_Rep1/2 + hColon + mouse_Colon + hSkin_Melanoma
**Big group**  (2 datasets, ~111 GB): IDC_Big_Rep1/2

## Layout

```
cell-benchmark/
├── README.md
├── scripts/
│   ├── urls.tsv          # 21-row manifest (dataset, role, rel_path, url)
│   ├── download.sh       # resumable parallel downloader
│   └── unpack.sh         # unzip *_outs.zip into raw/<dataset>/outs/
├── raw/                  # downloaded files (see scripts/urls.tsv for layout)
└── logs/                 # per-file curl logs
```

After unpack, each dataset directory contains:
```
raw/<Dataset>/
├── *_outs.zip
├── *_he_image.ome.tif (or *_he_unaligned_image.ome.tif for IDC_Big)
├── *_he_imagealignment.csv
└── outs/
    ├── cells.csv.gz
    ├── cell_feature_matrix.h5
    ├── transcripts.parquet
    ├── cell_boundaries.parquet
    ├── nucleus_boundaries.parquet
    ├── gene_panel.json
    ├── morphology_mip.ome.tif
    ├── morphology_focus.ome.tif (newer datasets)
    └── analysis/
```

## Usage

```bash
# Default: download the 5 H&E→Omics datasets (~60 GB, ~3 parallel streams)
./scripts/download.sh

# Add IDC_Big (Panel Integration), total ~171 GB
./scripts/download.sh --group all

# Single dataset
./scripts/download.sh --dataset hSkin_Melanoma

# More parallelism
DL_JOBS=6 ./scripts/download.sh --group all

# Unpack outs bundles
./scripts/unpack.sh
```

The downloader is resumable (`curl -C -`) and skips files whose local size matches
the server's `Content-Length`, so re-running after interruption is safe.

## Notes on H&E + alignment files

- **hColon**: `outs.zip` is from the `Add_on` panel sample; the post-Xenium H&E +
  alignment CSV are only released with the matching `Base` panel sample (same
  physical slice, narrower gene panel). Both reference the same H&E image.
- **hSkin**: `outs.zip` is the `xeniumranger` reprocessed bundle (version 1.7.0
  with extended panel); H&E + alignment come from the original 1.6.0 release of
  the same section.
- **IDC_Big**: 10x only ships the *unaligned* H&E. Use the `imagealignment.csv`
  (3×3 affine, H&E pixel → Xenium DAPI um) to register it during preprocessing.
- All other datasets ship both `he_image.ome.tif` (post-Xenium, registered) and
  `he_imagealignment.csv` from the same release tag as `outs.zip`.

## Provenance

URLs and Content-Length verified against `cf.10xgenomics.com` on 2026-05-21
by scraping the public dataset pages on `10xgenomics.com/datasets/`.
