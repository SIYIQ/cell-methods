# HST_STFlow

STFlow comparison variant of the HST-middle benchmark pipeline.

This directory wraps the official STFlow model (`/home/sb202604/STFlow`) in
the HST-middle / HST_novae data + evaluation harness so it can be compared
head-to-head with `HST-middle/h0mini_official_offline` and `HST_novae` on
the same nine HEST-bench tasks under the same protocol.

## What is shared with HST-middle / HST_novae

The data pipeline is byte-for-byte identical so the only meaningful
variable between the three runs is the gene-prediction architecture:

  * offline cache layout (`/home/sb202604/hest-bench-pretreat`)
  * `load_slide` (ImageNet normalization, log1p adata, barcode alignment)
  * `select_top_hvgs_official` (HEST 200-HVG selection)
  * patient-aware k-fold CV in `benchmark_tasks.py`
  * grid + halo partitioning for large slides
  * per-slide PCC eval, best-by-train / best-by-test checkpointing
  * per-task / global `summary.json` aggregation
  * 3 random seeds × k folds, AdamW + cosine LR schedule
  * `FlexibleImageEncoder` (h0_mini partial unfreeze, gradient checkpointing)

## What is different

The model is STFlow itself, imported verbatim from `/home/sb202604/STFlow`:

  * `stflow.model.denoiser.Denoiser` (SpatialTransformer + Fourier time
    embedding + KNN graph built inside the model from `coords`)
  * `stflow.flow.interpolant.Interpolant` (ZINB prior + linear flow
    matching with the official zinb prior parameters)
  * loss = STFlow's native MSE flow-matching (no MGM, no Pearson loss,
    no uncertainty head)
  * inference = STFlow's Euler-step sampling loop (`n_sample_steps=5`)

Files that change vs. HST-middle / HST_novae:

  * `model.py`     — `STFlowModel` (FlexibleImageEncoder + STFlow's
                     Denoiser + Interpolant).
  * `utils.py`     — same as HST_novae minus `build_graph_data` /
                     `build_spatial_edges` / `build_morph_edges` (STFlow
                     builds its own KNN graph internally).
  * `train.py`     — drop-in replacement for HST_novae/train.py; calls
                     `STFlowModel.train_step` / `STFlowModel.predict`.
  * `train_spotsplit.py` — same grid+halo scaffolding, no MGM / loss_type
                           branches.
  * `config*.yaml` — same training schedule / encoder unfreeze knobs as
                     HST_novae; STFlow architecture keys (`stflow_*`)
                     stay at STFlow defaults across all tasks so STFlow's
                     native design is unchanged.

The STFlow repo itself is **not modified**: this directory only imports
`stflow.model.denoiser` and `stflow.flow.interpolant`. Set the
environment variable `STFLOW_ROOT` to override the default location
(`/home/sb202604/STFlow`).

## Usage

```
python train.py --config config_Task1_IDC.yaml
python train.py --config config_Task1_IDC.yaml --fold 0 --seed 42
```

Per-task configs live in `config_TaskN_*.yaml`; the global default
`config.yaml` carries the 200-gene HVG variant.
