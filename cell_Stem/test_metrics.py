"""Evaluate saved cell_Stem predictions and report per-dataset MSE / MAE.

Scans completed runs under ``runs/<dataset>/<version>/<dataset>_fold<k>_seed<seed>/``
(produced by ``predict.py``), loads the saved ``pred.npy`` and the corresponding
ground-truth test-half expression, then computes:

  - Per-gene MSE / MAE
  - Mean MSE / MAE averaged across genes
  - Per-run, per-direction, and per-dataset/version aggregates
  - PCC (for reference, matching the metric used during training)

Stem predicts in ``log2(count+1)`` space, so both predictions and targets are
compared on that scale.

Usage
-----
    # Evaluate all completed runs (uses saved pred.npy; fast, CPU-only)
    CUDA_VISIBLE_DEVICES=7 python test_metrics.py --config config.yaml

    # Restrict to a specific dataset
    python test_metrics.py --config config.yaml --dataset hSkin_Melanoma

    # Restrict to a specific version (default: latest version per dataset)
    python test_metrics.py --config config.yaml --version v3

    # Specify output JSON path
    python test_metrics.py --config config.yaml --output test_metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from glob import glob
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import yaml

warnings.filterwarnings('ignore', message='Received a view of an AnnData')


# Keep in sync with predict.py's DATASETS registry.
DATASETS = {
    "hSkin_Melanoma": {
        "mode": "single",
        "dir": "hSkin_Melanoma",
        "cell_file": "adata_S1.h5ad",
    },
    "hColon_Non_diseased": {
        "mode": "single",
        "dir": "hColon_Non_diseased",
        "cell_file": "adata_S1.h5ad",
    },
    "mouse_Colon": {
        "mode": "single",
        "dir": "mouse_Colon",
        "cell_file": "adata_S1.h5ad",
    },
    "Human_Breast_Cancer": {
        "mode": "pair",
        "members": [
            {"label": "Rep1", "dir": "Human_Breast_Cancer_Rep1", "cell_file": "adata_Rep1.h5ad"},
            {"label": "Rep2", "dir": "Human_Breast_Cancer_Rep2", "cell_file": "adata_Rep2.h5ad"},
        ],
    },
}


def extract_gene_expr_log2(adata, gene_names):
    """Return ``log2(count+1)`` matrix [N, len(gene_names)]."""
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


def compute_mse_mae(pred, target):
    """Return per-gene and overall MSE/MAE for prediction / target arrays."""
    pred_np = np.asarray(pred, dtype=np.float32)
    target_np = np.asarray(target, dtype=np.float32)

    diff = pred_np - target_np
    mse_per_gene = np.mean(diff ** 2, axis=0)
    mae_per_gene = np.mean(np.abs(diff), axis=0)
    mse_per_gene = np.where(np.isfinite(mse_per_gene), mse_per_gene, 0.0)
    mae_per_gene = np.where(np.isfinite(mae_per_gene), mae_per_gene, 0.0)

    return {
        'mse_per_gene': mse_per_gene.tolist(),
        'mae_per_gene': mae_per_gene.tolist(),
        'mse_mean': float(np.mean(mse_per_gene)),
        'mae_mean': float(np.mean(mae_per_gene)),
        'mse_overall': float(np.mean(diff ** 2)),
        'mae_overall': float(np.mean(np.abs(diff))),
        'n_cells': int(pred_np.shape[0]),
        'n_genes': int(pred_np.shape[1]),
    }


def compute_pcc(pred, target):
    """Per-gene Pearson r and mean (matches utils.compute_pcc)."""
    pred_np = np.asarray(pred, dtype=np.float32)
    target_np = np.asarray(target, dtype=np.float32)
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


def load_test_truth(ds_name, test_half, gene_names, cfg):
    """Load the test-half ground-truth log2(count+1) expression matrix."""
    use_pc = cfg.get("use_processed_cell", False)
    min_counts = int(cfg.get("min_counts", 10))

    if use_pc:
        pc_root = Path(cfg.get(
            "processed_cell_root", "/home/sb202604/cell-benchmark/processed_cell"
        ))
        adata_path = pc_root / ds_name / f"adata_{test_half}.h5ad"
        adata = ad.read_h5ad(adata_path)
        if "raw" in adata.layers:
            adata.X = adata.layers["raw"].copy()
    else:
        proc_root = Path(cfg["processed_root"])
        spec = DATASETS[ds_name]
        if spec["mode"] == "single":
            ds_dir = proc_root / spec["dir"]
            adata = ad.read_h5ad(ds_dir / "cells.h5ad")
            splits = json.load(open(ds_dir / "splits.json"))
            test_idx = np.asarray(splits["spatial_ood"][test_half], dtype=np.int64)
            adata = adata[test_idx].copy()
        else:
            member = next(m for m in spec["members"] if m["label"] == test_half)
            adata = ad.read_h5ad(proc_root / member["dir"] / "cells.h5ad")

    if "is_gene" in adata.var.columns:
        adata = adata[:, adata.var["is_gene"].astype(bool).values].copy()
    sc.pp.filter_cells(adata, min_counts=min_counts)
    return extract_gene_expr_log2(adata, gene_names)


def discover_runs(output_root, datasets=None, version=None):
    """Discover completed (dataset, version, run_dir) tuples.

    If ``version`` is None, only the latest version folder per dataset is
    returned. Set ``version`` to a specific value (e.g. ``v3``) to evaluate
    that version only.
    """
    discovered = []
    ds_names = [datasets] if isinstance(datasets, str) else datasets
    if ds_names is None:
        ds_names = sorted(DATASETS.keys())

    for ds_name in ds_names:
        ds_dir = Path(output_root) / ds_name
        if not ds_dir.exists():
            continue

        version_dirs = sorted([
            d for d in ds_dir.iterdir()
            if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit()
        ])
        if not version_dirs:
            continue

        if version is not None:
            selected = [d for d in version_dirs if d.name == version]
        else:
            selected = [max(version_dirs, key=lambda d: int(d.name[1:]))]

        for ver_dir in selected:
            run_dirs = sorted([
                d for d in ver_dir.iterdir()
                if d.is_dir() and d.name.startswith(f"{ds_name}_fold") and "_seed" in d.name
            ])
            for run_dir in run_dirs:
                result_path = run_dir / "result.json"
                pred_path = run_dir / "pred.npy"
                if result_path.exists() and pred_path.exists():
                    discovered.append({
                        'dataset': ds_name,
                        'version': ver_dir.name,
                        'run_dir': str(run_dir),
                    })
    return discovered


def load_run_config(run_dir, global_cfg):
    """Load the configuration that was actually used for a run.

    Priority: run_dir/config.yaml > run_dir/config.json > global_cfg.
    This matters when different versions used different settings (e.g.
    ``use_processed_cell``).
    """
    run_dir = Path(run_dir)
    run_yaml = run_dir / "config.yaml"
    run_json = run_dir / "config.json"
    if run_yaml.exists():
        with open(run_yaml) as f:
            return yaml.safe_load(f)
    if run_json.exists():
        with open(run_json) as f:
            return json.load(f)
    return global_cfg


def evaluate_run(run_dir, global_cfg):
    """Evaluate a single run directory and return metrics dict."""
    run_dir = Path(run_dir)
    cfg = load_run_config(run_dir, global_cfg)

    result_path = run_dir / "result.json"
    pred_path = run_dir / "pred.npy"

    with open(result_path) as f:
        result = json.load(f)

    pred = np.load(pred_path).astype(np.float32)
    gene_names = result["gene_names"]
    ds_name = result["dataset"]
    test_half = result["test_half"]
    train_half = result["train_half"]

    true = load_test_truth(ds_name, test_half, gene_names, cfg)
    if pred.shape != true.shape:
        raise ValueError(
            f"Shape mismatch in {run_dir}: pred {pred.shape} vs true {true.shape}"
        )

    metrics = compute_mse_mae(pred, true)
    _, mean_pcc = compute_pcc(pred, true)
    metrics['pcc'] = float(mean_pcc)

    return {
        'dataset': ds_name,
        'fold': int(result.get('fold', -1)),
        'seed': int(result.get('seed', -1)),
        'version': run_dir.parent.name,
        'run_dir': str(run_dir),
        'train_half': train_half,
        'test_half': test_half,
        'gene_names': gene_names,
        'metrics': metrics,
    }


def aggregate_runs(runs):
    """Aggregate run-level results into dataset/version-level statistics."""
    if not runs:
        return None

    mse_values = [r['metrics']['mse_mean'] for r in runs]
    mae_values = [r['metrics']['mae_mean'] for r in runs]
    pcc_values = [r['metrics']['pcc'] for r in runs]

    def _agg(values):
        arr = np.asarray(values, dtype=np.float32)
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            'n': len(arr),
            'values': values,
        }

    summary = {
        'dataset': runs[0]['dataset'],
        'version': runs[0]['version'],
        'n_runs': len(runs),
        'overall_mse': _agg(mse_values),
        'overall_mae': _agg(mae_values),
        'overall_pcc': _agg(pcc_values),
        'directions': {},
        'runs': runs,
    }

    directions = set((r['train_half'], r['test_half']) for r in runs)
    for train_half, test_half in directions:
        dir_runs = [
            r for r in runs
            if r['train_half'] == train_half and r['test_half'] == test_half
        ]
        key = f"train_{train_half}_test_{test_half}"
        summary['directions'][key] = {
            'n_runs': len(dir_runs),
            'mse': _agg([r['metrics']['mse_mean'] for r in dir_runs]),
            'mae': _agg([r['metrics']['mae_mean'] for r in dir_runs]),
            'pcc': _agg([r['metrics']['pcc'] for r in dir_runs]),
        }

    return summary


def print_summary(dataset_summaries):
    """Print a concise table of results."""
    print(f"\n{'='*90}")
    print("TEST METRICS SUMMARY (MSE / MAE per dataset)")
    print(f"{'='*90}")
    print(f"{'Dataset':<24} {'Version':<8} {'Runs':>5} {'MSE':>14} {'MAE':>14} {'PCC':>14}")
    print("-" * 90)

    for ds_name in sorted(dataset_summaries.keys()):
        summary = dataset_summaries[ds_name]
        print(
            f"{ds_name:<24} {summary['version']:<8} {summary['n_runs']:>5} "
            f"{summary['overall_mse']['mean']:>7.4f}±{summary['overall_mse']['std']:<5.4f} "
            f"{summary['overall_mae']['mean']:>7.4f}±{summary['overall_mae']['std']:<5.4f} "
            f"{summary['overall_pcc']['mean']:>7.4f}±{summary['overall_pcc']['std']:<5.4f}"
        )

    print(f"{'='*90}")
    print(f"Total datasets evaluated: {len(dataset_summaries)}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute MSE/MAE for saved cell_Stem predictions.'
    )
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Config file used during training (default: config.yaml)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Evaluate a single dataset, e.g. hSkin_Melanoma')
    parser.add_argument('--version', type=str, default=None,
                        help='Version folder to evaluate, e.g. v3 (default: latest)')
    parser.add_argument('--output', type=str, default='test_metrics.json',
                        help='Output JSON path for detailed metrics')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_root = Path(cfg.get('output_root', './runs'))
    datasets = [args.dataset] if args.dataset else cfg.get('datasets', list(DATASETS.keys()))

    runs = discover_runs(output_root, datasets=datasets, version=args.version)
    if not runs:
        print(f"No completed runs found under {output_root} matching the criteria.")
        return

    print(f"Discovered {len(runs)} completed run(s). Starting evaluation ...")

    run_results = []
    failed = []
    for i, run_info in enumerate(runs, start=1):
        run_dir = run_info['run_dir']
        print(f"\n[{i}/{len(runs)}] {run_dir}")
        try:
            result = evaluate_run(run_dir, cfg)
            run_results.append(result)
            print(
                f"  Overall -> MSE: {result['metrics']['mse_mean']:.4f}, "
                f"MAE: {result['metrics']['mae_mean']:.4f}, "
                f"PCC: {result['metrics']['pcc']:.4f}, "
                f"cells={result['metrics']['n_cells']}"
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed.append({'run_dir': run_dir, 'error': str(e)})

    # Aggregate per (dataset, version)
    dataset_groups = {}
    for r in run_results:
        key = (r['dataset'], r['version'])
        dataset_groups.setdefault(key, []).append(r)

    dataset_summaries = {}
    for (ds_name, version), group_runs in dataset_groups.items():
        dataset_summaries[ds_name] = aggregate_runs(group_runs)

    print_summary(dataset_summaries)

    output = {
        'config': args.config,
        'output_root': str(output_root),
        'version': args.version,
        'n_datasets': len(dataset_summaries),
        'dataset_summaries': dataset_summaries,
        'failed': failed,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed metrics saved to {output_path}")


if __name__ == '__main__':
    main()
