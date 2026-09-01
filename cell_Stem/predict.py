"""Stem single-cell Xenium training + sampling script.

Adapts HST_Stem to the single-cell benchmark used by cell_HST_uni_lp.
For each dataset it:
  1. Loads S1/S2 (or Rep1/Rep2) splits from /home/sb202604/cell-benchmark/processed.
  2. Selects 50 dataset-level HVGs with ``select_top_hvgs_official``.
  3. Caches UNI[1024] + CONCH[512] embeddings per half once and concatenates
     them into a 1536-D conditioning vector.
  4. Trains the Stem DiT on the train half for ``train_steps`` iterations.
  5. Samples ``sample_num_per_cond`` trajectories per test cell, averages them,
     and reports per-gene Pearson r against log2(count+1) targets.

Outputs mirror cell_HST_*_lp: ``runs/<dataset>/v<N>/<dataset>_fold<k>_seed42/result.json``.
"""

import os
import sys
import argparse
import json
import shutil
import warnings
import time
from pathlib import Path

_N_THREADS = os.environ.get('HST_NUM_THREADS', '4')
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ.setdefault(_v, _N_THREADS)

import numpy as np
import torch
import yaml
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
import anndata as ad

sys.path.insert(0, '/home/sb202604/Stem')
from Stem.models import Stem_models
from Stem.diffusion import create_diffusion

from utils import (
    select_top_hvgs_official, compute_pcc,
    load_uni, load_conch, freeze, update_ema,
    load_or_encode_features, load_cached_he_features, preprocess_half,
)

warnings.filterwarnings('ignore', message='Received a view of an AnnData')


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
DATASETS = {
    "hSkin_Melanoma": {"mode": "single", "dir": "hSkin_Melanoma", "cell_file": "adata_S1.h5ad"},
    "hColon_Non_diseased": {"mode": "single", "dir": "hColon_Non_diseased", "cell_file": "adata_S1.h5ad"},
    "mouse_Colon": {"mode": "single", "dir": "mouse_Colon", "cell_file": "adata_S1.h5ad"},
    "Human_Breast_Cancer": {
        "mode": "pair",
        "members": [
            {"label": "Rep1", "dir": "Human_Breast_Cancer_Rep1", "cell_file": "adata_Rep1.h5ad"},
            {"label": "Rep2", "dir": "Human_Breast_Cancer_Rep2", "cell_file": "adata_Rep2.h5ad"},
        ],
    },
}


def load_dataset(ds_name: str, cfg: dict):
    """Return a dict with two halves: {label, dir, adata_raw, cell_idx}."""
    spec = DATASETS[ds_name]

    if cfg.get("use_processed_cell", False):
        processed_cell_root = Path(cfg.get("processed_cell_root", "/home/sb202604/cell-benchmark/processed_cell"))
        if spec["mode"] == "single":
            ds_dir = processed_cell_root / spec["dir"]
            adata_s1 = ad.read_h5ad(ds_dir / "adata_S1.h5ad")
            adata_s2 = ad.read_h5ad(ds_dir / "adata_S2.h5ad")
            for adata in (adata_s1, adata_s2):
                if "raw" in adata.layers:
                    adata.X = adata.layers["raw"].copy()
            return {
                "mode": "single",
                "name": ds_name,
                "halves": [
                    {"label": "S1", "dir": spec["dir"], "adata": adata_s1, "cell_file": "adata_S1.h5ad"},
                    {"label": "S2", "dir": spec["dir"], "adata": adata_s2, "cell_file": "adata_S2.h5ad"},
                ],
            }
        else:
            halves = []
            for m in spec["members"]:
                half_file = processed_cell_root / ds_name / m["cell_file"]
                print(f"  load {m['label']} from {half_file}")
                adata = ad.read_h5ad(half_file)
                if "raw" in adata.layers:
                    adata.X = adata.layers["raw"].copy()
                halves.append({"label": m["label"], "dir": m["dir"], "adata": adata, "cell_file": m["cell_file"]})
            return {"mode": "pair", "name": ds_name, "halves": halves}

    processed_root = Path(cfg["processed_root"])
    if spec["mode"] == "single":
        ds_dir = processed_root / spec["dir"]
        adata = ad.read_h5ad(ds_dir / "cells.h5ad")
        splits = json.load(open(ds_dir / "splits.json"))
        s1_idx = np.asarray(splits["spatial_ood"]["S1"], dtype=np.int64)
        s2_idx = np.asarray(splits["spatial_ood"]["S2"], dtype=np.int64)
        return {
            "mode": "single",
            "name": ds_name,
            "halves": [
                {"label": "S1", "dir": spec["dir"], "adata": adata[s1_idx].copy(), "cell_idx": s1_idx},
                {"label": "S2", "dir": spec["dir"], "adata": adata[s2_idx].copy(), "cell_idx": s2_idx},
            ],
        }
    else:
        halves = []
        for m in spec["members"]:
            half_dir = processed_root / m["dir"]
            print(f"  load {m['label']} from {half_dir}")
            adata = ad.read_h5ad(half_dir / "cells.h5ad")
            halves.append({"label": m["label"], "dir": m["dir"], "adata": adata, "cell_idx": None})
        return {"mode": "pair", "name": ds_name, "halves": halves}


# ---------------------------------------------------------------------------
# Train / sample utilities
# ---------------------------------------------------------------------------
def build_train_tensors(train_half, gene_names, uni_model, uni_preprocess,
                        conch_model, conch_preprocess, device, cfg):
    """Load train half expression and image conditioning."""
    cache_root = Path(cfg.get("cache_root", "./cache"))
    use_pc = cfg.get("use_processed_cell", False)

    if use_pc:
        cache_path = cache_root / train_half["dir"] / f"{train_half['label']}_he.npy"
        cond = load_cached_he_features(train_half["adata"], cache_path=cache_path, feature_key="he")
    else:
        cache_path = cache_root / train_half["dir"] / f"{train_half['label']}_uni_conch_cls.npy"
        cond = load_or_encode_features(
            Path(cfg["processed_root"]) / train_half["dir"],
            cache_path,
            uni_model, uni_preprocess,
            conch_model, conch_preprocess,
            device,
            uni_batch=int(cfg.get("uni_batch_size", 64)),
            conch_batch=int(cfg.get("conch_batch_size", 32)),
            cell_idx=train_half.get("cell_idx"),
        )
    expr, cond = preprocess_half(
        train_half["adata"], cond, gene_names,
        min_counts=int(cfg.get("min_counts", 10)),
    )

    # Drop rows that are all-zero across the selected gene list
    keep = expr.sum(dim=1) > 0
    expr = expr[keep]
    cond = cond[keep]
    return expr, cond


def train_stem(model, diffusion, X, C, device, args, logger):
    """Step-based training, no DDP, single GPU."""
    ema = deepcopy(model).to(device)
    freeze(ema)
    update_ema(ema, model, decay=0.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=0.0)

    dataset = TensorDataset(X, C)
    loader = DataLoader(dataset, batch_size=args['batch_size'],
                        shuffle=True, num_workers=0, pin_memory=True,
                        drop_last=True)
    n_steps = int(args['train_steps'])
    log_every = int(args.get('log_every', 500))
    ckpt_every = int(args.get('ckpt_every', 0))
    augment_p = float(args.get('augment_p', 0.0))

    # Save checkpoint dir
    checkpoint_dir = args.get('checkpoint_dir')
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    model.train()
    step = 0
    running = 0.0
    last_log = time.time()
    while step < n_steps:
        for x, y_cond in loader:
            if step >= n_steps:
                break
            x = x.unsqueeze(1).to(device, non_blocking=True)
            y_cond = y_cond.to(device, non_blocking=True)

            if augment_p > 0:
                y_cond = y_cond + augment_p * torch.randn_like(y_cond)

            t = torch.randint(0, diffusion.num_timesteps, (x.size(0),), device=device)
            loss_dict = diffusion.training_losses(model, x, t, dict(y=y_cond))
            loss = loss_dict['loss'].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema(ema, model)
            running += float(loss.detach())
            step += 1

            if step % log_every == 0:
                avg = running / log_every
                running = 0.0
                dt = time.time() - last_log
                last_log = time.time()
                logger(f"step {step}/{n_steps}  loss={avg:.4f}  {log_every/max(dt,1e-6):.1f} it/s")

            if checkpoint_dir is not None and ckpt_every > 0 and step % ckpt_every == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"{step:07d}.pt")
                torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": optimizer.state_dict()}, ckpt_path)
                logger(f"saved checkpoint {ckpt_path}")

    model.eval()
    ema.eval()
    return ema


@torch.no_grad()
def sample_stem(ema, diffusion_sampler, cond, gene_size, device,
                K=10, batch_size=128):
    """Sample K trajectories per cell, return mean log2(x+1) prediction."""
    ema.eval()
    n = cond.shape[0]
    accum = torch.zeros(n, gene_size, dtype=torch.float32)
    for _ in range(K):
        out = torch.zeros(n, gene_size, dtype=torch.float32)
        for i in range(0, n, batch_size):
            y_batch = cond[i:i + batch_size].to(device)
            z = torch.randn(y_batch.shape[0], 1, gene_size, device=device)
            samples = diffusion_sampler.p_sample_loop(
                ema.forward, z.shape, z,
                clip_denoised=False,
                model_kwargs=dict(y=y_batch),
                progress=False, device=device,
            )
            out[i:i + batch_size] = samples.squeeze(1).cpu().float()
        accum += out
    return (accum / K).numpy()


# ---------------------------------------------------------------------------
# Per (dataset, fold) entry point
# ---------------------------------------------------------------------------
def run_one_fold(ds_name, fold_idx, half1, half2, cfg, output_root, gene_names,
                 uni_model, uni_preprocess, conch_model, conch_preprocess,
                 device, config_path=None, seed=42):
    train_half, test_half = (half1, half2) if fold_idx == 0 else (half2, half1)
    use_pc = cfg.get("use_processed_cell", False)
    run_name = f"{ds_name}_fold{fold_idx}_seed{seed}"
    run_dir = os.path.join(output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2, default=str)
    if config_path is not None and os.path.exists(config_path):
        shutil.copy(config_path, os.path.join(run_dir, 'config.yaml'))

    print(f"\n{'='*60}")
    print(f"[Stem] Dataset: {ds_name} | Fold: {fold_idx} | train={train_half['label']} test={test_half['label']}")
    print(f"{'='*60}")

    cache_root = Path(cfg.get("cache_root", "./cache"))
    uni_batch = int(cfg.get('uni_batch_size', 64))
    conch_batch = int(cfg.get('conch_batch_size', 32))

    print('Caching/loading train embeddings ...')
    X, C = build_train_tensors(
        train_half, gene_names,
        uni_model, uni_preprocess, conch_model, conch_preprocess,
        device, cfg,
    )
    print(f'  X (log2 expr): {tuple(X.shape)}  C (cond): {tuple(C.shape)}')

    gene_size = X.shape[1]
    cond_size = C.shape[1]
    model = Stem_models[cfg.get('model', 'Stem')](
        input_size=gene_size,
        depth=int(cfg['DiT_num_blocks']),
        hidden_size=int(cfg['hidden_size']),
        num_heads=int(cfg['num_heads']),
        label_size=cond_size,
    ).to(device)
    diffusion = create_diffusion(timestep_respacing="")
    n_train_params = sum(p.numel() for p in model.parameters())
    print(f'  Stem DiT params: {n_train_params:,}')

    def log_fn(msg):
        print(msg)

    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    train_args = {
        'lr': float(cfg.get('lr', 1e-4)),
        'batch_size': int(cfg.get('batch_size', 256)),
        'train_steps': int(cfg.get('train_steps', 30000)),
        'log_every': int(cfg.get('log_every', 500)),
        'augment_p': float(cfg.get('augment_p', 0.0)),
        'ckpt_every': int(cfg.get('ckpt_every', 0)),
        'checkpoint_dir': checkpoint_dir,
    }
    print(f'  training {train_args["train_steps"]} steps, batch={train_args["batch_size"]}, lr={train_args["lr"]}')
    ema = train_stem(model, diffusion, X, C, device, train_args, log_fn)

    # Save final EMA checkpoint
    final_ckpt = os.path.join(checkpoint_dir, 'final.pt')
    torch.save({"ema": ema.state_dict()}, final_ckpt)

    # Sampling on test half
    print('Caching/loading test embeddings ...')
    if use_pc:
        test_cache = cache_root / test_half["dir"] / f"{test_half['label']}_he.npy"
        cond_test = load_cached_he_features(test_half["adata"], cache_path=test_cache, feature_key="he")
    else:
        test_cache = cache_root / test_half["dir"] / f"{test_half['label']}_uni_conch_cls.npy"
        cond_test = load_or_encode_features(
            Path(cfg["processed_root"]) / test_half["dir"],
            test_cache,
            uni_model, uni_preprocess, conch_model, conch_preprocess,
            device, uni_batch, conch_batch,
            cell_idx=test_half.get("cell_idx"),
        )
    expr_test, cond_test = preprocess_half(
        test_half["adata"], cond_test, gene_names,
        min_counts=int(cfg.get("min_counts", 10)),
    )
    expr_test_np = expr_test.numpy()

    samp_steps = int(cfg.get('num_sampling_steps', 250))
    diffusion_sampler = create_diffusion(timestep_respacing=str(samp_steps))
    K = int(cfg.get('sample_num_per_cond', 10))
    samp_batch = int(cfg.get('sampling_batch_size', 128))

    print(f'  sampling {test_half["label"]}: cells={cond_test.shape[0]}  K={K}  steps={samp_steps}')
    pred = sample_stem(ema, diffusion_sampler, cond_test, gene_size, device,
                        K=K, batch_size=samp_batch)
    per_gene_pcc, mean_pcc = compute_pcc(pred, expr_test_np)
    print(f'    test PCC = {mean_pcc:.4f}')

    np.save(os.path.join(run_dir, 'pred.npy'), pred)
    np.save(os.path.join(run_dir, 'pcc_per_gene.npy'), np.asarray(per_gene_pcc))

    result = {
        'dataset': ds_name,
        'fold': fold_idx,
        'seed': seed,
        'train_half': train_half['label'],
        'test_half': test_half['label'],
        'n_genes': len(gene_names),
        'gene_names': gene_names,
        'test_pcc': float(mean_pcc),
        'stem_info': {
            'DiT_num_blocks': int(cfg['DiT_num_blocks']),
            'hidden_size': int(cfg['hidden_size']),
            'num_heads': int(cfg['num_heads']),
            'cond_size': int(cond_size),
            'train_steps': int(cfg.get('train_steps', 30000)),
            'sample_num_per_cond': int(K),
            'num_sampling_steps': int(samp_steps),
        },
        'protocol': 'stem_diffusion',
        'status': 'completed',
    }
    with open(os.path.join(run_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--fold', type=int, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    output_root = cfg['output_root']
    os.makedirs(output_root, exist_ok=True)
    device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    use_pc = cfg.get('use_processed_cell', False)
    print(f"Loading UNI from {cfg['uni_local_path']}")
    uni_model, uni_preprocess = load_uni(cfg['uni_local_path'], device)
    if use_pc:
        print("Using precomputed UNI features from processed_cell/ (CONCH disabled)")
        conch_model, conch_preprocess = None, None
    else:
        print(f"Loading CONCH from {cfg['conch_local_path']}")
        conch_model, conch_preprocess = load_conch(cfg['conch_local_path'], device)

    seed = int(cfg.get('seeds', [42])[0])
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    all_results = []
    for ds_name in cfg['datasets']:
        if args.dataset is not None and ds_name != args.dataset:
            continue

        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        ds_info = load_dataset(ds_name, cfg)
        half1, half2 = ds_info['halves']

        gene_names = select_top_hvgs_official(
            [half1['adata'], half2['adata']], n_top=int(cfg.get('n_top_hvgs', 50))
        )
        print(f"  selected {len(gene_names)} HVGs")

        ds_dir = os.path.join(output_root, ds_name)
        os.makedirs(ds_dir, exist_ok=True)
        existing = [d for d in os.listdir(ds_dir)
                    if d.startswith('v') and d[1:].isdigit()
                    and os.path.isdir(os.path.join(ds_dir, d))]
        version = f'v{max(int(d[1:]) for d in existing) + 1}' if existing else 'v1'
        version_dir = os.path.join(ds_dir, version)
        os.makedirs(version_dir, exist_ok=True)
        shutil.copy(args.config, os.path.join(version_dir, 'config.yaml'))
        print(f"  saving runs to {version_dir}/")

        for fold in range(2):
            if args.fold is not None and fold != args.fold:
                continue
            try:
                result = run_one_fold(
                    ds_name, fold, half1, half2, cfg, version_dir,
                    gene_names, uni_model, uni_preprocess,
                    conch_model, conch_preprocess, device,
                    config_path=args.config, seed=seed,
                )
                all_results.append(result)
            except Exception as e:
                print(f"ERROR in {ds_name} fold={fold}: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    'dataset': ds_name, 'fold': fold, 'seed': seed,
                    'error': str(e),
                })

        fold_pccs = [r['test_pcc'] for r in all_results
                     if r.get('dataset') == ds_name and 'error' not in r]
        if fold_pccs:
            summary = {
                'dataset': ds_name,
                'version': version,
                'mean': float(np.mean(fold_pccs)),
                'std': float(np.std(fold_pccs, ddof=1)) if len(fold_pccs) > 1 else 0.0,
                'n': len(fold_pccs),
                'values': fold_pccs,
                'protocol': 'stem_diffusion',
            }
            with open(os.path.join(version_dir, 'summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"  dataset summary: {summary['mean']:.4f} ± {summary['std']:.4f}")

    # Global summary
    by_ds = {}
    for r in all_results:
        if 'error' in r:
            continue
        by_ds.setdefault(r['dataset'], []).append(r['test_pcc'])

    global_summary = {
        'all_results': all_results,
        'dataset_summary': {
            ds: {
                'mean': float(np.mean(pccs)),
                'std': float(np.std(pccs, ddof=1)) if len(pccs) > 1 else 0.0,
                'n': len(pccs),
                'values': pccs,
            } for ds, pccs in by_ds.items()
        },
        'protocol': 'stem_diffusion',
    }
    with open(os.path.join(output_root, 'summary.json'), 'w') as f:
        json.dump(global_summary, f, indent=2)
    print(f"\nGlobal summary saved to {os.path.join(output_root, 'summary.json')}")


if __name__ == '__main__':
    main()
