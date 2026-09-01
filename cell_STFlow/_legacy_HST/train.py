"""HST_STFlow cross-validation training script.

Mirrors HST-middle/h0mini_official_offline/train.py and HST_novae/train.py
for everything around k-fold CV, HVG selection, offline cache, grid+halo
training, best-by-train / best-by-test checkpointing, early stopping, and
per-task/global summary.json output. The only differences are:

  - imports `STFlowModel` instead of `HeteroST` / `NovaeST`
  - removes MGM kwargs (STFlow trains with native flow-matching MSE only)
  - graph construction is internal to STFlow's SpatialTransformer, so we
    don't pass `build_graph_data` / `build_hetero_data` parameters

The patient-aware k-fold CV, offline cache, HVG selection, grid+halo,
augmentation, early stopping, and result aggregation are all unchanged
so this script is a drop-in counterpart to the other two variants — the
only difference between the runs is the gene-prediction model.
"""

import os
import argparse
import json
import random
import shutil
import warnings

_N_THREADS = os.environ.get('HST_NUM_THREADS', '4')
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ.setdefault(_v, _N_THREADS)

import h5py
import numpy as np
import scanpy as sc
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
import yaml

torch.set_num_threads(int(_N_THREADS))
try:
    torch.set_num_interop_threads(int(_N_THREADS))
except RuntimeError:
    pass

from model import STFlowModel
from utils import (
    load_slide, select_top_hvgs_official, select_top_hvgs, select_top_hvgs_on_spots,
    extract_gene_expr, compute_gene_weights_on_spots,
    compute_pcc, augment_images,
    evaluate_slides, set_offline_cache,
)
from train_spotsplit import run_epoch_spotsplit
from benchmark_tasks import (
    BENCHMARK_TASKS, make_fold_split, determine_n_folds,
    get_patient_map, get_slide_paths,
)

warnings.filterwarnings('ignore', message='Received a view of an AnnData')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_spot_split(slide_paths, ratios, seed):
    """All spots on train slides are used for training (no val/test spot split)."""
    splits = []
    for pf, st_path in slide_paths:
        with h5py.File(pf, 'r') as f:
            n = len(f['barcode'])
        splits.append({
            'is_train': torch.ones(n, dtype=torch.bool),
            'is_val':   torch.zeros(n, dtype=torch.bool),
            'is_test':  torch.zeros(n, dtype=torch.bool),
        })
    return splits


def run_epoch(model, slide_specs, gene_names, gene_weights, device,
              optimizer=None, img_batch_size=128, full_graph=False,
              aug_cfg=None, loss_type='mse',
              epoch=None, total_epochs=None):
    """Run one epoch full-slide. If optimizer is None, eval mode.

    STFlow trains with its native MSE flow-matching loss (no aux losses,
    no MGM), and predicts via the Euler sampling loop in `model.predict`.
    """
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    loss_breakdown = {}
    n_slides = len(slide_specs)
    accum = {'train': {'pred': [], 'tgt': []},
             'val':   {'pred': [], 'tgt': []},
             'test':  {'pred': [], 'tgt': []}}

    for patches_path, st_path, split in slide_specs:
        imgs, adata, coords = load_slide(patches_path, st_path)
        gene_expr = extract_gene_expr(adata, gene_names)

        is_train_spot = split['is_train']
        is_val_spot   = split['is_val']
        is_test_spot  = split['is_test']

        imgs = imgs.to(device)
        gene_expr = gene_expr.to(device)
        coords = coords.to(device)
        is_train_spot = is_train_spot.to(device)
        is_val_spot   = is_val_spot.to(device)
        is_test_spot  = is_test_spot.to(device)

        if is_train and aug_cfg is not None and aug_cfg.get('enabled', False):
            imgs = augment_images(
                imgs,
                p_flip_h=aug_cfg.get('p_flip_h', 0.5),
                p_flip_v=aug_cfg.get('p_flip_v', 0.5),
                p_rot90=aug_cfg.get('p_rot90', 0.5),
                brightness=aug_cfg.get('brightness', 0.1),
                contrast=aug_cfg.get('contrast', 0.1),
                saturation=aug_cfg.get('saturation', 0.05),
                hue=aug_cfg.get('hue', 0.02),
            )

        if is_train:
            pred, loss = model.train_step(
                imgs, gene_expr, coords, img_batch_size=img_batch_size,
            )
            if is_train_spot.any():
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
        else:
            pred = model.predict(imgs, coords, img_batch_size=img_batch_size)

        with torch.no_grad():
            for k, mask in [('train', is_train_spot),
                            ('val', is_val_spot),
                            ('test', is_test_spot)]:
                if mask.any():
                    accum[k]['pred'].append(pred[mask].detach().cpu())
                    accum[k]['tgt'].append(gene_expr[mask].detach().cpu())

    avg_loss = total_loss / n_slides if is_train else 0.0

    pccs = {}
    for k in ['train', 'val', 'test']:
        if not accum[k]['pred']:
            pccs[k] = float('nan')
            continue
        pred_all = torch.cat(accum[k]['pred'], dim=0)
        tgt_all = torch.cat(accum[k]['tgt'], dim=0)
        _, mean_pcc = compute_pcc(pred_all, tgt_all)
        pccs[k] = float(mean_pcc)

    return avg_loss, pccs['train'], pccs['val'], pccs['test'], loss_breakdown


def train_one_fold(task_name, fold_idx, n_folds, seed, cfg, output_root, gene_names=None,
                   config_path=None, resume=False):
    set_seed(seed)

    run_name = f"{task_name}_fold{fold_idx}_seed{seed}"
    run_dir = os.path.join(output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2, default=str)

    if config_path is not None and os.path.exists(config_path):
        shutil.copy(config_path, os.path.join(run_dir, 'config.yaml'))

    train_samples, test_samples = make_fold_split(task_name, fold_idx, n_folds, seed=seed)
    train_paths = get_slide_paths(train_samples)
    test_paths = get_slide_paths(test_samples)

    print(f"\n{'='*60}")
    print(f"Task: {task_name} | Fold: {fold_idx}/{n_folds} | Seed: {seed}")
    print(f"Train slides ({len(train_paths)}): {[os.path.basename(p[0]) for p in train_paths]}")
    print(f"Test slides  ({len(test_paths)}): {[os.path.basename(p[0]) for p in test_paths]}")
    print(f"{'='*60}")

    train_adata = []
    for pf, st_path in train_paths:
        _, adata, _ = load_slide(pf, st_path)
        train_adata.append(adata)

    ratios = cfg.get('spot_split_ratios', [0.7, 0.1, 0.2])
    splits = make_spot_split(train_paths, ratios, seed)
    train_masks_np = [s['is_train'].numpy() for s in splits]

    if gene_names is not None:
        n_genes = len(gene_names)
        print(f"Using task-level HVGs ({n_genes} genes) selected from all slides.")
    else:
        gene_names = select_top_hvgs_on_spots(
            train_adata, train_masks_np, n_top=cfg.get('n_top_hvgs', 200)
        )
        n_genes = len(gene_names)
        print(f"Selected {n_genes} HVGs from train spots.")

    gene_weights = compute_gene_weights_on_spots(
        train_adata, train_masks_np, gene_names
    )

    slide_specs = [
        (pf, st_path, splits[i])
        for i, (pf, st_path) in enumerate(train_paths)
    ]

    device = cfg['device']
    model = STFlowModel(
        n_genes=n_genes,
        encoder_out_dim=int(cfg.get('encoder_out_dim', cfg.get('hidden_dim', 512))),
        hidden_dim=int(cfg.get('stflow_hidden_dim', 128)),
        pairwise_hidden_dim=int(cfg.get('stflow_pairwise_hidden_dim', 128)),
        n_layers=int(cfg.get('stflow_n_layers', 4)),
        n_heads=int(cfg.get('stflow_n_heads', 4)),
        dropout=float(cfg.get('stflow_dropout', 0.2)),
        attn_dropout=float(cfg.get('stflow_attn_dropout', 0.2)),
        n_neighbors=int(cfg.get('stflow_n_neighbors', 8)),
        activation=cfg.get('stflow_activation', 'swiglu'),
        n_sample_steps=int(cfg.get('stflow_n_sample_steps', 5)),
        prior_sampler=cfg.get('stflow_prior_sampler', 'zinb'),
        zinb_logits=float(cfg.get('stflow_zinb_logits', 0.1)),
        zinb_total_count=float(cfg.get('stflow_zinb_total_count', 1.0)),
        zinb_zi_logits=float(cfg.get('stflow_zinb_zi_logits', 0.0)),
        encoder_name=cfg.get('encoder_name', 'resnet50'),
        encoder_unfreeze=cfg.get('encoder_unfreeze', False),
        encoder_local_path=cfg.get('encoder_local_path', None),
        encoder_unfreeze_last_n_blocks=cfg.get('encoder_unfreeze_last_n_blocks', None),
    ).to(device)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_train:,} / {n_total:,}")

    encoder_lr = float(cfg.get('encoder_lr', cfg['lr']))
    backbone_ids = {id(p) for p in model.image_encoder.backbone.parameters()}
    backbone_params, other_params = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (backbone_params if id(p) in backbone_ids else other_params).append(p)
    optimizer = AdamW(
        [
            {'params': backbone_params, 'lr': encoder_lr},
            {'params': other_params, 'lr': cfg['lr']},
        ],
        weight_decay=float(cfg['wd']),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg['epochs'],
        eta_min=float(cfg.get('lr_min', 1e-6)),
    )

    best_val_pcc = -float('inf')
    best_epoch = 0
    best_test_pcc = -float('inf')
    best_test_epoch = 0
    patience_counter = 0
    early_stop_patience = cfg.get('early_stop_patience', 20)

    eval_interval = int(cfg.get('eval_interval', 5))
    grid_cfg = cfg.get('grid_training', None)
    if grid_cfg and grid_cfg.get('enabled', False):
        print(f"Grid + Halo training enabled: "
              f"n_spots_per_patch={grid_cfg.get('n_spots_per_patch', 512)}, "
              f"n_patches_per_slide={grid_cfg.get('n_patches_per_slide', -1)}, "
              f"cycle_mode={grid_cfg.get('cycle_mode', False)}")

    writer = SummaryWriter(log_dir=os.path.join(run_dir, 'tb'))

    eval_test_pcc_history = []
    ckpt_dir = os.path.join(run_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)
    best_train_ckpt_path = os.path.join(ckpt_dir, 'ckpt_best_train.pt')
    best_test_ckpt_path = os.path.join(ckpt_dir, 'ckpt_best_test.pt')

    # ---------- Resume from in-progress checkpoint ----------
    start_epoch = 1
    if resume:
        result_path = os.path.join(run_dir, 'result.json')
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                prev_state = json.load(f)
            status = prev_state.get('status')
            if status == 'in_progress':
                ckpt_candidates = []
                for cand in [best_train_ckpt_path, best_test_ckpt_path]:
                    if os.path.exists(cand):
                        ckpt_candidates.append(
                            (cand, torch.load(cand, map_location=device, weights_only=False))
                        )
                if ckpt_candidates:
                    # Prefer the checkpoint saved at the latest epoch.
                    ckpt_path, ckpt = max(ckpt_candidates, key=lambda x: x[1].get('epoch', 0))
                    print(f"\nResuming {run_name} from checkpoint: {os.path.basename(ckpt_path)}")
                    model.load_state_dict(ckpt['model'])
                    optimizer.load_state_dict(ckpt['optimizer'])
                    scheduler.load_state_dict(ckpt['scheduler'])

                    current_epoch = int(prev_state.get('current_epoch', ckpt['epoch']))
                    start_epoch = current_epoch + 1
                    best_val_pcc = float(prev_state.get('best_train_pcc', best_val_pcc))
                    best_epoch = int(prev_state.get('best_train_epoch', 0))
                    if prev_state.get('best_test_pcc') is not None:
                        best_test_pcc = float(prev_state['best_test_pcc'])
                        best_test_epoch = int(prev_state.get('best_test_epoch', 0))
                    eval_test_pcc_history = list(prev_state.get('eval_test_pcc_history', []))
                    patience_counter = max(0, current_epoch - best_epoch)

                    print(f"  Resuming at epoch {start_epoch}/{cfg['epochs']} | "
                          f"best_train_epoch={best_epoch} | best_test_epoch={best_test_epoch} | "
                          f"patience_counter={patience_counter}")
                else:
                    print(f"result.json says in_progress but no checkpoint found in {ckpt_dir}; "
                          f"starting from scratch.")
            elif status == 'completed':
                print(f"\nFold {run_name} already completed; skipping.")
                return prev_state
            else:
                print(f"Fold {run_name} status is '{status}'; starting from scratch.")

    px_um = cfg.get('pixel_size_um', 0.46)
    d_um = cfg.get('dist_thresh_um', 150.0)

    for epoch in range(start_epoch, cfg['epochs'] + 1):
        if grid_cfg and grid_cfg.get('enabled', False):
            cycle_mode = bool(grid_cfg.get('cycle_mode', False))
            train_loss, train_pcc, val_pcc, test_pcc, _ = run_epoch_spotsplit(
                model, slide_specs, gene_names, gene_weights, device, optimizer,
                img_batch_size=cfg['img_batch_size'],
                full_graph=cfg.get('full_graph', False),
                aug_cfg=cfg.get('augmentation', None),
                epoch=epoch,
                total_epochs=cfg['epochs'],
                grid_cfg=grid_cfg,
                pixel_size_um=px_um,
                dist_thresh_um=d_um,
                cycle_mode=cycle_mode,
            )
        else:
            train_loss, train_pcc, val_pcc, test_pcc, _ = run_epoch(
                model, slide_specs, gene_names, gene_weights, device, optimizer,
                img_batch_size=cfg['img_batch_size'],
                full_graph=cfg.get('full_graph', False),
                aug_cfg=cfg.get('augmentation', None),
                epoch=epoch,
                total_epochs=cfg['epochs'],
            )

        do_eval = (epoch % eval_interval == 0) or (epoch == cfg['epochs'])
        if do_eval:
            _, eval_train_pcc, eval_val_pcc, _, _ = run_epoch(
                model, slide_specs, gene_names, gene_weights, device, optimizer=None,
                img_batch_size=cfg['img_batch_size'],
                full_graph=cfg.get('full_graph', False),
                aug_cfg=None,
            )
            grid_cfg_eval = cfg.get('grid_training') if cfg.get('eval_use_grid', True) else None
            eval_test_pcc, eval_per_slide_test_pcc = evaluate_slides(
                model, test_paths, gene_names, device,
                img_batch_size=cfg['img_batch_size'],
                grid_cfg=grid_cfg_eval,
            )
            eval_test_pcc_history.append({
                'epoch': epoch,
                'test_pcc': float(eval_test_pcc),
                'per_slide_test_pcc': {k: float(v) for k, v in eval_per_slide_test_pcc.items()},
            })
        else:
            eval_train_pcc = eval_val_pcc = float('nan')
            eval_test_pcc = float('nan')

        train_metric = eval_train_pcc if not np.isnan(eval_train_pcc) else train_pcc
        scheduler.step()

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('PCC/train_epoch', train_pcc, epoch)
        writer.add_scalar('LR/backbone', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('LR/other', optimizer.param_groups[1]['lr'], epoch)
        if not np.isnan(eval_train_pcc):
            writer.add_scalar('PCC/train_eval', eval_train_pcc, epoch)
            writer.add_scalar('PCC/test_eval', eval_test_pcc, epoch)

        print(f"Epoch {epoch}/{cfg['epochs']} | "
              f"TrLoss: {train_loss:.4f} | TrPCC: {train_pcc:.4f} | "
              f"EvalTrain: {eval_train_pcc:.4f} | "
              f"EvalTest: {eval_test_pcc:.4f}")

        new_best_train = (not np.isnan(train_metric) and train_metric > best_val_pcc)
        new_best_test = (not np.isnan(eval_test_pcc) and eval_test_pcc > best_test_pcc)

        if new_best_train:
            best_val_pcc = float(train_metric)
            best_epoch = epoch
            patience_counter = 0
            print(f"  -> New best train PCC: {best_val_pcc:.4f}")
        else:
            patience_counter += 1

        if new_best_test:
            best_test_pcc = float(eval_test_pcc)
            best_test_epoch = epoch
            print(f"  -> New best test PCC: {best_test_pcc:.4f}")

        def _save_ckpt(path):
            tmp = path + '.tmp'
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'gene_names': gene_names,
                'gene_weights': gene_weights,
                'train_pcc': float(train_metric) if not np.isnan(train_metric) else None,
                'test_pcc': float(eval_test_pcc) if not np.isnan(eval_test_pcc) else None,
                'seed': seed,
            }, tmp)
            os.replace(tmp, path)

        if do_eval and new_best_train:
            _save_ckpt(best_train_ckpt_path)
        if do_eval and new_best_test:
            _save_ckpt(best_test_ckpt_path)

        if do_eval:
            progress = {
                'task': task_name,
                'fold': fold_idx,
                'n_folds': n_folds,
                'seed': seed,
                'current_epoch': epoch,
                'best_epoch': best_epoch,
                'best_train_epoch': best_epoch,
                'best_test_epoch': best_test_epoch,
                'best_train_pcc': float(best_val_pcc),
                'best_test_pcc': float(best_test_pcc) if best_test_epoch > 0 else None,
                'eval_test_pcc_history': eval_test_pcc_history,
                'n_train_slides': len(train_paths),
                'n_test_slides': len(test_paths),
                'n_genes': n_genes,
                'gene_names': gene_names,
                'status': 'in_progress',
            }
            with open(os.path.join(run_dir, 'result.json'), 'w') as f:
                json.dump(progress, f, indent=2)

        if early_stop_patience > 0 and patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience_counter} epochs)")
            break

    writer.close()

    if os.path.exists(best_test_ckpt_path) and best_test_epoch > 0:
        print(f"\n{'='*60}")
        print(f"Loading best-by-test checkpoint")
        print(f"  Epoch {best_test_epoch}: Test PCC = {best_test_pcc:.4f}")
        print(f"  (Train best was epoch {best_epoch}: Train PCC = {best_val_pcc:.4f})")
        print(f"{'='*60}")
        best_ckpt = torch.load(best_test_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt['model'])
        has_best_test = True
    else:
        print("\nWarning: No best-by-test checkpoint found.")
        has_best_test = False

    grid_cfg_eval = cfg.get('grid_training') if cfg.get('eval_use_grid', True) else None
    test_pcc, per_slide_pcc = evaluate_slides(
        model, test_paths, gene_names, device,
        img_batch_size=cfg['img_batch_size'],
        grid_cfg=grid_cfg_eval,
    )
    print(f"Final Test PCC: {test_pcc:.4f}")
    for sid, pcc in per_slide_pcc.items():
        print(f"  {sid}: {pcc:.4f}")

    result = {
        'task': task_name,
        'fold': fold_idx,
        'n_folds': n_folds,
        'seed': seed,
        'best_epoch': best_test_epoch if has_best_test else best_epoch,
        'best_train_epoch': best_epoch,
        'best_test_epoch': best_test_epoch,
        'best_train_pcc': float(best_val_pcc),
        'best_test_pcc_selected': float(best_test_pcc) if has_best_test else None,
        'test_pcc': float(test_pcc),
        'per_slide_test_pcc': per_slide_pcc,
        'eval_test_pcc_history': eval_test_pcc_history,
        'n_train_slides': len(train_paths),
        'n_test_slides': len(test_paths),
        'n_genes': n_genes,
        'gene_names': gene_names,
        'status': 'completed',
    }
    with open(os.path.join(run_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--task', type=str, default=None,
                        help='Run only a specific task (e.g. Task1_IDC)')
    parser.add_argument('--fold', type=int, default=None,
                        help='Run only a specific fold index')
    parser.add_argument('--seed', type=int, default=None,
                        help='Run only a specific seed')
    parser.add_argument('--resume', action='store_true',
                        help='Resume in-progress folds from saved checkpoints')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    offline_cache_root = cfg.get('offline_cache_root', None)
    if offline_cache_root:
        set_offline_cache(offline_cache_root)

    seeds = cfg.get('seeds', [42, 43, 44])
    output_root = cfg['output_root']
    os.makedirs(output_root, exist_ok=True)

    all_results = []

    for task_name in BENCHMARK_TASKS:
        if args.task is not None and task_name != args.task:
            continue

        task_dir = os.path.join(output_root, task_name)
        os.makedirs(task_dir, exist_ok=True)

        existing_versions = [
            d for d in os.listdir(task_dir)
            if d.startswith('v') and d[1:].isdigit()
            and os.path.isdir(os.path.join(task_dir, d))
        ]
        if existing_versions:
            max_v = max(int(d[1:]) for d in existing_versions)
            if args.resume:
                version = f'v{max_v}'
            else:
                version = f'v{max_v + 1}'
        else:
            version = 'v1'

        version_dir = os.path.join(task_dir, version)
        os.makedirs(version_dir, exist_ok=True)
        shutil.copy(args.config, os.path.join(version_dir, 'config.yaml'))
        if args.resume:
            print(f"\nTask {task_name}: resuming runs in {version_dir}/")
        else:
            print(f"\nTask {task_name}: saving runs to {version_dir}/")

        gene_list_path = cfg.get('gene_list_path')
        if gene_list_path is not None:
            with open(gene_list_path, 'r') as f:
                task_gene_names = json.load(f)['genes']
            print(f"Task {task_name}: loaded {len(task_gene_names)} genes from {gene_list_path}")
        else:
            all_samples = BENCHMARK_TASKS[task_name]['samples']
            all_paths = get_slide_paths(all_samples)
            raw_adata_list = []
            for pf, st_path in all_paths:
                raw_adata_list.append(sc.read_h5ad(st_path))
            task_gene_names = select_top_hvgs_official(
                raw_adata_list, n_top=cfg.get('n_top_hvgs', 200)
            )
            print(f"Task {task_name}: selected {len(task_gene_names)} HVGs (official method) from all {len(all_paths)} slides.")

        patient_map = get_patient_map(task_name)
        n_folds = determine_n_folds(task_name, patient_map)

        for fold in range(n_folds):
            if args.fold is not None and fold != args.fold:
                continue
            for seed in seeds:
                if args.seed is not None and seed != args.seed:
                    continue
                try:
                    result = train_one_fold(
                        task_name, fold, n_folds, seed, cfg, version_dir,
                        gene_names=task_gene_names,
                        config_path=args.config,
                        resume=args.resume,
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"ERROR in {task_name} fold={fold} seed={seed}: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results.append({
                        'task': task_name,
                        'fold': fold,
                        'seed': seed,
                        'error': str(e),
                    })

        task_pccs = [r['test_pcc'] for r in all_results
                     if r.get('task') == task_name and 'error' not in r]
        if task_pccs:
            mean_pcc = float(np.mean(task_pccs))
            std_pcc = float(np.std(task_pccs, ddof=1)) if len(task_pccs) > 1 else 0.0
            task_summary = {
                'task': task_name,
                'version': version,
                'mean': mean_pcc,
                'std': std_pcc,
                'n': len(task_pccs),
                'values': task_pccs,
            }
            with open(os.path.join(version_dir, 'summary.json'), 'w') as f:
                json.dump(task_summary, f, indent=2)
            print(f"Summary saved to {os.path.join(version_dir, 'summary.json')}")

    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")

    task_results = {}
    for r in all_results:
        if 'error' in r:
            continue
        task = r['task']
        if task not in task_results:
            task_results[task] = []
        task_results[task].append(r['test_pcc'])

    summary_lines = []
    for task_name in BENCHMARK_TASKS:
        pccs = task_results.get(task_name, [])
        if pccs:
            mean_pcc = float(np.mean(pccs))
            std_pcc = float(np.std(pccs, ddof=1)) if len(pccs) > 1 else 0.0
            line = f"{task_name}: {mean_pcc:.4f} ± {std_pcc:.4f} (n={len(pccs)})"
        else:
            line = f"{task_name}: NO RESULTS"
        print(line)
        summary_lines.append(line)

    summary = {
        'all_results': all_results,
        'task_summary': {
            task: {
                'mean': float(np.mean(pccs)) if pccs else None,
                'std': float(np.std(pccs, ddof=1)) if len(pccs) > 1 else 0.0,
                'n': len(pccs),
                'values': pccs,
            }
            for task, pccs in task_results.items()
        },
        'summary_text': summary_lines,
    }
    with open(os.path.join(output_root, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nGlobal summary saved to {os.path.join(output_root, 'summary.json')}")


if __name__ == '__main__':
    main()
