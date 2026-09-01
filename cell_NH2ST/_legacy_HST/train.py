"""HST_NH2ST cross-validation training script.

Data pipeline (load_slide, HVG selection, CV split, evaluation) is identical
to HST-middle/h0mini_official_offline. Model and training logic preserves the
original NH2ST design: NGHist2ST with contrastive learning + MSE reconstruction.

Pure PyTorch training loop (no PyTorch Lightning).
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

import numpy as np
import scanpy as sc
import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
import yaml

from models import NGHist2ST
from dataset import HSTNH2STDataset, collate_fn
from utils import (
    load_slide, select_top_hvgs_official, extract_gene_expr,
    compute_pcc, set_offline_cache,
)
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


def train_epoch(model, dataloader, optimizer, device):
    """One training epoch."""
    model.train()
    total_loss = 0.0
    n_samples = 0

    for patches, exps, neighbor_patches, neighbor_exps, positions in dataloader:
        patches = patches.to(device)
        exps = exps.to(device)
        neighbor_patches = neighbor_patches.to(device)
        neighbor_exps = neighbor_exps.to(device)

        optimizer.zero_grad()
        outputs = model(patches, exps, neighbor_patches, neighbor_exps)
        loss = model.compute_loss(outputs, exps)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * patches.size(0)
        n_samples += patches.size(0)

    return total_loss / n_samples if n_samples > 0 else 0.0


def evaluate_epoch(model, dataloader, device):
    """Evaluate on validation/test set."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for patches, exps, neighbor_patches, neighbor_exps, positions in dataloader:
            patches = patches.to(device)
            exps = exps.to(device)
            neighbor_patches = neighbor_patches.to(device)
            neighbor_exps = neighbor_exps.to(device)

            outputs = model(patches, exps, neighbor_patches, neighbor_exps)
            pred = outputs[5]

            all_preds.append(pred.cpu())
            all_targets.append(exps.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    metrics = model.compute_metrics(preds, targets)
    return metrics


def evaluate_test_slides(model, test_paths, gene_names, cfg, device):
    """Evaluate on test slides (full-slide inference)."""
    model.eval()
    per_slide_pcc = {}

    for patches_path, st_path in test_paths:
        imgs, adata, coords = load_slide(patches_path, st_path)
        gene_expr = extract_gene_expr(adata, gene_names)

        ds = HSTNH2STDataset(
            [(patches_path, st_path)], gene_names,
            neighbor_k=cfg.get('neighbor_k', 8),
            dist_thresh_um=cfg.get('dist_thresh_um', 150.0),
            pixel_size_um=cfg.get('pixel_size_um', 0.46),
            mode='test',
        )
        loader = DataLoader(ds, batch_size=cfg.get('batch_size', 32),
                            shuffle=False, num_workers=0, collate_fn=collate_fn)

        all_preds = []
        with torch.no_grad():
            for batch in loader:
                patch, exp, neighbor, neighbor_exp, position = batch
                patch = patch.to(device)
                exp = exp.to(device)
                neighbor = neighbor.to(device)
                neighbor_exp = neighbor_exp.to(device)
                outputs = model(patch, exp, neighbor, neighbor_exp)
                all_preds.append(outputs[5].cpu())

        preds = torch.cat(all_preds, dim=0)
        _, pcc = compute_pcc(preds, gene_expr)
        basename = os.path.splitext(os.path.basename(patches_path))[0]
        per_slide_pcc[basename] = float(pcc)

    if not per_slide_pcc:
        return float('nan'), {}
    return float(np.mean(list(per_slide_pcc.values()))), per_slide_pcc


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

    device = cfg['device']

    # HVG selection
    if gene_names is not None:
        n_genes = len(gene_names)
        print(f"Using task-level HVGs ({n_genes} genes).")
    else:
        train_adata = []
        for pf, st_path in train_paths:
            _, adata, _ = load_slide(pf, st_path)
            train_adata.append(adata)
        gene_names = select_top_hvgs_official(train_adata, n_top=cfg.get('n_top_hvgs', 50))
        n_genes = len(gene_names)
        print(f"Selected {n_genes} HVGs (official method).")

    # Split train slides into train/val
    # FIX: when only 1 slide, use it for both train and val
    if len(train_paths) <= 1:
        train_slide_paths = train_paths
        val_slide_paths = train_paths
    else:
        n_train_slides = max(1, int(len(train_paths) * 0.8))
        indices = list(range(len(train_paths)))
        random.shuffle(indices)
        train_slide_idx = indices[:n_train_slides]
        val_slide_idx = indices[n_train_slides:]
        train_slide_paths = [train_paths[i] for i in train_slide_idx]
        val_slide_paths = [train_paths[i] for i in val_slide_idx]

    # Build datasets
    train_ds = HSTNH2STDataset(
        train_slide_paths, gene_names,
        neighbor_k=cfg.get('neighbor_k', 8),
        dist_thresh_um=cfg.get('dist_thresh_um', 150.0),
        pixel_size_um=cfg.get('pixel_size_um', 0.46),
        mode='train',
        aug_cfg=cfg.get('augmentation', None),
    )
    val_ds = HSTNH2STDataset(
        val_slide_paths, gene_names,
        neighbor_k=cfg.get('neighbor_k', 8),
        dist_thresh_um=cfg.get('dist_thresh_um', 150.0),
        pixel_size_um=cfg.get('pixel_size_um', 0.46),
        mode='test',
    )

    batch_size = cfg.get('batch_size', 32)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)

    print(f"Train spots: {len(train_ds)}  Val spots: {len(val_ds)}")

    # Model
    model = NGHist2ST(
        num_genes=n_genes,
        emb_dim=cfg.get('emb_dim', 512),
        depth1=cfg.get('depth1', 2),
        num_heads1=cfg.get('num_heads1', 8),
        mlp_ratio1=cfg.get('mlp_ratio1', 2.0),
        dropout1=cfg.get('dropout1', 0.1),
        res_neighbor=tuple(cfg.get('res_neighbor', [5, 5])),
        learning_rate=cfg.get('lr', 0.0001),
        temperature1=cfg.get('temperature1', 0.05),
        temperature2=cfg.get('temperature2', 0.05),
        loss_ratio1=cfg.get('loss_ratio1', 1.0),
        loss_ratio2=cfg.get('loss_ratio2', 0.5),
    ).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,}")

    # Optimizer / Scheduler (preserve NH2ST design: Adam + StepLR)
    optimizer = optim.Adam(model.parameters(), lr=cfg.get('lr', 0.0001), weight_decay=cfg.get('wd', 0.0))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.get('step_size', 50), gamma=cfg.get('gamma', 0.9))

    # Training loop
    epochs = cfg.get('epochs', 400)
    eval_interval = cfg.get('eval_interval', 5)
    best_val_pcc = -float('inf')
    best_epoch = 0
    best_state = None
    patience_counter = 0
    early_stop_patience = cfg.get('early_stop_patience', 40)

    writer = SummaryWriter(log_dir=os.path.join(run_dir, 'tb'))

    ckpt_dir = os.path.join(run_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, 'best.pt')
    last_ckpt_path = os.path.join(ckpt_dir, 'last.pt')

    # ---------- Resume from in-progress checkpoint ----------
    start_epoch = 1
    if resume:
        result_path = os.path.join(run_dir, 'result.json')
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                prev_state = json.load(f)
            status = prev_state.get('status')
            if status == 'in_progress':
                if os.path.exists(last_ckpt_path):
                    print(f"\nResuming {run_name} from checkpoint: last.pt")
                    ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
                    model.load_state_dict(ckpt['model'])
                    optimizer.load_state_dict(ckpt['optimizer'])
                    if 'scheduler' in ckpt:
                        scheduler.load_state_dict(ckpt['scheduler'])
                    start_epoch = int(ckpt['epoch']) + 1
                    best_val_pcc = float(ckpt.get('best_val_pcc', best_val_pcc))
                    best_epoch = int(ckpt.get('best_epoch', 0))
                    patience_counter = int(ckpt.get('patience_counter', 0))
                    best_state = ckpt.get('best_state', None)
                    print(f"  Resuming at epoch {start_epoch}/{epochs} | "
                          f"best_epoch={best_epoch} | patience_counter={patience_counter}")
                else:
                    print(f"result.json says in_progress but no last.pt found in {ckpt_dir}; "
                          f"starting from scratch.")
            elif status == 'completed':
                print(f"\nFold {run_name} already completed; skipping.")
                return prev_state
            else:
                print(f"Fold {run_name} status is '{status}'; starting from scratch.")

    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        do_eval = (epoch % eval_interval == 0) or (epoch == epochs)
        if do_eval:
            train_metrics = evaluate_epoch(model, train_loader, device)
            val_metrics = evaluate_epoch(model, val_loader, device)

            writer.add_scalar('PCC/train', train_metrics['mean_pcc'], epoch)
            writer.add_scalar('PCC/val', val_metrics['mean_pcc'], epoch)
            writer.add_scalar('MSE/val', val_metrics['mse'], epoch)

            print(f"Epoch {epoch}/{epochs} | "
                  f"TrLoss: {train_loss:.4f} | "
                  f"TrPCC: {train_metrics['mean_pcc']:.4f} | "
                  f"ValPCC: {val_metrics['mean_pcc']:.4f} | "
                  f"ValMSE: {val_metrics['mse']:.4f}")

            if val_metrics['mean_pcc'] > best_val_pcc:
                best_val_pcc = val_metrics['mean_pcc']
                best_epoch = epoch
                patience_counter = 0
                best_state = model.state_dict().copy()
                print(f"  -> New best val PCC: {best_val_pcc:.4f}")
            else:
                patience_counter += 1

            # Save in-progress checkpoint so training can be resumed after interruption
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_pcc': float(best_val_pcc),
                'best_epoch': best_epoch,
                'patience_counter': patience_counter,
                'best_state': best_state,
                'gene_names': gene_names,
                'seed': seed,
            }, last_ckpt_path)

            # Track in-progress status for resume
            progress = {
                'task': task_name,
                'fold': fold_idx,
                'n_folds': n_folds,
                'seed': seed,
                'current_epoch': epoch,
                'best_epoch': best_epoch,
                'best_val_pcc': float(best_val_pcc),
                'status': 'in_progress',
            }
            with open(os.path.join(run_dir, 'result.json'), 'w') as f:
                json.dump(progress, f, indent=2)

            if early_stop_patience > 0 and patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience_counter} epochs)")
                break
        else:
            print(f"Epoch {epoch}/{epochs} | TrLoss: {train_loss:.4f}")

    writer.close()

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nLoaded best model from epoch {best_epoch} (val PCC = {best_val_pcc:.4f})")

    # Final test evaluation
    print(f"\n{'='*60}")
    print("Final Test Evaluation")
    print(f"{'='*60}")
    test_pcc, per_slide_pcc = evaluate_test_slides(model, test_paths, gene_names, cfg, device)
    print(f"Final Test PCC: {test_pcc:.4f}")
    for sid, pcc in per_slide_pcc.items():
        print(f"  {sid}: {pcc:.4f}")

    # Save checkpoint
    torch.save({
        'epoch': best_epoch,
        'model': model.cpu().state_dict(),
        'optimizer': optimizer.state_dict(),
        'gene_names': gene_names,
        'best_val_pcc': float(best_val_pcc),
        'test_pcc': float(test_pcc),
        'seed': seed,
    }, best_ckpt_path)

    result = {
        'task': task_name,
        'fold': fold_idx,
        'n_folds': n_folds,
        'seed': seed,
        'best_epoch': best_epoch,
        'best_val_pcc': float(best_val_pcc),
        'test_pcc': float(test_pcc),
        'per_slide_test_pcc': per_slide_pcc,
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

        # Task-level HVG selection
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
                raw_adata_list, n_top=cfg.get('n_top_hvgs', 50)
            )
            print(f"Task {task_name}: selected {len(task_gene_names)} HVGs from all {len(all_paths)} slides.")

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

        # Per-task summary
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

    # Global summary
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
            line = f"{task_name}: {mean_pcc:.4f} +- {std_pcc:.4f} (n={len(pccs)})"
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
