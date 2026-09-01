"""HST_MERGE cross-validation training script.

Data pipeline (load_slide, HVG selection, CV split, evaluation) is identical
to HST-middle/h0mini_official_offline. Model and training logic preserves the
original MERGE design:

  Stage 1: CNN (ResnetMLP + CNN_Predictor) trained with MSE loss.
  Stage 2: GNN (GATNet) refines CNN embeddings on a hierarchical graph.

Both stages run for each (task, fold, seed) combination.
"""

import os
import argparse
import json
import random
import shutil
import time
import warnings

_N_THREADS = os.environ.get('HST_NUM_THREADS', '4')
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ.setdefault(_v, _N_THREADS)

import numpy as np
import scanpy as sc
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from pathlib import Path

from model import CNN_Predictor, GATNet
from graph import graph_construction
from utils import (
    load_slide, select_top_hvgs_official, extract_gene_expr,
    compute_pcc, augment_images, set_offline_cache,
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


# ---------------------------------------------------------------------------
# Dataset for CNN stage
# ---------------------------------------------------------------------------

class SpotDataset(Dataset):
    """Spot-level dataset: (patch, gene_expr) pairs."""
    def __init__(self, patches, gene_expr):
        self.patches = patches
        self.gene_expr = gene_expr

    def __len__(self):
        return self.patches.shape[0]

    def __getitem__(self, idx):
        return self.patches[idx], self.gene_expr[idx]


# ---------------------------------------------------------------------------
# CNN stage
# ---------------------------------------------------------------------------

def cnn_train_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    n_samples = 0
    for patches, labels in dataloader:
        patches = patches.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(patches)
        loss = F.mse_loss(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * patches.size(0)
        n_samples += patches.size(0)

    return running_loss / n_samples if n_samples > 0 else 0.0


def cnn_evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for patches, labels in dataloader:
            patches = patches.to(device)
            labels = labels.to(device)
            outputs = model(patches)
            all_preds.append(outputs.cpu())
            all_targets.append(labels.cpu())

    if not all_preds:
        return float('nan'), float('nan'), -float('inf')

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    mse = F.mse_loss(preds, targets).item()
    mae = F.l1_loss(preds, targets).item()
    _, mean_pcc = compute_pcc(preds, targets)
    return mse, mae, mean_pcc


def train_cnn(train_patches, train_expr, val_patches, val_expr,
              num_genes, cfg, device):
    """Train CNN_Predictor. Returns trained model and validation metrics."""
    batch_size = cfg.get('cnn_batch_size', 8)
    epochs = cfg.get('cnn_epochs', 15)
    lr = cfg.get('cnn_lr', 5e-5)
    dropout = cfg.get('cnn_dropout', 0.2)
    pretrained_path = cfg.get('cnn_pretrained_path', None)

    train_ds = SpotDataset(train_patches, train_expr)
    val_ds = SpotDataset(val_patches, val_expr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = CNN_Predictor(
        num_genes=num_genes, device=device,
        dropout=dropout, pretrained_path=pretrained_path
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=cfg.get('cnn_step_size', 2), gamma=cfg.get('cnn_gamma', 0.5)
    )

    best_val_pcc = -float('inf')
    best_state = None
    has_val = len(val_loader.dataset) > 0

    for epoch in range(1, epochs + 1):
        train_loss = cnn_train_epoch(model, train_loader, optimizer, device)
        if has_val:
            val_mse, val_mae, val_pcc = cnn_evaluate(model, val_loader, device)
            if val_pcc > best_val_pcc:
                best_val_pcc = val_pcc
                best_state = model.state_dict().copy()
        else:
            val_mse, val_mae, val_pcc = float('nan'), float('nan'), float('nan')
            best_state = model.state_dict().copy()
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  [CNN] Epoch {epoch}/{epochs}  "
                  f"train_loss={train_loss:.4f}  val_mse={val_mse:.4f}  "
                  f"val_mae={val_mae:.4f}  val_pcc={val_pcc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def extract_cnn_features(model, patches, device, batch_size=128):
    """Extract 256-dim features by removing the last FC layer."""
    model.eval()
    # Remove last FC layer: keep up to dropout
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor = feature_extractor.to(device)

    features = []
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch = patches[i:i + batch_size].to(device)
            feat = feature_extractor(batch)
            features.append(feat.cpu())
    return torch.cat(features, dim=0)


# ---------------------------------------------------------------------------
# GNN stage
# ---------------------------------------------------------------------------

def gnn_train_epoch(gnn, dataloader, optimizer, alpha=0):
    gnn.train()
    train_mse, train_corr = [], []

    for batch in dataloader:
        slide_index, edge_indices, labels, patch_embeddings = batch
        labels = labels.squeeze()
        edge_indices = edge_indices.squeeze()
        patch_embeddings = patch_embeddings.squeeze()

        output = gnn(patch_embeddings, edge_indices)
        output = output.view_as(labels)

        mse = F.mse_loss(output, labels)

        # Per-gene Pearson correlation
        output_t = output.T
        labels_t = labels.T
        corr = []
        for g in range(labels_t.shape[0]):
            c = pearsonr(output_t[g].cpu().detach().numpy(),
                         labels_t[g].cpu().detach().numpy())[0]
            corr.append(0.0 if np.isnan(c) else float(c))
        corr = torch.tensor(corr).mean()

        loss = mse + alpha * (1 - corr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_mse.append(mse.item())
        train_corr.append(corr.item())

    return np.mean(train_mse), np.mean(train_corr)


def gnn_evaluate(gnn, dataloader, num_genes):
    gnn.eval()
    test_mse, test_mae, test_corr = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            slide_index, edge_indices, labels, patch_embeddings = batch
            labels = labels.squeeze()
            edge_indices = edge_indices.squeeze()
            patch_embeddings = patch_embeddings.squeeze()

            output = gnn(patch_embeddings, edge_indices)
            output = output.view_as(labels)

            mse = F.mse_loss(output, labels)
            mae = F.l1_loss(output, labels)

            output_t = output.T
            labels_t = labels.T
            corr = []
            for g in range(num_genes):
                c = pearsonr(output_t[g].cpu().detach().numpy(),
                             labels_t[g].cpu().detach().numpy())[0]
                corr.append(0.0 if np.isnan(c) else float(c))
            corr = np.mean(corr)

            test_mse.append(mse.item())
            test_mae.append(mae.item())
            test_corr.append(corr)

    return np.mean(test_mse), np.mean(test_mae), np.mean(test_corr)


def train_gnn(graph_dataset, num_genes, cfg, device):
    """Train GATNet on graph data. Returns trained model."""
    epochs = cfg.get('gnn_epochs', 400)
    lr = cfg.get('gnn_lr', 0.001)
    num_heads = cfg.get('gnn_attn_heads', 8)
    drop_edge = cfg.get('gnn_drop_edge', 0.2)
    alpha = cfg.get('gnn_alpha', 0.0)
    warmup_steps = cfg.get('gnn_warmup_steps', 10)

    # Split graph dataset into train/val (80/20 by slides)
    n_slides = len(graph_dataset)
    n_train = max(1, int(n_slides * 0.8))
    indices = list(range(n_slides))
    random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    # Build single dataset and dataloader with batch_size=1 (one slide at a time)
    dataloader = DataLoader(graph_dataset, batch_size=1, shuffle=True, num_workers=0)

    gnn = GATNet(num_genes=num_genes, num_heads=num_heads, drop_edge=drop_edge).to(device)
    optimizer = optim.Adam(gnn.parameters(), lr=lr, weight_decay=0.0)

    # Warmup scheduler
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_pcc = -float('inf')
    best_state = None
    patience_counter = 0
    patience = cfg.get('gnn_early_stop_patience', 50)

    for epoch in range(1, epochs + 1):
        train_mse, train_corr = gnn_train_epoch(gnn, dataloader, optimizer, alpha=alpha)
        scheduler.step()

        if epoch % 40 == 0 or epoch == epochs:
            val_mse, val_mae, val_corr = gnn_evaluate(gnn, dataloader, num_genes)
            print(f"  [GNN] Epoch {epoch}/{epochs}  "
                  f"train_mse={train_mse:.4f}  train_corr={train_corr:.4f}  "
                  f"val_mse={val_mse:.4f}  val_mae={val_mae:.4f}  val_corr={val_corr:.4f}")

            if val_corr > best_val_pcc:
                best_val_pcc = val_corr
                best_state = gnn.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience > 0 and patience_counter >= patience:
                print(f"  [GNN] Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        gnn.load_state_dict(best_state)
    return gnn


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_test_slides(cnn_model, gnn_model, test_paths, gene_names, cfg, device):
    """Evaluate on test slides: CNN feature extraction + GNN prediction."""
    cnn_model.eval()
    gnn_model.eval()

    per_slide_pcc = {}
    for patches_path, st_path in test_paths:
        imgs, adata, coords = load_slide(patches_path, st_path)
        gene_expr = extract_gene_expr(adata, gene_names)

        imgs = imgs.to(device)
        gene_expr = gene_expr.to(device)

        with torch.no_grad():
            # CNN feature extraction
            features = extract_cnn_features(cnn_model, imgs, device)
            # Build graph
            graph_cfg = {
                'hierarchical': cfg.get('gnn_hierarchical', True),
                'spatial_clusters': cfg.get('gnn_spatial_clusters', 5),
                'feature_clusters': cfg.get('gnn_feature_clusters', 5),
            }
            graph_ds = graph_construction(
                [coords], [features], [gene_expr], graph_cfg, device=device
            )
            # GNN prediction
            _, edge_idx, labels, embeddings = graph_ds[0]
            pred = gnn_model(embeddings, edge_idx)
            pred = pred.view_as(labels)

        _, pcc = compute_pcc(pred.cpu(), gene_expr.cpu())
        basename = os.path.splitext(os.path.basename(patches_path))[0]
        per_slide_pcc[basename] = float(pcc)

    if not per_slide_pcc:
        return float('nan'), {}
    return float(np.mean(list(per_slide_pcc.values()))), per_slide_pcc


# ---------------------------------------------------------------------------
# Main CV loop
# ---------------------------------------------------------------------------

def train_one_fold(task_name, fold_idx, n_folds, seed, cfg, output_root, gene_names=None,
                   config_path=None):
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

    # Load train slides for HVG selection
    train_adata = []
    for pf, st_path in train_paths:
        _, adata, _ = load_slide(pf, st_path)
        train_adata.append(adata)

    # HVG selection
    if gene_names is not None:
        n_genes = len(gene_names)
        print(f"Using task-level HVGs ({n_genes} genes).")
    else:
        gene_names = select_top_hvgs_official(train_adata, n_top=cfg.get('n_top_hvgs', 50))
        n_genes = len(gene_names)
        print(f"Selected {n_genes} HVGs (official method).")

    # Load all train slide data
    all_train_patches = []
    all_train_expr = []
    all_val_patches = []
    all_val_expr = []
    all_slide_patches = []
    slide_coords = []
    slide_expr = []

    # Simple slide-level split: 80% slides for train, 20% for val
    n_train_slides = max(1, int(len(train_paths) * 0.8))
    train_slide_idx = list(range(len(train_paths)))
    random.shuffle(train_slide_idx)
    train_idx_set = set(train_slide_idx[:n_train_slides])

    for i, (pf, st_path) in enumerate(train_paths):
        imgs, adata, coords = load_slide(pf, st_path)
        gene_expr = extract_gene_expr(adata, gene_names)

        all_slide_patches.append(imgs)
        slide_coords.append(coords)
        slide_expr.append(gene_expr)

        if i in train_idx_set:
            all_train_patches.append(imgs)
            all_train_expr.append(gene_expr)
        else:
            all_val_patches.append(imgs)
            all_val_expr.append(gene_expr)

    # Concatenate patches and expressions
    train_patches = torch.cat(all_train_patches, dim=0) if all_train_patches else torch.empty(0, 3, 224, 224)
    train_expr = torch.cat(all_train_expr, dim=0) if all_train_expr else torch.empty(0, n_genes)
    val_patches = torch.cat(all_val_patches, dim=0) if all_val_patches else torch.empty(0, 3, 224, 224)
    val_expr = torch.cat(all_val_expr, dim=0) if all_val_expr else torch.empty(0, n_genes)

    print(f"\nTrain spots: {train_patches.shape[0]}  Val spots: {val_patches.shape[0]}")

    # ========== Stage 1: CNN ==========
    print(f"\n{'='*60}")
    print("Stage 1: CNN Training")
    print(f"{'='*60}")
    cnn_model = train_cnn(train_patches, train_expr, val_patches, val_expr,
                          n_genes, cfg, device)

    # ========== Extract CNN features ==========
    print(f"\n{'='*60}")
    print("Extracting CNN features")
    print(f"{'='*60}")
    slide_features = []
    for imgs in all_slide_patches:
        features = extract_cnn_features(cnn_model, imgs, device)
        slide_features.append(features)

    # ========== Stage 2: GNN ==========
    print(f"\n{'='*60}")
    print("Stage 2: GNN Training")
    print(f"{'='*60}")
    graph_cfg = {
        'hierarchical': cfg.get('gnn_hierarchical', True),
        'spatial_clusters': cfg.get('gnn_spatial_clusters', 5),
        'feature_clusters': cfg.get('gnn_feature_clusters', 5),
    }
    graph_dataset = graph_construction(
        slide_coords, slide_features, slide_expr, graph_cfg, device=device
    )
    gnn_model = train_gnn(graph_dataset, n_genes, cfg, device)

    # ========== Final Evaluation ==========
    print(f"\n{'='*60}")
    print("Final Test Evaluation")
    print(f"{'='*60}")
    test_pcc, per_slide_pcc = evaluate_test_slides(
        cnn_model, gnn_model, test_paths, gene_names, cfg, device
    )
    print(f"Final Test PCC: {test_pcc:.4f}")
    for sid, pcc in per_slide_pcc.items():
        print(f"  {sid}: {pcc:.4f}")

    # Save models
    ckpt_dir = os.path.join(run_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(cnn_model.cpu().state_dict(), os.path.join(ckpt_dir, 'cnn_best.pt'))
    torch.save(gnn_model.cpu().state_dict(), os.path.join(ckpt_dir, 'gnn_best.pt'))

    result = {
        'task': task_name,
        'fold': fold_idx,
        'n_folds': n_folds,
        'seed': seed,
        'best_test_pcc': float(test_pcc),
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
    args = parser.parse_args()

    import yaml
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
            version = f'v{max_v + 1}'
        else:
            version = 'v1'

        version_dir = os.path.join(task_dir, version)
        os.makedirs(version_dir, exist_ok=True)
        shutil.copy(args.config, os.path.join(version_dir, 'config.yaml'))
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
        task_pccs = [r['best_test_pcc'] for r in all_results
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
        task_results[task].append(r['best_test_pcc'])

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
