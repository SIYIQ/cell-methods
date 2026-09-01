"""Data utilities for HST_STFlow.

This mirrors HST-middle/h0mini_official_offline/utils.py and HST_novae/utils.py
one-to-one for everything that touches data loading, preprocessing, HVG
selection, augmentation, grid+halo partitioning, and PCC evaluation. The
only differences from HST_novae are:

  - There is no graph-construction step on our side: STFlow's
    `SpatialTransformer` builds its own KNN neighbors from `coords`
    inside the model, so `build_graph_data` / `build_spatial_edges` /
    `build_morph_edges` are dropped.
  - `evaluate_full_slide` / `evaluate_slides` call `model.predict(...)`
    (the STFlow flow-matching sampling loop) instead of a single forward
    pass through a GAT/HGT encoder.

The offline cache layout, ImageNet normalization, log1p adata, HVG
intersection, augmentation, grid+halo, and PCC formulas are kept
bit-for-bit identical so the only meaningful variable between the runs
is the gene-prediction architecture.
"""

import os
import h5py
import numpy as np
import scanpy as sc
import torch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Offline cache produced by preprocess_offline.py. Layout matches HST-middle:
#   <root>/<task_folder>/slides/<sample>.pt    {'imgs': fp16/fp32 [N,3,224,224]}
#   <root>/<task_folder>/adata/<sample>.h5ad   log1p'd, reordered
_OFFLINE_CACHE_ROOT = None


def set_offline_cache(root):
    global _OFFLINE_CACHE_ROOT
    _OFFLINE_CACHE_ROOT = root
    if root is not None:
        print(f"[offline cache] enabled at: {root}")
    else:
        print("[offline cache] disabled")


def _try_load_cache(patches_h5_path):
    if _OFFLINE_CACHE_ROOT is None:
        return None
    parts = os.path.normpath(patches_h5_path).split(os.sep)
    if len(parts) < 3:
        return None
    task_folder = parts[-3]
    sample = os.path.splitext(parts[-1])[0]
    pt_path = os.path.join(_OFFLINE_CACHE_ROOT, task_folder, 'slides', f'{sample}.pt')
    ad_path = os.path.join(_OFFLINE_CACHE_ROOT, task_folder, 'adata', f'{sample}.h5ad')
    if not (os.path.exists(pt_path) and os.path.exists(ad_path)):
        return None
    data = torch.load(pt_path, map_location='cpu', weights_only=False)
    imgs = data['imgs']
    if imgs.dtype != torch.float32:
        imgs = imgs.float()
    adata = sc.read_h5ad(ad_path)
    coords = torch.from_numpy(adata.obsm['spatial']).float()
    return imgs, adata, coords


def load_slide(patches_h5_path, st_h5ad_path):
    """Load patches and gene expression for a single slide, matched by barcode.

    Identical semantics to HST-middle / HST_novae: ImageNet-normalized
    images, log1p'd adata, barcode-aligned. Offline cache used when
    available.
    """
    cached = _try_load_cache(patches_h5_path)
    if cached is not None:
        return cached

    with h5py.File(patches_h5_path, 'r') as f:
        h5_barcodes = [b[0].decode('utf-8') for b in f['barcode'][:]]
        h5_coords = f['coords'][:]
        h5_imgs = f['img'][:]

    adata = sc.read_h5ad(st_h5ad_path)
    adata_barcodes = list(adata.obs_names)
    barcode_to_idx = {b: i for i, b in enumerate(adata_barcodes)}
    order = [barcode_to_idx[b] for b in h5_barcodes]
    adata = adata[order]

    imgs = torch.from_numpy(h5_imgs).permute(0, 3, 1, 2).float() / 255.0
    imgs = (imgs - IMAGENET_MEAN) / IMAGENET_STD
    sc.pp.log1p(adata)
    coords = torch.from_numpy(adata.obsm['spatial']).float()
    return imgs, adata, coords


def _filter_control_probes(var_names):
    import pandas as pd
    s = pd.Series(var_names)
    mask = ~(s.str.startswith('NegControlProbe_') | s.str.startswith('UnassignedCodeword_'))
    return s[mask].tolist()


def select_top_hvgs_official(adata_list, n_top=200, min_cells_pct=0.10):
    """Official HEST gene selection (unchanged from HST-middle / HST_novae)."""
    import pandas as pd

    common_genes = None
    for adata in adata_list:
        my_adata = adata.copy()
        if min_cells_pct:
            sc.pp.filter_genes(my_adata, min_cells=np.ceil(min_cells_pct * len(my_adata.obs)))
        curr_genes = np.array(my_adata.to_df().columns)
        if common_genes is None:
            common_genes = curr_genes
        else:
            common_genes = np.intersect1d(common_genes, curr_genes)

    common_genes = [g for g in common_genes
                    if 'BLANK' not in g and 'Control' not in g
                    and not g.startswith('NegControlProbe_')
                    and not g.startswith('UnassignedCodeword_')]

    stacked = None
    for adata in adata_list:
        df = adata.to_df()[common_genes]
        if stacked is None:
            stacked = df
        else:
            stacked = pd.concat([stacked, df])

    stacked_adata = sc.AnnData(stacked.astype(np.float32))
    sc.pp.filter_genes(stacked_adata, min_cells=0)
    sc.pp.log1p(stacked_adata)
    sc.pp.highly_variable_genes(stacked_adata, n_top_genes=n_top)
    hvg_mask = stacked_adata.var['highly_variable'].values
    gene_names = stacked_adata.var_names[hvg_mask].tolist()[:n_top]
    return gene_names


def select_top_hvgs(adata_list, n_top=200):
    adata_concat = sc.concat(adata_list, label='sample', join='outer')
    if hasattr(adata_concat.X, 'toarray'):
        adata_concat.X = np.nan_to_num(adata_concat.X.toarray(), nan=0.0)
    else:
        adata_concat.X = np.nan_to_num(adata_concat.X, nan=0.0)
    keep_genes = _filter_control_probes(adata_concat.var_names)
    adata_concat = adata_concat[:, keep_genes]
    if n_top is None:
        return adata_concat.var_names.tolist()
    sc.pp.highly_variable_genes(adata_concat, n_top_genes=n_top, flavor='seurat')
    hvg_mask = adata_concat.var['highly_variable'].values
    gene_names = adata_concat.var_names[hvg_mask].tolist()
    return gene_names


def select_top_hvgs_on_spots(adata_list, train_masks, n_top=200):
    adata_train = sc.concat(
        [adata[mask] for adata, mask in zip(adata_list, train_masks)],
        label='sample', join='outer'
    )
    if hasattr(adata_train.X, 'toarray'):
        adata_train.X = np.nan_to_num(adata_train.X.toarray(), nan=0.0)
    else:
        adata_train.X = np.nan_to_num(adata_train.X, nan=0.0)
    import pandas as pd
    vn = pd.Series(adata_train.var_names)
    keep = ~(vn.str.startswith('NegControlProbe_') | vn.str.startswith('UnassignedCodeword_'))
    adata_train = adata_train[:, vn[keep].tolist()]
    if n_top is None:
        return adata_train.var_names.tolist()
    sc.pp.highly_variable_genes(adata_train, n_top_genes=n_top, flavor='seurat')
    hvg_mask = adata_train.var['highly_variable'].values
    return adata_train.var_names[hvg_mask].tolist()


def extract_gene_expr(adata, gene_names):
    common = [g for g in gene_names if g in adata.var_names]
    if len(common) == len(gene_names):
        expr = adata[:, gene_names].X
        if hasattr(expr, 'toarray'):
            expr = expr.toarray()
        return torch.from_numpy(expr).float()
    import pandas as pd
    expr_mat = adata[:, common].X
    if hasattr(expr_mat, 'toarray'):
        expr_mat = expr_mat.toarray()
    df = pd.DataFrame(expr_mat, index=adata.obs_names, columns=common)
    df = df.reindex(columns=gene_names, fill_value=0.0)
    return torch.from_numpy(df.values).float()


def compute_gene_weights_on_spots(adata_list, train_masks, gene_names):
    """Per-gene variance weights kept for parity with HST-middle / HST_novae.

    STFlow's native loss is MSE (no gene-weighting), so train.py does not
    consume `gene_weights` for the loss — we still compute it so the train
    pipeline and checkpoint payloads have the same fields and ordering as
    the comparison runs.
    """
    all_expr = []
    for adata, mask in zip(adata_list, train_masks):
        common = [g for g in gene_names if g in adata.var_names]
        expr = adata[mask][:, common].X
        if hasattr(expr, 'toarray'):
            expr = expr.toarray()
        if len(common) < len(gene_names):
            import pandas as pd
            df = pd.DataFrame(expr, columns=common)
            df = df.reindex(columns=gene_names, fill_value=0.0)
            expr = df.values
        all_expr.append(expr)
    all_expr = np.concatenate(all_expr, axis=0)
    var_per_gene = np.var(all_expr, axis=0)
    mean_var = var_per_gene.mean()
    weights = var_per_gene / mean_var
    weights = np.clip(weights, 0.5, 2.0)
    return torch.from_numpy(weights).float()


def compute_pcc(pred, target):
    pred_np = pred.detach().cpu().numpy() if torch.is_tensor(pred) else pred
    target_np = target.detach().cpu().numpy() if torch.is_tensor(target) else target
    pcc_list = []
    for g in range(pred_np.shape[1]):
        std_pred = np.std(pred_np[:, g])
        std_target = np.std(target_np[:, g])
        if std_pred == 0 or std_target == 0:
            pcc_list.append(0.0)
            continue
        r = np.corrcoef(pred_np[:, g], target_np[:, g])[0, 1]
        pcc_list.append(0.0 if np.isnan(r) else float(r))
    return pcc_list, np.mean(pcc_list)


def augment_images(imgs, p_flip_h=0.5, p_flip_v=0.5, p_rot90=0.5,
                   brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02):
    N = imgs.shape[0]
    aug_imgs = imgs.clone()
    if torch.cuda.is_available() and not aug_imgs.is_cuda:
        aug_imgs = aug_imgs.cuda()
    if torch.rand(1).item() < p_flip_h:
        aug_imgs = torch.flip(aug_imgs, dims=[3])
    if torch.rand(1).item() < p_flip_v:
        aug_imgs = torch.flip(aug_imgs, dims=[2])
    if torch.rand(1).item() < p_rot90:
        k = torch.randint(1, 4, (1,)).item()
        aug_imgs = torch.rot90(aug_imgs, k=k, dims=[2, 3])
    if brightness > 0:
        delta = torch.empty(N, 1, 1, 1, device=aug_imgs.device).uniform_(-brightness, brightness)
        aug_imgs = aug_imgs * (1 + delta)
    if contrast > 0:
        delta = torch.empty(N, 1, 1, 1, device=aug_imgs.device).uniform_(-contrast, contrast)
        mean = aug_imgs.mean(dim=(2, 3), keepdim=True)
        aug_imgs = (aug_imgs - mean) * (1 + delta) + mean
    if saturation > 0:
        delta = torch.empty(N, 1, 1, 1, device=aug_imgs.device).uniform_(-saturation, saturation)
        gray = aug_imgs.mean(dim=1, keepdim=True)
        aug_imgs = (aug_imgs - gray) * (1 + delta) + gray
    aug_imgs = torch.clamp(aug_imgs, -3.0, 3.0)
    return aug_imgs


# =============================================================================
# Evaluation helpers
# =============================================================================

def evaluate_full_slide(model, patches_path, st_path, gene_names, device,
                        img_batch_size=128, pixel_size_um=0.46, dist_thresh_um=150.0,
                        grid_cfg=None):
    """Evaluate a single slide using full-graph or grid-patch inference.

    Returns (mean_pcc, pred_tensor, target_tensor).
    """
    imgs, adata, coords = load_slide(patches_path, st_path)
    gene_expr = extract_gene_expr(adata, gene_names)
    N = gene_expr.shape[0]

    if grid_cfg and grid_cfg.get('enabled', False):
        n_spots_per_patch = int(grid_cfg.get('n_spots_per_patch', 512))
        halo_dist_thresh_um = float(
            grid_cfg.get('halo_dist_thresh_um', dist_thresh_um)
        )

        # Deterministic grid: use fixed epoch=0 for reproducible eval.
        grids = build_spatial_grid(
            coords, n_spots_per_patch, epoch=0, n_shift_steps=1
        )

        all_pred = [None] * N
        all_target = [None] * N
        covered = [False] * N

        basename = os.path.basename(patches_path)
        for gidx, grid in enumerate(grids):
            interior_idx = grid['interior']
            halo_idx = compute_halo_nodes(
                coords, interior_idx,
                pixel_size_um=pixel_size_um,
                dist_thresh_um=halo_dist_thresh_um,
            )
            all_idx = np.concatenate([interior_idx, halo_idx])
            n_interior = len(interior_idx)

            if len(all_idx) == 0:
                continue

            all_idx_t = torch.from_numpy(all_idx)
            subset_imgs = imgs[all_idx_t].to(device)
            subset_coords = coords[all_idx_t].to(device)

            pred = model.predict(
                subset_imgs, subset_coords,
                img_batch_size=img_batch_size
            )

            pred_interior = pred[:n_interior]

            for i, orig_idx in enumerate(interior_idx):
                all_pred[orig_idx] = pred_interior[i]
                all_target[orig_idx] = gene_expr[orig_idx]
                covered[orig_idx] = True

        # Fallback: full-slide inference for any uncovered spots.
        uncovered = [i for i, c in enumerate(covered) if not c]
        if uncovered:
            print(f"  [{basename}] {len(uncovered)} spots uncovered by grid; "
                  f"falling back to full-slide for them")
            imgs_f = imgs.to(device)
            coords_f = coords.to(device)
            pred_f = model.predict(
                imgs_f, coords_f,
                img_batch_size=img_batch_size
            )
            for i in uncovered:
                all_pred[i] = pred_f[i]
                all_target[i] = gene_expr[i]
                covered[i] = True

        pred = torch.stack(all_pred)
        target = torch.stack(all_target)
        _, mean_pcc = compute_pcc(pred, target)
        return mean_pcc, pred.detach().cpu(), target.detach().cpu()

    # ---- Full-graph mode ----
    imgs = imgs.to(device)
    gene_expr = gene_expr.to(device)
    coords = coords.to(device)

    pred = model.predict(imgs, coords, img_batch_size=img_batch_size)
    _, mean_pcc = compute_pcc(pred, gene_expr)
    return mean_pcc, pred.detach().cpu(), gene_expr.detach().cpu()


def evaluate_slides(model, slide_paths, gene_names, device, img_batch_size=128,
                    grid_cfg=None):
    """Evaluate multiple slides and return mean PCC.

    slide_paths: list of (patches_path, st_path).
    grid_cfg: if provided and enabled, use grid-patch inference per slide.
    Returns (mean_pcc_over_slides, dict_of_per_slide_pcc).
    """
    model.eval()
    per_slide_pcc = {}
    for patches_path, st_path in slide_paths:
        pcc, _, _ = evaluate_full_slide(
            model, patches_path, st_path, gene_names, device,
            img_batch_size=img_batch_size,
            grid_cfg=grid_cfg,
        )
        basename = os.path.splitext(os.path.basename(patches_path))[0]
        per_slide_pcc[basename] = float(pcc)
    if not per_slide_pcc:
        return float('nan'), {}
    return float(np.mean(list(per_slide_pcc.values()))), per_slide_pcc


# =============================================================================
# Grid + Halo partitioning (identical to HST-middle / HST_novae)
# =============================================================================


def build_spatial_grid(coords, n_spots_per_patch, epoch=0, n_shift_steps=4):
    coords_np = coords.cpu().numpy() if torch.is_tensor(coords) else np.asarray(coords)
    N = len(coords_np)
    if N <= n_spots_per_patch:
        return [{'interior': np.arange(N), 'bounds': None}]

    x_min, x_max = coords_np[:, 0].min(), coords_np[:, 0].max()
    y_min, y_max = coords_np[:, 1].min(), coords_np[:, 1].max()
    n_grids = max(1, int(np.ceil(N / n_spots_per_patch)))
    x_range = max(x_max - x_min, 1.0)
    y_range = max(y_max - y_min, 1.0)
    aspect = x_range / y_range
    n_y = max(1, int(np.round(np.sqrt(n_grids / aspect))))
    n_x = max(1, int(np.ceil(n_grids / n_y)))
    cell_size_x = x_range / n_x
    cell_size_y = y_range / n_y
    rng = np.random.default_rng(seed=epoch * 7919)
    shift_x = rng.uniform(0, cell_size_x)
    shift_y = rng.uniform(0, cell_size_y)
    x_start = x_min - cell_size_x + shift_x
    x_end = x_max + cell_size_x + shift_x
    y_start = y_min - cell_size_y + shift_y
    y_end = y_max + cell_size_y + shift_y
    x_edges = np.linspace(x_start, x_end, n_x + 3)
    y_edges = np.linspace(y_start, y_end, n_y + 3)
    grids = []
    for i in range(n_x + 2):
        for j in range(n_y + 2):
            mask = (
                (coords_np[:, 0] >= x_edges[i]) & (coords_np[:, 0] < x_edges[i + 1]) &
                (coords_np[:, 1] >= y_edges[j]) & (coords_np[:, 1] < y_edges[j + 1])
            )
            indices = np.where(mask)[0]
            if len(indices) > 0:
                grids.append({
                    'interior': indices,
                    'bounds': (float(x_edges[i]), float(x_edges[i + 1]),
                               float(y_edges[j]), float(y_edges[j + 1])),
                })
    return grids


def compute_halo_nodes(coords, interior_indices, pixel_size_um=0.46,
                       dist_thresh_um=150.0):
    coords_np = coords.cpu().numpy() if torch.is_tensor(coords) else np.asarray(coords)
    N = len(coords_np)
    coords_um = coords_np * pixel_size_um
    interior_coords = coords_um[interior_indices]
    coords_t = torch.from_numpy(coords_um).float()
    interior_t = torch.from_numpy(interior_coords).float()
    with torch.no_grad():
        if torch.cuda.is_available():
            coords_t = coords_t.cuda()
            interior_t = interior_t.cuda()
        dists = torch.cdist(coords_t, interior_t, p=2)
        min_dists = dists.min(dim=1)[0]
        is_interior = torch.zeros(N, dtype=torch.bool, device=min_dists.device)
        is_interior[torch.from_numpy(interior_indices).long().to(min_dists.device)] = True
        halo_mask = (min_dists < dist_thresh_um) & (~is_interior)
        halo_indices = torch.where(halo_mask)[0].cpu().numpy()
    return halo_indices


def sample_grid_patches(grids, n_patches_per_slide, epoch, slide_hash,
                         cycle_mode=False):
    if n_patches_per_slide <= 0 or n_patches_per_slide >= len(grids):
        return grids
    if cycle_mode:
        total = len(grids)
        start = ((epoch - 1) * n_patches_per_slide) % total
        end = start + n_patches_per_slide
        if end <= total:
            indices = list(range(start, end))
        else:
            indices = list(range(start, total)) + list(range(0, end - total))
        return [grids[i] for i in indices]
    rng = np.random.default_rng(seed=epoch * 10000 + slide_hash)
    n = min(n_patches_per_slide, len(grids))
    indices = rng.choice(len(grids), size=n, replace=False)
    return [grids[i] for i in indices]
