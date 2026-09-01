import os
import json
import h5py
import numpy as np
import scanpy as sc
import torch
from torch_geometric.data import HeteroData
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Offline cache produced by preprocess_offline.py. Layout:
#   <root>/<task_folder>/slides/<sample>.pt    {'imgs': fp16/fp32 [N,3,224,224]}
#   <root>/<task_folder>/adata/<sample>.h5ad   log1p'd, reordered
# Enabled by calling set_offline_cache(root) before training.
_OFFLINE_CACHE_ROOT = None


def set_offline_cache(root):
    """Enable / disable the offline preprocessed cache.

    Pass the directory produced by preprocess_offline.py (e.g.
    ``/home/sb202604/hest-bench-pretreat``) to enable. Pass None to disable.
    When enabled, load_slide() consults the cache first and falls back to
    the online uint8 -> float + log1p path on miss.
    """
    global _OFFLINE_CACHE_ROOT
    _OFFLINE_CACHE_ROOT = root
    if root is not None:
        print(f"[offline cache] enabled at: {root}")
    else:
        print("[offline cache] disabled")


def _try_load_cache(patches_h5_path):
    """Try loading a slide from the offline cache.

    The cache layout mirrors hest-bench: paths shaped like
    ``/.../hest-bench/<TASK>/patches/<SAMPLE>.h5`` map to
    ``<cache_root>/<TASK>/slides/<SAMPLE>.pt`` and
    ``<cache_root>/<TASK>/adata/<SAMPLE>.h5ad``.

    Returns (imgs, adata, coords) on hit, None on miss.
    """
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

    If set_offline_cache(...) was called, attempts to load from the
    preprocessed cache first; falls back to the online path on miss.
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

    # Match and reorder h5 to adata
    barcode_to_idx = {b: i for i, b in enumerate(adata_barcodes)}
    order = [barcode_to_idx[b] for b in h5_barcodes]
    adata = adata[order]

    # Preprocess images: [N, 224, 224, 3] uint8 -> [N, 3, 224, 224] float
    imgs = torch.from_numpy(h5_imgs).permute(0, 3, 1, 2).float() / 255.0
    imgs = (imgs - IMAGENET_MEAN) / IMAGENET_STD

    # Preprocess genes: official HEST/ST-Flow uses log1p only (no normalize_total)
    sc.pp.log1p(adata)

    # Spatial coords from adata (fullres pixels)
    coords = torch.from_numpy(adata.obsm['spatial']).float()

    return imgs, adata, coords


def _filter_control_probes(var_names):
    """Remove Xenium control probes (NegControlProbe_*, UnassignedCodeword_*)."""
    import pandas as pd
    s = pd.Series(var_names)
    mask = ~(s.str.startswith('NegControlProbe_') | s.str.startswith('UnassignedCodeword_'))
    return s[mask].tolist()


def select_top_hvgs_official(adata_list, n_top=200, min_cells_pct=0.10):
    """Official HEST gene selection: min_cells filter -> common genes -> concat -> log1p -> HVG.

    Matches the logic in hest.utils.get_k_genes used to derive var_50genes.json.
    Operates on raw (un-normalized) AnnData; performs log1p internally.
    """
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
    """Select top n highly variable genes across concatenated adata."""
    adata_concat = sc.concat(adata_list, label='sample', join='outer')
    # Fill NaN from outer join (different gene sets across platforms)
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
    """Select top HVGs using only the TRAIN spots from each slide."""
    adata_train = sc.concat(
        [adata[mask] for adata, mask in zip(adata_list, train_masks)],
        label='sample', join='outer'
    )
    # Fill NaN from outer join
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
    """Extract expression matrix for given genes, returning [N, n_genes]."""
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


def build_spatial_edges(coords, pixel_size_um=0.46, dist_thresh_um=150.0):
    """Build spatial adjacency edges on GPU when available."""
    if not torch.is_tensor(coords):
        coords = torch.from_numpy(np.asarray(coords)).float()
    with torch.no_grad():
        coords_um = coords * pixel_size_um
        if torch.cuda.is_available() and not coords_um.is_cuda:
            coords_um = coords_um.cuda()
        dists = torch.cdist(coords_um, coords_um, p=2)
        mask = (dists < dist_thresh_um) & (dists > 0)
        src, dst = torch.where(mask)
        edge_index = torch.stack([src, dst], dim=0)
    return edge_index


def build_morph_edges(img_embeds, top_k=5, sim_thresh=0.6):
    """Build morphological similarity edges on GPU using matrix ops."""
    if not torch.is_tensor(img_embeds):
        img_embeds = torch.from_numpy(np.asarray(img_embeds)).float()
    with torch.no_grad():
        if torch.cuda.is_available() and not img_embeds.is_cuda:
            img_embeds = img_embeds.cuda()
        embeds_norm = torch.nn.functional.normalize(img_embeds, p=2, dim=1)
        sims = embeds_norm @ embeds_norm.t()
        N = sims.shape[0]
        sims.fill_diagonal_(-float('inf'))
        k = min(top_k, N - 1)
        topk_vals, topk_idx = torch.topk(sims, k=k, dim=1)
        mask = topk_vals > sim_thresh
        src = torch.arange(N, device=sims.device).view(-1, 1).expand_as(mask)[mask]
        dst = topk_idx[mask]
        if len(src) == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=sims.device)
        edge_index = torch.stack([src, dst], dim=0)
    return edge_index


def build_hetero_data(x_img, x_gene, coords, img_embeds):
    """Build HeteroData with three edge types."""
    N = x_img.shape[0]
    data = HeteroData()
    data['image'].x = x_img
    data['gene'].x = x_gene
    device = x_img.device
    corr = torch.arange(N, device=device)
    data['image', 'corresponds_to', 'gene'].edge_index = torch.stack([corr, corr], dim=0)
    data['gene', 'corresponds_to', 'image'].edge_index = torch.stack([corr, corr], dim=0)
    spa = build_spatial_edges(coords)
    if spa.device != device:
        spa = spa.to(device)
    data['image', 'spatially_adjacent', 'image'].edge_index = spa
    data['gene', 'spatially_adjacent', 'gene'].edge_index = spa.clone()
    morph = build_morph_edges(img_embeds)
    if morph.device != device:
        morph = morph.to(device)
    data['image', 'morphologically_similar', 'image'].edge_index = morph
    if morph.numel() > 0:
        data['image', 'morphologically_similar_rev', 'image'].edge_index = morph.flip(0)
    return data


def encode_images_batched(encoder, imgs, batch_size=128):
    """Encode images in smaller batches to avoid OOM."""
    embeddings = []
    for i in range(0, len(imgs), batch_size):
        batch = imgs[i:i + batch_size]
        embeddings.append(encoder(batch))
    return torch.cat(embeddings, dim=0)


def compute_gene_weights_on_spots(adata_list, train_masks, gene_names):
    """Compute per-gene variance weights using only TRAIN spots."""
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
    """Compute per-gene Pearson correlation and mean."""
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
    """Vectorized augmentation on GPU."""
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


def sample_mask(n_spots, mask_ratio, device):
    """Sample a Bernoulli spot-level mask for MGM training."""
    mask = torch.rand(n_spots, device=device) < mask_ratio
    if mask_ratio > 0 and not mask.any():
        idx = torch.randint(0, n_spots, (1,), device=device)
        mask[idx] = True
    return mask


def compute_mask_ratio(epoch, total_epochs, mgm_cfg):
    """Curriculum masking schedule."""
    base_ratio = float(mgm_cfg.get('mask_ratio', 0.5))
    cur_cfg = (mgm_cfg.get('curriculum') or {})
    if not cur_cfg.get('enabled', False):
        return base_ratio
    start = float(cur_cfg.get('start_ratio', base_ratio))
    end = float(cur_cfg.get('end_ratio', 1.0))
    warmup = int(cur_cfg.get('warmup_epochs', 0))
    schedule = cur_cfg.get('schedule', 'linear')
    if epoch <= warmup:
        return start
    progress = (epoch - warmup) / max(1, total_epochs - warmup)
    progress = min(1.0, max(0.0, progress))
    if schedule == 'cosine':
        import math
        return start + (end - start) * (1.0 - math.cos(math.pi * progress)) / 2.0
    return start + (end - start) * progress


# =============================================================================
# Evaluation-only helpers
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
            subset_gene_expr = gene_expr[all_idx_t].to(device)
            subset_coords = coords[all_idx_t].to(device)

            with torch.no_grad():
                subset_img_embeds = encode_images_batched(
                    model.image_encoder, subset_imgs,
                    batch_size=img_batch_size
                )
                data = build_hetero_data(
                    subset_img_embeds, subset_gene_expr,
                    subset_coords, subset_img_embeds
                )
                data = data.to(device)
                pred = model(subset_img_embeds, data)

            pred_interior = pred[:n_interior]
            target_interior = subset_gene_expr[:n_interior]

            for i, orig_idx in enumerate(interior_idx):
                all_pred[orig_idx] = pred_interior[i]
                all_target[orig_idx] = target_interior[i]
                covered[orig_idx] = True

        # Fallback: full-slide inference for any uncovered spots.
        uncovered = [i for i, c in enumerate(covered) if not c]
        if uncovered:
            print(f"  [{basename}] {len(uncovered)} spots uncovered by grid; "
                  f"falling back to full-slide for them")
            imgs_f = imgs.to(device)
            gene_expr_f = gene_expr.to(device)
            coords_f = coords.to(device)
            with torch.no_grad():
                img_embeds_f = encode_images_batched(
                    model.image_encoder, imgs_f,
                    batch_size=img_batch_size
                )
                data_f = build_hetero_data(
                    img_embeds_f, gene_expr_f,
                    coords_f, img_embeds_f
                )
                data_f = data_f.to(device)
                pred_f = model(img_embeds_f, data_f)
            for i in uncovered:
                all_pred[i] = pred_f[i]
                all_target[i] = gene_expr_f[i]
                covered[i] = True

        pred = torch.stack(all_pred)
        target = torch.stack(all_target)
        _, mean_pcc = compute_pcc(pred, target)
        return mean_pcc, pred.detach().cpu(), target.detach().cpu()

    # ---- Full-graph mode ----
    imgs = imgs.to(device)
    gene_expr = gene_expr.to(device)
    coords = coords.to(device)

    with torch.no_grad():
        img_embeds = encode_images_batched(
            model.image_encoder, imgs, batch_size=img_batch_size
        )
        data = build_hetero_data(img_embeds, gene_expr, coords, img_embeds)
        data = data.to(device)
        pred = model(img_embeds, data)

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
            grid_cfg=grid_cfg
        )
        basename = os.path.splitext(os.path.basename(patches_path))[0]
        per_slide_pcc[basename] = float(pcc)
    if not per_slide_pcc:
        return float('nan'), {}
    return float(np.mean(list(per_slide_pcc.values()))), per_slide_pcc


# =============================================================================
# Grid + Halo partitioning functions (for large-slide memory control)
# =============================================================================


def build_spatial_grid(coords, n_spots_per_patch, epoch=0, n_shift_steps=4):
    """Partition spots into spatial grid cells with sliding-window offset.

    Each epoch shifts the grid boundaries by a random offset (deterministic
    given epoch) to implement a sliding-window effect.
    """
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
    """Compute halo (ghost) nodes for a set of interior nodes."""
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
    """Sample a subset of grid patches for training.

    Supports two modes:
    - Random mode (default): randomly sample n_patches_per_slide grids.
    - Cycle mode: sequentially cycle through all grids across epochs.
    """
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
