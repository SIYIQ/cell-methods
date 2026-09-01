"""Grid + Halo training epoch logic for HST_STFlow.

Mirrors HST-middle/h0mini_official_offline/train_spotsplit.py and
HST_novae/train_spotsplit.py for everything around grid partitioning,
halo nodes, and augmentation. The only differences are:

  - There is no graph-construction call (build_hetero_data /
    build_graph_data): STFlow's SpatialTransformer builds its own KNN
    neighbors from coords inside the model.
  - The loss is whatever STFlow's Denoiser computes natively
    (`STFlowModel.train_step` → MSE flow-matching). The `loss_type` and
    `mgm_cfg` kwargs accepted by the HST-middle / HST_novae versions are
    ignored here so the train.py call signature stays familiar.
"""

import os
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from utils import (
    load_slide, extract_gene_expr,
    compute_pcc, augment_images,
    build_spatial_grid, compute_halo_nodes, sample_grid_patches,
)


def run_epoch_spotsplit(model, slide_specs, gene_names, gene_weights, device,
                        optimizer=None, img_batch_size=128, full_graph=False,
                        aug_cfg=None, loss_type='mse',
                        epoch=None, total_epochs=None, grid_cfg=None,
                        cycle_mode=False,
                        # accepted for call-signature parity with HST_novae /
                        # HST-middle even though STFlow's SpatialTransformer
                        # does its own KNN graph construction inside the model.
                        pixel_size_um=0.46, dist_thresh_um=150.0,
                        morph_top_k=5, morph_sim_thresh=0.6,
                        **_unused):
    """Run one epoch with per-spot train/val/test split and optional grid+halo."""
    is_train_mode = optimizer is not None
    if is_train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    loss_breakdown = {}
    n_slides = len(slide_specs)
    accum = {'train': {'pred': [], 'tgt': []},
             'val':   {'pred': [], 'tgt': []},
             'test':  {'pred': [], 'tgt': []}}

    grid_enabled = (grid_cfg is not None
                    and grid_cfg.get('enabled', False)
                    and is_train_mode)

    for patches_path, st_path, split in slide_specs:
        imgs, adata, coords = load_slide(patches_path, st_path)
        gene_expr = extract_gene_expr(adata, gene_names)

        is_train_spot = split['is_train']
        is_val_spot   = split['is_val']
        is_test_spot  = split['is_test']

        if grid_enabled:
            n_patches_per_slide = int(grid_cfg.get('n_patches_per_slide', -1))
            n_spots_per_patch = int(grid_cfg.get('n_spots_per_patch', 512))
            n_shift_steps = int(grid_cfg.get('n_shift_steps', 4))
            halo_dist_thresh_um = float(
                grid_cfg.get('halo_dist_thresh_um', dist_thresh_um)
            )

            grids = build_spatial_grid(
                coords, n_spots_per_patch,
                epoch=epoch, n_shift_steps=n_shift_steps
            )

            slide_hash = hash(patches_path) % 10000
            sampled_grids = sample_grid_patches(
                grids, n_patches_per_slide, epoch, slide_hash,
                cycle_mode=cycle_mode,
            )

            basename = os.path.basename(patches_path)
            print(f"  [{basename}] {len(grids)} grids total, "
                  f"processing {len(sampled_grids)} this epoch "
                  f"({'cycle' if cycle_mode else 'random'})")

            for gidx, grid in enumerate(sampled_grids):
                interior_idx = grid['interior']
                halo_idx = compute_halo_nodes(
                    coords, interior_idx,
                    pixel_size_um=pixel_size_um,
                    dist_thresh_um=halo_dist_thresh_um,
                )

                print(f"    grid {gidx+1}/{len(sampled_grids)}: "
                      f"interior={len(interior_idx)}, halo={len(halo_idx)}")

                all_idx = np.concatenate([interior_idx, halo_idx])
                n_interior = len(interior_idx)

                all_idx_t = torch.from_numpy(all_idx)
                subset_imgs = imgs[all_idx_t]
                subset_gene_expr = gene_expr[all_idx_t]
                subset_coords = coords[all_idx_t]
                subset_is_train = is_train_spot[all_idx_t]
                interior_is_train = subset_is_train[:n_interior]

                if not interior_is_train.any():
                    continue

                if aug_cfg is not None and aug_cfg.get('enabled', False):
                    subset_imgs = augment_images(
                        subset_imgs,
                        p_flip_h=aug_cfg.get('p_flip_h', 0.5),
                        p_flip_v=aug_cfg.get('p_flip_v', 0.5),
                        p_rot90=aug_cfg.get('p_rot90', 0.5),
                        brightness=aug_cfg.get('brightness', 0.1),
                        contrast=aug_cfg.get('contrast', 0.1),
                        saturation=aug_cfg.get('saturation', 0.05),
                        hue=aug_cfg.get('hue', 0.02),
                    )

                subset_imgs = subset_imgs.to(device)
                subset_gene_expr = subset_gene_expr.to(device)
                subset_coords = subset_coords.to(device)
                interior_is_train = interior_is_train.to(device)

                pred, loss = model.train_step(
                    subset_imgs, subset_gene_expr, subset_coords,
                    img_batch_size=img_batch_size,
                )

                # STFlow's MSE flow-matching loss is computed over all
                # spots in the sample (no mask), so we just backprop it.
                # `interior_is_train` still controls which spots feed the
                # PCC accumulator below — matching HST-middle's grid+halo
                # bookkeeping where only interior train spots count for
                # in-epoch metrics.
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

                pred_interior = pred[:n_interior]
                gene_expr_interior = subset_gene_expr[:n_interior]

                with torch.no_grad():
                    interior_is_val = is_val_spot[all_idx_t[:n_interior]].to(device)
                    interior_is_test = is_test_spot[all_idx_t[:n_interior]].to(device)
                    for k, mask in [('train', interior_is_train),
                                    ('val', interior_is_val),
                                    ('test', interior_is_test)]:
                        if mask.any():
                            accum[k]['pred'].append(
                                pred_interior[mask].detach().cpu()
                            )
                            accum[k]['tgt'].append(
                                gene_expr_interior[mask].detach().cpu()
                            )

            continue

        # ---- Full-slide mode (eval or non-grid training) ----
        if not is_train_mode:
            basename = os.path.basename(patches_path)
            print(f"    [{basename}] full-slide eval ({gene_expr.shape[0]} spots)")

        imgs = imgs.to(device)
        gene_expr = gene_expr.to(device)
        coords = coords.to(device)

        is_train_spot = is_train_spot.to(device)
        is_val_spot   = is_val_spot.to(device)
        is_test_spot  = is_test_spot.to(device)

        if is_train_mode and aug_cfg is not None and aug_cfg.get('enabled', False):
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

        if is_train_mode:
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

    avg_loss = total_loss / n_slides if is_train_mode else 0.0

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
