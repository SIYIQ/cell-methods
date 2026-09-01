"""Offline preprocessing for hest-bench data.

For every slide under SRC (e.g. /home/sb202604/hest-bench/<TASK>/) this script:
  1. Calls utils.load_slide(patches_h5, st_h5ad) — the EXACT same logic used
     in training (uint8 -> float32, permute, ImageNet normalize; sc.pp.log1p
     on adata; barcode reorder).
  2. Saves the (imgs, adata) pair to DST so training only has to do
     torch.load + sc.read_h5ad at runtime — zero CPU preprocessing per epoch.

Gene-selection logic is NOT touched: var_50genes.json / mean_50genes.json /
splits/ are simply copied across, and the runtime HVG selection
(select_top_hvgs_official, etc.) can still be applied to the cached
log1p'd adata if desired — the .h5ad is byte-equivalent to what
load_slide returns.

Layout produced::

    DST/<TASK>/
    ├── slides/<SLIDE>.pt       # {'imgs': fp16/fp32 [N,3,224,224]}
    ├── adata/<SLIDE>.h5ad      # log1p'd, reordered to match h5 barcode order
    ├── var_50genes.json        # copied
    ├── mean_50genes.json       # copied (if exists)
    ├── splits/                 # copied
    └── _meta.json              # script provenance

Usage::

    python preprocess_offline.py \
        --src /home/sb202604/hest-bench \
        --dst /home/sb202604/hest-bench-pretreat \
        --dtype fp16
"""

# IMPORTANT: must set thread limits BEFORE importing numpy / torch / scanpy
# so the BLAS pools come up small.
import os
import argparse


def _parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument('--src', default='/home/sb202604/hest-bench')
    p.add_argument('--dst', default='/home/sb202604/hest-bench-pretreat')
    p.add_argument('--tasks', nargs='*', default=None,
                   help='Subset of task subdirs to process (default: all).')
    p.add_argument('--dtype', choices=['fp16', 'fp32'], default='fp16',
                   help='Stored image dtype. fp16 halves disk; loss is below '
                        'the ImageNet-normalized signal range.')
    p.add_argument('--num-threads', type=int, default=4,
                   help='CPU threads for BLAS/torch during preprocessing.')
    p.add_argument('--no-skip-existing', action='store_true',
                   help='Reprocess slides even if outputs already exist.')
    p.add_argument('--verify-first', action='store_true',
                   help='Print mean/std/min/max of first cached slide.')
    return p.parse_args()


_args = _parse_cli()

# Lock down threads BEFORE heavy imports
N = str(_args.num_threads)
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
         'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[v] = N

import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import torch

torch.set_num_threads(_args.num_threads)
try:
    torch.set_num_interop_threads(_args.num_threads)
except RuntimeError:
    pass  # already initialized in some envs

# Reuse the project's exact load_slide so semantics match training 1:1
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_slide  # noqa: E402


def _slide_outputs(dst_task_dir: Path, slide_name: str):
    return (
        dst_task_dir / 'slides' / f'{slide_name}.pt',
        dst_task_dir / 'adata' / f'{slide_name}.h5ad',
    )


def _slide_done(dst_task_dir: Path, slide_name: str) -> bool:
    pt_path, ad_path = _slide_outputs(dst_task_dir, slide_name)
    return pt_path.exists() and ad_path.exists()


def process_slide(patches_h5: Path, st_h5ad: Path, dst_task_dir: Path,
                  dtype: str) -> dict:
    """Run load_slide and save outputs. Returns small stats dict."""
    slide_name = patches_h5.stem
    pt_path, ad_path = _slide_outputs(dst_task_dir, slide_name)
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    ad_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    imgs, adata, _coords = load_slide(str(patches_h5), str(st_h5ad))
    t_load = time.time() - t0

    if dtype == 'fp16':
        imgs_store = imgs.half().contiguous()
    else:
        imgs_store = imgs.contiguous()

    # Atomic writes
    tmp_pt = pt_path.with_suffix('.pt.tmp')
    torch.save({'imgs': imgs_store}, tmp_pt)
    os.replace(tmp_pt, pt_path)

    tmp_ad = ad_path.with_suffix('.h5ad.tmp')
    adata.write_h5ad(tmp_ad)
    os.replace(tmp_ad, ad_path)

    return {
        'slide': slide_name,
        'n_spots': int(imgs.shape[0]),
        'load_sec': round(t_load, 2),
        'total_sec': round(time.time() - t0, 2),
        'pt_mb': round(pt_path.stat().st_size / 1024 / 1024, 1),
        'h5ad_mb': round(ad_path.stat().st_size / 1024 / 1024, 1),
    }


def copy_meta(src_task_dir: Path, dst_task_dir: Path):
    """Copy gene lists / splits / other static files as-is."""
    for name in ('var_50genes.json', 'mean_50genes.json'):
        src = src_task_dir / name
        if src.exists():
            shutil.copy2(src, dst_task_dir / name)
    splits_src = src_task_dir / 'splits'
    if splits_src.is_dir():
        splits_dst = dst_task_dir / 'splits'
        if splits_dst.exists():
            shutil.rmtree(splits_dst)
        shutil.copytree(splits_src, splits_dst)


def main():
    args = _args
    src_root = Path(args.src)
    dst_root = Path(args.dst)
    dst_root.mkdir(parents=True, exist_ok=True)

    if args.tasks:
        task_names = args.tasks
    else:
        task_names = sorted(
            d.name for d in src_root.iterdir()
            if d.is_dir() and (d / 'patches').is_dir() and (d / 'adata').is_dir()
        )

    print(f'Source : {src_root}')
    print(f'Dest   : {dst_root}')
    print(f'Tasks  : {task_names}')
    print(f'Dtype  : {args.dtype}    Threads: {args.num_threads}    '
          f'Skip-existing: {not args.no_skip_existing}')
    print('=' * 70)

    grand_stats = []
    for task in task_names:
        src_task = src_root / task
        dst_task = dst_root / task
        dst_task.mkdir(parents=True, exist_ok=True)

        # Collect slide pairs: patches/<NAME>.h5  +  adata/<NAME>.h5ad
        patches_dir = src_task / 'patches'
        adata_dir = src_task / 'adata'
        if not patches_dir.is_dir() or not adata_dir.is_dir():
            print(f'[{task}] missing patches/ or adata/, skipping')
            continue

        slide_names = sorted(
            p.stem for p in patches_dir.iterdir()
            if p.suffix == '.h5' and (adata_dir / f'{p.stem}.h5ad').exists()
        )
        if not slide_names:
            print(f'[{task}] no matching (patches, adata) pairs')
            continue

        print(f'\n[{task}] {len(slide_names)} slides')
        copy_meta(src_task, dst_task)

        t_task = time.time()
        task_stats = []
        for i, name in enumerate(slide_names, 1):
            if not args.no_skip_existing and _slide_done(dst_task, name):
                print(f'  ({i}/{len(slide_names)}) {name}  [skip — already cached]')
                continue
            patches_h5 = patches_dir / f'{name}.h5'
            st_h5ad = adata_dir / f'{name}.h5ad'
            try:
                stats = process_slide(patches_h5, st_h5ad, dst_task, args.dtype)
                task_stats.append(stats)
                print(f'  ({i}/{len(slide_names)}) {name}  '
                      f'spots={stats["n_spots"]}  '
                      f'pt={stats["pt_mb"]}MB  h5ad={stats["h5ad_mb"]}MB  '
                      f'{stats["total_sec"]}s')
            except Exception as e:
                print(f'  ({i}/{len(slide_names)}) {name}  FAILED: {e}')
                traceback.print_exc()
        print(f'[{task}] done in {time.time() - t_task:.1f}s')
        grand_stats.extend([{**s, 'task': task} for s in task_stats])

        # Per-task provenance
        with open(dst_task / '_meta.json', 'w') as f:
            json.dump({
                'src': str(src_task),
                'dtype': args.dtype,
                'load_slide_signature': (
                    'uint8 -> permute(0,3,1,2)/255 -> ImageNet normalize; '
                    'sc.pp.log1p(adata); barcode reorder to h5 order'
                ),
                'slides': task_stats,
            }, f, indent=2)

    # Optional verification on the first produced slide
    if args.verify_first and grand_stats:
        first = grand_stats[0]
        pt = dst_root / first['task'] / 'slides' / f'{first["slide"]}.pt'
        d = torch.load(pt, map_location='cpu')
        x = d['imgs'].float()
        print('\nVerify first slide:')
        print(f"  {first['task']}/{first['slide']}  shape={tuple(x.shape)}  "
              f"dtype={d['imgs'].dtype}")
        print(f"  mean={x.mean().item():.4f}  std={x.std().item():.4f}  "
              f"min={x.min().item():.4f}  max={x.max().item():.4f}")

    print('\nAll done.')


if __name__ == '__main__':
    main()
