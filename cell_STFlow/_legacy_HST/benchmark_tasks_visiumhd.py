"""HEST-1k VisiumHD task definitions and patient-aware k-fold splitting.

Four benchmark tasks from the HEST-1k VisiumHD subset:
  - IDC  (Breast, VisiumHD)
  - LUAD (Lung, VisiumHD)
  - COAD (Colon, VisiumHD)
  - SOC  (Ovary, VisiumHD)

Each task is evaluated with patient-aware k-fold cross-validation.
Since no external patient metadata is available, each slide is treated as
its own patient group to avoid data leakage across slides.
"""

import json
import os
from collections import defaultdict

import numpy as np
from sklearn.model_selection import GroupKFold

# -----------------------------------------------------------------------------
# Root paths
# -----------------------------------------------------------------------------
HEST_BENCH_ROOT = '/home/sb202604/hest-bench-visiumhd'
PRETREAT_ROOT = '/home/sb202604/hest-bench-visiumhd-pretreat'

# No external metadata directory for this subset; patient info is derived
# directly from the sample IDs (each slide = one patient).
METADATA_DIR = None

# Maps task name -> subfolder name under HEST_BENCH_ROOT.
TASK_FOLDER = {
    'Task1_IDC':  'IDC',
    'Task2_LUAD': 'LUAD',
    'Task3_COAD': 'COAD',
    'Task4_SOC':  'SOC',
}


# -----------------------------------------------------------------------------
# Task definitions: sample IDs per task
# -----------------------------------------------------------------------------
BENCHMARK_TASKS = {
    'Task1_IDC': {
        'oncotree': 'IDC',
        'organ': 'Breast',
        'technology': 'VisiumHD',
        'samples': ['TENX161', 'TENX162', 'TENX180'],
    },
    'Task2_LUAD': {
        'oncotree': 'LUAD',
        'organ': 'Lung',
        'technology': 'VisiumHD',
        'samples': ['TENX163', 'TENX168', 'TENX169', 'TENX170', 'TENX171'],
    },
    'Task3_COAD': {
        'oncotree': 'COAD',
        'organ': 'Colon',
        'technology': 'VisiumHD',
        'samples': ['TENX128', 'TENX153', 'TENX154', 'TENX155', 'TENX156', 'TENX175'],
    },
    'Task4_SOC': {
        'oncotree': 'SOC',
        'organ': 'Ovary',
        'technology': 'VisiumHD',
        'samples': ['TENX182', 'TENX183', 'TENX184', 'TENX185'],
    },
}


def _read_patient(sample_id):
    """Read patient field from metadata JSON.

    For the VisiumHD subset there is no external metadata, so each slide is
    treated as a unique patient to guarantee patient-aware splitting.
    """
    if METADATA_DIR is not None and os.path.isdir(METADATA_DIR):
        fpath = os.path.join(METADATA_DIR, f'{sample_id}.json')
        if os.path.exists(fpath):
            with open(fpath) as f:
                d = json.load(f)
            p = d.get('patient')
            if p is not None and not (isinstance(p, float) and np.isnan(p)):
                return str(p).strip()
    # Fallback: each slide is its own patient group
    return f'__slide_{sample_id}'


def get_patient_map(task_name):
    """Return {sample_id: patient_str} for a given task."""
    samples = BENCHMARK_TASKS[task_name]['samples']
    return {s: _read_patient(s) for s in samples}


def determine_n_folds(task_name, patient_map):
    """Determine the number of folds based on patient count.

    Rules:
      - 2 patients  -> 2-fold (leave-one-patient-out)
      - 3 patients  -> 3-fold
      - 4 patients  -> 4-fold
      - >=20 patients -> 6-fold (grouped, ~4 patients per group)
      - otherwise   -> min(n_patients, 4)
    """
    unique_patients = set(patient_map.values())
    n_patients = len(unique_patients)

    if n_patients == 2:
        return 2
    if n_patients == 3:
        return 3
    if n_patients == 4:
        return 4
    if n_patients >= 20:
        return 6
    return min(n_patients, 4)


def make_fold_split(task_name, fold_idx, n_folds, seed=42):
    """Generate train/test slide split for a given task, fold, and seed.

    Parameters
    ----------
    task_name : str
    fold_idx : int
        Which fold to use as test (0 to n_folds-1).
    n_folds : int
        Number of folds.
    seed : int
        Random seed for GroupKFold shuffle.

    Returns
    -------
    train_samples : list of str
    test_samples : list of str
    """
    samples = BENCHMARK_TASKS[task_name]['samples'][:]
    patient_map = get_patient_map(task_name)
    groups = [patient_map[s] for s in samples]

    # GroupKFold for patient-aware splitting
    gkf = GroupKFold(n_splits=n_folds)
    splits = list(gkf.split(samples, groups=groups))

    train_idx, test_idx = splits[fold_idx]
    train_samples = [samples[i] for i in train_idx]
    test_samples = [samples[i] for i in test_idx]
    return train_samples, test_samples


def _sample_to_task_folder():
    """Build {sample_id: task_folder} from BENCHMARK_TASKS + TASK_FOLDER."""
    idx = {}
    for task_name, info in BENCHMARK_TASKS.items():
        folder = TASK_FOLDER.get(task_name)
        if folder is None:
            continue
        for sid in info['samples']:
            idx[sid] = folder
    return idx


_SAMPLE_TO_TASK_FOLDER = _sample_to_task_folder()


def get_slide_paths(sample_ids):
    """Return list of (patches_path, st_path) tuples for given sample IDs.

    Paths point at the hest-bench-visiumhd layout::

        <HEST_BENCH_ROOT>/<TASK_FOLDER>/patches/<SAMPLE>.h5
        <HEST_BENCH_ROOT>/<TASK_FOLDER>/adata/<SAMPLE>.h5ad

    Each sample's task folder is resolved via BENCHMARK_TASKS + TASK_FOLDER.

    The utils.load_slide() function will automatically consult the offline
    cache (PRETREAT_ROOT) when set_offline_cache(PRETREAT_ROOT) is called.
    """
    paths = []
    for sid in sample_ids:
        folder = _SAMPLE_TO_TASK_FOLDER.get(sid)
        if folder is None:
            print(f"WARNING: {sid} not registered in any BENCHMARK_TASKS entry; skipping")
            continue
        pf = os.path.join(HEST_BENCH_ROOT, folder, 'patches', f'{sid}.h5')
        sf = os.path.join(HEST_BENCH_ROOT, folder, 'adata', f'{sid}.h5ad')
        if os.path.exists(pf) and os.path.exists(sf):
            paths.append((pf, sf))
        else:
            print(f"WARNING: Missing data for {sid}: patch={os.path.exists(pf)} st={os.path.exists(sf)}")
    return paths


def list_all_runs(seeds=[42, 43, 44]):
    """Enumerate all (task, fold, seed) combinations.

    Returns list of dicts with keys: task, n_folds, fold, seed.
    """
    runs = []
    for task_name in BENCHMARK_TASKS:
        patient_map = get_patient_map(task_name)
        n_folds = determine_n_folds(task_name, patient_map)
        for fold in range(n_folds):
            for seed in seeds:
                runs.append({
                    'task': task_name,
                    'n_folds': n_folds,
                    'fold': fold,
                    'seed': seed,
                })
    return runs


if __name__ == '__main__':
    # Quick sanity check
    for task_name in BENCHMARK_TASKS:
        patient_map = get_patient_map(task_name)
        n_folds = determine_n_folds(task_name, patient_map)
        print(f"{task_name}: {len(BENCHMARK_TASKS[task_name]['samples'])} samples, "
              f"{len(set(patient_map.values()))} patients -> {n_folds}-fold")
        for fold in range(n_folds):
            train, test = make_fold_split(task_name, fold, n_folds)
            print(f"  fold {fold}: train={train}, test={test}")

    # Verify all slide paths exist
    print("\n--- Verifying slide paths ---")
    all_samples = []
    for task_name in BENCHMARK_TASKS:
        all_samples.extend(BENCHMARK_TASKS[task_name]['samples'])
    paths = get_slide_paths(all_samples)
    print(f"Total samples: {len(all_samples)}, found paths: {len(paths)}")
    if len(paths) == len(all_samples):
        print("All slide paths verified successfully!")
