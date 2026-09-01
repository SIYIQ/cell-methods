"""HEST-Bench task definitions and patient-aware k-fold splitting.

Based on the 9 benchmark tasks from the HEST benchmark paper.
Each task is evaluated with patient-aware k-fold cross-validation:
  - 2 patients -> 2-fold
  - 3 patients -> 3-fold
  - 4 patients -> 4-fold
  - 24 patients -> 6-fold (grouped)

Each fold is run with 3 random seeds.
"""

import json
import os
from collections import defaultdict

import numpy as np
from sklearn.model_selection import GroupKFold

METADATA_DIR = '/home/sb202604/HEST-benchmark/metadata'
HEST_BENCH_ROOT = '/home/sb202604/hest-bench'

# Maps task name -> subfolder name under HEST_BENCH_ROOT. Needed because the
# hest-bench layout uses slightly different folder names than the task labels.
TASK_FOLDER = {
    'Task1_IDC':   'IDC',
    'Task2_PRAD':  'PRAD',
    'Task3_PAAD':  'PAAD',
    'Task4_SKCM':  'SKCM',
    'Task5_COAD':  'COAD',
    'Task6_READ':  'READ',
    'Task7_ccRCC': 'CCRCC',
    'Task8_LUAD':  'LUNG',
    'Task9_IDC_LN': 'LYMPH_IDC',
}


def _read_patient(sample_id):
    """Read patient field from metadata JSON."""
    fpath = os.path.join(METADATA_DIR, f'{sample_id}.json')
    if not os.path.exists(fpath):
        return None
    with open(fpath) as f:
        d = json.load(f)
    p = d.get('patient')
    # Normalize: treat None/nan as a unique patient per sample
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return f'__unknown_{sample_id}'
    return str(p).strip()


# ---------------------------------------------------------------------------
# Task definitions: sample IDs per task
# ---------------------------------------------------------------------------
BENCHMARK_TASKS = {
    'Task1_IDC': {
        'oncotree': 'IDC',
        'organ': 'Breast',
        'technology': 'Xenium',
        'samples': ['TENX95', 'TENX99', 'NCBI783', 'NCBI785'],
    },
    'Task2_PRAD': {
        'oncotree': 'PRAD',
        'organ': 'Prostate',
        'technology': 'Visium',
        # MEND155 is missing from the dataset
        'samples': [f'MEND{i}' for i in list(range(139, 155)) + list(range(156, 163))],
    },
    'Task3_PAAD': {
        'oncotree': 'PAAD',
        'organ': 'Pancreas',
        'technology': 'Xenium',
        'samples': ['TENX116', 'TENX126', 'TENX140'],
    },
    'Task4_SKCM': {
        'oncotree': 'SKCM',
        'organ': 'Skin',
        'technology': 'Xenium',
        'samples': ['TENX115', 'TENX117'],
    },
    'Task5_COAD': {
        'oncotree': 'COAD',
        'organ': 'Colon',
        'technology': 'Visium',
        'samples': ['TENX111', 'TENX147', 'TENX148', 'TENX149'],
    },
    'Task6_READ': {
        'oncotree': 'READ',
        'organ': 'Rectum',
        'technology': 'Visium',
        'samples': ['ZEN36', 'ZEN40', 'ZEN48', 'ZEN49'],
    },
    'Task7_ccRCC': {
        'oncotree': 'ccRCC',
        'organ': 'Kidney',
        'technology': 'Visium',
        'samples': [f'INT{i}' for i in range(1, 25)],
    },
    'Task8_LUAD': {
        'oncotree': 'LUAD',
        'organ': 'Lung',
        'technology': 'Xenium',
        'samples': ['TENX118', 'TENX141'],
    },
    'Task9_IDC_LN': {
        'oncotree': 'IDC-LymphNode',
        'organ': 'LymphNode',
        'technology': 'Visium',
        'samples': ['NCBI681', 'NCBI682', 'NCBI683', 'NCBI684'],
    },
}


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
      - 24 patients -> 6-fold (grouped, ~4 patients per group)

    Special case: Task5_COAD follows the HEST-bench official 2-fold split
    (TENX111 vs TENX147/148/149) shipped in <task>/splits/. Patient info is
    incomplete (TENX111 patient=nan, others Patient 1) so patient-aware
    GroupKFold isn't usable; the official csv split is the correct protocol.
    """
    if task_name == 'Task5_COAD':
        return 2  # HEST official 2-fold

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

    if task_name == 'Task5_COAD':
        # HEST official 2-fold split (from <task>/splits/{train,test}_{0,1}.csv):
        #   fold 0: train=TENX111         | test=TENX147,TENX148,TENX149
        #   fold 1: train=TENX147,148,149 | test=TENX111
        coad_folds = [
            (['TENX111'], ['TENX147', 'TENX148', 'TENX149']),
            (['TENX147', 'TENX148', 'TENX149'], ['TENX111']),
        ]
        return coad_folds[fold_idx]

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

    Paths point at the hest-bench layout::

        <HEST_BENCH_ROOT>/<TASK_FOLDER>/patches/<SAMPLE>.h5
        <HEST_BENCH_ROOT>/<TASK_FOLDER>/adata/<SAMPLE>.h5ad

    Each sample's task folder is resolved via BENCHMARK_TASKS + TASK_FOLDER.
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
