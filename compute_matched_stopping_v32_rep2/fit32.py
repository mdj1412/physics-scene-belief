"""Seal two stage-wise ridge policies using source data alone (v32)."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
V30 = ROOT.parent / 'learned_probe_value_v30'
ARMS = {'broad': (1, 1), 'repeated': (2, 2)}
PENALTY = 3.0


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_arm(root, arm, dimension):
    groups = sorted((root / arm).glob('block_[0-9][0-9][0-9]'))
    if not groups:
        raise ValueError(f'No source data for {arm}')
    xs, ys, provenance = [], [], []
    for i, directory in enumerate(groups):
        if directory.name != f'block_{i:03d}':
            raise ValueError(f'Non-contiguous training blocks in {arm}')
        path, metadata = directory / 'data.npz', directory / 'meta.json'
        meta = json.loads(metadata.read_text())
        with np.load(path, allow_pickle=False) as z:
            x = z['features'].astype(np.float64)
            gain = z['gain'].astype(np.float64)
        n = x.shape[1]
        if (x.shape != (2, n, dimension) or gain.shape != (2, n, *ARMS[arm]) or
                not 1 <= n <= 8 or (n < 8 and i < len(groups) - 1) or
                not np.isfinite(x).all() or not np.isfinite(gain).all() or
                np.abs(gain).max() > 1.000001 or float(meta['elapsed_s']) <= 0):
            raise ValueError(f'Invalid source tensor or timing at {directory}')
        xs.append(x)
        ys.append(gain.mean(axis=(2, 3)))
        provenance.append({'block': i, 'n_environments': n, 'data_sha256': sha(path),
                           'metadata_sha256': sha(metadata), 'elapsed_s': float(meta['elapsed_s'])})
    return np.concatenate(xs, axis=1), np.concatenate(ys, axis=1), provenance


def verify_fit(models, x, y):
    # Independent reference uses an augmented least squares design and SVD.
    errors = []
    for m, xx, yy in zip(models, x, y):
        avg, sd = xx.mean(axis=0), np.maximum(xx.std(axis=0), 1e-8)
        z = (xx - avg) / sd
        penalty_rows = np.sqrt(len(xx) * PENALTY) * np.eye(xx.shape[1])
        beta = np.linalg.lstsq(np.vstack((z, penalty_rows)),
                               np.r_[yy - yy.mean(), np.zeros(xx.shape[1])], rcond=None)[0]
        predicted = (xx - m['mean']) / m['sd'] @ m['beta'] + m['intercept']
        expected = z @ beta + yy.mean()
        error = float(np.max(np.abs(predicted - expected)))
        np.testing.assert_allclose(predicted, expected, rtol=2e-9, atol=1e-11)
        np.testing.assert_allclose(m['beta'], beta, rtol=2e-9, atol=1e-11)
        errors.append(error)
    return errors


def fit_and_lock(results=None):
    sys.path.insert(0, str(V30))
    from score30 import NAMES, fit, torch
    root = Path(results) if results is not None else ROOT / 'results_train'
    output = root / 'model_lock.json'
    locked = {'features': NAMES, 'stages': ['2_to_4', '4_to_8'], 'models': {}, 'training': {},
              'penalty': PENALTY, 'device': torch.cuda.get_device_name(),
              'fit_source_sha256': sha(V30 / 'score30.py'), 'lock_builder_sha256': sha(Path(__file__)),
              'target': 'mean common-decoder Brier before minus after, per paid prefix',
              'training_times': 'block elapsed_s plus data-to-model wall time after shared kernel warm-up'}
    if output.exists():
        raise FileExistsError(f'Existing frozen lock {output}')
    size = len(NAMES)
    # Warm shared runtime and the 15x15 solve, independent of either training corpus.
    torch.cuda.synchronize()
    start = time.perf_counter()
    fit(np.zeros((8, size)), np.zeros(8), PENALTY)
    torch.cuda.synchronize()
    locked['shared_fit_warmup_s'] = time.perf_counter() - start
    for arm, axes in ARMS.items():
        # Read/hash and independent CPU audit are recorded separately from model fitting.
        start = time.perf_counter()
        x, y, sources = read_arm(root, arm, size)
        load_s = time.perf_counter() - start
        torch.cuda.synchronize()
        start = time.perf_counter()
        models = [fit(xx, yy, PENALTY) for xx, yy in zip(x, y)]
        torch.cuda.synchronize()
        fit_s = time.perf_counter() - start
        err = verify_fit(models, x, y)
        labels_s = sum(row['elapsed_s'] for row in sources)
        locked['models'][arm] = models
        locked['training'][arm] = {'blocks': sources, 'n_environments': x.shape[1],
            'futures': axes[0], 'batches': axes[1], 'labeled_cells': int(y.size * np.prod(axes)),
            'labels_s': labels_s, 'source_read_audit_s': load_s, 'fit_s': fit_s,
            'labels_plus_fit_s': labels_s + fit_s, 'labels_plus_read_plus_fit_s': labels_s + load_s + fit_s,
            'mean_targets': y.mean(axis=1).tolist(), 'target_sd': y.std(axis=1).tolist(),
            'cpu_svd_max_prediction_error': err}
    tmp = output.with_suffix('.tmp')
    tmp.write_text(json.dumps(locked, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, output)
    return locked


if __name__ == '__main__':
    fit_and_lock()
