"""Fit two direction-free score maps on v34 fold 0's source only."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'bidirectional_probe_v34'
EPS = 1e-12
RIDGE = 0.005


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_fit():
    probabilities, outcomes, hashes = [], [], {}
    for block in range(8):
        for orientation in ('plus', 'minus'):
            for schedule in ('same', 'alternating'):
                path = (SOURCE / 'results_policy' / f'ridged_coarse_{orientation}'
                        / f'block_{block:03d}' / schedule / 'metrics.npz')
                hashes[str(path.relative_to(SOURCE))] = hash_file(path)
                with np.load(path) as batch:
                    sensor = (batch['actions'][:, :, 1] == 3).astype(int)
                    h = np.take_along_axis(batch['probabilities'], sensor[:, :, None, None],
                                           axis=2)[:, :, 0]
                    y = batch['labels'].astype(float)
                    assert h.shape == (8, 128, 32) and y.shape == (8, 128)
                    probabilities.append(h)
                    outcomes.append(y[..., None])
    h, y = np.stack(probabilities), np.stack(outcomes)
    assert h.min() >= -1e-12 and h.max() <= 1 + 1e-12
    eta = logit(np.clip(h, EPS, 1 - EPS))

    def d_offset(c):
        return np.mean(expit(eta + c) - y) + RIDGE*c

    def d_scale(a):
        return np.mean((expit(a*eta) - y)*eta) + RIDGE*(a-1)

    offset = brentq(d_offset, -20, 20, xtol=1e-12)
    # Optimize inverse temperature directly (positive root gives the unique optimum).
    scale = brentq(d_scale, 1e-8, 100., xtol=1e-12)
    return dict(source='v34 fold 0 beta training family ridged_coarse',
                source_strata=32, environments_per_stratum=8, worlds_per_environment=128,
                sensor_draws_per_world=32, source_prediction_count=int(h.size),
                fixed4_action_column=1, intercept=offset, inverse_temperature=scale,
                temperature=1/scale, probability_clip=EPS, ridge_weight=RIDGE,
                criterion='pooled binary log loss, quadratic ridge around identity in intercept or inverse temperature',
                source_model_gradient=[d_offset(offset),d_scale(scale)],
                fit_code_sha256=hash_file(Path(__file__)),
                source_beta_code_sha256=hash_file(SOURCE/'anchored34.py'),
                metrics_files_sha256=hashes)


if __name__ == '__main__':
    result = source_fit()
    out = ROOT / 'source_cal35_frozen.json'
    serialized = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if out.exists():
        assert out.read_text() == serialized, 'frozen output mismatch'
    else:
        out.write_text(serialized)
    print({k: result[k] for k in ('source_strata', 'intercept', 'temperature', 'inverse_temperature', 'source_model_gradient')})
    print('sha256', hash_file(out))
