"""Reconcile every catalog endpoint with the original independent scoring paths."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.stats import t

root = Path(__file__).resolve().parent
project = root.parent
load = lambda p: json.loads(p.read_text())
source = load(root / 'soft_choice.json')
target = load(root / 'soft_transfer.json')
src45 = load(project / 'image_pressure_v45/score45.json')['source_axis']
tgt46 = load(project / 'normal_variance_v46/score46.json')
checks = []
for new, old, columns in [(source, src45['pooled_panel'], [0,4]),
                           (target, tgt46['absolute_panel'], [0,1])]:
    pooled = np.array(new['per_catalog_family']).mean(axis=1)
    difference = float(np.abs(pooled[:, [0,-1]] - np.array(old)[:, columns]).max())
    assert difference < 1e-12, difference
    checks.append(difference)
actual = np.array(target['per_catalog_family']).mean(axis=1)
i = target['grid'].index(source['alpha'])
assert i == np.argmin(source['source_mean']) and source['alpha'] == target['alpha']
for comp, result in zip([0,-1],target['selected_minus_full_and_hard']):
    z = actual[:,i]-actual[:,comp]
    half = t.isf(.0125,11)*np.sqrt(np.sum((z-z.mean())**2)/(11*12))
    np.testing.assert_allclose(result, [z.mean(),z.mean()-half,z.mean()+half], atol=1e-13)
assert source['code_sha256'] == hashlib.sha256((root/'soft_likelihood.py').read_bytes()).hexdigest()
print(json.dumps({'max_abs_catalog_brier_error_v45_v46':checks,'paired_t_recomputed':True,
                  'source_choice_and_code_freeze':True},indent=2))
