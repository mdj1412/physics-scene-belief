"""Reproduce frozen source scores and stress-test name-free paid sensor input logics."""
import hashlib
import json
import numpy as np
from scipy.special import expit
from pathlib import Path

import inference43 as infer
import model43 as m

root = Path(__file__).resolve().parent
model = m.load_model()
d = np.load(root/'source_matrix43.npz')
s = json.loads((root/'source_validation43.json').read_text())
choice = d['panel'] >= 10
x = d['feature'][choice]
truth = d['label'][choice, :, 1]
z, w = d['cue'][choice, None], d['weight'][choice]
pred = dict(original=expit(x[:,0]), robust=m.robust_from_features(model,x),
            privileged=d['privileged'][choice])
errors = []
for k,p in pred.items():
    cases = {'zero':p[:,None], 'temperature':m.temperature(model,p,k)[:,None],
             'fixed':infer.r.v38.apply(p[:,None],z*.325,infer.r.v38.BETA),
             'optimizer':infer.r.v38.apply(p[:,None],z*w,infer.r.v38.BETA)}
    for action, p_eval in cases.items():
        measured = float((truth*(1-p_eval)**2 + (1-truth)*p_eval**2).mean())
        errors.append(abs(measured-s['simple_selection_scores'][k][action]['shifted']))
assert max(errors) < 1.e-12
# Labels and hidden metadata are not accepted by production methods. Poison unpurchased readings.
rng = np.random.default_rng(4300233)
actions = np.tile([1,3],(2,8))
base = rng.uniform(size=(2,16,2))
xx = rng.normal(0,.01,(2,16,11))
obs = [{a:rng.normal(size=(16,2,k)) for a,k in [(1,7),(3,2)]} for _ in range(2)]
p = m.robust_probability(model,base,xx,actions,obs,.019)
for e in range(2):
    for a in obs[e]:
        obs[e][a][actions[e]!=a] = np.nan
np.testing.assert_array_equal(p,m.robust_probability(model,base,xx,actions,obs,.019))
out = dict(source_recomputed_simple_risk_max_error=max(errors),inactive_sensor_max_abs_error=0,
           fitted_feature_count=int(s['feature_dim']),
           source_sha256=hashlib.sha256((root/'source_matrix43.npz').read_bytes()).hexdigest(),
           rows_checked=int(choice.sum()),checks=12)
(root/'model_audit43.json').write_text(json.dumps(out,indent=2)+'\n')
print(out)
