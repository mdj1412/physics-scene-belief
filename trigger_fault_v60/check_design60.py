"""Independent pre-score checks for hazard permutations and balanced decisions."""
from pathlib import Path
import json
import os
import numpy as np
from score60 import ROOT,assign,cat_metrics

assert os.getenv('SLURM_JOB_ID') and os.getenv('CUDA_VISIBLE_DEVICES')==''
old=ROOT.parent/'sensor_reset_v59'
with np.load(old/'score59_cells.npz') as f:
    rates=f['route'][:,None]
    causes=f['frac_class_physical'][:,None]
    fine=f['exact_by_e']
reference=json.loads((old/'transfer59.json').read_text())['assignments']
errors=[]
for kind in ('matched','independent','reversed'):
    assigned=assign(rates,kind)
    assert np.array_equal(assigned[:,:,:,0],rates[:,:,:,0])
    if kind=='independent':
        assert np.array_equal(assigned[:,:,:,1,0],assigned[:,:,:,1,1])
    vals=cat_metrics(fine,assigned[:,0],assign(causes,kind)[:,0])
    ref=reference[kind]
    for metric,other in [('mean_Brier','mean'),('action_BA','accuracy')]:
        errors.append(float(abs(vals[metric].mean(axis=0)-ref[other]).max()))
assert max(errors)<1e-12
print('v59 legacy reproduction for 3 assignments, maximum absolute discrepancy:',max(errors))
