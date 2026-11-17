"""Audit the same routes on which the two policies made identical follow-up purchases."""
import json
import os
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
assert os.getenv('SLURM_JOB_ID') and os.getenv('CUDA_VISIBLE_DEVICES') == ''
ms, outer, bought = [], [], []
for catalog in range(24):
    with np.load(root/'timing62'/f'panel_{catalog:02d}.npz') as f:
        ms.append(f['ms']); outer.append(f['outer']); bought.append(f['bought'])
        methods = f['methods'].tolist()
ms = np.concatenate(ms)
outer = np.concatenate(outer)
bought = np.concatenate(bought)
stats = {}
for index in (1, 2, 3):
    same_outer = (outer[:, 0] == outer[:, index]).all(-1)
    same_bought = (bought[:, 0] == bought[:, index]).all(-1)
    same = same_outer & same_bought
    if not same.any():
        raise ValueError('No paired block with matching decisions')
    t = ms[same][:, [0, index]]
    primary, comparator = np.quantile(t, .9, axis=0)
    stats[methods[index]] = dict(total=len(ms), same_outer=int(same_outer.sum()),
        same_bought=int(same_bought.sum()), same_both=int(same.sum()),
        same_both_p90_ms=[primary, comparator], same_both_delta_p90=primary-comparator,
        same_both_paired_median_ms=np.median(t[:, 0]-t[:, 1]))
(root/'route_timing62.json').write_text(json.dumps(stats,indent=2)+'\n')
