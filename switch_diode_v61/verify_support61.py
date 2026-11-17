"""Explain the original hash collision and verify final objects before analyzing outcomes."""
from pathlib import Path
import json
import os
import numpy as np

root=Path(__file__).resolve().parent
assert os.getenv('SLURM_JOB_ID') and os.getenv('CUDA_VISIBLE_DEVICES')==''
keynames=('theta','cal_theta','old_cal_theta')
def collect(path):
    records={key:set() for key in keynames}
    files=sorted((path/'worlds').glob('panel_*.npz'))
    assert len(files)==24
    for panel in files:
        with np.load(panel) as f:
            for key in keynames:
                records[key].update(map(bytes,np.ascontiguousarray(f[key]).reshape(-1,6)))
    return records
previous=collect(root.parent/'trigger_fault_v60')
old=collect(root/'excluded_5100')
fresh=collect(root)
def matrix(left,right):
    return {a:{b:len(x&y) for b,y in right.items()} for a,x in left.items()}
data=dict(original_5100_vs_v60=matrix(old,previous),accepted_6000_vs_v60=matrix(fresh,previous),
          accepted_6000_vs_excluded_5100=matrix(fresh,old))
assert data['original_5100_vs_v60']['cal_theta']['old_cal_theta']>0
for lines in [data['accepted_6000_vs_v60'],data['accepted_6000_vs_excluded_5100']]:
    assert sum(sum(row.values()) for row in lines.values())==0
(root/'support61.json').write_text(json.dumps(data,indent=2)+'\n')
