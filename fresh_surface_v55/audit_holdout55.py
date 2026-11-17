"""Physics-only and support checks. No acquisition policy or decision loss enters."""
import hashlib
import json
from pathlib import Path
import os
import numpy as np

from generate55 import ROOT,N,design,field_mm,pop55,load,dest

assert os.environ.get('SLURM_JOB_ID')
manifest=json.loads((ROOT/'independence55.json').read_text())
n_flip=n_solver=total=active_flip=base_mismatch=0
ranges=[]
for i in range(N):
    expected=manifest['panels'][i]['sha256']
    assert hashlib.sha256(dest('panels',i).read_bytes()).hexdigest()==expected
    pop=pop55(i); d=design(i); data=load('panels',i)
    active=d['state']!=0
    changed=data['y_keep']!=data['y_flip']
    assert not np.any(changed&~active)
    n_flip+=int(changed.sum());active_flip+=int(active.sum())
    for name in ('y_keep','y_flip'):
        n_solver+=int((data[name]!=data[name+'_ultra']).sum());total+=data[name].size
    assert np.array_equal(data['ow'][0],data['ow'][-1]) and np.array_equal(data['nw'][0],data['nw'][-1])
    assert np.all((data['imp']==0)==(data['nf'][0,...,3]==data['nf'][-1,...,3]))
    ranges.extend([field_mm(h,s) for s in set(pop['surface']) for h in np.linspace(0.,40.,401)])
    for e in range(4):
        assert np.all(np.diff(np.sort(data['new_ids'][e],axis=1),axis=1)>0)
        assert np.all(np.diff(np.sort(data['old_ids'][e],axis=1),axis=1)>0)
    base_mismatch+=int(np.any(data['y_keep'][~active]!=data['y_flip'][~active]))
stats=dict(sha_verified_panels=N,material_spatial_scale_range=np.ptp(np.asarray(ranges),axis=0).tolist(),
           material_scale_orientation_min=np.min(ranges,axis=0).tolist(),material_scale_orientation_max=np.max(ranges,axis=0).tolist(),
           changed_under_reversal=n_flip,active_objects=active_flip,deployment_objects=total//2,
           fine_ultra_mismatches=n_solver,binary_endpoints=total,inactive_reversal_disagreements=base_mismatch)
(ROOT/'physics_audit55.json').write_text(json.dumps(stats,indent=2))
print(stats)
