"""Post-design stress test: decouple or reverse the hazard-to-deployment assignment."""
import os
from pathlib import Path
import numpy as np
import json
from score59 import MIX, METHODS, COMPARISONS, risk_by_catalog, contrasts

root=Path(__file__).resolve().parent
assert os.getenv('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
with np.load(root/'score59_cells.npz') as f:
    rates=f['route']; fine=f['exact_by_e']
picks=np.random.default_rng(59102477).integers(24,size=(5000,24))
w=np.vstack([np.ones((1,24))/24,np.stack([(picks==c).mean(1) for c in range(24)],axis=1)])
out={}
for key in ('matched','independent','reversed'):
    # In common-mode pilots the true physical mismatch is iid Bernoulli(.29) in A and B.
    # The material and mean are held fixed; only hazard strength assignment changes.
    local=rates.copy()
    if key=='independent':
        local[:,:,1]=local[:,:,1].mean(axis=2,keepdims=True)
    if key=='reversed':
        local[:,:,1]=local[:,:,1,::-1]
    avg=local.mean(2)
    mean=w@risk_by_catalog(fine,avg).mean((1,2))
    acc=w@((avg[:,:,0]+1-avg[:,:,1]).mean(1)/2)
    common=local[:,:,1]
    ac=(common[:,:,0]+1-common[:,:,1]).mean((0,1))/2
    out[key]=dict(mean=mean[0],accuracy=acc[0],common_accuracy=ac,
                  mean_contrasts=contrasts(mean,1,COMPARISONS+[(0,8),(0,9),(4,8)]),
                  accuracy_contrasts=contrasts(acc,1))
(root/'transfer59.json').write_text(json.dumps(dict(methods=METHODS,assignments=out),indent=2,default=lambda x:np.asarray(x).tolist())+'\n')
