"""Sensitivity to the deployment frequency of extreme and intermediate retention sessions."""
import json
import os
import numpy as np
from score50 import ROOT
assert os.getenv('SLURM_JOB_ID')
loss=np.load(ROOT/'catalog_scores50.npz')['matched'].mean(axis=1)[:,:,2]
b=np.random.default_rng(500669).integers(0,len(loss),(16000,len(loss)))
def cutoff(delta):
    easy=delta[..., [0,4]].mean(-1)
    middle=delta[..., [1,2,3]].mean(-1)
    return middle/(middle-easy)
result={}
for m in (0,6):
    d=loss[:,:,m]-loss[:,:,4]
    root=cutoff(d.mean(0)); boot=cutoff(d[b].mean(1))
    result[str(m)]=dict(equal_extreme_diff=d[:,[0,4]].mean(),equal_middle_diff=d[:,1:4].mean(),
                        crossing=float(root),pointwise95=np.quantile(boot,[.025,.975]).tolist())
(ROOT/'mix50.json').write_text(json.dumps(result,indent=1))
