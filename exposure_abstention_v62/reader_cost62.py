"""Measure the full CPU pixel-to-routing portion alone, without running the GPU model."""
import json
import os
import time
from pathlib import Path
import numpy as np
from bench62 import ROOT,METHODS,choice,plan,orders

assert os.getenv('SLURM_JOB_ID') and os.getenv('CUDA_VISIBLE_DEVICES')==''
schedule=plan()
values=np.empty((24,4,6,6))
for i in range(24):
    with np.load(ROOT/'fixtures'/f'panel_{i:02}.npz') as f:
        frame,wave,electronic,index=f['frames'],f['wave'],f['electronic'],f['reset_id']
    m=np.arange(2)
    for b in range(4):
        d,c,t,z=schedule[i,b]
        fr=frame[d,c,z,m]
        el=np.stack([electronic[:,c[j],z[j],j] for j in m])
        for method in range(6):
            choice(fr,wave[d],el,index,method)
        # Six Williams arrangements, randomly permuted labels per catalog.
        names=np.random.default_rng([6208931,i,b]).permutation(6)
        for row in range(6):
            order=names[(np.array([0,1,5,2,4,3])+row)%6]
            for method in order:
                a=time.perf_counter_ns()
                choice(fr,wave[d],el,index,method)
                values[i,b,row,method]=(time.perf_counter_ns()-a)/1.e6
q=np.quantile(values,.9,axis=(0,1,2))
ix=np.random.default_rng(6208811).integers(24,size=(4000,24))
boot=np.quantile(values[ix],.9,axis=(1,2,3))
deltas=boot[:,[0]]-boot[:,1:]
out=dict(methods=METHODS,p90_ms=q.tolist(),delta_reset_minus_others_ms=(q[0]-q[1:]).tolist(),
         pointwise95=np.quantile(deltas,[.025,.975],axis=0).T.tolist(),
         clips_per_call=32,measured_per_method=576,device='CPU only, separate benchmark on node56')
(ROOT/'reader_cost62.json').write_text(json.dumps(out,indent=2)+'\n')
np.savez_compressed(ROOT/'reader_cost62_raw.npz',ms=values)
