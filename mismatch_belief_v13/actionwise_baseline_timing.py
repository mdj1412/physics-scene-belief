"""Strong baseline timing on the same replication cases as actionwise gates."""
import os,sys,time,gc
from pathlib import Path
import numpy as np
import torch
import physics,common
CPU='--cpu' in sys.argv
if CPU:physics.DEV='cpu'
common.R=Path('results_actionwise')
from common import *
from runtime import Engine
def sync():
    if not CPU:torch.cuda.synchronize()
assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(2)
x=public('test');obs=observations('test');stored=np.load(R/'decision_predictions.npz');out={};ids=np.linspace(0,len(x)-1,48,dtype=int)
for nm in ['qmc16_probe','expanded_qmc16_probe','expanded_qmc64_probe','expanded_qmc64_native']:
    m=Engine(nm);rows=[]
    for rep in range(-1,3):
        for i in (ids[:2] if rep<0 else ids):
            sync();start=time.perf_counter();ctx=m.start(x[i:i+1]);a=ctx['action'];p,_=m.finish(ctx,obs[a][i:i+1,0] if a else None);sync();elapsed=time.perf_counter()-start
            assert abs(p.cpu().numpy()[0]-stored[nm][i,0]).max()<1e-8
            if rep>=0:rows.append(dict(case=int(i),repeat=rep,seconds=elapsed))
    out[nm]=dict(rows=rows,median_s=float(np.median([r['seconds'] for r in rows])),p90_s=float(np.quantile([r['seconds'] for r in rows],.9)))
    del m;gc.collect()
write(R/('baseline_timing_cpu.json' if CPU else 'baseline_timing_gpu.json'),out)
