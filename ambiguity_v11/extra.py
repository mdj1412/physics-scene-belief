"""LMMSE baseline: public factual evidence only, no actual outcome access."""
import os,time,json
from pathlib import Path
import torch
from model import Bank,VSD
from run import R,SPLITS,sha,write
assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(2)
b=Bank(41)
for split in ['test','misspec']:
    for i in range(SPLITS[split][0]):
        start=time.time();p=R/split/'public'/f'{i:04d}.json';w=b.posterior(json.loads(p.read_text()));z=b.z-w@b.z;gains=[0.]
        for a in [1,2,3]:
            x=b.mu[a]-w@b.mu[a];cx=x.T@(w[:,None]*x)+VSD[a-1]**2*torch.eye(x.shape[1],device=x.device,dtype=x.dtype)
            cross=z.T@(w[:,None]*x);gain=.5*torch.trace(cross@torch.linalg.solve(cx,cross.T));gains.append(float(gain))
        action=max(range(4),key=lambda j:gains[j]);write(R/split/'extra_policies'/f'{i:04d}.json',dict(action=action,gains=gains,seconds=time.time()-start,public_sha256=sha(p)))
    write(R/split/'extra_freeze.json',dict(files={p.name:sha(p) for p in (R/split/'extra_policies').glob('*.json')},code_sha256=sha('extra.py'),time=time.time()))
print('LMMSE policies frozen',flush=True)
