"""Select authentic direction-changing episodes without access to outcome labels."""
import json
import os
from pathlib import Path
import numpy as np
from bench62 import ROOT, plan, deployment, select54, router

assert os.getenv('SLURM_JOB_ID') and os.getenv('CUDA_VISIBLE_DEVICES')==''
schedules=plan()
rng=np.random.default_rng(6209911)
pool=[]
eligible={}
for i in range(24):
    raw,surface=deployment(i,schedules[i,0][2])
    gate=select54(raw[2][:,:3],raw[3][:,:3],mode='robust')
    with np.load(ROOT/'pilots'/f'panel_{i:02d}.npz') as f:
        codes=f['codes']
    eligible[i]=0
    for row in range(4):
        d,cause,target,z=schedules[i,row]
        decisions=[]
        for mode in (2,1):
            paid=codes[d,mode,1,cause,z,np.arange(2)]
            decisions.append(router.route(paid,'risk_r1c1').T)
        selective,baseline=decisions
        changed=selective!=baseline
        actual=(selective[:,surface]|gate)!=(baseline[:,surface]|gate)
        valid=(changed.sum(1)==1)&actual.any(1)
        eligible[i]+=valid.sum().item()
        for ep in np.flatnonzero(valid):
            m=int(np.flatnonzero(changed[ep])[0])
            pool.append(dict(catalog=i,episode=int(ep),board=int(ep%8),cell=d,material=m,
                             cause=cause.tolist(),regime=target.tolist(),source=z.tolist(),
                             direction=int(selective[ep,m]),fixed_sel=selective[ep].tolist(),
                             fixed_v61=baseline[ep].tolist(),changed_sessions=int(actual[ep].sum()),
                             u=rng.random()))
by_direction={
    str(d):dict(catalogs=sorted({r['catalog'] for r in pool if r['direction']==d}),
                switch_cases=sum(r['direction']==d for r in pool))
    for d in (0,1)}
(ROOT/'rare_eligible62.json').write_text(json.dumps(by_direction,indent=2)+'\n')
