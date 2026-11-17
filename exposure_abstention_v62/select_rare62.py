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
maxcases=min(16,sum(x>0 for x in eligible.values()))
selected=[]
cntb=np.zeros(8,int); cntd=np.zeros(4,int); cntdir=np.zeros(2,int)
for j in range(maxcases):
    counts=lambda a:(cntdir[a['direction']],cntb[a['board']],cntd[a['cell']],a['u'])
    case=min(pool,key=counts)
    selected.append(case)
    cntb[case['board']]+=1;cntd[case['cell']]+=1;cntdir[case['direction']]+=1
    pool=[a for a in pool if a['catalog']!=case['catalog']]
out=dict(eligible_by_catalog=eligible,board_counts=cntb.tolist(),cell_counts=cntd.tolist(),
         direction_counts=cntdir.tolist(),records=selected)
(ROOT/'rare_selection62.json').write_text(json.dumps(out,indent=2)+'\n')
print({a['catalog']:a['episode'] for a in selected},flush=True)
