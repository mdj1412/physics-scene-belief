"""Second, independent deployment noise seed; unchanged four-pair policies."""
import os
import json
import numpy as np
from score52 import ROOT,P,N,NREP,EPS,PRE,GMEAN,METHODS,weighted_cdf_inverse
from physics_risk52 import V50

assert os.getenv('SLURM_JOB_ID')
cells=np.empty((N,3,2,5,4,NREP,4),np.float32)
for i in range(N):
    name=f'panel_{i:02d}.npz'; rng=np.random.default_rng([520271130,i])
    with np.load(ROOT/'decisions52'/name) as bank:
        alpha=bank['alpha'][:,:,P]; branch=bank['action'][:,:,P]; vote=bank['new'][:,:,P,...,1]; joint=bank['target_alpha'][:,:,P,...,0]
    with np.load(V50/'forecast50'/name) as bank:
        p=bank['p'][[list(bank['methods'].astype(str)).index(k) for k in ('local_plus','local_minus','local_unsigned_fit')]]
    with np.load(V50/'labels50'/name) as bank: labels=bank['labels']
    shape=(4,NREP,128); env=np.arange(4)[:,None,None]; obj=np.arange(128)[None,None,:]
    f,h,align,variant,rad,x,noise=[rng.integers(s,size=shape) for s in (4,2,2,2,2,3,4)]
    weak=rng.choice(3,size=shape,p=(.5,.25,.25)); flip=np.where(weak,3-weak,0)
    plus,minus,uns=p[:,f,h,weak,align,variant,env,obj,noise]
    y0=labels[f,h,weak,rad,x,env,obj]; y1=labels[f,h,flip,rad,x,env,obj]
    u,v=rng.random((2,*shape))
    for j,event,c in np.ndindex(3,2,5):
        after=4-c if event else c; actual=np.where((u>=PRE[after])^(v<EPS[j]),y1,y0)
        for m in range(4):
            if m<2:
                a=alpha[c,after,...,m,None]; is_unsigned=branch[c,after,...,m,None]==2
            else:
                a=(vote if m==2 else joint)[c,after,...,None]; is_unsigned=False
            forecast=np.where(is_unsigned,uns,a*plus+(1-a)*minus)
            cells[i,j,event,c,:,:,m]=np.square(forecast-actual).mean(-1)
q=[]
for m in range(4):
    sorted_events=[np.sort(cells[:,:,ev,...,m].ravel().astype(float)) for ev in range(2)]
    q.append(weighted_cdf_inverse(sorted_events,(1-GMEAN,GMEAN)))
out=dict(methods=[METHODS[k] for k in (0,1,2,6)],p90=q,selector_minus=(np.array(q[0])-q[1:]).tolist(),
         scope='Secondary independent deployment noise/realized outcomes on same supports and same paid prefixes; not new physical materials.')
(ROOT/'tail_repeat52.json').write_text(json.dumps(out,indent=1))
