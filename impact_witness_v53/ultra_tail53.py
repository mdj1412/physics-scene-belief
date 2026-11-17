"""Second-resolution sensitivity using identical observed policies and identical held-out draws."""
import json
import os
import numpy as np
from score53 import ROOT,V50,N,R,E,G,KAPPA,quantiles,bootstrap,quadratic
assert os.getenv('SLURM_JOB_ID')
names=['impact_joint','impact_selector','impact_vote_selector','impact_post_vote','reference4_joint']
rows=[]; unique_n=unique_flips=0
for i in range(N):
    file=f'panel_{i:02d}.npz'
    with np.load(ROOT/'decisions53'/file) as d:
        ix=[list(d['methods'].astype(str)).index(n) for n in names]
        a=d['alpha'][...,ix]; u=d['unsigned'][...,ix]
    with np.load(V50/'forecast50'/file) as f:
        p=f['p'][[list(f['methods'].astype(str)).index(key) for key in ('local_plus','local_minus','local_unsigned_fit')]]
    with np.load(V50/'labels50_ultra'/file) as f,np.load(V50/'labels50'/file) as pf:
        labels=f['labels']; yprimary=pf['labels']
        mask=np.ones(labels.shape[:3],bool);mask[1:,:,0]=False
        unique_n+=labels[mask].size; unique_flips+=(labels[mask]!=yprimary[mask]).sum()
    rng=np.random.default_rng([52090211,i]) # same deployment draw as physics_risk52.panel
    shape=(4,R,128);env=np.arange(4)[:,None,None];obj=np.arange(128)[None,None,:]
    family,h,axis,var,radius,off,noise=[rng.integers(k,size=shape) for k in (4,2,2,2,2,3,4)]
    w=rng.choice(3,size=shape,p=(.5,.25,.25));rev=np.where(w,3-w,0)
    plus,minus,blind=p[:,family,h,w,axis,var,env,obj,noise]
    y=labels[family,h,w,radius,off,env,obj];z=labels[family,h,rev,radius,off,env,obj]
    pol,err=rng.random((2,*shape));gaps=((plus-minus)**2).mean(-1)
    real=np.empty_like(a,dtype=np.float32)
    for e,event,k in np.ndindex(3,2,5):
        truth=np.where((pol>=KAPPA[4-k if event else k])^(err<E[e]),z,y)
        base=np.stack([((b-truth)**2).mean(-1) for b in (plus,minus,blind)]+[gaps],-1)
        real[e,event,k]=quadratic(base[...,None,:],a[e,event,k],u[e,event,k])
    rows.append(real)
ss=np.stack(rows); p=quantiles(ss,G.mean()); boot=bootstrap(ss);d=boot[:,0,None]-boot[:,1:3]
out=dict(methods=names, p90=p, ultra_primary_q90=np.column_stack((p[0]-p[1:3],np.quantile(d,[.025/4,1-.025/4],axis=0).T)),
         nonduplicated_trajectories=int(unique_n), nonduplicated_switches=int(unique_flips), nonduplicated_switch_rate=unique_flips/unique_n)
(ROOT/'ultra_tail53.json').write_text(json.dumps(out,default=lambda x:np.asarray(x).tolist(),indent=1));print(out)
