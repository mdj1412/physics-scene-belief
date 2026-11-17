"""Secondary deployment audit: one observed calibration prefix per session and one actual binary label per object."""
import json
import os
import numpy as np
from score50 import ROOT, PRED, METHODS, KAPPA, RHO, interval
from estimator50 import infer

assert os.getenv('SLURM_JOB_ID')
rng=np.random.default_rng(50116622)
# Assign one hidden environment to each cohort, balanced as far as 48/15 permits.
schedule=rng.permutation(np.r_[np.tile(np.arange(15),3),rng.choice(15,3,replace=False)]).reshape(12,4)
risk=np.zeros((12,4,10))
checks=[]
for i in range(12):
    tag=f'panel_{i:02d}.npz'
    with np.load(ROOT/'forecast50'/tag) as ar,np.load(ROOT/'labels50'/tag) as yl,np.load(ROOT/'cal50'/tag) as ca:
        bank=ar['p'][[list(ar['methods'].astype(str)).index(m) for m in PRED]]
        outcomes=yl['labels']; weak=ca['weak_mm']; full=ca['full_mm']
    for e,cell in enumerate(schedule[i]):
        ir,ic=divmod(cell,5)
        rho,kappa=RHO[ir],KAPPA[ic]
        rep=rng.integers(256)
        a=float(infer(weak[ic,e,rep,:4],full[ic,e,rep,:4]))
        obj=np.arange(128)
        f,h=rng.integers(4,size=128),rng.integers(2,size=128)
        s=rng.choice(3,p=(.5,.25,.25),size=128)
        sfull=np.where((s!=0)&(rng.random(128)>=kappa),3-s,s)
        off=rng.choice(3,p=(.5,.25,.25),size=128)
        remote=np.where((h==1)|(rng.random(128)<rho),s,off)
        align,variant,rad=rng.integers(2,size=(3,128))
        noise,offset=rng.integers(4,size=128),rng.integers(3,size=128)
        truth=outcomes[f,h,sfull,rad,offset,e,obj]
        probs=[]
        for m in range(6):
            state=remote if m in (3,4) else s
            probs.append(bank[m,f,h,state,align,variant,e,obj,noise])
        orig,lp,lm,fp,fm,uns=probs
        deploy=(lp if a>=.5 else lm,lp,fp,fp if a>=.5 else fm,uns,(lp+lm)/2,
                a*lp+(1-a)*lm,a*fp+(1-a)*fm,kappa*lp+(1-kappa)*lm,orig)
        risk[i,e]=[((pred-truth)**2).mean() for pred in deploy]
        checks.append([i,e,int(cell),int(rep),a,int(truth.sum())])
v=risk.mean(1)
out=dict(methods=METHODS,n_objects=6144,n_environments=48,n_cal_pairs=192,counts=np.bincount(schedule.ravel(),minlength=15),
         loss=v.mean(0),differences=interval(v[:,0,None]-v[:,1:6],5),soft_minus_hard=interval(v[:,6]-v[:,0]),
         draws=checks,scope='Secondary realized allocation: a single paid prefix per environment and one binary outcome per new object. More variable and slightly unequal cell weights; no retuning.')
(ROOT/'rollout50.json').write_text(json.dumps(out,default=lambda x:np.asarray(x).tolist(),indent=1))
