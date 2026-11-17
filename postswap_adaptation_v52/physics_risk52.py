"""Blind deployment inputs and held-out losses before consulting any v52 policies."""
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
V50=ROOT.parent/'polarity_calibration_v50'
sys.path.insert(0,str(V50))
from score50 import PRED, components, integrate
EPS=(0.,.25,.5)
DRIFT=(0.,.25,.5,1.)
PRE=(0.,.25,.5,.75,1.)
BUDGET=(0,1,2,4,8)
NREP=256
N=12


def panel(i):
    name=f'panel_{i:02d}.npz'
    with np.load(V50/'forecast50'/name) as f:
        names=list(f['methods'].astype(str))
        p=f['p'][[names.index(key) for key in PRED]]
    with np.load(V50/'worlds'/name) as w:
        surface=w['surface'].astype(str); law=w['law'].astype(str)
        assert np.all(surface[[2,3,0,1]]!=surface) and np.array_equal(law[[2,3,0,1]],law)
        assert int(w['cal_support_panel'])!=int(w['support_panel'])
    with np.load(V50/'labels50'/name) as d: y=d['labels']
    with np.load(V50/'labels50_ultra'/name) as d: yf=d['labels']
    # conditional population expectations, analytic integration over all geometry, noise and 128 objects
    exact=np.empty((2,3,2,5,4,4))
    for resolution,labels in enumerate((y,yf)):
        kernel,gap=components(p,labels)
        for ei,e in enumerate(EPS):
            for event in (0,1):
                for c,k in enumerate(PRE):
                    newk=1-k if event else k
                    risk,gaps=integrate(kernel,gap,0.,e+(1-2*e)*newk)
                    exact[resolution,ei,event,c]=np.stack([risk[1],risk[2],risk[5],gaps[0]],-1)
    # One purchase, one selected forecast, one binary impact for each object, common random numbers.
    rng=np.random.default_rng([52090211,i])
    dims=(4,NREP,128)
    env=np.arange(4)[:,None,None]; obj=np.arange(128)[None,None,:]
    f,h,align,variant,rad,off,noise=(rng.integers(n,size=dims) for n in (4,2,2,2,2,3,4))
    weak=rng.choice(3,size=dims,p=(.5,.25,.25)); opp=np.where(weak,3-weak,0)
    uref,uflip=rng.random((2,*dims))
    fi=(f,h,weak,align,variant,env,obj,noise)
    lp,lm,un=[p[m][fi] for m in (1,2,5)]
    real_gap=np.mean((lp-lm)**2,axis=-1)
    same=y[f,h,weak,rad,off,env,obj];rev=y[f,h,opp,rad,off,env,obj]
    obs=np.empty((3,2,5,4,NREP,4))
    example=None
    for ei,e in enumerate(EPS):
        for event in (0,1):
            for c,k in enumerate(PRE):
                changed=(uref>=(1-k if event else k))^(uflip<e)
                truth=np.where(changed,rev,same)
                obs[ei,event,c]=np.stack([np.mean((v-truth)**2,-1) for v in (lp,lm,un)]+[real_gap],-1)
                if ei==1 and event==1 and c==0:
                    alpha=rng.uniform(size=(4,NREP,1))
                    prob=alpha*lp+(1-alpha)*lm
                    explicit=((prob-truth)**2).mean(-1)
                    a=alpha[:,:,0]
                    lp2,lm2,_,dg=np.moveaxis(obs[ei,event,c],-1,0)
                    np.testing.assert_allclose(explicit,lm2+a*(lp2-lm2)-a*(1-a)*dg,atol=1e-13)
                # K=.5 post reference and no-drift versus drift is a design negative control.
    np.testing.assert_array_equal(obs[:,0,2],obs[:,1,2])
    dest=ROOT/'risk52'/name
    dest.parent.mkdir(exist_ok=True)
    np.savez_compressed(dest,exact=exact,realized=obs,surface=surface,law=law)
    print('risk',i,flush=True)


if __name__=='__main__':
    assert os.getenv('SLURM_JOB_ID')
    for i in range(N):panel(i)
