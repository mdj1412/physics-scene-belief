"""Frozen action rule using only measured pre/post displacement pairs."""
from functools import lru_cache
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
V50 = ROOT.parent / 'polarity_calibration_v50'
sys.path.insert(0, str(V50))
from estimator50 import DEAD_ZONE, MAGNITUDE, STD, PRIORS, RETENTIONS, densities

BUDGET = (0, 1, 2, 4, 8)
KAPPA = (0., .25, .5, .75, 1.)
MODES = ('likelihood', 'vote')
DRIFT_PRIOR = .25
REFERENCE_ERROR_PRIOR = .10


def load_model():
    return np.asarray(json.loads((ROOT/'source_risk52.json').read_text())['coefficients'])


def readable(x,y):
    return (np.abs(x)>=DEAD_ZONE[0]) & (np.abs(y)>=DEAD_ZONE[1])


def likelihood_grid(weak,full,method):
    x,y=np.asarray(weak,dtype=float),np.asarray(full,dtype=float)
    if x.shape!=y.shape or x.ndim<1 or (not np.isfinite(x).all()) or (not np.isfinite(y).all()):
        raise ValueError('expected paired finite observations')
    active=readable(x,y)
    if method=='vote':
        same=(active & ((x>0)==(y>0))).sum(-1)[...,None]
        opposite=active.sum(-1)[...,None]-same
        loglik=same*np.log(RETENTIONS)+opposite*np.log1p(-RETENTIONS)
    elif method=='likelihood':
        u=densities(x,MAGNITUDE[0],STD[0]);v=densities(y,MAGNITUDE[1],STD[1])
        null=.5*u[...,0]*v[...,0]
        keep=null+.25*(u[...,1]*v[...,1]+u[...,2]*v[...,2])
        flip=null+.25*(u[...,1]*v[...,2]+u[...,2]*v[...,1])
        pairs=flip[...,None]+(keep-flip)[...,None]*RETENTIONS
        loglik=np.where(active[...,None],np.log(np.maximum(pairs,np.finfo(float).tiny)),0.).sum(-2)
    else:
        raise ValueError(method)
    return np.exp(loglik-loglik.max(-1,keepdims=True))


def mean_grid(lik):
    mass=lik*PRIORS
    return (mass@RETENTIONS)/mass.sum(-1)


def posterior_single(w,f,method):
    if method=='vote':
        y=np.asarray(f);x=np.asarray(w)
        active=readable(x,y)
        return (1+(active&((x>0)==(y>0))).sum(-1))/(2+active.sum(-1))
    return mean_grid(likelihood_grid(w,f,method))


def decide(old_w,old_f,new_w,new_f,method,model):
    oldlik=likelihood_grid(old_w,old_f,method)
    newlik=likelihood_grid(new_w,new_f,method)
    old=mean_grid(oldlik);new=mean_grid(newlik)
    # Integrate on post-swap reference retention. State under flip = 1 - old k.
    preserve=(1-DRIFT_PRIOR)*oldlik*newlik*PRIORS
    flipped=DRIFT_PRIOR*oldlik[...,::-1]*newlik*PRIORS
    norm=(preserve+flipped).sum(-1)
    post=((preserve+flipped)@RETENTIONS)/norm
    drift=flipped.sum(-1)/norm
    q=REFERENCE_ERROR_PRIOR+(1-2*REFERENCE_ERROR_PRIOR)*post
    # model rows indexed by true weak-to-impact retention endpoint (0,1).
    # Row entries: constant L(p_minus), linear, quadratic, L(unsigned).
    c=model[0]+q[...,None]*(model[1]-model[0])
    def proxy(a):
        return c[...,0]+a*(c[...,1]+a*c[...,2])
    risks=np.stack((proxy(old),proxy(new),c[...,3]),axis=-1)
    if np.shape(new_w)[-1]==0:
        risks[...,1]=np.inf
    # Near-exact ties go to first listed option; no parameter selected on target labels.
    action=np.argmax(risks<=risks.min(-1,keepdims=True)+1e-14,axis=-1).astype(np.int8)
    alpha=np.where(action==0,old,np.where(action==1,new,.5))
    return dict(action=action,alpha=alpha,old=old,new=new,drift_prob=drift,post_target=post,target_alpha=q)


@lru_cache(maxsize=16)
def bank(panel):
    name=f'panel_{panel:02d}.npz'
    with np.load(V50/'cal50'/name) as data, np.load(V50/'worlds'/name) as world:
        w,f=data['weak_mm'],data['full_mm']; surface=world['surface'];law=world['law']
        assert np.all(surface!=surface[[2,3,0,1]]) and np.all(law==law[[2,3,0,1]])
        assert int(world['support_panel']) != int(world['cal_support_panel'])
        assert np.all(np.diff(np.sort(data['cal_object_id'],axis=-1),axis=-1)>0)
    order=np.random.default_rng(520096+panel).permutation(w.shape[-2])
    return w,f,order


def witnesses(panel,k_old_idx,k_new_idx,kpost):
    w,f,order=bank(panel)
    return (w[k_old_idx,[2,3,0,1],:,:4],f[k_old_idx,[2,3,0,1],:,:4],
            w[k_new_idx,:,:,:kpost][:,order],f[k_new_idx,:,:,:kpost][:,order])
