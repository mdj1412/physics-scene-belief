"""Label-blind environment-level choice among zero, fixed, and v38 correction."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
BETA = .8289760825845313
FIXED = .325
METHODS = ('zero', 'fixed', 'optimizer')


def load_selector():
    return json.loads((ROOT/'selector_model42.json').read_text())


def features(base, z, w, actions, first3_y, fourth_y, dirs):
    """Each row uses a single observed deployment batch and four calibration responses."""
    p=np.asarray(base,dtype=np.float64)
    if p.ndim==2:
        p=p[:,:,None]
    z=np.asarray(z)
    w=np.asarray(w)
    n=p.shape[1]
    is_probe=(np.asarray(actions)==3)[:,:,None]
    prop=is_probe.sum(axis=1)
    mean=p.mean(axis=1)
    pv=(p*(1-p)).mean(axis=1)
    probe=(p*is_probe).sum(axis=1)/np.maximum(prop,1)
    query=(p*(~is_probe)).sum(axis=1)/np.maximum(n-prop,1)
    side=(p>.5).mean(axis=1)
    cov=(p*(1-p)*(p-.5)).mean(axis=1)/np.maximum(pv,1e-15)
    far=np.asarray(first3_y); near=np.asarray(fourth_y)
    dirs=np.asarray(dirs)
    pos=dirs>0; neg=dirs<0
    vx=far[:,:,0]; vy=far[:,:,1]
    far_asym=(vy*pos).sum(1)/pos.sum(1)+(vy*neg).sum(1)/neg.sum(1)
    signed=(vy*dirs).mean(1)
    def spread(x): return np.broadcast_to(x[:,None],mean.shape)
    cue,mag,gain=spread(z),spread(abs(z)),spread(w)
    occupancy=[mean-.5,pv,probe-query,side-.5,cov,cue,mag,gain,cue*gain,
               cue*(mean-.5),cue*(probe-query),gain*(mean-.5),cue*cov,(mean-.5)**2]
    contact=[spread(vx.mean(1)),spread(signed),spread(far_asym),
             spread(near[:,0]),spread(near[:,1]),cue*spread(near[:,1]),
             (mean-.5)*spread(near[:,1])]
    return np.stack(occupancy+contact,axis=-1)


def moments(p,z,scale,labels=None):
    """Exact step moments for q = logit_shift(p); target uses source labels only."""
    p=np.asarray(p)
    if p.ndim==2:p=p[:,:,None]
    k=(BETA*np.asarray(z)*scale)[:,None,None]
    ex=np.expm1(k)
    ratio=np.divide(ex,k,out=np.ones_like(k),where=k!=0)
    # d=q-p=k*h. Mean h ≥ 0, including continuous extension at k=0.
    h=np.maximum(p*(1-p),0)*ratio/(1+p*ex)
    hd=h.mean(1)
    step=k[:,0,:]*hd
    sq=(k[:,0,:]**2)*(h*h).mean(1)
    if labels is None: return step,sq
    target=(h*(labels[:,:,None]-p)).mean(1)/np.maximum(hd,1.e-15)
    return step,sq,target


def select(model,base,z,w,actions,first3_y,fourth_y,dirs):
    x=features(base,z,w,actions,first3_y,fourth_y,dirs)
    if model['mode']=='zero':
        chosen=np.zeros(x.shape[:2],dtype=np.int8)
    else:
        standardized=np.clip((x[...,model['indices']]-model['center'])/model['scale'],-5,5)
        residual=np.einsum('erf,fa->era',standardized,np.asarray(model['coef']))+model['intercept']
        scores=[np.zeros(x.shape[:2])]
        for i,weight in enumerate((FIXED,w)):
            step,sq=moments(base,z,weight)
            scores.append(sq-2*step*residual[...,i]+model['threshold'])
        chosen=np.argmin(np.stack(scores,axis=-1),axis=-1).astype(np.int8)
    return chosen[:,0] if np.ndim(base)==2 else chosen
