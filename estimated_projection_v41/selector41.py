"""Frozen environment-level selection between v38 corrections using purchased observations."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
BETA=.8289760825845313
FIXED=.325


def load_selector():
    return json.loads((ROOT/'selector_model41.json').read_text())


def design(base,z,w,actions,first3_y,fourth_y,dirs):
    """Features E,R,K and the exact positive weight in the paired Brier difference."""
    p=np.asarray(base,np.float64)
    if p.ndim==2:
        p=p[:,:,None]
    z=np.asarray(z,np.float64)
    w=np.asarray(w,np.float64)
    trial=z*(w-FIXED)*BETA
    k=trial[:,None,None]
    ex=np.expm1(k)
    ratio=np.divide(ex,k,out=np.ones_like(k),where=k!=0)
    odds=np.exp(BETA*FIXED*z)[:,None,None]
    pf=(p*odds)/(1-p+p*odds)
    h=np.maximum(0.,pf*(1-pf))*ratio/(1+pf*ex)
    mid=pf+.5*k*h
    hbar=h.mean(1)
    den=np.maximum(h.sum(1),1e-15)
    shape=hbar.shape
    def broad(x):
        return np.broadcast_to(np.asarray(x).reshape(len(z),1),shape)
    branch=np.asarray(actions)==3
    def mean(selected):
        return (p*selected[:,:,None]).sum(1)/np.maximum(selected.sum(1)[:,None],1)
    p_q=mean(~branch);p_probe=mean(branch)
    p_all=p.mean(1); mq=(h*mid).sum(1)/den
    side=(h*((p>.5).astype(np.float64))).sum(1)/den
    skew=(h*(p-.5)**2).sum(1)/den
    fraction_probe=(h*branch[:,:,None]).sum(1)/den
    cue=broad(z)
    wm=broad(w-.325)
    vy=np.asarray(first3_y);dr=np.asarray(dirs)
    y4=np.asarray(fourth_y)
    signed_y=vy[:,:,1]*dr
    far_mean=signed_y.mean(1)
    far_gap=(vy[:,:,1]*(dr>0)).sum(1)/(dr>0).sum(1)+(vy[:,:,1]*(dr<0)).sum(1)/(dr<0).sum(1)
    passive=[p_all-.5,p_q-p_probe,mq-.5,hbar,side-.5,skew,fraction_probe]
    calibration=[cue,wm,cue*wm,broad(y4[:,0]),broad(y4[:,1]),broad(far_mean),broad(far_gap),broad(vy[:,:,0].mean(1))]
    interaction=[cue*(p_all-.5),cue*(p_q-p_probe),cue*(mq-.5),wm*(mq-.5),
                 broad(y4[:,1])*(p_all-.5),cue*broad(y4[:,1]),(p_all-.5)**2,
                 (mq-.5)**2]
    return np.stack(passive+calibration+interaction,axis=-1),trial[:,None]*hbar,h,mid


def predict(model,base,z,w,actions,first3_y,fourth_y,dirs):
    features,weight,_,_=design(base,z,w,actions,first3_y,fourth_y,dirs)
    idx=model['feature_index']
    scaled=np.clip((features[:,:,idx]-np.asarray(model['center']))/np.asarray(model['scale']),-5,5)
    residual=np.einsum('erk,k->er',scaled,np.asarray(model['coef']))+model['intercept']
    return -2*weight*residual


def select(model,base,z,w,actions,first3_y=None,fourth_y=None,dirs=None):
    score=predict(model,base,z,w,actions,first3_y,fourth_y,dirs)
    return (score[:,0] if np.ndim(base)==2 else score)<0
