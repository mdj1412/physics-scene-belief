"""Prespecified held-out object contours and orientation-dependent surface friction."""
from pathlib import Path
import sys
import numpy as np

BASE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(BASE/'learned_probe_value_v30'))
from contact30 import (draw_theta, profile, sigmas, stream, t, torch, FY, VYS, QUERY_Y,
                       QUERY_INCOMING, FACTUAL_INCOMING, PROBE_INCOMING, Q, P)

# Per-family geometry multiplier and signed change in local slope at |y|=25 mm.
OBJECTS={'rounded':(.90,-.060),'ridged':(1.10,.060)}
# Reference friction multiplier, directional amplitude, tangential velocity scale.
SURFACES={'fine':(.98,.17,.34),'coarse':(1.04,.22,.50)}
LAWS=tuple(f'{o}_{s}_{d}' for o in OBJECTS for s in SURFACES for d in ('plus','minus'))

def split(law):
    obj,surface,direction=law.split('_')
    return OBJECTS[obj],SURFACES[surface],{'plus':1.,'minus':-1.}[direction]

def draw_object(n,seed,lane,alpha,law):
    theta=draw_theta(n,seed,lane,alpha).clone()
    (scale,_),(friction,_,_),_=split(law)
    theta[:,3]*=scale
    theta[:,4]*=friction
    return theta

def contour(theta,y,law):
    (_,d),_,_=split(law)
    y=t(y)
    return profile(theta,y)+d*y.pow(3)/(3*.025**2)

def contact(theta,height,incoming,law,*,isotropic=False):
    (_,d),(_,amplitude,tau),direction=split(law)
    _,_,slope,curve,friction,restitution=theta.unbind(-1)
    local=slope+curve*(height/.025)+d*(height/.025)**2
    norm=(1+local.square()).sqrt()
    vn=(1-local*incoming)/norm
    vt=(incoming+local)/norm
    normal_impulse=(1+restitution)*vn.clamp_min(0)
    coefficient=friction if isotropic else friction*(1+direction*amplitude*torch.tanh(vt/tau))
    stick=vt.abs()<=coefficient*normal_impulse
    impulse=torch.where(stick,vt,vt.sign()*coefficient*normal_impulse)
    velocity=torch.stack((1-(normal_impulse+impulse*local)/norm,
                          incoming+(normal_impulse*local-impulse)/norm),dim=-1)
    return velocity,stick

def signals(theta,law):
    v,stick=contact(theta,0,FACTUAL_INCOMING,law)
    p,probe_stick=contact(theta,0,PROBE_INCOMING,law)
    assert bool(stick.all()) and not bool(probe_stick.any()), law
    return {0:torch.cat((contour(theta,FY,law),v),dim=-1),Q:contour(theta,VYS[0],law),P:p}

def calibration_pool(seed,alpha,law):
    signal=signals(draw_object(1024,seed,10,alpha,law),law)
    x=signal[0]+t(stream(seed,11).normal(size=(1024,11)))*sigmas()[0]
    return x.cpu().numpy(),signal[P].cpu().numpy()

def calibration(seed,alpha,law):
    x,p=calibration_pool(seed,alpha,law)
    ids=stream(seed,98).permutation(1024)[:8]
    y=p[ids]+stream(seed,99).normal(size=(8,2))*sigmas()[P].cpu().numpy()
    return x,ids,y

def deployment_x(seed,alpha,law):
    theta=draw_object(128,seed,20,alpha,law)
    sig=signals(theta,law)[0]
    return (sig+t(stream(seed,21).normal(size=(128,11)))*sigmas()[0]).cpu().numpy()

def deployment_truth(seed,alpha,law):
    theta=draw_object(128,seed,20,alpha,law)
    label=contact(theta,QUERY_Y,QUERY_INCOMING,law)[1]
    neutral=contact(theta,QUERY_Y,QUERY_INCOMING,law,isotropic=True)[1]
    sig=signals(theta,law)
    obs={a:(sig[a][:,None]+t(stream(seed,40+a).normal(size=(128,32,sig[a].shape[-1])))*sigmas()[a]).cpu().numpy()
         for a in (Q,P)}
    return label.cpu().numpy(),neutral.cpu().numpy(),obs
