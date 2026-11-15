"""Height-localized directional friction, using the v35 shape and sensor models."""
import sys
from pathlib import Path
import numpy as np

PROJECT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(PROJECT/'frozen_direction_v35'))
import target35 as v35
from target35 import (torch, t, sigmas, stream, draw_object, contour, perturbation,
                      FY, VYS, Q, P, QUERY_Y, QUERY_INCOMING, PROBE_INCOMING)
MATCH_HEIGHT=.023
MATCH_SPEED=.50
STRENGTH={'strong':1.,'weak':.3,'null':0.,'global_null':0.}

def impact(theta,height,speed,law,regime):
    obj,surface,sign=law
    _,_,slope,curve,mu,restitution=theta.unbind(-1)
    slope=slope+curve*height/.025+perturbation(height,obj)[0]
    norm=(1+slope.square()).sqrt()
    vt=(speed+slope)/norm
    jn=(1+restitution)*((1-slope*speed)/norm).clamp_min(0.)
    frac=(.21*(2/np.pi)*torch.atan(vt/.40) if surface=='atan'
          else .16*vt/(.30**2+vt.square()).sqrt())
    u=min(abs(height)/MATCH_HEIGHT,1.)
    amplitude=0. if regime=='global_null' else 1+(STRENGTH[regime]-1)*u*u*(3-2*u)
    effective=mu*(1+sign*amplitude*frac)
    stick=vt.abs()<=effective*jn
    jt=torch.where(stick,vt,vt.sign()*effective*jn)
    return torch.stack((1.-(jn+jt*slope)/norm,speed+(jn*slope-jt)/norm),-1),stick

def calibration(seed,alpha,law,regime,start):
    theta=draw_object(1024,seed,10,alpha,law)
    x=v35.factual(theta,seed,11,law)
    ids=stream(seed,98).permutation(1024)[:4]
    dirs=v35.directions('alternating',start,4)
    far,far_stick=impact(theta[ids],0,t(dirs)*abs(PROBE_INCOMING),law,regime)
    assert not bool(far_stick.any())
    near,near_stick=impact(theta[ids[-1:]],MATCH_HEIGHT,MATCH_SPEED,law,regime)
    noise=t(stream(seed,99).normal(size=(4,2)))*sigmas()[P]
    far_y=(far+noise).cpu().numpy()
    near_y=far_y.copy()
    near_y[-1]=(near[0]+noise[-1]).cpu().numpy()
    return ((x,ids,far_y,dirs),(x,ids,near_y,dirs),
            (theta[ids[-1]].cpu().numpy(),near[0].cpu().numpy(),bool(near_stick[0])))

def deployment(seed,alpha,law,regime):
    theta=draw_object(128,seed,20,alpha,law)
    x=v35.factual(theta,seed,21,law)
    y=impact(theta,QUERY_Y,QUERY_INCOMING,law,regime)[1]
    probe,stick=impact(theta,0,PROBE_INCOMING,law,regime)
    assert not bool(stick.any())
    sensors={Q:contour(theta,VYS[0],law),P:probe}
    obs={a:(m[:,None]+t(stream(seed,40+a).normal(size=(128,32,m.shape[1])))*sigmas()[a]).cpu().numpy()
         for a,m in sensors.items()}
    return x,y.cpu().numpy(),theta.cpu().numpy(),obs
