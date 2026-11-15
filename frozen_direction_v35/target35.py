"""Prospectively specified continuous contours and directional friction laws."""
import sys
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'bidirectional_probe_v34'))
from target34 import directions
from target33 import (draw_theta, profile, t, torch, sigmas, stream, FY, VYS,
                      QUERY_Y, QUERY_INCOMING, FACTUAL_INCOMING, PROBE_INCOMING, Q, P)

OBJECTS = ('sinusoid', 'quartic')
SURFACES = ('atan', 'algebraic', 'isotropic')
FAMILIES = [(o, s) for o in OBJECTS for s in SURFACES[:2]] + [(o,'isotropic') for o in OBJECTS]

def draw_object(n, seed, lane, alpha, law):
    obj, surface, sign = law
    assert obj in OBJECTS and surface in SURFACES and sign in (-1,1)
    theta = draw_theta(n, seed, lane, alpha).clone()
    theta[:,3] *= {'sinusoid':.95, 'quartic':1.05}[obj]
    theta[:,4] *= {'atan':1.02, 'algebraic':.99, 'isotropic':1.0}[surface]
    return theta

def perturbation(y, obj):
    """Dimensionless slope and its integral in metres, both zero at y=0."""
    u = t(y)/.025
    if obj == 'sinusoid':
        return -.045*torch.sin(u), -.045*.025*(1.-torch.cos(u))
    return .040*u.pow(3), .040*.025*u.pow(4)/4.

def contour(theta, y, law):
    return profile(theta,y) + perturbation(y,law[0])[1]

def contact(theta, height, incoming, law, *, isotropic=False):
    obj, surface, sign = law
    _, _, slope, curve, friction, restitution = theta.unbind(-1)
    local = slope + curve*(height/.025) + perturbation(height,obj)[0]
    norm = (1+local.square()).sqrt()
    vn, vt = (1-local*incoming)/norm, (incoming+local)/norm
    jn = (1+restitution)*vn.clamp_min(0)
    if isotropic or surface == 'isotropic':
        mu = friction
    elif surface == 'atan':
        mu = friction*(1+sign*.21*(2/np.pi)*torch.atan(vt/.40))
    else:
        mu = friction*(1+sign*.16*vt/(.30**2+vt.square()).sqrt())
    stick = vt.abs() <= mu*jn
    jt = torch.where(stick, vt, vt.sign()*mu*jn)
    return torch.stack((1.-(jn+jt*local)/norm, incoming+(jn*local-jt)/norm),-1), stick

def factual(theta, seed, lane, law):
    v, stick = contact(theta,0,FACTUAL_INCOMING,law)
    assert bool(stick.all())
    signal = torch.cat((contour(theta,FY,law),v),-1)
    return (signal+t(stream(seed,lane).normal(size=signal.shape))*sigmas()[0]).cpu().numpy()

def calibration(seed, alpha, law, schedule, start):
    theta = draw_object(1024,seed,10,alpha,law)
    x = factual(theta,seed,11,law)
    ids = stream(seed,98).permutation(1024)[:4]
    signs = directions(schedule,start,4)
    signal,stick = contact(theta[ids],0,t(signs)*abs(PROBE_INCOMING),law)
    assert not bool(stick.any())
    # v34 draws eight rows; the first four remain identical.
    y = signal + t(stream(seed,99).normal(size=(4,2)))*sigmas()[P]
    return x,ids,y.cpu().numpy(),signs

def deployment_x(seed,alpha,law):
    return factual(draw_object(128,seed,20,alpha,law),seed,21,law)

def deployment_truth(seed,alpha,law):
    theta = draw_object(128,seed,20,alpha,law)
    _,label = contact(theta,QUERY_Y,QUERY_INCOMING,law)
    probe,stick = contact(theta,0,PROBE_INCOMING,law)
    assert not bool(stick.any())
    signals = {P:probe, Q:contour(theta,VYS[0],law)}
    obs = {a:(val[:,None]+t(stream(seed,40+a).normal(size=(128,32,val.shape[-1])))*sigmas()[a]).cpu().numpy()
           for a,val in signals.items()}
    return label.cpu().numpy(),theta.cpu().numpy(),obs
