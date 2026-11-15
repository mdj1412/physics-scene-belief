"""Evaluation-only sensor and intervention generator. No target-law tag reaches the policy."""
from pathlib import Path
import sys
import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'learned_probe_value_v30'))
from contact30 import (calibration_data, evaluation_data, factual, collision, draw_theta, means,
                       profile, sigmas, stream, t, torch, FY, VYS, QUERY_Y, QUERY_INCOMING,
                       FACTUAL_INCOMING, PROBE_INCOMING, Q, P)

LAWS = ('split_static', 'speed_weak', 'directional')


def directional(theta, height, incoming):
    _, _, slope, curvature, mu, restitution = theta.unbind(-1)
    s = slope + curvature * height / .025
    normalizer = (1 + s.square()).sqrt()
    vn = (1 - s * incoming) / normalizer
    vt = (s + incoming) / normalizer
    impulse = (1 + restitution) * vn.clamp_min(0)
    # Positive coefficient in both directions. Negative vt persists for calibration probes.
    coefficient = mu * (1 + .23 * torch.tanh(vt / .28))
    stick = vt.abs() <= coefficient * impulse
    friction_impulse = torch.where(stick, vt, vt.sign() * coefficient * impulse)
    velocity = torch.stack((1 - (impulse + friction_impulse*s)/normalizer,
                            incoming + (impulse*s-friction_impulse)/normalizer), -1)
    return velocity, stick


def signals(theta):
    velocity, sticks = directional(theta, 0, FACTUAL_INCOMING)
    probe, sliding = directional(theta, 0, PROBE_INCOMING)
    assert bool(sticks.all()) and not bool(sliding.any())
    return {0:torch.cat((profile(theta,FY),velocity),-1), Q:profile(theta,VYS[0]), P:probe}


def calibration(seed,alpha,law):
    if law != 'directional':
        return calibration_data(seed,alpha,law)
    theta=draw_theta(1024,seed,10,alpha)
    signal=signals(theta)
    x=signal[0]+t(stream(seed,11).normal(size=(1024,11)))*sigmas()[0]
    ids=stream(seed,98).permutation(1024)[:8]
    y=signal[P][ids]+t(stream(seed,99).normal(size=(8,2)))*sigmas()[P]
    return x.cpu().numpy(),ids,y.cpu().numpy()


def deployment_x(seed,alpha,law):
    if law != 'directional':
        return factual(128,seed,20,alpha,law)
    signal=signals(draw_theta(128,seed,20,alpha))[0]
    return (signal+t(stream(seed,21).normal(size=(128,11)))*sigmas()[0]).cpu().numpy()


def deployment_truth(seed,alpha,law):
    if law != 'directional':
        label,nominal,obs,_=evaluation_data(128,seed,alpha,law)
        return label,nominal,obs
    theta=draw_theta(128,seed,20,alpha)
    label=directional(theta,QUERY_Y,QUERY_INCOMING)[1]
    nominal=collision(theta,QUERY_Y,QUERY_INCOMING,'coulomb')[1]
    sig=signals(theta)
    obs={a:(sig[a][:,None]+t(stream(seed,40+a).normal(size=(128,32,sig[a].shape[-1])))*sigmas()[a]).cpu().numpy()
         for a in (Q,P)}
    return label.cpu().numpy(),nominal.cpu().numpy(),obs
