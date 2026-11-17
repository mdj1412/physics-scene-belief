"""Source-trained geometry residual and disjoint-source probability temperatures."""
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parent
QUERY, PROBE = 1, 3


def load_model(path=ROOT/'frozen_model43.json'):
    artifact = json.loads(Path(path).read_text())
    for key in ('mean', 'scale', 'skip', 'w1', 'b1', 'w2'):
        artifact[key] = np.asarray(artifact[key], dtype=np.float64)
    return artifact


def features(base, xx, actions, sensors, height):
    """Only gather chosen sensor entries. No target names, labels or calibration response."""
    base = np.asarray(base)
    if base.ndim != 3 or xx.shape[0:2] != base.shape[:2] or actions.shape != base.shape[:2]:
        raise ValueError('Expected [environment, object, noise] probabilities')
    e, n, k = base.shape
    query = np.zeros((e, n, k, 7))
    probe = np.zeros((e, n, k, 2))
    for j in range(e):
        q, p = actions[j] == QUERY, actions[j] == PROBE
        if not np.all(q|p):
            raise ValueError('Only the two paid actions are supported')
        query[j,q] = sensors[j][QUERY][q]
        probe[j,p] = sensors[j][PROBE][p]
    # A relative profile prevents object translation from hiding slope in local shape.
    centered = (query - query[..., 3:4]) * 1000.
    # Factual camera sees nine coordinate samples at known heights and two velocities.
    factual = np.broadcast_to(xx[:, :, None, :], (e,n,k,xx.shape[-1]))
    factual_geom = (factual[..., :9] - factual[..., 4:5]) * 1000.
    mask = np.broadcast_to((actions == QUERY)[:, :, None, None], (e,n,k,1))
    local = np.stack(((centered[...,4]-centered[...,2])/4.,
                      (centered[...,5]-centered[...,1])/8.,
                      (centered[...,6]-centered[...,0])/12.,
                      (centered[...,2]+centered[...,4])/4.,
                      (centered[...,1]+centered[...,5])/16.),axis=-1)
    raw_logit = logit(np.clip(base, 1.e-5, 1.-1.e-5))
    out = np.concatenate((raw_logit[...,None], mask,
                          np.broadcast_to(np.array(height)*1000., (e,n,k,1)),
                          factual_geom, factual[...,9:], centered, local, probe), axis=-1)
    if not np.all(np.isfinite(out)):
        raise ValueError('Nonfinite paid observation supplied')
    return out


def robust_from_features(artifact, raw):
    x = np.clip((raw - artifact['mean'])/artifact['scale'], -8., 8.)
    residual = (x @ artifact['skip'] + artifact['bias']
                + np.tanh(x @ artifact['w1'] + artifact['b1']) @ artifact['w2'])
    return expit(raw[...,0] + residual)


def robust_probability(artifact, base, xx, actions, obs, height):
    return robust_from_features(artifact, features(base, xx, actions, obs, height))


def temperature(artifact, base, decoder):
    return expit(logit(np.clip(base, 1.e-5, 1.-1.e-5)) / artifact['temperature'][decoder])


def correct(policy, base, decoder, model, physics, gain_table, records):
    """Apply the frozen per-decoder temperature or precisely the historic signed logit rule."""
    if policy == 'zero':
        return base
    if policy == 'temperature':
        return temperature(model, base, decoder)
    if policy == 'fixed':
        z, weight = physics.v38.policy(gain_table, records, 'fixed', True)
    elif policy == 'optimizer':
        x, y, dirs = physics.v38.observations(records)
        z, weight = physics.continuous.source_optimum(gain_table, x, y[:,:3], dirs[:,:3], y[:,3])
    else:
        raise ValueError(policy)
    return physics.v38.apply(base, (z*weight).reshape((-1,) + (1,)*(base.ndim-1)), physics.v38.BETA)
