"""Frozen common-prior acquisition and physics decoding without target metadata."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'estimated_projection_v41'))
import runner41 as r

PAIR = (43031001, 43031002)


def make_bank():
    """Return one (inner, outer) Sobol engine shared across all deployments."""
    return r.v38.Engine(list(PAIR), 0., 0.)


def actions43(xx, engine):
    actions = np.stack([r.v38.v33.decisions(r.v38.batch_gap(engine, x)) for x in xx])
    if not np.all((actions == r.P).sum(axis=1) == 13):
        raise ValueError('Expected 13 physical probes per 128 objects')
    return actions


def make_template(engine, height, direction=None):
    return r.templates([engine.inner], height, direction)[0]


def decode43(engine, template, xx, actions, obs):
    return r.decode([engine.inner] * len(xx), [template] * len(xx), xx, actions, obs)


def paid_sensors(obs, actions):
    # Unselected entries never reach an operating decoder, even as unused arrays.
    paid = []
    for e, sensors in enumerate(obs):
        current = {}
        for a in (r.Q, r.P):
            values = np.array(sensors[a], copy=True)
            values[actions[e] != a] = np.nan
            current[a] = values
        paid.append(current)
    return paid
