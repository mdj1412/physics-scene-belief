"""Frozen decision to use fixed impact or the v54 source-risk gate."""
import numpy as np

CROSSING = .4348074444131813
W = np.array([.10, .35, .40, CROSSING, .47, .52, .80])
BUDGETS = (0, 1, 4, 8, 16)
PRIMARY = (1, 5)

def estimate(reference, impact):
    reference, impact = np.asarray(reference), np.asarray(impact)
    if reference.shape != impact.shape or reference.ndim == 0:
        raise ValueError("Reference and impact must be paired along the last axis.")
    valid = (reference != 0) & (impact != 0)
    count = valid.sum(axis=-1)
    flips = (valid & (reference != impact)).sum(axis=-1)
    rate = np.divide(flips, count, out=np.full(count.shape, np.nan), where=count > 0)
    w = np.clip((rate - .125) / .375, 0., 1.)
    return count, flips, rate, w

def choose(reference, impact):
    # Undefined estimate (no readable pair) and exact ties both retain the v54 gate.
    return estimate(reference, impact)[3] > CROSSING
