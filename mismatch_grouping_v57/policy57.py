"""Fixed acquisition-menu inference from eight paid signed observations."""
import numpy as np

MIXTURE = np.array([[.42, 0., .58], [0., .84, .16]])
SECOND_MOMENT = np.array([.145, .0925])
MEAN = .29
# gate minus fixed-I mean Brier in A, B from v56 identification56.json.
SOURCE_GAPS = np.array([4.175119807325477e-05, -8.417543970276335e-05])
SOURCE_CUTOFF = np.log(-SOURCE_GAPS[1]/SOURCE_GAPS[0])
# Singletons and empty groups have identical likelihoods in both regimes.
# For a complete pair, probabilities below are for 0,1,2 flips respectively.
PAIR_COUNTS = np.stack((1-2*MEAN+SECOND_MOMENT,2*(MEAN-SECOND_MOMENT),SECOND_MOMENT),axis=1)
LLR = np.log(PAIR_COUNTS[0]/PAIR_COUNTS[1])

def log_odds(reference, impact, design):
    reference,impact=np.asarray(reference),np.asarray(impact)
    if reference.shape != impact.shape or reference.shape[-1:]!=(8,):
        raise ValueError("Need eight paired reference and impact readings.")
    if design not in ('8x1','4x2'):
        raise ValueError("Unknown pilot design.")
    valid=(reference!=0)&(impact!=0)
    mismatch=valid&(reference!=impact)
    if design=='8x1':
        # Each single active comparison has Bernoulli(.29) under both hypotheses.
        return np.zeros(reference.shape[:-1],dtype=float)
    pair_shape=reference.shape[:-1]+(4,2)
    both=valid.reshape(pair_shape).all(-1)
    count=mismatch.reshape(pair_shape).sum(-1)
    return np.where(both,LLR[count],0.).sum(-1)

def choose(reference, impact, design='4x2', rule='map'):
    """Return True to deploy fixed 3R+I. A tie retains the v54 gate."""
    odds=log_odds(reference,impact,design)
    if rule=='map':
        return odds > 0.
    if rule=='risk':
        return odds > SOURCE_CUTOFF
    raise ValueError("Unknown decision rule.")
