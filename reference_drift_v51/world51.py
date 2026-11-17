"""Counterfactual sign transfer between independent calibration and deployment specimens.

Array state indexes follow v50: 0=inert, 1=negative, 2=positive.
A replacement event is common to a session, whereas impact disagreement is sampled
independently for each active object, conditional on the current reference sign.
"""
import numpy as np

EPSILON = (0., .1, .25, .5, .75, 1.)
DRIFT = (0., .1, .25, .5, .75, 1.)
KAPPA = (0., .25, .5, .75, 1.)


def paired_calibration_rows(surface, law):
    surface, law = np.asarray(surface), np.asarray(law)
    candidates=(surface[:,None]!=surface[None,:]) & (law[:,None]==law[None,:])
    if not np.all(candidates.sum(1)==1):
        raise ValueError('Each deployment surface must have one opposite-surface partner with the same law')
    index=candidates.argmax(1)
    np.testing.assert_array_equal(index[index],np.arange(len(surface)))
    return index


def signed_state(weak, reverse):
    return np.where((weak!=0)&reverse,3-weak,weak)


def marginal_retention(kappa,epsilon,drift):
    post_reference=kappa*(1.-drift)+(1.-kappa)*drift
    return epsilon+(1.-2.*epsilon)*post_reference


def full_states(weak,kappa,u_weak_to_reference,u_drift,u_reference_to_impact,epsilon,drift):
    """Return observed fixture states. Policies receive only scalar paid displacements.

    Drift complements the post-replacement weak/reference law for the *entire* session.
    A complementary coupling of the two potential reference states is used for pairing.
    """
    if u_drift.shape!=kappa.shape or weak.shape[:-1]!=kappa.shape:
        raise ValueError('Supply exactly one kappa and drift draw per session')
    shifted=u_drift<drift
    ref_reverse=np.logical_xor(u_weak_to_reference>=kappa[...,None],shifted[...,None])
    reference=signed_state(weak,ref_reverse)
    impact=signed_state(reference,u_reference_to_impact<epsilon)
    return reference,impact,shifted
