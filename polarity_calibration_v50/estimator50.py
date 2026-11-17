"""Estimate a session retention probability from observable paired displacements in mm."""
import numpy as np

# Frozen design values based on the preceding weak/load instrument specification.
MAGNITUDE = (.084, .420)
STD = (.035, .075)
DEAD_ZONE = (.018, .120)
NODES, QUADRATURE_WEIGHTS = np.polynomial.legendre.leggauss(16)
RETENTIONS = .5 * (NODES + 1.)
PRIORS = .5 * QUADRATURE_WEIGHTS


def densities(observed, magnitude, noise_sd):
    """Per-reading likelihoods up to a common factor for zero, positive and negative."""
    states = np.array([0., 1., -1.]) * magnitude
    logs = -.5 * ((observed[..., None] - states) / noise_sd) ** 2
    return np.exp(logs - logs.max(axis=-1, keepdims=True))


def infer(weak_mm, full_mm):
    """Posterior mean kappa, integrating out inactive state and both deformation signs.

    Arrays have matching dimensions (..., paid_pairs). Multiple sessions can be passed
    in leading axes. A hard flip uses the strict criterion infer(...) < 0.5.
    """
    weak, full = np.asarray(weak_mm, dtype=float), np.asarray(full_mm, dtype=float)
    if weak.shape != full.shape or weak.ndim < 1:
        raise ValueError("expected two arrays with matching (..., paid_pairs) shape")
    if weak.shape[-1] == 0:
        return np.full(weak.shape[:-1], .5)
    if not np.all(np.isfinite(weak)) or not np.all(np.isfinite(full)):
        raise ValueError("paid pairs must be finite")
    x = densities(weak, MAGNITUDE[0], STD[0])
    y = densities(full, MAGNITUDE[1], STD[1])
    null = .5*x[..., 0]*y[..., 0]
    keep = null + .25*(x[..., 1]*y[..., 1] + x[..., 2]*y[..., 2])
    flip = null + .25*(x[..., 1]*y[..., 2] + x[..., 2]*y[..., 1])
    readable = (abs(weak) >= DEAD_ZONE[0]) & (abs(full) >= DEAD_ZONE[1])
    keep = np.where(readable, keep, 1.)
    flip = np.where(readable, flip, 1.)
    likelihood = keep[..., None]*RETENTIONS + flip[..., None]*(1.-RETENTIONS)
    loglik = np.log(np.maximum(likelihood, np.finfo(float).tiny)).sum(axis=-2)
    mass = PRIORS * np.exp(loglik-loglik.max(axis=-1, keepdims=True))
    centered = np.sum(mass*(RETENTIONS-.5), axis=-1)/np.sum(mass, axis=-1)
    return .5 + np.where(abs(centered)<1.e-12, 0., centered)
