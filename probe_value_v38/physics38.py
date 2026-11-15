"""Target-only v38 contact population with finite, disjoint prototype catalogs."""
from functools import lru_cache
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "query_matched_probe_v36"))
from target36 import FY, VYS, Q, P, QUERY_INCOMING, QUERY_Y, PROBE_INCOMING
from target35 import FACTUAL_INCOMING, directions, draw_theta, profile, sigmas, stream, t, torch

OBJECTS = ("shoulder", "bead")
SURFACES = ("weakening", "strengthening")
REGIMES = ("strong", "weak", "null", "global_null")
STRENGTH = dict(strong=1., weak=.3, null=0., global_null=0.)
MATCH_HEIGHT = .023
MATCH_SPEED = 1.10
SUPPORT_SIZE = 2048
SUPPORT_SEED = 1838210000
ALPHAS = json.loads((PROJECT / "bidirectional_probe_v34/policy_design34.json").read_text())["alphas"]


@lru_cache(maxsize=64)
def _catalog(shape, surface, alpha_index, lane):
    # Split, cell and alpha have independent support RNGs. Both directions use the same support.
    catalog_seed = (SUPPORT_SEED + 100000 * OBJECTS.index(shape) + 10000 * SURFACES.index(surface)
                    + 100 * alpha_index + lane)
    theta = draw_theta(SUPPORT_SIZE, catalog_seed, lane, ALPHAS[alpha_index]).clone()
    theta[:, 3] *= {"shoulder": .97, "bead": 1.03}[shape]
    return theta


def latent_support(alpha, law, lane):
    shape, surface, direction = law
    if lane not in (10, 20) or direction not in (-1, 1):
        raise ValueError("Lane 10 is calibration; lane 20 is deployment; direction must be +/-1")
    return _catalog(shape, surface, ALPHAS.index(list(map(float, alpha))), lane)


def draw_indices(n, seed, lane):
    # Sample prototypes with replacement; public pool row ids are distinct and unrelated
    # to the prototype ids, which are unobserved by every online method.
    return stream(seed, 3800 + lane).integers(SUPPORT_SIZE, size=n)


def draw_object(n, seed, lane, alpha, law):
    return latent_support(alpha, law, lane)[draw_indices(n, seed, lane)]


def perturbation(y, shape):
    """Return local slope and its analytic integral (both zero at y=0)."""
    u = t(y) / .025
    if shape == "shoulder":
        return .072 * u.square() / (1 + u.square()), .072 * .025 * (u - torch.atan(u))
    if shape == "bead":
        decay = torch.exp(-.5 * (u / .80).square())
        return -.078 * u * decay, -.078 * .025 * .80**2 * (1 - decay)
    raise ValueError(shape)


def contour(theta, y, law):
    return profile(theta, y) + perturbation(y, law[0])[1]


def coefficients(mu, tangential, height, surface, direction, regime):
    """Effective static threshold and sliding coefficient for one incoming tangential speed."""
    s = tangential.abs()
    u = min(abs(float(height)) / MATCH_HEIGHT, 1.)
    # Half-cosine spatial roll-off saturates beyond the fourth probe height.
    envelope = 1. - (1. - STRENGTH[regime]) * (.5 - .5 * np.cos(np.pi * u))
    if regime == "global_null":
        envelope = 0.
    if surface == "weakening":
        asymmetry = .19 * torch.sign(tangential) * (-torch.expm1(-s / .47))
        static = 1. + .04 / (1. + (s / .40).square())
        kinetic = .89 + .08 * torch.exp(-s / .48)
    elif surface == "strengthening":
        # Peak directional contrast at |vt|=.75; the isotropic sliding coefficient rises.
        relative = tangential / .75
        asymmetry = .192 * (2. * relative) / (1. + relative.square())
        static = 1. + .05 * s / (.45 + s)
        kinetic = .88 + .11 * s / (.50 + s)
    else:
        raise ValueError(surface)
    common = mu * (1. + direction * envelope * asymmetry)
    return common * static, common * kinetic


def impact(theta, height, speed, law, regime):
    shape, surface, direction = law
    _, _, slope, curvature, friction, restitution = theta.unbind(-1)
    local = slope + curvature * (height / .025) + perturbation(height, shape)[0]
    norm = torch.sqrt(1. + local.square())
    vt = (speed + local) / norm
    normal_impulse = (1. + restitution) * ((1. - local * speed) / norm).clamp_min(0.)
    mu_s, mu_k = coefficients(friction, vt, height, surface, direction, regime)
    stick = vt.abs() <= mu_s * normal_impulse
    jt = torch.where(stick, vt, vt.sign() * mu_k * normal_impulse)
    outgoing = torch.stack((1. - (normal_impulse + jt * local) / norm,
                            speed + (normal_impulse * local - jt) / norm), -1)
    return outgoing, stick


def factual_clean(theta, law):
    # All zero-speed factual trials stick. Their velocity is common across hidden surface regimes.
    vel, stick = impact(theta, 0., FACTUAL_INCOMING, law, "global_null")
    if not bool(stick.all()):
        raise ValueError("A factual object left the designed sticking range")
    return torch.cat((contour(theta, FY, law), vel), -1)


def factual(theta, seed, lane, law):
    mean = factual_clean(theta, law)
    return (mean + t(stream(seed, lane).normal(size=tuple(mean.shape))) * sigmas()[0]).cpu().numpy()


def deployment_clean(theta, law, regime):
    probe, stick = impact(theta, 0., PROBE_INCOMING, law, regime)
    if bool(stick.any()):
        raise ValueError("Unexpected sticking in conventional deployment probe")
    return {Q: contour(theta, VYS[0], law), P: probe}


def calibration(seed, alpha, law, regime, start, speeds=None, height=MATCH_HEIGHT):
    if float(height) != MATCH_HEIGHT or (speeds is not None and tuple(speeds) != (MATCH_SPEED,)):
        raise ValueError("The fourth probe in v38 is fixed at 23 mm and +1.10")
    theta = draw_object(1024, seed, 10, alpha, law)
    x = factual(theta, seed, 11, law)
    ids = stream(seed, 98).permutation(1024)[:4]
    dirs = directions("alternating", start, 4)
    dirs[-1] = 1  # The fourth command is positive regardless of the starting sign.
    far, far_stick = impact(theta[ids[:3]], 0., t(dirs[:3]) * abs(PROBE_INCOMING), law, regime)
    if bool(far_stick.any()):
        raise ValueError("Unexpected sticking in a conventional calibration probe")
    near, near_stick = impact(theta[ids[3:]], height, MATCH_SPEED, law, regime)
    noise = t(stream(seed, 99).normal(size=(4, 2))) * sigmas()[P]
    y = (torch.cat((far, near), 0) + noise).cpu().numpy()
    record = (x, ids, y, dirs)
    fourth = theta[ids[3]].cpu().numpy()
    if speeds is None:
        return record, fourth, near[0].cpu().numpy(), bool(near_stick[0])
    return [record], fourth, near.cpu().numpy(), near_stick.cpu().numpy()


def deployment(seed, alpha, law, regime, repeats=32):
    theta = draw_object(128, seed, 20, alpha, law)
    x = factual(theta, seed, 21, law)
    labels = impact(theta, QUERY_Y, QUERY_INCOMING, law, regime)[1]
    means = deployment_clean(theta, law, regime)
    obs = {}
    for a, mean in means.items():
        eps = t(stream(seed, 40 + a).normal(size=(len(mean), repeats, mean.shape[-1])))
        obs[a] = (mean[:, None] + eps * sigmas()[a]).cpu().numpy()
    return x, labels.cpu().numpy(), theta.cpu().numpy(), obs
