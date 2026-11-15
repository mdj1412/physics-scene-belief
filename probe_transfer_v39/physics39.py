"""v39 target generator: fresh analytic contours and independent finite catalogs."""
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "probe_value_v38"))
from physics38 import (
    ALPHAS, FY, VYS, P, Q, PROBE_INCOMING, QUERY_INCOMING, QUERY_Y, FACTUAL_INCOMING,
    MATCH_HEIGHT, MATCH_SPEED, SUPPORT_SIZE, coefficients, directions, draw_theta,
    profile, sigmas, stream, t, torch,
)

OBJECTS = ("shelf", "ridge")
SURFACES = ("weakening", "strengthening")
REGIMES = ("strong", "weak", "null")
CATALOGS = 12
SUPPORT_SEED = 1939100000


def catalog_seed(shape, surface, catalog_id, alpha_index, lane):
    if not 0 <= catalog_id < CATALOGS or lane not in (10, 20):
        raise ValueError("Unknown catalog replicate or split")
    cell = OBJECTS.index(shape) * len(SURFACES) + SURFACES.index(surface)
    return SUPPORT_SEED + cell * 1000000 + catalog_id * 10000 + alpha_index * 100 + lane


@lru_cache(maxsize=128)
def _catalog(shape, surface, catalog_id, alpha_index, lane):
    seed = catalog_seed(shape, surface, catalog_id, alpha_index, lane)
    return draw_theta(SUPPORT_SIZE, seed, lane, ALPHAS[alpha_index])


def latent_support(alpha, law, lane):
    shape, surface, sign, catalog_id = law
    if sign not in (-1, 1):
        raise ValueError("Invalid orientation")
    return _catalog(shape, surface, catalog_id, ALPHAS.index(list(map(float, alpha))), lane)


def draw_indices(n, seed, lane):
    return stream(seed, 3900 + lane).integers(SUPPORT_SIZE, size=n)


def draw_object(n, seed, lane, alpha, law):
    return latent_support(alpha, law, lane)[draw_indices(n, seed, lane)]


def perturbation(y, shape):
    """Return contour slope and its analytic integral anchored at zero height."""
    u = t(y) / .025
    if shape == "shelf":
        k = np.pi / 2
        return .043 * (1 - torch.cos(k*u)), .043 * .025 * (u - torch.sin(k*u)/k)
    if shape == "ridge":
        den = 1 + u.square()
        return -.153 * u / den.square(), -.153 * .025 * .5 * u.square() / den
    raise ValueError(shape)


def contour(theta, y, law):
    return profile(theta, y) + perturbation(y, law[0])[1]


def impact(theta, height, speed, law, regime):
    shape, surface, sign, _ = law
    _, _, slope, curvature, friction, restitution = theta.unbind(-1)
    local = slope + curvature * (height/.025) + perturbation(height, shape)[0]
    norm = (1+local.square()).sqrt()
    vt = (speed+local) / norm
    jn = (1+restitution) * ((1-local*speed)/norm).clamp_min(0.)
    mu_s, mu_k = coefficients(friction, vt, height, surface, sign, regime)
    stick = vt.abs() <= mu_s * jn
    jt = torch.where(stick, vt, vt.sign()*mu_k*jn)
    return torch.stack((1-(jn+jt*local)/norm, speed+(jn*local-jt)/norm), -1), stick


def factual(theta, seed, lane, law):
    velocity, stick = impact(theta, 0., FACTUAL_INCOMING, law, "global_null")
    if not bool(stick.all()):
        raise ValueError("Factual support violates sticking assumption")
    mean = torch.cat((contour(theta, FY, law), velocity), -1)
    noise = t(stream(seed, lane).normal(size=tuple(mean.shape))) * sigmas()[0]
    return (mean + noise).cpu().numpy()


def calibration(seed, alpha, law, regime, start):
    theta = draw_object(1024, seed, 10, alpha, law)
    x = factual(theta, seed, 11, law)
    ids = stream(seed, 98).permutation(1024)[:4]
    dirs = directions("alternating", start, 4)
    dirs[-1] = 1
    far, far_stick = impact(theta[ids[:3]], 0., t(dirs[:3])*abs(PROBE_INCOMING), law, regime)
    if bool(far_stick.any()):
        raise ValueError("Conventional calibration probe stuck")
    near, near_stick = impact(theta[ids[3:]], MATCH_HEIGHT, MATCH_SPEED, law, regime)
    noise = t(stream(seed, 99).normal(size=(4,2))) * sigmas()[P]
    y = (torch.cat((far, near), 0) + noise).cpu().numpy()
    return (x,ids,y,dirs), theta[ids[3]].cpu().numpy(), near[0].cpu().numpy(), bool(near_stick[0])


def deployment(seed, alpha, law, regime, repeats=32):
    theta = draw_object(128, seed, 20, alpha, law)
    x = factual(theta, seed, 21, law)
    _, label = impact(theta, QUERY_Y, QUERY_INCOMING, law, regime)
    vel, probe_stick = impact(theta, 0., PROBE_INCOMING, law, regime)
    if bool(probe_stick.any()):
        raise ValueError("Conventional deployment probe stuck")
    means = {Q:contour(theta,VYS[0],law), P:vel}
    obs = {}
    for a,mean in means.items():
        noise = t(stream(seed, 40+a).normal(size=(len(mean),repeats,mean.shape[-1])))
        obs[a] = (mean[:,None]+noise*sigmas()[a]).cpu().numpy()
    return x, label.cpu().numpy(), theta.cpu().numpy(), obs
