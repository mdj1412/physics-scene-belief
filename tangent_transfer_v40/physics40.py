"""Paired finite worlds with matched contact tangents and held-out friction responses."""
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "probe_transfer_v39"))
from physics39 import (
    ALPHAS, FACTUAL_INCOMING, FY, MATCH_HEIGHT, MATCH_SPEED, P, PROBE_INCOMING,
    Q, QUERY_INCOMING, QUERY_Y, REGIMES, SUPPORT_SIZE, VYS, directions,
    draw_theta, perturbation as family_profile, profile, sigmas, stream, t, torch,
)

FAMILIES = ("shelf", "ridge")
CONTOURS = ("base", "hollow", "bulge")
SURFACES = ("saturating", "crossover")
SUPPORT_SEED = 2040100000
OFFSET_METERS = {"base": 0., "hollow": -.0009, "bulge": .0009}
NEAR_STRENGTH = {"strong": 1., "weak": .3, "null": 0.}


def catalog_seed(catalog_id, alpha_index, lane):
    if catalog_id < 0 or lane not in (10, 20) or not 0 <= alpha_index < len(ALPHAS):
        raise ValueError("Invalid catalog, alpha or split")
    return SUPPORT_SEED + catalog_id * 10000 + alpha_index * 100 + lane


@lru_cache(maxsize=32)
def _catalog(catalog_id, alpha_index, lane):
    seed = catalog_seed(catalog_id, alpha_index, lane)
    return draw_theta(SUPPORT_SIZE, seed, lane, ALPHAS[alpha_index])


def latent_support(alpha, law, lane):
    return _catalog(law[4], ALPHAS.index(list(map(float, alpha))), lane)


def draw_indices(n, seed, lane):
    return stream(seed, 4000+lane).integers(SUPPORT_SIZE, size=n)


def draw_object(n, seed, lane, alpha, law):
    return latent_support(alpha, law, lane)[draw_indices(n, seed, lane)]


def lobe(y, left, right):
    """Compact C1 quartic of unit height with zero derivative at both endpoints."""
    x = ((y-left)/(right-left)).clamp(0., 1.)
    return 32*x*(1-x)*(1-2*x)/(right-left), 16*x.square()*(1-x).square()


def perturbation(y, family, variant):
    ampl = OFFSET_METERS[variant]
    slope, value = family_profile(y, family)
    if ampl:
        y = t(y)
        before = lobe(y, .017, .023)
        after = lobe(y, .025, .033)
        slope = slope + ampl*(before[0]+after[0])
        value = value + ampl*(before[1]+after[1])
    return slope, value


def contour(theta, y, law):
    return profile(theta, y) + perturbation(y, law[0], law[1])[1]


def coefficients(mu, vt, height, surface, sign, regime):
    s = vt.abs()
    if regime == "global_null":
        envelope = 0.
    else:
        height_fraction = min(abs(float(height))/MATCH_HEIGHT, 1.)
        near = NEAR_STRENGTH[regime]
        envelope = near+(1-near)*np.cos(.5*np.pi*height_fraction)**2
    if surface == "saturating":
        direction = .18*torch.tanh(vt/.62).pow(3)
        speed_gain = torch.tanh(s/.58).square()
        static, kinetic = 1.015+.025*speed_gain, .88+.075*speed_gain
    elif surface == "crossover":
        direction = .20*torch.tanh(vt/.32)*torch.tanh((s-.70)/.16)
        shoulder = torch.exp(-((s-.68)/.24).square())
        static, kinetic = 1.035-.02*shoulder, .94-.055*shoulder
    else:
        raise ValueError(surface)
    common = mu*(1+sign*envelope*direction)
    return common*static, common*kinetic


def impact(theta, height, speed, law, regime):
    family, variant, surface, sign, _ = law
    _, _, slope, curvature, friction, restitution = theta.unbind(-1)
    local = slope + curvature*(height/.025) + perturbation(height, family, variant)[0]
    norm = (1+local.square()).sqrt()
    vt = (speed+local)/norm
    jn = (1+restitution)*((1-local*speed)/norm).clamp_min(0.)
    static, kinetic = coefficients(friction, vt, height, surface, sign, regime)
    stick = vt.abs() <= static*jn
    jt = torch.where(stick, vt, vt.sign()*kinetic*jn)
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
