"""Evaluator-only generator for jointly held-out shape, surface field and rate law.

Acquisition/decoding must receive only the bought observations (plus the explicitly
privileged direction in its own row), never the context or both potential views.
All original sensor scales, probe commands and law-independent source hypothesis
coordinates come from v41 to keep the comparison on the same physical task.
"""
from functools import lru_cache
import math
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'estimated_projection_v41'))
import world41 as w

t = w.t
torch = w.torch
Q, P = w.Q, w.P
ALPHAS = w.ALPHAS
HEIGHTS = (.019, .031)
VARIANTS = ('base', 'hollow', 'bulge')
REGIMES = ('strong', 'weak', 'null')
OFFSETS_MM = (-1., -.5, -.25, -.125, 0., .125, .25, .5, 1.)
SOURCE_SHAPES = ('flare', 'roll')
SOURCE_SURFACES = ('uniform', 'linear')
SOURCE_LAWS = ('saturating', 'crossover')
TARGET_SHAPES = ('fold', 'ripple')
TARGET_SURFACES = ('inlay', 'woven')
TARGET_LAWS = ('rational', 'gaussian_rate')
AMPLITUDE = .00075  # metres
KNOTS = (.011, .019, .023, .031, .039)
WEIGHTS = {'fold': (.70, -.90, .45, .85), 'ripple': (-.65, .50, .88, -.76)}
VARIANT_SIGN = {'base': 0., 'hollow': -1., 'bulge': 1.}


def base_profile(y, shape):
    """First derivative (dimensionless) and lateral displacement (metres)."""
    if shape in SOURCE_SHAPES:
        return w.base_profile(y, shape)
    u = t(y) / .025
    if shape == 'fold':
        # Rounded hinge: logistic slope through an off-origin inflection.
        origin = t(-3.75)
        k = 5.
        shift = k * u + origin
        ref = origin.sigmoid()
        g = .085 * (shift.sigmoid() - ref)
        f = .085 * .025 * ((torch.nn.functional.softplus(shift)
                             - torch.nn.functional.softplus(origin)) / k - ref*u)
        return g, f
    if shape == 'ripple':
        k = 1.65
        g = .061 * (torch.sin(k*u) - .43*torch.sin(2*k*u))
        f = .061*.025 * ((1.-torch.cos(k*u))/k - .43*(1.-torch.cos(2*k*u))/(2*k))
        return g, f
    raise ValueError(shape)


def deformation(y, shape):
    if shape in SOURCE_SHAPES:
        return w.deformation(y, shape)
    y = t(y)
    g, f = torch.zeros_like(y), torch.zeros_like(y)
    for left, right, weight in zip(KNOTS[:-1], KNOTS[1:], WEIGHTS[shape]):
        inside = (y > left) & (y < right)
        u = ((y-left)/(right-left)).clamp(0., 1.)
        # Analytic d/du. The explicit mask yields exactly zero at knot positions.
        if shape == 'fold':
            env = torch.sin(math.pi*u).square()
            denv = math.pi*torch.sin(2*math.pi*u)
            mod = 1.+.38*(2*u-1)
            dmod = .76
        else:
            env = 16*u.square()*(1.-u).square()
            denv = 32*u*(1.-u)*(1.-2*u)
            mod = .55+.45*torch.cos(2*math.pi*u)
            dmod = -.9*math.pi*torch.sin(2*math.pi*u)
        f += inside * (AMPLITUDE*weight) * env * mod
        g += inside * (AMPLITUDE*weight/(right-left)) * (denv*mod+env*dmod)
    return g, f


def perturbation(y, shape, variant):
    g, f = base_profile(y, shape)
    sign = VARIANT_SIGN[variant]
    if sign:
        dg, df = deformation(y, shape)
        return g+sign*dg, f+sign*df
    return g, f


def surface_field(height, surface):
    """Return spatially varying isotropic material scale and anisotropy multiplier."""
    h = float(height) * 1000.  # mm
    if surface == 'uniform':
        return 1., 1.
    if surface == 'linear':
        u = float(np.clip(h / 36., -1., 1.))
        return 1.+.025*u, 1.-.15*u
    if surface == 'inlay':
        # Localized material insert centred between contact heights.
        insert = math.exp(-.5*((h-22.)/5.5)**2)
        return 1.+.079*insert, 1.-1.45*insert
    if surface == 'woven':
        # Repeating surface ribs with independent scale and orientation response.
        phase = 2.*math.pi*h/24.
        return 1.+.047*math.sin(phase), .38+.62*math.cos(phase)
    raise ValueError(surface)


def coefficients(mu, vt, height, surface, law, sign, regime):
    scale, anisotropy = surface_field(height, surface)
    if law in SOURCE_LAWS:
        return w.coefficients(mu*scale, vt, height, law, sign*anisotropy, regime)
    s = vt.abs()
    if law == 'rational':
        rate = s.square()/(.58**2+s.square())
        direction = .145 * vt.pow(3)/(.61**3+s.pow(3))
        static, kinetic = 1.045-.046*rate, .900+.035*rate
    elif law == 'gaussian_rate':
        band = torch.exp(-.5*((s-.64)/.25).square())
        direction = .16 * torch.erf(vt/.34) * (1.-1.55*band)
        static, kinetic = 1.007+.043*band, .930-.061*band
    else:
        raise ValueError(law)
    near = {'strong': 1., 'weak': .3, 'null': 0., 'global_null': 0.}[regime]
    u = min(abs(float(height))/w.MATCH_HEIGHT, 1.)
    envelope = 0. if regime=='global_null' else near+(1.-near)*math.cos(math.pi*u/2)**2
    common = mu*scale*(1.+sign*anisotropy*envelope*direction)
    return common*static, common*kinetic


def impact(theta, height, speed, context, regime):
    shape, variant, _, surface, law, sign, _ = context
    _, _, slope, curvature, mu, e = theta.unbind(-1)
    local = slope + curvature*(height/.025) + perturbation(height, shape, variant)[0]
    norm = (1.+local.square()).sqrt()
    tangent = (speed+local)/norm
    jn = (1.+e)*((1.-local*speed)/norm).clamp_min(0.)
    static, kinetic = coefficients(mu, tangent, height, surface, law, sign, regime)
    stick = tangent.abs() <= static*jn
    jt = torch.where(stick, tangent, tangent.sign()*kinetic*jn)
    return torch.stack((1.-(jn+jt*local)/norm, speed+(jn*local-jt)/norm),-1), stick


@lru_cache(maxsize=40)
def _catalog(panel, alpha, lane, is_target):
    if not 0 <= panel < 1000 or lane not in (10, 20):
        raise ValueError((panel, lane))
    # Bits of a deterministic seed tuple; supports cannot overlap across splits/lanes.
    sd = np.random.SeedSequence([430109, int(is_target), panel, lane]+
                                [int(round((v+2.)*1000000)) for v in alpha])
    return w.draw_theta(w.SUPPORT_SIZE, int(sd.generate_state(1)[0]), lane, list(alpha))


def draw_object(n, seed, lane, alpha, context):
    bank = _catalog(int(context[-1]), tuple(map(float, alpha)), lane, context[0] in TARGET_SHAPES)
    return bank[w.stream(seed, 4300+lane).integers(w.SUPPORT_SIZE,size=n)]


def contour(theta, y, context):
    return w.profile(theta, y) + perturbation(y, context[0], context[1])[1]


def factual(theta, seed, lane, context):
    vel, _ = impact(theta, 0., w.FACTUAL_INCOMING, context, 'global_null')
    mean = torch.cat((contour(theta, w.FY, context), vel), -1)
    return (mean+t(w.stream(seed,lane).normal(size=tuple(mean.shape)))*w.sigmas()[0]).cpu().numpy()


def calibration(seed, alpha, context, regime, start):
    theta = draw_object(1024, seed, 10, alpha, context)
    x = factual(theta, seed, 11, context)
    ids = w.stream(seed,98).permutation(1024)[:4]
    dirs = w.directions('alternating', start, 4)
    dirs[-1] = 1
    far, _ = impact(theta[ids[:3]], 0., t(dirs[:3])*abs(w.PROBE_INCOMING), context, regime)
    near, stick = impact(theta[ids[3:]], w.MATCH_HEIGHT, w.MATCH_SPEED, context, regime)
    noise = t(w.stream(seed,99).normal(size=(4,2)))*w.sigmas()[P]
    record = (x, ids, (torch.cat((far,near),0)+noise).cpu().numpy(), dirs)
    return record, theta[ids[3]].cpu().numpy(), near[0].cpu().numpy(), bool(stick[0])


def deployment(seed, alpha, context, regime, repeats=8):
    theta = draw_object(128, seed, 20, alpha, context)
    x = factual(theta, seed, 21, context)
    height = context[2]
    _, label = impact(theta, height, w.QUERY_INCOMING, context, regime)
    probe, _ = impact(theta, 0., w.PROBE_INCOMING, context, regime)
    means = {Q: contour(theta, w.view_heights(height), context), P: probe}
    obs = {}
    for a, mean in means.items():
        eps = t(w.stream(seed,40+a).normal(size=(len(mean),repeats,mean.shape[-1])))
        obs[a] = (mean[:,None]+eps*w.sigmas()[a]).cpu().numpy()
    return x, label.cpu().numpy(), theta.cpu().numpy(), obs


def offset_labels(theta, context, regime):
    theta = t(theta)
    return torch.stack([impact(theta, context[2]+.001*d, w.QUERY_INCOMING, context, regime)[1]
                        for d in OFFSETS_MM]).cpu().numpy()


def oracle_direction(shape, height):
    raw = deformation(w.view_heights(height),shape)[1]/w.sigmas()[Q]
    return (raw/raw.norm()).cpu().numpy()
