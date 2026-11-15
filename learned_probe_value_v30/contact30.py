"""Source and previously unseen synthetic friction laws, with unchanged sensors."""
from pathlib import Path
import sys

import numpy as np

V29 = Path(__file__).resolve().parent.parent / "adaptive_stopping_v29"
sys.path.insert(0, str(V29))
from runner29 import P, Q, Engine, t, torch
from runner27 import draw_theta, stream
from physics25 import (FACTUAL_INCOMING, FY, PROBE_INCOMING, QUERY_INCOMING, QUERY_Y,
                       VYS, impact, profile, sigmas)

LAWS = ("split_static", "speed_weak")


def collision(theta, height, incoming, law):
    if law == "coulomb":
        return impact(theta, height, incoming)
    _, _, slope, curve, friction, restitution = theta.unbind(-1)
    slope = slope + curve * height / .025
    norm = (1 + slope.square()).sqrt()
    vn = (1 - slope * incoming) / norm
    vt = (slope + incoming) / norm
    jn = (1 + restitution) * vn.clamp_min(0)
    if law == "split_static":
        mu_s, mu_k = 1.20 * friction, .80 * friction
    elif law == "speed_weak":
        speed_factor = (-vt.abs() / .45).exp()
        mu_s = friction * (1 + .22 * speed_factor)
        mu_k = friction * (.64 + .25 * speed_factor)
    else:
        raise ValueError(law)
    stick = vt.abs() <= mu_s * jn
    jt = torch.where(stick, vt, vt.sign() * mu_k * jn)
    return torch.stack([1 - (jn + jt * slope) / norm,
                        incoming + (jn * slope - jt) / norm], -1), stick


def means(theta, law):
    v0, factual_stick = collision(theta, 0, FACTUAL_INCOMING, law)
    probe, probe_stick = collision(theta, 0, PROBE_INCOMING, law)
    if not factual_stick.all() or probe_stick.any():
        raise ValueError("Unexpected factual or calibration contact regime")
    return (torch.cat([profile(theta, FY), v0], -1), profile(theta, VYS[0]),
            profile(theta, VYS[1]), probe)


def factual(n, seed, lane, alpha, law):
    theta = draw_theta(n, seed, lane, alpha)
    observed = means(theta, law)[0]
    noise = t(stream(seed, lane+1).normal(size=tuple(observed.shape)))
    return (observed + noise * sigmas()[0]).cpu().numpy()


def calibration_data(seed, alpha, law):
    n = 1024
    x = factual(n, seed, 10, alpha, law)
    order = stream(seed, 98).permutation(n)[:8]
    theta = draw_theta(n, seed, 10, alpha)[order]
    y = means(theta, law)[P]
    noise = t(stream(seed, 99).normal(size=tuple(y.shape)))
    return x, order, (y+noise*sigmas()[P]).cpu().numpy()


def evaluation_data(n, seed, alpha, law, repeats=32):
    theta = draw_theta(n, seed, 20, alpha)
    actual, reference = means(theta, law), means(theta, "coulomb")
    label = collision(theta, QUERY_Y, QUERY_INCOMING, law)[1]
    source = collision(theta, QUERY_Y, QUERY_INCOMING, "coulomb")[1]
    obs = {}
    for a in (Q, P):
        m = actual[a]
        noise = t(stream(seed, 40+a).normal(size=(n,repeats,m.shape[1])))
        obs[a] = (m[:,None]+noise*sigmas()[a]).cpu().numpy()
    shifts = [torch.linalg.vector_norm(actual[a]-reference[a],dim=1).cpu().numpy() for a in (0,P)]
    return label.cpu().numpy(), source.cpu().numpy(), obs, shifts


def matched_decoder(pair, alpha, law):
    decoder = Engine(pair, *alpha)
    bank = decoder.inner
    bank.mu = dict(enumerate(means(bank.theta, law)))
    bank.white = {a: m / bank.sigma[a] for a,m in bank.mu.items()}
    bank.norm = {a: .5*m.square().sum(-1) for a,m in bank.white.items()}
    bank.label = collision(bank.theta, QUERY_Y, QUERY_INCOMING, law)[1].to(bank.theta.dtype)[:,None]*np.sqrt(2.)
    return decoder
