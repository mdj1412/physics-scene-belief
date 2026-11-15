"""Paired intervention on two public proxy/latent-width associations in the V25 simulator."""
from pathlib import Path
import sys

import numpy as np

SOURCE = Path(__file__).resolve().parent.parent / "within_calibration_v25"
sys.path.insert(0, str(SOURCE))
from physics25 import (BANK_SIZE, UNIT_DIM, CALIBRATION, DT, Engine as V25Engine,
                       FACTUAL_INCOMING, PROBE_INCOMING, WorldBank as V25Bank,
                       impact, labels, observations, parameters, sigmas, t, torch)
from common import R, save

LEVELS = (1., .75, .5, 0., -.5, -1.)
ORIGIN = (1., 1.)
# A donor below the threshold reverses the conditional center, keeping the v25 jitter.
# The mixture of the two centers is unchanged marginally under balanced cell sampling.
def variants(u_flip, alpha_c, alpha_f):
    """Variant 0=source, 1=restitution flip, 2=offset flip, 3=both; monotone common draws."""
    if not (-1 <= alpha_c <= 1 and -1 <= alpha_f <= 1):
        raise ValueError("Proxy alignment must be in [-1, 1].")
    u_flip = np.asarray(u_flip)
    if u_flip.ndim != 2 or u_flip.shape[1] != 2:
        raise ValueError("Expected [world, 2] independent flip uniforms.")
    return (2 * (u_flip[:, 0] < (1-alpha_c)/2).astype(np.uint8)
            + (u_flip[:, 1] < (1-alpha_f)/2).astype(np.uint8))


def change_proxy(theta, which):
    """Preserve curvature, friction, slope, within-center jitter. Recompute physics from new theta."""
    out = theta.clone()
    cell = theta[:, 0]
    curve_sign = 2 * (cell >= 2).to(DT) - 1
    friction_sign = 2 * cell.remainder(2) - 1
    out[:, 1] -= 2 * .00115 * curve_sign * ((which & 2) != 0)
    out[:, 5] -= 2 * .095 * friction_sign * ((which & 1) != 0)
    return out


def independent_uniforms(seed, count, tag):
    # Distinct from v25 base and noise streams; IID row order prevents accidental pairing
    # of the same Sobol index bits across separately scrambled 9D and 2D sequences.
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(tag)]))
    return rng.random((count, 2))


class ShiftBank(V25Bank):
    def __init__(self, seed, alpha_c, alpha_f):
        super().__init__(seed)
        if (alpha_c, alpha_f) == ORIGIN:
            return
        u_flip = independent_uniforms(seed, self.N, 260401)
        assignment = torch.as_tensor(variants(u_flip, alpha_c, alpha_f),
                                     dtype=torch.uint8, device=self.theta.device)
        self.theta = change_proxy(self.theta, assignment)
        self.mu = dict(enumerate(observations(self.theta)))
        self.white = {a: m/self.sigma[a] for a, m in self.mu.items()}
        self.norm = {a: .5*m.square().sum(-1) for a, m in self.white.items()}
        self.label = labels(self.theta).to(DT)[:, None] * np.sqrt(2.)


class Engine(V25Engine):
    def __init__(self, pair, alpha_c=1., alpha_f=1.):
        super().__init__(pair)
        if (alpha_c, alpha_f) != ORIGIN:
            self.inner = ShiftBank(pair[0], alpha_c, alpha_f)
            self.outer = ShiftBank(pair[1], alpha_c, alpha_f)


def generate(split, n, seed, repeats=32):
    """Cache four coherent worlds per row. All noise is independent of flip and paired across variants.

    Source cell assignment, physical parameters and noise use the exact v25 call order.
    Private files may only be used for evaluation. The assignment file is the experiment
    coordinator's schedule; neither its uniforms nor cell nor variant is an online feature.
    """
    if n <= 0 or n % 4:
        raise ValueError("Requested count must be a positive multiple of four.")
    rng = np.random.default_rng(seed)
    cell = rng.permutation(np.tile(np.arange(4), n//4))
    u = rng.uniform(size=(n, UNIT_DIM))
    u[:, 0] = (cell//2 + u[:, 0])/2
    u[:, 1] = (cell % 2 + u[:, 1])/2
    origin = parameters(t(u))
    sd = sigmas()
    reference = observations(origin)
    noise0 = t(rng.normal(size=(n, 11))) * sd[0]
    errors = [None] + [t(rng.normal(size=(n, repeats, reference[a].shape[-1]))) * sd[a]
                       for a in (1, 2, 3)]
    u_flip = independent_uniforms(seed, n, 260402)
    x = []
    theta = []
    target = []
    ys = {a: [] for a in (1, 2, 3)}
    for code in range(4):
        which = torch.full((n,), code, dtype=torch.uint8, device=origin.device)
        th = origin if code == 0 else change_proxy(origin, which)
        mu = reference if code == 0 else observations(th)
        z = labels(th).cpu().numpy()
        assert bool(impact(th, 0, FACTUAL_INCOMING)[1].all())
        assert not bool(impact(th, 0, PROBE_INCOMING)[1].any())
        # Offset translates the profile, but cannot change the local slope or any stick label.
        if code >= 2:
            assert np.array_equal(z, target[code-2])
        for a in (1, 2, 3):
            ys[a].append((mu[a][:, None] + errors[a]).cpu().numpy())
        x.append((mu[0]+noise0).cpu().numpy())
        theta.append(th.cpu().numpy())
        target.append(z)
    save(R/split/"public.npz", x=np.stack(x, axis=1))
    save(R/split/"assignment.npz", u_flip=u_flip, cell=cell)
    save(R/split/"private.npz", label=np.stack(target, axis=1), cell=cell, u_flip=u_flip,
         theta=np.stack(theta, axis=1), **{f"obs{a}":np.stack(ys[a],axis=1) for a in ys})
