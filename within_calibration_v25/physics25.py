"""One calibrated Coulomb contact simulator with independently varying parameter spreads."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "factual_tail_v23"))
from physics23 import DT, FY, VYS, PhysicalBank, impact, profile, t, torch
from policy23 import TailGate
from policy22 import action
from common import R, save


FACTUAL_INCOMING = 0.0
QUERY_INCOMING = 0.425
PROBE_INCOMING = -1.10
QUERY_Y = 0.025
CALIBRATION = (0.0016, 0.095, 0.00030, 0.00120, 0.027)
BANK_SIZE = 16384
UNIT_DIM = 9
CELL_NAMES = ("narrow/narrow", "narrow/wide", "wide/narrow", "wide/wide")


def parameters(unit):
    """Draw offset/slope/curvature/friction/restitution; column zero is evaluation metadata."""
    wide_curve = (unit[:, 0] >= 0.5).to(DT)
    wide_mu = (unit[:, 1] >= 0.5).to(DT)
    curve_sign = 2 * (unit[:, 2] >= 0.5).to(DT) - 1
    mu_sign = 2 * (unit[:, 3] >= 0.5).to(DT) - 1
    # Standoff offset and restitution are overlapping proxies for the two widths.
    # Neither cue reveals the independent sign of curvature or friction.
    offset = 0.00115 * (2 * wide_curve - 1) + 0.0034 * (unit[:, 6] - 0.5)
    slope = 0.070 * (unit[:, 7] - 0.5)
    curvature = curve_sign * (0.022 + 0.318 * wide_curve) + 0.024 * (unit[:, 4] - 0.5)
    friction = 0.251 + mu_sign * (0.011 + 0.127 * wide_mu) + 0.015 * (unit[:, 5] - 0.5)
    restitution = 0.70 + 0.095 * (2 * wide_mu - 1) + 0.18 * (unit[:, 8] - 0.5)
    cell = 2 * wide_curve + wide_mu
    return torch.stack([cell, offset, slope, curvature, friction, restitution], dim=-1)


def observations(theta):
    return (torch.cat([profile(theta, FY), impact(theta, 0, FACTUAL_INCOMING)[0]], dim=-1),
            profile(theta, VYS[0]), profile(theta, VYS[1]),
            impact(theta, 0, PROBE_INCOMING)[0])


def labels(theta):
    return impact(theta, QUERY_Y, QUERY_INCOMING)[1]


def sigmas():
    factual_profile, factual_velocity, query_view, factual_view, probe = CALIBRATION
    return (t([factual_profile] * 9 + [factual_velocity] * 2),
            t([query_view] * 7), t([factual_view] * 7), t([probe] * 2))


class WorldBank(PhysicalBank):
    """Candidate bank. Evaluation-only cell metadata never enters a likelihood or label."""
    def __init__(self, seed):
        self.N = BANK_SIZE
        unit = torch.quasirandom.SobolEngine(UNIT_DIM, scramble=True, seed=seed).draw(self.N, dtype=DT)
        self.theta = parameters(t(unit))
        self.sigma = sigmas()
        self.mu = dict(enumerate(observations(self.theta)))
        self.white = {a: m / self.sigma[a] for a, m in self.mu.items()}
        self.norm = {a: 0.5 * m.square().sum(-1) for a, m in self.white.items()}
        self.label = labels(self.theta).to(DT)[:, None] * np.sqrt(2.0)


class Engine(TailGate):
    def __init__(self, pair):
        super().__init__(pair, 0)
        self.inner = WorldBank(pair[0])
        self.outer = WorldBank(pair[1])
        for k in (64, 128):
            n = k // 8
            jitter = torch.rand(2, n, dtype=DT, generator=torch.Generator().manual_seed(798311 + k))
            self.strata[k] = ((torch.arange(n, dtype=DT) + jitter) / n).to(self.inner.theta.device)

    def choose(self, state, k):
        return action(self.risk(state, f"keep{k}m2048"))

    def choose_pair(self, state, k):
        risks = self.risk(state, f"keep{k}m2048")
        return 1 if float(risks[0] - risks[2]) <= 1e-12 else 3


def generate(split, n, seed, repeats=32):
    if n % 4:
        raise ValueError("World count must be divisible by four for the balanced factorial draw.")
    rng = np.random.default_rng(seed)
    cells = rng.permutation(np.tile(np.arange(4), n // 4))
    u = rng.uniform(size=(n, UNIT_DIM))
    u[:, 0] = (cells // 2 + u[:, 0]) / 2
    u[:, 1] = (cells % 2 + u[:, 1]) / 2
    theta = parameters(t(u))
    assert np.array_equal(theta[:, 0].cpu().numpy(), cells)
    mu = observations(theta)
    sd = sigmas()
    factual = mu[0] + t(rng.normal(size=(n, 11))) * sd[0]
    # Sanity check: factual tangential impulse always sticks; every probe slides.
    assert bool(impact(theta, 0, FACTUAL_INCOMING)[1].all())
    assert not bool(impact(theta, 0, PROBE_INCOMING)[1].any())
    private = dict(label=labels(theta).cpu().numpy(), cell=cells, theta=theta.cpu().numpy())
    for a in (1, 2, 3):
        eps = t(rng.normal(size=(n, repeats, mu[a].shape[-1])))
        y = mu[a][:, None] + eps * sd[a]
        private[f"obs{a}"] = y.cpu().numpy()[:, None]
    save(R / split / "public.npz", x=factual.cpu().numpy()[:, None])
    save(R / split / "private.npz", **private)
