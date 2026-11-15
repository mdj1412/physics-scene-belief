"""Population inference from independent, unlabeled factual observations in the v26 world law."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "proxy_shift_v26"))
from physics26 import DT, FACTUAL_INCOMING, UNIT_DIM, change_proxy, impact, parameters, sigmas, t, torch
from physics25 import FY, profile


def factual_mean(theta):
    velocity, sticks = impact(theta, 0, FACTUAL_INCOMING)
    if not bool(sticks.all()):
        raise ValueError("Factual-stick symmetry requires all candidate factual impacts to stick.")
    return torch.cat([profile(theta, FY), velocity], dim=-1)


class FactualShiftEstimator:
    """Uniform independent population priors on alpha_c and alpha_f in [-1,1].

    Calibration must be an iid batch from one environment. The estimator knows the
    simulator family and calibration, not the environment's flip rates or hidden width
    groups. Its bank is completely independent of calibration, evaluation and policy banks.
    """

    @torch.inference_mode()
    def __init__(self, seed=270431, bank_size=8192):
        if bank_size < 4 or bank_size & (bank_size - 1):
            raise ValueError("Use a power-of-two bank_size >= 4 for scrambled Sobol.")
        self.bank_size = bank_size
        self.seed = int(seed)
        self.sigma = sigmas()[0]
        half = bank_size // 2
        unit = torch.quasirandom.SobolEngine(UNIT_DIM, scramble=True, seed=self.seed).draw(
            half, dtype=DT)
        # Integrate the independent equiprobable width_f exactly. Both partners have
        # identical slope, curvature, geometry and restitution jitter; they have opposite
        # restitution centers. Flipping the restitution center permutes the partners' X.
        lower = unit.clone()
        lower[:, 1] *= .5
        upper = lower.clone()
        upper[:, 1] += .5
        th = parameters(t(torch.cat([lower, upper])))
        which = torch.zeros(bank_size, dtype=torch.uint8, device=th.device)
        means = []
        self.symmetry_error = 0.
        for variant in (0, 2):
            mu = factual_mean(change_proxy(th, which + variant))
            flipped = factual_mean(change_proxy(th, which + variant + 1))
            sym_error = (flipped - mu.roll(half, dims=0)).abs().max().item()
            self.symmetry_error = max(self.symmetry_error, sym_error)
            means.append(mu/self.sigma)
        if self.symmetry_error > 1e-12:
            raise ArithmeticError(f"Factual friction-width symmetry failed: {self.symmetry_error}")
        self.centers = torch.stack(means)
        self.energy = .5*self.centers.square().sum(dim=-1)
        self.grid = np.linspace(-1., 1., 401)

    @torch.inference_mode()
    def component_loglikelihood(self, factual):
        """N by 2 log densities at alpha_c=+1,-1, up to a common per-row constant.

        alpha_f=+1 and -1 have identical factual densities for each of the columns.
        Sample chunks bound memory independently of calibration cohort size.
        """
        x = t(factual)
        if x.ndim != 2 or x.shape[1] != 11 or len(x) == 0 or not bool(torch.isfinite(x).all()):
            raise ValueError("Expected a non-empty finite array of shape [worlds,11].")
        pairs = []
        for group in x.split(128):
            weights = torch.matmul(group/self.sigma, self.centers.transpose(1, 2)) - self.energy[:, None, :]
            pairs.append(torch.logsumexp(weights, dim=-1).T - np.log(self.bank_size))
        return torch.cat(pairs).cpu().numpy()

    def posterior(self, component_logs):
        ll = np.asarray(component_logs, dtype=np.float64)
        if ll.ndim != 2 or ll.shape[1] != 2 or not len(ll) or not np.isfinite(ll).all():
            raise ValueError("Expected N pairs of finite component log densities.")
        # A single new world has density .5*(q_plus+q_minus)*(1+alpha_c*s);
        # s is in [-1,1]. logaddexp handles even boundary alpha=+-1 and outliers.
        align = (1 + self.grid) / 2
        with np.errstate(divide="ignore"):
            logw = np.log(align)
            logw_swap = np.log1p(-align)
        center = np.logaddexp(ll[:, 0], ll[:, 1]) - np.log(2.)
        profile_loglik = np.logaddexp(ll[:, 0, None]-center[:, None]+logw,
                                      ll[:, 1, None]-center[:, None]+logw_swap).sum(axis=0)
        peak = float(profile_loglik.max())
        density = np.exp(profile_loglik - peak)
        quadrature = np.ones_like(self.grid)
        quadrature[[0, -1]] *= .5
        weights = density * quadrature
        weights /= weights.sum()
        mass = .5*(density[:-1]+density[1:])
        cdf = np.r_[0., np.cumsum(mass)/mass.sum()]
        mean = float(weights@self.grid)
        lower, upper = np.interp([.025,.975], cdf, self.grid).tolist()
        score_at_zero = np.tanh((ll[:, 0] - ll[:, 1])*.5)
        log_evidence = peak + np.log(np.sum(density * quadrature) / (len(self.grid)-1))
        # KL(posterior || uniform prior); the friction posterior stays equal to its prior.
        information_gain = max(0., float(weights @ profile_loglik - log_evidence))
        return dict(alpha=[mean, 0.], ci95=[[lower, upper], [-.95, .95]],
                    sd=[float(np.sqrt(weights@((self.grid-mean)**2))), float(np.sqrt(1/3))],
                    posterior_prob_negative=[float(np.interp(0.,self.grid,cdf)), .5],
                    posterior_prob_alpha_below_half=[float(np.interp(.5,self.grid,cdf)), .75],
                    posterior_kl_nats=[information_gain, 0.],
                    log_bayes_factor_vs_source=float(log_evidence-profile_loglik[-1]),
                    per_world_observed_information_at_zero=[float(np.mean(score_at_zero**2)), 0.],
                    n_worlds=len(ll), bank_size=self.bank_size, bank_seed=self.seed,
                    max_abs_factual_friction_pair_residual=self.symmetry_error)

    def fit(self, factual):
        """Estimate the deployment population using only the current context's factual X."""
        return self.posterior(self.component_loglikelihood(factual))
