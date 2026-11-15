"""Expected flip cost of the Q/P ranking after an extra block of calibration probes.

Only model-generated reference observations, the current estimated posterior, and
already-revealed calibration likelihoods enter the scorer. No evaluation X/Y.
"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "calibration_probe_v28"))
from estimator28 import PROBE_INCOMING, impact
from runner28 import Engine, draw_theta, factual_batch, sigmas, t, torch


REFERENCE_SEED = 29150873
RISK_BANK_PAIR = (29104243, 29114771)
OUTCOME_SEED = 29175401
N_REFERENCE = 96
N_OUTCOMES = 256
MAX_HORIZON = 4
C_GRID = np.linspace(-1., 1., 5)
F_GRID = np.linspace(-1., 1., 9)
POST_GRID = np.linspace(-1., 1., 81)


def posterior_grid(factual_logs, probe_ids, joint_logs):
    """Quadrature on the v28 posterior with a single factor per calibration world."""
    ll = np.asarray(factual_logs, dtype=float)
    ids = np.asarray(probe_ids, dtype=int)
    joint = np.asarray(joint_logs, dtype=float).reshape(-1, 4)
    if ll.ndim != 2 or ll.shape[1] != 2 or len(joint) != len(ids):
        raise ValueError("Inconsistent observed calibration likelihoods")
    if np.any(ids < 0) or np.any(ids >= len(ll)) or len(np.unique(ids)) != len(ids):
        raise ValueError("Invalid paid calibration row indices")
    unprobed = np.ones(len(ll), dtype=bool)
    unprobed[ids] = False
    mixture = np.stack([(1+POST_GRID)/2, (1-POST_GRID)/2], axis=1)
    def rescale(x):
        return np.exp(x-x.max(axis=1, keepdims=True))
    profile = rescale(ll[unprobed]) @ mixture.T
    # Keeping each future response attached to its factual input here avoids
    # double-counting X. The hypothetical bank below deliberately forecasts Y only.
    tiny = np.finfo(float).tiny
    log_mass = np.broadcast_to(np.log(profile.clip(min=tiny)).sum(0)[:,None],
                               (len(POST_GRID),len(POST_GRID))).copy()
    for coef in rescale(joint):
        log_mass += np.log((mixture @ coef.reshape(2,2) @ mixture.T).clip(min=tiny))
    mass = np.exp(log_mass-log_mass.max())
    mass[[0,-1],:] *= .5
    mass[:,[0,-1]] *= .5
    return mass / mass.sum()


def linear_indices(grid, x):
    x = np.clip(np.atleast_1d(x),grid[0],grid[-1])
    lo = np.clip(np.searchsorted(grid,x,side="right")-1,0,len(grid)-2)
    frac = (x-grid[lo])/(grid[lo+1]-grid[lo])
    return lo, frac[:,None]


def interpolate(table, alpha_c, alpha_f):
    i, u = linear_indices(C_GRID,alpha_c)
    j, v = linear_indices(F_GRID,alpha_f)
    return ((1-u)*((1-v)*table[i,j]+v*table[i,j+1])
            + u*((1-v)*table[i+1,j]+v*table[i+1,j+1]))


def outcome_coefficients(scores):
    """Power coefficients of product_i (1 + alpha_f * likelihood_score_i)."""
    n, horizon = scores.shape
    coef = np.zeros((n,horizon+1))
    coef[:,0] = 1.
    for i in range(horizon):
        coef[:,1:i+2] += scores[:,i:i+1]*coef[:,:i+1]
    return coef


def moments_for_horizon(mass, horizon):
    powers=POST_GRID[:,None]**np.arange(horizon+2)[None,:]
    f_moments=mass.sum(axis=0)@powers
    c_times_f=(POST_GRID@mass)@powers[:,:horizon+1]
    return f_moments,c_times_f


class RankFlipScorer:
    """Two-dimensional interpolated K256 risk with posterior predictive lookahead."""

    @torch.inference_mode()
    def __init__(self, estimator, cache_path=None):
        # Static component densities and risks can be constructed once per deployment.
        path=Path(cache_path) if cache_path is not None else None
        if path is not None and path.is_file():
            with np.load(path) as f:
                self.table=f['table']; self.x_score=f['x_score']; self.z=f['z']
                if int(f['estimator_seed'])!=estimator.seed or int(f['bank_size'])!=estimator.bank_size:
                    raise ValueError("Risk cache was generated with different estimator")
        else:
            x=factual_batch(N_REFERENCE,REFERENCE_SEED,60,(0.,0.))
            ll=estimator.component_loglikelihood(x)
            self.x_score=np.tanh((ll[:,0]-ll[:,1])/2)
            self.table=np.empty((len(C_GRID),len(F_GRID),len(x)))
            for i,ac in enumerate(C_GRID):
                for j,af in enumerate(F_GRID):
                    engine=Engine(RISK_BANK_PAIR,float(ac),float(af))
                    for k,row in enumerate(x):
                        state=engine.fit(t(row))
                        risks=engine.risk(state,'keep256m2048')
                        self.table[i,j,k]=float(risks[0]-risks[2])
            # Neutral proposal. The outcome library uses the same physical sensor noise
            # as the purchased observation, but integrates over calibration-world X.
            count=N_OUTCOMES*MAX_HORIZON
            theta=draw_theta(count,OUTCOME_SEED,60,(0.,0.))
            mu,stick=impact(theta,0,PROBE_INCOMING)
            assert not bool(stick.any())
            noise=np.random.default_rng(OUTCOME_SEED+1).normal(size=(count,2))
            probe_white=mu/sigmas()[3]+t(noise)
            # Since the probe's position y=0, curvature-spread flips have no effect
            # on either probe velocity. Exact geometry cancellation is checked.
            all_centers=estimator.joint_centers[:,:,-2:]
            assert torch.equal(all_centers[:2],all_centers[2:])
            centers=all_centers[:2]
            norm=.5*centers.square().sum(-1)
            scores=[]
            for batch in probe_white.split(64):
                log_y=torch.logsumexp(batch@centers.transpose(-1,-2)-norm[:,None,:],-1)
                scores.append(torch.tanh((log_y[0]-log_y[1])/2).cpu().numpy())
            self.z=np.concatenate(scores).reshape(N_OUTCOMES,MAX_HORIZON)
            if path is not None:
                with path.open('wb') as f:
                    np.savez(f,table=self.table,x_score=self.x_score,z=self.z,
                             estimator_seed=estimator.seed,bank_size=estimator.bank_size)
        assert self.table.shape==(len(C_GRID),len(F_GRID),N_REFERENCE)
        assert self.z.shape==(N_OUTCOMES,MAX_HORIZON) and np.isfinite(self.table).all()
        self.coefficients={h:outcome_coefficients(self.z[:,:h]) for h in range(1,MAX_HORIZON+1)}

    def score(self,estimate,horizon=1,factual_logs=None,probe_ids=None,joint_logs=None):
        """Forecast the action-rank change after horizon new probes on random worlds.

        Future Y is forecast marginally on random worlds, without using their individual
        factual vectors. The acquisition table indexes product-prior posterior means,
        so the small cross-coupling covariance in the actual K256 bank is omitted in
        the lookup, while hypothetical means use full joint hyperposterior moments.
        """
        if horizon not in self.coefficients:
            raise ValueError("Unsupported lookahead horizon")
        ac,af=estimate['alpha']
        if factual_logs is not None:
            mass=posterior_grid(factual_logs,probe_ids,joint_logs)
            mf,mcf=moments_for_horizon(mass,horizon)
        elif horizon==1:
            mf=np.array([1.,af,af*af+estimate['sd'][1]**2])
            mcf=np.array([ac,ac*af+estimate['covariance']])
        else:
            raise ValueError("Horizon >1 requires the observed component logs")
        coef=self.coefficients[horizon]
        ratio=coef@mf[:-1]
        if np.min(ratio)<-1.e-10 or not np.isfinite(ratio).all():
            raise ArithmeticError("Invalid predictive density")
        # Product factors are non-negative on [-1,1]. Endpoint integration may
        # produce machine-zero likelihood for an impossible hypothetical outcome.
        pos=ratio>1.e-14
        prob=ratio[pos]/ratio[pos].sum()
        next_c=(coef[pos]@mcf)/ratio[pos]
        next_f=(coef[pos]@mf[1:])/ratio[pos]
        current_gap=interpolate(self.table,ac,af)
        after_gap=interpolate(self.table,next_c,next_f)
        flip=(current_gap>1e-12)!=(after_gap>1e-12)
        x_weight=np.maximum(0.,1+next_c[:,None]*self.x_score[None,:])
        x_weight/=x_weight.sum(1,keepdims=True)
        frequency=float(prob@(x_weight*flip).sum(1))
        regret=float(prob@(x_weight*flip*np.abs(after_gap)).sum(1))
        return dict(priority=regret,score=regret,flip_probability=frequency,
                    mean_switched_gap=regret/frequency if frequency>0 else 0.,
                    horizon=int(horizon),hypothetical_ess=float(1/(prob@prob)),
                    posterior_mean_grid_error=max(abs(mf[1]-af),abs(mcf[0]-ac)))
