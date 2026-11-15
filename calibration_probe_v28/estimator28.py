"""Update the v27 factual population posterior with paired calibration probe responses."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "population_shift_v27"))
from estimator27 import FactualShiftEstimator, factual_mean
from physics26 import (DT, UNIT_DIM, PROBE_INCOMING, V25Bank, V25Engine, change_proxy,
                       independent_uniforms, labels, observations, parameters, sigmas, t, torch)
from physics25 import impact


def independent_masses(alpha):
    """One new world; v26 code order is aligned, friction flip, profile flip, both."""
    aligned_c, aligned_f = (1+np.asarray(alpha, dtype=float))/2
    return np.outer([aligned_c, 1-aligned_c], [aligned_f, 1-aligned_f]).ravel()


def summaries(nodes, density):
    """Summaries of a continuous marginal density using composite trapezoid quadrature."""
    q = np.ones(len(nodes))
    q[[0,-1]] = .5
    mass = q*density
    mass /= mass.sum()
    cdf = np.r_[0.,np.cumsum(density[:-1]+density[1:])]
    cdf /= cdf[-1]
    mean = mass@nodes
    prior_mass = q/q.sum()
    pos = mass>0
    kl = (mass[pos]*np.log(mass[pos]/prior_mass[pos])).sum()
    return dict(mean=float(mean),sd=float(np.sqrt(mass@(nodes-mean)**2)),
                ci95=np.interp([.025,.975],cdf,nodes).tolist(),
                prob_negative=float(np.interp(0.,nodes,cdf)),kl=float(max(0,kl)))


class ProbeShiftEstimator(FactualShiftEstimator):
    """Independent uniform hyperpriors for two couplings, updated by joint x,y_probe data."""

    @torch.inference_mode()
    def __init__(self, seed=270431, bank_size=8192):
        super().__init__(seed, bank_size)
        # Recreate precisely the v27 symmetrized quadrature particles so replacing a
        # factual factor with a joint factor does not introduce a second numerical law.
        half = bank_size//2
        unit = torch.quasirandom.SobolEngine(UNIT_DIM, scramble=True, seed=self.seed).draw(half,dtype=DT)
        lower = unit.clone()
        lower[:,1] *= .5
        upper = lower.clone()
        upper[:,1] += .5
        theta = parameters(t(torch.cat([lower,upper])))
        self.probe_sigma = sigmas()[3]
        joint_centers = []
        for code in range(4):
            which = torch.full((bank_size,),code,dtype=torch.uint8,device=theta.device)
            th = change_proxy(theta, which)
            fact = factual_mean(th)/self.sigma
            expected = self.centers[code//2]
            if code%2:
                expected = expected.roll(half,dims=0)
            if not torch.allclose(fact,expected,rtol=0,atol=1e-11):
                raise ArithmeticError("Mismatched factual integration banks.")
            probe,stick = impact(th,0,PROBE_INCOMING)
            if bool(stick.any()):
                raise ValueError("Calibration probe crossed out of the sliding regime.")
            joint_centers.append(torch.cat([fact,probe/self.probe_sigma],dim=1))
        self.joint_centers = torch.stack(joint_centers)
        self.joint_energy = .5*self.joint_centers.square().sum(-1)
        self.grid_f = np.linspace(-1.,1.,201)
        with np.errstate(divide="ignore"):
            self.log_c = np.log(np.stack([(1+self.grid)/2,(1-self.grid)/2]))
            self.log_f = np.log(np.stack([(1+self.grid_f)/2,(1-self.grid_f)/2]))

    @torch.inference_mode()
    def joint_component_loglikelihood(self, factual_selected, probe_y):
        """Per-row log p(X,Y_probe|flip code) up to a shared constant for four codes."""
        xx,yy = t(factual_selected),t(probe_y)
        if xx.ndim!=2 or xx.shape[1]!=11 or yy.shape!=(len(xx),2):
            raise ValueError("Expected paired selected arrays [K,11], [K,2].")
        if not bool(torch.isfinite(xx).all() and torch.isfinite(yy).all()):
            raise ValueError("Non-finite calibration sensor reading.")
        if not len(xx):
            return np.empty((0,4))
        raw = torch.cat([xx/self.sigma,yy/self.probe_sigma],dim=1)
        result = []
        for group in raw.split(64):
            log_density = group@self.joint_centers.transpose(1,2) - self.joint_energy[:,None,:]
            result.append(torch.logsumexp(log_density,-1).T - np.log(self.bank_size))
        return torch.cat(result).cpu().numpy()

    @staticmethod
    def _check_ids(probe_ids, n):
        ids = np.asarray(probe_ids)
        if ids.ndim!=1 or (len(ids) and ids.dtype.kind not in "iu"):
            raise ValueError("probe_ids must be a vector of integer calibration-world indices.")
        ids = ids.astype(np.int64)
        if len(np.unique(ids))!=len(ids) or np.any((ids<0)|(ids>=n)):
            raise ValueError("Probe observations must come from distinct, valid calibration worlds.")
        return ids

    def posterior_from_components(self, factual_logs, probe_ids, joint_logs):
        ll = np.asarray(factual_logs,dtype=float)
        if ll.ndim!=2 or not len(ll) or ll.shape[1]!=2 or not np.all(np.isfinite(ll)):
            raise ValueError("Expected finite factual component logs [N,2].")
        ids = self._check_ids(probe_ids, len(ll))
        joint = np.asarray(joint_logs,dtype=float)
        if joint.shape!=(len(ids),4) or not np.all(np.isfinite(joint)):
            raise ValueError("Expected four finite joint component log likelihoods per probe.")
        if not len(ids):
            # Reproduce v27 full factual-only output and preserve exact symmetry at budget zero.
            out = super().posterior(ll)
            w = independent_masses(out["alpha"])
            return dict(out, n_probes=0, covariance=0.,variant_probabilities=w.tolist(),
                        predictive_tv_vs_product=0.,quadrature_shape=[len(self.grid),len(self.grid_f)])
        mask = np.ones(len(ll),dtype=bool)
        mask[ids] = False
        remaining = ll[mask]
        if len(remaining):
            # Per-row centering removes high-dimensional common X constants.
            remaining = remaining - remaining.max(axis=1,keepdims=True)
            profile_log = np.logaddexp(remaining[:,0,None]+self.log_c[0],
                                       remaining[:,1,None]+self.log_c[1]).sum(0)
        else:
            profile_log = np.zeros(len(self.grid))
        # Exactly one likelihood factor per calibration world: X only, or (X,Y_probe).
        log_grid = np.repeat(profile_log[:,None],len(self.grid_f),axis=1)
        for row in joint:
            centered = row - row.max()
            by_profile = np.logaddexp(centered[::2,None]+self.log_f[0],
                                      centered[1::2,None]+self.log_f[1])
            log_grid += np.logaddexp(self.log_c[0,:,None]+by_profile[0],
                                     self.log_c[1,:,None]+by_profile[1])
        density = np.exp(log_grid-log_grid.max())
        qc = np.ones(len(self.grid));qc[[0,-1]]=.5
        qf = np.ones(len(self.grid_f));qf[[0,-1]]=.5
        c = summaries(self.grid,density@qf)
        f = summaries(self.grid_f,qc@density)
        mass = density*qc[:,None]*qf
        mass /= mass.sum()
        cov = float((self.grid-c['mean'])@mass@(self.grid_f-f['mean']))
        alpha = [c["mean"],f["mean"]]
        # E[(1+s_c alpha_c)(1+s_f alpha_f)]/4 includes signed hyperposterior covariance.
        product = independent_masses(alpha)
        weights = product + .25*cov*np.array([1.,-1.,-1.,1.])
        if not np.all(weights>=-1.e-12) or not np.isclose(weights.sum(),1.):
            raise ArithmeticError("Invalid posterior predictive component weights.")
        weights = np.maximum(weights,0);weights/=weights.sum()
        axis=(c,f)
        return dict(alpha=alpha,ci95=[a["ci95"] for a in axis],sd=[a["sd"] for a in axis],
                    posterior_prob_negative=[a["prob_negative"] for a in axis],
                    posterior_kl_nats=[a["kl"] for a in axis],covariance=cov,
                    n_worlds=len(ll),n_probes=len(ids),variant_probabilities=weights.tolist(),
                    predictive_tv_vs_product=float(.5*np.abs(weights-product).sum()),
                    bank_size=self.bank_size,bank_seed=self.seed,
                    max_abs_factual_friction_pair_residual=self.symmetry_error,
                    quadrature_shape=[len(self.grid),len(self.grid_f)])

    def fit(self, factual, probe_ids, probe_y):
        xx=np.asarray(factual,dtype=float)
        ids=self._check_ids(probe_ids,len(xx))
        ll=self.component_loglikelihood(xx)
        joint=self.joint_component_loglikelihood(xx[ids],probe_y)
        return self.posterior_from_components(ll,ids,joint)


class PredictiveBank(V25Bank):
    """Sample the four full posterior predictive masses with shared v26 random numbers."""

    def __init__(self,seed,weights):
        super().__init__(seed)
        w = np.asarray(weights,dtype=float).reshape(2,2)
        u = independent_uniforms(seed,self.N,260401)
        q_c = w[1].sum()
        cflip = u[:,0]<q_c
        totals=w.sum(1)
        q_f = np.divide(w[:,1],totals,out=np.zeros(2),where=totals>0)
        fflip = u[:,1]<q_f[cflip.astype(int)]
        assignment = 2*cflip.astype(np.uint8)+fflip.astype(np.uint8)
        self.theta = change_proxy(self.theta,t(torch.from_numpy(assignment)).to(torch.uint8))
        self.mu = dict(enumerate(observations(self.theta)))
        self.white = {a:m/self.sigma[a] for a,m in self.mu.items()}
        self.norm = {a:.5*m.square().sum(-1) for a,m in self.white.items()}
        self.label = labels(self.theta).to(DT)[:,None]*np.sqrt(2.)


class PosteriorPredictiveEngine(V25Engine):
    """Same K256 acquisition and decoder as v26, now using four predictive masses."""

    def __init__(self,pair,estimate):
        w=np.asarray(estimate["variant_probabilities"],dtype=float)
        if w.shape!=(4,) or not np.isfinite(w).all() or np.any(w<0) or not np.isclose(w.sum(),1):
            raise ValueError("Expected four normalized posterior predictive masses.")
        super().__init__(pair)
        if not np.array_equal(w,[1,0,0,0]):
            self.inner = PredictiveBank(pair[0],w)
            self.outer = PredictiveBank(pair[1],w)
