"""Paired deployment evaluation for a calibration specimen with an impact-direction readout."""
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parent
V52 = ROOT.parent / 'postswap_adaptation_v52'
V50 = ROOT.parent / 'polarity_calibration_v50'
sys.path.insert(0, str(V52))
from score52 import weighted_cdf_inverse

E = np.array([0., .25, .5])
G = np.array([0., .25, .5, 1.])
KAPPA = np.linspace(0., 1., 5)
N, R = 12, 256
OLD = ['v52_selector','v52_vote_selector','v52_post_vote','v52_joint','unsigned']
PRIMARY = ['impact_selector','impact_vote_selector']

def quadratic(risk, alpha, unsigned):
    plus, minus, blind, spread = np.moveaxis(risk, -1, 0)
    return np.where(unsigned, blind, minus + alpha * (plus - minus) - alpha * (1. - alpha) * spread)

def old_actions(file):
    with np.load(V52 / 'decisions52' / file) as d:
        a = np.empty((2, 5, 4, R, len(OLD)))
        u = np.zeros_like(a, bool)
        for event, k in np.ndindex(2, 5):
            ix = (k, 4 - k if event else k, 3)
            a[event, k, ..., :2] = d['alpha'][ix]
            u[event, k, ..., :2] = d['action'][ix] == 2
            a[event, k, ..., 2] = d['new'][ix][..., 1]
            a[event, k, ..., 3] = d['target_alpha'][ix][..., 0]
            a[event, k, ..., 4] = .5
            u[event, k, ..., 4] = True
    return a, u

def secondary_deployment(i):
    file = f'panel_{i:02}.npz'
    with np.load(V50 / 'forecast50' / file) as f:
        forecasts = f['p'][[list(f['methods'].astype(str)).index(key) for key in
                             ('local_plus', 'local_minus', 'local_unsigned_fit')]]
    with np.load(V50 / 'labels50' / file) as f:
        labels = f['labels']
    rng = np.random.default_rng([530049721, i])
    shape = (4, R, 128)
    env = np.arange(4)[:, None, None]
    obj = np.arange(128)[None, None, :]
    family, height, axis, variant, radius, offset, noise = [rng.integers(k, size=shape) for k in (4,2,2,2,2,3,4)]
    weak = rng.choice(3, size=shape, p=(.5,.25,.25))
    rev = np.where(weak, 3-weak, 0)
    plus, minus, un = forecasts[:,family,height,weak,axis,variant,env,obj,noise]
    y = labels[family,height,weak,radius,offset,env,obj]
    yrev = labels[family,height,rev,radius,offset,env,obj]
    pol, err = rng.random((2, *shape))
    risk = np.empty((3,2,5,4,R,4))
    spread = ((plus-minus)**2).mean(-1)
    for j, event, k in np.ndindex(3,2,5):
        actual = np.where((pol >= KAPPA[4-k if event else k]) ^ (err < E[j]), yrev, y)
        risk[j,event,k] = np.stack([((p-actual)**2).mean(-1) for p in (plus, minus, un)]+[spread], -1)
    return risk

def panel(i):
    file = f'panel_{i:02}.npz'
    with np.load(ROOT / 'decisions53' / file) as d:
        names = list(d['methods'].astype(str))
        a, u = d['alpha'], d['unsigned'].astype(bool)
        assert a.shape == u.shape == (3,2,5,4,R,len(names))
        assert np.isfinite(a).all() and np.all((a>=0)&(a<=1))
    old_a, old_u = old_actions(file)
    # Four-reference control must coincide with the archived frozen prediction, prefix by prefix.
    np.testing.assert_allclose(a[...,names.index('reference4_joint')], np.broadcast_to(old_a[...,3], a.shape[:-1]), atol=2e-11, rtol=0)
    a = np.concatenate((a, np.broadcast_to(old_a, (3, *old_a.shape))), -1)
    u = np.concatenate((u, np.broadcast_to(old_u, (3, *old_u.shape))), -1)
    with np.load(V52 / 'risk52' / file) as rr:
        ex, raw, surface = rr['exact'], rr['realized'], rr['surface']
    quarter = np.empty((*ex.shape[:-1], a.shape[-1]))
    real = np.empty_like(a, dtype=np.float32)
    replay = np.empty_like(real)
    sec = secondary_deployment(i)
    for m in range(a.shape[-1]):
        quarter[...,m] = quadratic(ex[...,None,:], a[...,m], u[...,m]).mean(-1)
        real[...,m] = quadratic(raw, a[...,m], u[...,m])
        replay[...,m] = quadratic(sec, a[...,m], u[...,m])
    assert np.all((real>=0)&(real<=1)) and np.all((replay>=0)&(replay<=1))
    return names + OLD, quarter, real, replay, surface

def pair_ci(delta, multiplicity):
    mu = delta.mean(0)
    half = t.ppf(1-.025/multiplicity, N-1)*delta.std(0, ddof=1)/np.sqrt(N)
    return np.stack((mu, mu-half, mu+half), -1)

def quantiles(raw, g):
    result = []
    for m in range(raw.shape[-1]):
        arrays = [np.sort(np.take(raw[...,m], ev, axis=2).ravel()) for ev in (0,1)]
        result.append(weighted_cdf_inverse(arrays, (1-g, g)))
    return np.asarray(result)

def bootstrap(raw):
    # Same catalog weights for every arm preserve the common session/impact realization.
    draws = np.random.default_rng(5300113).integers(N, size=(5000,N))
    weight = (draws[:,:,None] == np.arange(N)).mean(1)
    g = G.mean()
    qs = []
    for m in range(raw.shape[-1]):
        groups = [[np.sort(raw[i,:,ev,...,m].ravel().astype(float)) for ev in (0,1)] for i in range(N)]
        lo = np.full(len(draws), min(a[0] for row in groups for a in row)-1e-7)
        hi = np.full(len(draws), max(a[-1] for row in groups for a in row)+1e-7)
        for _ in range(27):
            mid = (lo+hi)/2
            frac = np.stack([((1-g)*np.searchsorted(a, mid, 'right')+g*np.searchsorted(b, mid, 'right'))/a.size for a,b in groups], 1)
            below = (weight*frac).sum(1)<.9
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        qs.append(hi)
    return np.stack(qs,1)

def summarize():
    if not os.getenv('SLURM_JOB_ID'):
        raise RuntimeError('Run on an allocated compute node.')
    panels = [panel(i) for i in range(N)]
    names = panels[0][0]
    assert all(p[0] == names for p in panels)
    exact,real,repeat,surfaces = [np.stack([p[i] for p in panels]) for i in range(1,5)]
    del panels
    # exact [catalog,resolution,e,event,k,env,method], realized [catalog,e,event,k,env,rep,method].
    cells = np.stack([(1-g)*exact[:,:,:,0]+g*exact[:,:,:,1] for g in G], 3)
    means = cells.mean((2,3,4,5)); population = means.mean(0)
    primary_q = quantiles(real, G.mean())
    repeat_q = quantiles(repeat, G.mean())
    qdist = bootstrap(real)
    def contrast(weights, family=1):
        a = np.zeros(len(names))
        for name, coefficient in weights.items(): a[names.index(name)] = coefficient
        mean_ci = pair_ci(means[:,0]@a, family)
        bs = qdist@a
        return dict(mean=mean_ci, p90=[primary_q@a,*np.quantile(bs, [.025/family,1-.025/family])],
                    ultra=pair_ci(means[:,1]@a, family), second_p90=float(repeat_q@a),
                    wins=int((means[:,0]@a<0).sum()))
    primary = {name:contrast({'impact_joint':1, name:-1}, 4) for name in PRIMARY}
    others = {name:contrast({'impact_joint':1, name:-1}) for name in names if name not in PRIMARY+['impact_joint']}
    controls = {
      'selector_impact_minus_reference':contrast({'impact_selector':1,'v52_selector':-1}),
      'same_menu_vote_impact_minus_reference':contrast({'impact_vote_selector':1,'v52_vote_selector':-1}),
      'joint_versus_selector_interaction':contrast({'impact_joint':1,'reference4_joint':-1,'impact_selector':-1,'v52_selector':1})}
    surfaces_mean = {}
    for s in np.unique(surfaces):
        az=np.stack([np.compress(surfaces[i]==s, cells[i,0], axis=3).mean((0,1,2,3)) for i in range(N)])
        surfaces_mean[str(s)] = az.mean(0)
    rng = np.random.default_rng(5301561)
    fleet = []; risk90 = []
    for r in range(R):
        ix = rng.permutation(np.tile(np.arange(12),4)).reshape(12,4)
        e, g = ix//4, G[ix%4]
        event = (rng.random((12,4)) < g).astype(int)
        k = rng.permutation(np.r_[np.tile(np.arange(5),9), rng.choice(5,3,replace=False)]).reshape(12,4)
        values = real[np.arange(12)[:,None],e,event,k,np.arange(4)[None,:],r].reshape(48,len(names))
        fleet.append(values.mean(0)); risk90.append(np.quantile(values,.9,axis=0))
    fleet, risk90 = np.asarray(fleet), np.asarray(risk90)
    best = names.index('impact_joint')
    cq = [[quantiles(real[:,j:j+1], g) for g in G] for j in range(len(E))]
    summary = dict(methods=names, e=E,g=G,kappa=KAPPA, catalogs=N, prefixes=R,mean=population[0],
                   ultra=population[1],p90=primary_q,secondary_seed_p90=repeat_q,primary=primary,
                   secondary=others,design_contrasts=controls, means_by_surface=surfaces_mean,
                   cell_means=cells[:,0].mean((0,3,4)),cell_p90=cq,
                   mean_by_k=cells[:,0].mean((0,1,2,4)),fleet_mean_worse=(fleet[:,best,None]>fleet).mean(0),
                   fleet_p90_worse=(risk90[:,best,None]>risk90).mean(0))
    # Identical held-out draws recover the historical benchmark, even at the quantile endpoint.
    archived=json.loads((V52 / 'score52.json').read_text())
    ix = [archived['methods'].index(s) for s in ['selector','vote_selector','post_vote_soft','continuous_joint','unsigned']]
    np.testing.assert_allclose(population[0,-len(OLD):], np.array(archived['mean'])[0,3,ix], atol=2e-10,rtol=0)
    np.testing.assert_allclose(primary_q[-len(OLD):], np.array(archived['absolute_p90'])[3,ix], atol=2e-8,rtol=0)
    np.savez_compressed(ROOT/'catalog53.npz',methods=names,exact=exact,surfaces=surfaces)
    np.savez_compressed(ROOT/'session53.npz',methods=names,realized=real,replay=repeat)
    (ROOT/'score53.json').write_text(json.dumps(summary, default=lambda x:np.asarray(x).tolist(), indent=1))
    for i,name in enumerate(names): print(name, population[0,i],primary_q[i],flush=True)
    print('PRIMARY',primary,flush=True)

if __name__ == '__main__': summarize()
