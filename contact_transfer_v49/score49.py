"""Paired factorial comparison. No learned coefficient or policy is updated here."""
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parent
METHODS = ("original", "direct_none", "direct_local", "direct_fixed", "gate_local", "fixed_shared", "gate_fixed")
WEIGHT = np.array([.5, .25, .25])
SPATIAL = (0., .5, 1.)
RETENTION = (0., .25, .5, .75, 1.)
N = 12


def joint_state_weights(rho, retention, fixed_remote):
    # W query-weak, S sensor station, T query-full. The prediction uses S alone.
    spatial = rho*np.eye(3) + (1-rho)*np.tile(WEIGHT, (3, 1)) if fixed_remote else np.eye(3)
    load = np.array([[1, 0, 0], [0, retention, 1-retention], [0, 1-retention, retention]])
    weights = spatial.T @ (WEIGHT[:, None]*load)
    np.testing.assert_allclose(weights.sum(0), WEIGHT)
    np.testing.assert_allclose(weights.sum(1), WEIGHT)
    return weights


def interval(x, comparisons=1):
    mean = x.mean(axis=0)
    half = x.std(axis=0, ddof=1)*t.ppf(1-.025/comparisons, len(x)-1)/np.sqrt(len(x))
    return np.stack([mean, mean-half, mean+half], axis=-1)


def panel(i):
    name = f'panel_{i:02d}.npz'
    with np.load(ROOT/'forecast49'/name) as a, np.load(ROOT/'labels49'/name) as b:
        assert tuple(a['methods'].astype(str)) == METHODS
        p = a['p']
        y = b['labels']
    with np.load(ROOT.parent/'preload_projection_v48'/'labels'/name) as b:
        y48 = b['labels'].astype(bool)
    assert p.shape == (7,4,2,3,2,2,4,128,4)
    assert y.shape == (4,2,3,2,3,4,128)
    assert np.isfinite(p).all() and np.min(p)>=-1e-9 and np.max(p)<=1+1e-9
    assert np.isin(y,[0,1]).all()
    # At 31 mm local and fixed are the same policy, measurement and noise realization.
    np.testing.assert_allclose(p[2,:,1], p[3,:,1], atol=1e-12, rtol=0)
    np.testing.assert_allclose(p[4,:,1], p[6,:,1], atol=1e-12, rtol=0)
    kernel = np.empty((4,2,len(METHODS),3,3))  # F,H,M,sensor weak state,query full state
    for f,h in np.ndindex(4,2):
        prob = p[:,f,h]
        label_rate = y[f,h].mean(axis=(1,2))
        mean_p = prob.mean((2,3,6))  # M,S,E,O
        mean_p2 = (prob**2).mean((2,3,4,5,6))
        cross = np.einsum('mseo,teo->mst',1-2*mean_p,label_rate)/(4*128)
        kernel[f,h] = mean_p2[:,:,None] + cross
        # Independent enumeration of actual squared errors over six binary solver outcomes;
        # rotate a policy and (sensor, full) state choice across catalogs, families, heights.
        m = (i+f+h) % len(METHODS)
        s,full = (i+f)%3,(i+h)%3
        observed = prob[m,s]
        raw = np.mean([((observed-y[f,h,full,r,x,...,None])**2).mean() for r,x in np.ndindex(2,3)])
        np.testing.assert_allclose(kernel[f,h,m,s,full],raw,rtol=0,atol=3e-12)
    diag = dict(stick=y.mean((0,1,3,4,5,6)), stick48=y48.mean((0,1,3,4,5,6)),
                different_solver=(y!=y48).mean((0,1,3,4,5,6)), sign_flip=(y[:,:,1]!=y[:,:,2]).mean(),
                sign_flip48=(y48[:,:,1]!=y48[:,:,2]).mean(),
                predictions=p.mean((1,2,4,5,6,7,8)))
    return kernel,diag


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'run in an allocation'
    parts = [panel(i) for i in range(N)]
    kernel = np.stack([k for k,_ in parts])
    # The physical station is fixed for methods direct_fixed, fixed_shared, gate_fixed.
    fixed = ('direct_fixed','fixed_shared','gate_fixed')
    values = np.empty((N,3,5,7,4,2))
    for r,rho in enumerate(SPATIAL):
        for c,keep in enumerate(RETENTION):
            for m,method in enumerate(METHODS):
                for h in range(2):
                    weights=joint_state_weights(rho,keep,method in fixed and h==0)
                    values[:,r,c,m,:,h]=np.einsum('nfst,st->nf',kernel[:,:,h,m],weights)
    scores=values.mean((-1,-2))
    # Brier expectation is bilinear in the two scenario probabilities, including the mixed term.
    for r,rho in enumerate(SPATIAL):
        for c,keep in enumerate(RETENTION):
            corner=(1-rho)*((1-keep)*scores[:,0,0]+keep*scores[:,0,-1])+rho*((1-keep)*scores[:,-1,0]+keep*scores[:,-1,-1])
            np.testing.assert_allclose(scores[:,r,c],corner,rtol=0,atol=1e-12)
    np.testing.assert_allclose(scores[:,0,:,4],scores[:,2,:,4],rtol=0,atol=1e-12)
    np.testing.assert_allclose(values[:,:,:,2,:,1],values[:,:,:,3,:,1],rtol=0,atol=1e-12)
    effects=scores[:,:,:,2,None]-scores[:,:,:,[3,4]]
    roots=[]
    for r,rho in enumerate(SPATIAL):
        bounds=effects[:,r,[0,-1],:].mean(0)
        root=np.divide(-bounds[0],bounds[1]-bounds[0],out=np.full(2,np.nan),where=abs(bounds[1]-bounds[0])>1e-12)
        roots.append([float(x) if 0<=x<=1 else None for x in root])
    diag={k:interval(np.stack([v[k] for _,v in parts])) for k in parts[0][1]}
    np.savez_compressed(ROOT/'per_catalog49.npz',risk_kernel=kernel,loss=scores,by_family_height=values,
                        rho=SPATIAL,kappa=RETENTION,methods=METHODS)
    out=dict(methods=METHODS,rho=SPATIAL,kappa=RETENTION,sample_unit='12 paired catalogs reused from v48',
             loss=scores.mean(0),loss_pointwise95=interval(scores),primary_local_minus_fixed_and_gate=interval(effects,30),
             counts_negative=(effects<0).sum(0),roots_in_unit_interval=roots,
             direct_local_minus_none=interval(scores[:,:,:,2]-scores[:,:,:,1],15),
             direct_local_minus_fixed_shared=interval(scores[:,:,:,2]-scores[:,:,:,5],15),
             by_height=values.mean((0,4)),solver_diagnostics_pointwise95=diag,
             algebra_checks='Independent binary squared error per catalog x family x height, equal-site replay, prior marginals and all bilinear corners pass.')
    (ROOT/'score49.json').write_text(json.dumps(out,default=lambda x:np.asarray(x).tolist(),indent=1))
    print('rho/kappa grids; primary difference point estimates',np.round(effects.mean(0),6))
    print('zero crossing kappa per rho vs fixed/gate',roots)


if __name__=='__main__':
    main()
