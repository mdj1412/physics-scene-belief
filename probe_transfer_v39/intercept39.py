"""Audit whether the label-free router's upward-only correction exceeds global recalibration."""
import argparse
import hashlib
import json

import numpy as np
from scipy.optimize import minimize_scalar

from forecast39 import ROOT,OLD,CELLS,BETA,SHRINK,design_matrix


def offset(p,v):
    return p*np.exp(v)/(1+p*np.expm1(v))


def fit_old():
    p_all=[]; y_all=[]
    for cell in CELLS:
        for block in sorted((OLD/'results'/cell).glob('block_*')):
            for sign in ('plus','minus'):
                with np.load(block/f'shared_{sign}.npz') as f:
                    fixed=offset(f['p_base'],(BETA*SHRINK*f['z'])[:,None,None])
                # Squared loss averaged over regimes has same optimizer as the
                # conditional mean of their binary labels; variance is additive.
                labels=[]
                for tier in ('strong','weak','null'):
                    with np.load(block/f'{tier}_{sign}.npz') as f:
                        labels.append(f['labels'])
                p_all.append(fixed)
                y_all.append(np.mean(labels,axis=0)[:,:,None])
    p=np.stack(p_all); y=np.stack(y_all)
    mse=lambda b:np.mean((offset(p,b)-y)**2)
    res=minimize_scalar(mse,bounds=(-2.,2.),method='bounded',options={'xatol':1e-12})
    assert res.success and -1.99<res.x<1.99
    return dict(alpha=float(res.x),loss_definition='Brier after adding a single global log-odds intercept to fixed; mean regime labels remove the same additive constant for every candidate',
                loss_change=float(res.fun-mse(0)),mean_fixed_minus_incidence=float((p-y).mean()),
                mean_intercepted_minus_incidence=float((offset(p,res.x)-y).mean()), bounded_minimum=[-2,2],
                corpus='v38 only, 32 blocks, 2 mirror orientations, 3 tiers, 8 environments, 128 objects, 32 readings')


def assess():
    src=ROOT/'results'
    meta=json.loads((src/'completion.json').read_text())
    assert meta['blocks']==96
    alpha=json.loads((ROOT/'intercept_fit39.json').read_text())['alpha']
    coeff=np.array(json.loads((ROOT/'forecast39.json').read_text())['observable']['coefficients'])
    cells=meta['cells']
    vals=np.empty((4,12,2,7))
    bygroup=np.empty((4,12,2,2,4))
    for c,cell in enumerate(cells):
        for i in range(12):
            for j in range(2):
                with np.load(src/cell/f'catalog_{i:02d}'/f'block_{j:02d}.npz') as f:
                    base=f['p_base'][:,None]
                    z=f['z'][:,None,:]
                    w=f['weights']
                    y=f['labels'][...,None]
                    fixed=offset(base,(BETA*SHRINK*z)[...,None,None])
                    opt=offset(base,(BETA*z*w)[...,None,None])
                    group=(design_matrix(z,w)@coeff)<0
                    np.testing.assert_array_equal(group,(z*(w-SHRINK)>0))
                    routed=np.where(group[...,None,None],opt,fixed)
                    np.testing.assert_allclose(routed,np.maximum(opt,fixed),rtol=0,atol=2e-15)
                    bias=offset(fixed,alpha)
                    methods=[fixed,opt,routed,bias]
                    vals[c,i,j]=[*[np.mean((m-y)**2) for m in methods],
                                  *[np.mean(m-y) for m in (fixed,routed,bias)]]
                    for side,mask in enumerate((group,~group)):
                        bygroup[c,i,j,side]=[mask.mean(),*[np.mean((m-y)*mask[...,None,None]) for m in (fixed,routed,bias)]]
    v=vals.mean(axis=2)
    contrasts=np.stack((v[...,3]-v[...,0],v[...,2]-v[...,3],v[...,1]-v[...,3]),axis=-1)
    rng=np.random.default_rng(390056)
    ix=rng.integers(12,size=(20000,4,12))
    bs=contrasts[np.arange(4)[None,:,None],ix].mean((1,2))
    groups=bygroup.mean((0,1,2))
    out=dict(global_intercept=alpha,methods=['fixed','optimizer','route','global_intercept_fixed'],brier=v.mean((0,1))[:4],
             forecast_minus_incidence=v.mean((0,1))[4:],residual_methods=['fixed','route','global_intercept_fixed'],
             residual_by_routed_action={'optimizer':groups[0,1:]/groups[0,0], 'fixed':groups[1,1:]/groups[1,0]},
             contrasts=['intercept_minus_fixed','route_minus_intercept','optimizer_minus_intercept'],
             delta=contrasts.mean((0,1)),pointwise_ci=np.quantile(bs,[.025,.975],axis=0).T,
             source_fit_sha256=hashlib.sha256((ROOT/'intercept_fit39.json').read_bytes()).hexdigest())
    (ROOT/'intercept39.json').write_text(json.dumps(out,indent=2,default=lambda v:v.tolist())+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fit-v38',action='store_true')
    if parser.parse_args().fit_v38:
        (ROOT/'intercept_fit39.json').write_text(json.dumps(fit_old(),indent=2)+'\n')
    else:
        assess()
