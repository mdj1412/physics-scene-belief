"""Score a frozen v54 acquisition and a rate-matched marginal lottery on new catalogs."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
V54 = ROOT.parent / 'adaptive_witness_v54'
sys.path.insert(0, str(V54))
from policy54 import select54
from policy53 import choice53

E = np.array([0., .25, .5])
K = np.linspace(0., 1., 5)
G = np.mean([0., .25, .5, 1.])
C, S, R, O = 24, 4, 256, 128
METHODS = ['gate', 'fixed3RI', 'rate_matched_random', 'fixed4R', 'within_catalog_random']
# Last axis in replay/exact: gate, always I, always R. The lotteries marginalize potential scores.
GATE, IMP, REF = 0, 1, 2
REPLAY_SEEDS = [55018591, 55029371]


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=1, default=lambda x: np.asarray(x).tolist()))


def coefficients(plus, minus, keep, flip):
    """Conditional risk when deployment weak and impact directions agree vs disagree."""
    spread = ((plus - minus)**2).mean(-1)
    return np.stack([np.stack([((plus-y)**2).mean(-1), ((minus-y)**2).mean(-1), spread], -1)
                     for y in (keep, flip)])


def quadratic(coeff, alpha):
    a, b, spread = np.moveaxis(coeff, -1, 0)
    return b + alpha*(a-b) - alpha*(1-alpha)*spread


def deploy_arrays(plus, minus, keep, flip, panel, seed):
    # Objects are sampled independently WITH replacement from this catalog/environment population.
    # The same deployment draws service both alternative fourth measurements and all policies.
    # Independent R indices yield independent sessions conditional on this catalog.
    rng = np.random.default_rng([seed, panel])
    shape = (S, R, O)
    ids = rng.integers(O, size=shape)
    env = np.arange(S)[:, None, None]
    x, z, y, ry = [a[env, ids] for a in (plus, minus, keep, flip)]
    orientation, mismatch = rng.random((2, *shape))
    out = {}
    spread=((x-z)**2).mean(-1)
    for ie, error in enumerate(E):
        for ik, k in enumerate(K):
            pick = (orientation >= k) ^ (mismatch < error)
            target = np.where(pick, ry, y)
            out[ie, ik] = np.stack([((x-target)**2).mean(-1), ((z-target)**2).mean(-1), spread], -1)
    return out


def panel_score(panel, dest):
    src = ROOT/'panels'/f'panel_{panel:02d}.npz'
    with np.load(src) as f:
        nw, nf = f['nw'],f['nf']
        assert nw.shape==nf.shape==(5,S,R,4)
        # No exposure of the selector to the old history, fourth weak readout, sign, errors or labels.
        gate = select54(nw[..., :3], nf[..., :3], mode='robust')
        assert gate.shape==(5,S,R)
        ow, of, imp = f['ow'],f['of'],f['imp']
        assert ow.shape==of.shape==nw.shape and imp.shape==(3,5,S,R)
        alpha = np.empty((3,2,5,S,R,3))
        for event, k in np.ndindex(2,5):
            j=4-k if event else k
            ar=choice53('reference4_joint',ow[k],of[k],nw[j],nf[j],imp[0,j],imp[0,j],None)[1]
            for ie in range(3):
                ai=choice53('impact_joint',ow[k],of[k],nw[j],nf[j],imp[ie,j],imp[ie,j],None)[1]
                alpha[ie,event,k]=np.stack([np.where(gate[j],ai,ar),ai,ar],-1)
        assert np.isfinite(alpha).all() and np.all((alpha>=0)&(alpha<=1))
        plus, minus, keep, flip = [np.asarray(f[name],float) for name in ('pplus','pminus','y_keep','y_flip')]
        assert all(a.shape==(S,O) for a in (plus,minus,keep,flip))
        assert np.isin(keep,[0,1]).all() and np.isin(flip,[0,1]).all()
        assert all(((a>=0)&(a<=1)).all() for a in (plus,minus))
        exact_coeff=coefficients(plus,minus,keep,flip)
        have_ultra=all(key in f for key in ('y_keep_ultra','y_flip_ultra'))
        if have_ultra:
            ultra_coeff=coefficients(plus,minus,f['y_keep_ultra'],f['y_flip_ultra'])
    exact=np.empty_like(alpha)
    real=np.empty((2,*alpha.shape), np.float32)
    ultra=np.empty_like(alpha) if have_ultra else None
    seeds=[deploy_arrays(plus,minus,keep,flip,panel,s) for s in REPLAY_SEEDS]
    for ie,event,k in np.ndindex(3,2,5):
        j=4-k if event else k
        q=E[ie]+(1-2*E[ie])*K[j]
        a=alpha[ie,event,k]
        def mean_loss(coeff):
            weighted=q*coeff[0]+(1-q)*coeff[1]
            return quadratic(weighted[:,None,None],a)
        exact[ie,event,k]=mean_loss(exact_coeff)
        if have_ultra: ultra[ie,event,k]=mean_loss(ultra_coeff)
        for s,risk in enumerate(seeds):
            real[s,ie,event,k]=quadratic(risk[ie,j][:,:,None],a)
        if k==2 and event==1:
            ax=a[0,0,:,None]
            forecast=minus[0]+ax*(plus[0]-minus[0])
            direct=(q*(forecast-keep[0])**2+(1-q)*(forecast-flip[0])**2).mean(-1)
            np.testing.assert_allclose(exact[ie,event,k,0,0],direct,atol=1e-13)
        for arm,choice in [(IMP,True),(REF,False)]:
            use=gate[j]==choice
            np.testing.assert_array_equal(exact[ie,event,k,...,GATE][use],exact[ie,event,k,...,arm][use])
    assert ((real>=0)&(real<=1)).all() and ((exact>=0)&(exact<=1)).all()
    # Keep exact mean by e, env and pre-k for distribution transfer without re-running predictions.
    def bycell(loss): return ((1-G)*loss[:,0]+G*loss[:,1]).mean(axis=3) # e, k, env, method
    save=dict(exact_cells=bycell(exact),real=real,gate=gate,
              rate=gate.mean(),rate_env=gate.mean((0,2)),source_sha256=sha256(src.read_bytes()).hexdigest())
    if have_ultra:save['ultra_cells']=bycell(ultra)
    dest.mkdir(exist_ok=True)
    tmp=dest/f'panel_{panel:02d}.tmp.npz'
    np.savez_compressed(tmp,**save)
    tmp.replace(dest/src.name)


def mean_matrix(base,rate,within):
    # base shape C, ... ,3. rate can be scalar while within has one entry per catalog.
    gate,imp,ref=np.moveaxis(base,-1,0)
    k=np.asarray(within).reshape((len(within),)+(1,)*(imp.ndim-1))
    return np.stack([gate,imp,ref+rate*(imp-ref),ref,ref+k*(imp-ref)],-1)


def quantile(arms,weights,sorts,rates):
    # arms is one action or a lottery over REF/IMP. For the global lottery, probability
    # of I is re-estimated from the same weighted catalog sample for each bootstrap.
    cols=[GATE] if arms=='gate' else [IMP] if arms=='fixed3RI' else [REF] if arms=='fixed4R' else [REF,IMP]
    B=len(weights)
    mix=(weights@rates)[:,None] if arms=='rate_matched_random' else rates[None,:]
    lo=np.broadcast_to(min(sorts[c,ev,a][0] for a in cols for c in range(C) for ev in (0,1))-1e-8,(B,)).copy()
    hi=np.broadcast_to(max(sorts[c,ev,a][-1] for a in cols for c in range(C) for ev in (0,1)),(B,)).copy()
    for it in range(28):
        mid=(lo+hi)*.5
        total=np.zeros(B)
        for a in cols:
            fractions=np.column_stack([((1-G)*np.searchsorted(sorts[c,0,a],mid,'right')+
                                      G*np.searchsorted(sorts[c,1,a],mid,'right'))/len(sorts[c,0,a]) for c in range(C)])
            factors=(mix if a==IMP else 1-mix) if len(cols)==2 else 1.
            total+=(fractions*weights*factors).sum(1)
        below=total<.90
        lo=np.where(below,mid,lo); hi=np.where(below,hi,mid)
    return hi


def sorted_cells(real):
    # real C,e,event,k,env,R,3
    return {(c,event,a):np.sort(real[c,:,event,...,a].ravel().astype(float)) for c,event,a in np.ndindex(C,2,3)}


def boot_means(weights,base,rates):
    tmp=weights@base
    rr=weights@rates
    g,i,r=tmp.T
    return np.stack([g,i,r+rr*(i-r),r,weights@(base[:,REF]+rates*(base[:,IMP]-base[:,REF]))],1)


def summ(dest):
    files=[dest/f'panel_{c:02d}.npz' for c in range(C)]
    assert all(p.is_file() for p in files)
    data=[]
    for p in files:
        with np.load(p) as f:data.append({k:f[k] for k in f.files})
    cells=np.stack([d['exact_cells'] for d in data])
    real=np.stack([d['real'] for d in data],1)
    rates=np.array([d['rate'] for d in data])
    r=rates.mean()
    by_e=mean_matrix(cells.mean((2,3)),r,rates)
    by_surface=mean_matrix(cells.mean((1,2)),r,rates)
    means=by_e.mean((0,1))
    real_by_e=np.stack([mean_matrix(((1-G)*raw[:,:,0]+G*raw[:,:,1]).mean((2,3,4)),r,rates) for raw in real])
    uniform=np.full((1,C),1/C)
    rng=np.random.default_rng(5505102)
    draws=rng.integers(C,size=(5000,C))
    weights=(draws[:,:,None]==np.arange(C)).mean(1)
    boot_mean=boot_means(weights,cells.mean((1,2,3)),rates)
    expected=boot_means(uniform,cells.mean((1,2,3)),rates)[0]
    np.testing.assert_allclose(expected,means,atol=1e-12)
    qs=[];boot_q=[]
    for s in range(2):
        sorts=sorted_cells(real[s])
        qs.append(np.array([quantile(arm,uniform,sorts,rates)[0] for arm in METHODS]))
        if s==0:
            boot_q=np.stack([quantile(arm,weights,sorts,rates) for arm in METHODS[:3]],1)
    q=np.stack(qs)
    def ints(x):return np.quantile(x,[.0125,.9875]).tolist()
    def contrasts(point, boot):
        return {METHODS[i]:[float(point[0]-point[i]),*ints(boot[:,0]-boot[:,i])] for i in (1,2)}
    base_mean = by_e.mean(1)
    result=dict(methods=METHODS, rates=[r,1,r,0,r], mean=means, session_p90=q[0], repeat_session_p90=q[1],
                mean_contrast=contrasts(means,boot_mean),p90_contrast=contrasts(q[0],boot_q),
                realized_mean=real_by_e.mean((1,2)),catalog_wins=(base_mean[:,0,None]<base_mean).sum(0),
                surface_wins=(by_surface[...,0,None]<by_surface).sum((0,1)),mean_by_e=by_e.mean(0),
                per_catalog_range=np.stack([(base_mean[:,0,None]-base_mean).min(0),(base_mean[:,0,None]-base_mean).max(0)]),
                catalog_impact_rate_range=[rates.min(),rates.max()],session_seeds=REPLAY_SEEDS,
                paired_bootstrap=5000,catalogs=C,real_sessions_per_event=int(S*R),gate_rate=float(r))
    # Check how the choice of 128 realized deployments changes mean-direction inference.
    mean_real_boot=[boot_means(weights,((1-G)*raw[:,:,0]+G*raw[:,:,1]).mean((1,2,3,4)),rates) for raw in real]
    result['realized_mean_contrasts']=[contrasts(v,b) for v,b in zip(result['realized_mean'],mean_real_boot)]
    save=dict(methods=METHODS,by_e=by_e,by_surface=by_surface,per_catalog_rate=rates,realized_by_e=real_by_e,
              seed=REPLAY_SEEDS,by_e_k_surface=mean_matrix(cells,r,rates),source_sha256=[d['source_sha256'] for d in data])
    if all('ultra_cells' in d for d in data):
        ul=np.stack([d['ultra_cells'] for d in data]).mean((1,2,3))
        uv=mean_matrix(ul,r,rates).mean(0)
        result['ultra_mean']=uv; result['ultra_mean_contrast']=contrasts(uv,boot_means(weights,ul,rates))
    np.savez_compressed(ROOT/'clusters55.npz',**save)
    np.savez_compressed(ROOT/'bootstrap55.npz',catalog_counts=np.round(weights*C).astype('i1'),mean=boot_mean,p90=boot_q)
    save_json(ROOT/'score55.json',result)
    print(json.dumps(result,default=lambda x:np.asarray(x).tolist()),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--panel',type=int)
    parser.add_argument('--summary',action='store_true')
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run scoring on an allocated compute node')
    assert sha256((V54/'policy54.py').read_bytes()).hexdigest()=='b4fbd2ba1d72c664337b2cd6bd0794af023a3901f87d67fa639211f7a779f3ea'
    dest=ROOT/'scored55'
    if args.panel is not None:
        if not 0<=args.panel<C:raise ValueError(args.panel)
        panel_score(args.panel,dest)
    elif args.summary:summ(dest)
    else:
        for i in range(C):panel_score(i,dest)
        summ(dest)
