"""Evaluate a frozen prefix-only acquisition using paired v53 potential outcomes."""
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
V53 = ROOT.parent / 'impact_witness_v53'
sys.path.insert(0, str(V53))
from score53 import V52, V50, E, G, KAPPA, N, R, quadratic, pair_ci, quantiles, bootstrap
from policy52 import readable, witnesses

GM = float(G.mean())
KPOST = np.stack([np.arange(5), np.arange(4,-1,-1)])
METHODS = ['adaptive','fixed4R','fixed3RI','vote_gate','no_old_gate','random_at_rate','random_half',
           'sign_lookup_loo','latent_cell_loo','both_futures_min_mean','adaptive_noisy',
           'adaptive_uniform_gate','nominal_gate_uniform_decoder','uniform4R','uniform3RI',
           'uniform_gate_nominal_decoder','robust_gate','fixed3RI_noisy']
SOURCES = ['reference4_joint','impact_joint','impact_joint_noisy','reference4_uniform','impact_joint_uniform']
REFERENCE, IMPACT, NOISY, UREF, UIMP = range(5)


def ultra_realized(file, panel, alphas):
    """Independent solver targets on the original deployment draws used by v52."""
    with np.load(V50/'forecast50'/file) as f:
        cols = list(f['methods'].astype(str))
        pred = f['p'][[cols.index('local_plus'), cols.index('local_minus')]]
    with np.load(V50/'labels50_ultra'/file) as f:
        label = f['labels']
    rng = np.random.default_rng([52090211,panel])
    shape = (4,R,128)
    env, obj = np.arange(4)[:,None,None],np.arange(128)[None,None,:]
    fam,h,axis,v,rad,off,meter = [rng.integers(n,size=shape) for n in (4,2,2,2,2,3,4)]
    weak = rng.choice(3,size=shape,p=(.5,.25,.25)); rev=np.where(weak,3-weak,0)
    plus,minus=pred[:,fam,h,weak,axis,v,env,obj,meter]
    y=label[fam,h,weak,rad,off,env,obj]; ry=label[fam,h,rev,rad,off,env,obj]
    pol,error = rng.random((2,*shape)); spread=((plus-minus)**2).mean(-1)
    raw=np.empty(alphas.shape,dtype=np.float32)
    for e,ev,k in np.ndindex(3,2,5):
        observed=np.where((pol>=KAPPA[KPOST[ev,k]])^(error<E[e]),ry,y)
        rminus=((minus-observed)**2).mean(-1)
        rplus=((plus-observed)**2).mean(-1)
        a=alphas[e,ev,k]
        raw[e,ev,k]=rminus[...,None]+a*(rplus-rminus)[...,None]-a*(1-a)*spread[...,None]
    return raw


def catalog(panel):
    file=f'panel_{panel:02d}.npz'
    with np.load(V53/'decisions53'/file) as f:
        names=list(f['methods'].astype(str))
        a=f['alpha'][...,[names.index(n) for n in SOURCES]]
    with np.load(V52/'risk52'/file) as f:
        ex=f['exact']
    exact=quadratic(ex[...,None,None,:],a[None],False)
    with np.load(ROOT/'routes54'/file) as f:
        route={n:f[n] for n in ('impact','vote','uniform','no_old','robust','delta')}
        for n,v in route.items():
            assert v.shape==(5,4,R) and np.isfinite(v).all(),(n,v.shape)
    bins=[]
    for k in range(5):
        _,_,w,r=witnesses(panel,0,k,3)
        ok=readable(w,r); n=ok.sum(-1); yes=(ok & ((w>0)==(r>0))).sum(-1)
        bins.append(n*(n+1)//2+yes)
    return exact,ultra_realized(file,panel,a),route,np.stack(bins)


def mix_event(x):
    # prefix-level input [C,e,event,k,env,R,...]
    return (1-GM)*x[:,:,0]+GM*x[:,:,1]


def leave_out_bins(ex, b):
    # For each bin use only other catalogs, weighted by evaluation condition frequencies.
    # ex [C,E,event,k,env,R,action]; bin [C,E,event,k,env,R]
    difference=ex[...,IMPACT]-ex[...,REFERENCE]
    route=np.zeros_like(b,dtype=bool)
    for value in np.unique(b):
        mask=b==value
        catalog_sums=mix_event(difference*mask).sum((1,2,3,4))
        for i in range(N):
            route[i][mask[i]]=catalog_sums.sum()-catalog_sums[i]<0
    return route


def summarize():
    if not os.getenv('SLURM_JOB_ID'):
        raise RuntimeError('Expected an allocated compute node.')
    data=[catalog(i) for i in range(N)]
    exact=np.stack([x[0] for x in data]) # C, resolution, e, event, k, env, rep, source
    ultra=np.stack([x[1] for x in data])
    def expand_route(raw):
        return np.broadcast_to(raw[:,None,KPOST],exact[:,0,...,0].shape)
    route={n:expand_route(np.stack([x[2][n] for x in data])) for n in data[0][2]}
    b=expand_route(np.stack([x[3] for x in data]))
    rng=np.random.default_rng(5401289)
    random=expand_route(rng.random((N,5,4,R)))
    rate=mix_event(route['impact']).mean()
    nshape=b.shape
    latent=np.broadcast_to(np.arange(30).reshape(1,3,2,5,1,1),nshape)
    decisions=[route['impact'],np.zeros(nshape,bool),np.ones(nshape,bool),route['vote'],route['no_old'],
               random<rate,random<.5,leave_out_bins(exact[:,0],b),leave_out_bins(exact[:,0],latent),
               exact[:,0,...,IMPACT]<exact[:,0,...,REFERENCE],route['impact'],route['uniform'],
               route['impact'],np.zeros(nshape,bool),np.ones(nshape,bool),
               route['uniform'],route['robust'],np.ones(nshape,bool)]
    assert len(decisions)==len(METHODS)
    def evaluate(potential):
        rows=[]
        for i,d in enumerate(decisions):
            j,k=(UREF,UIMP) if 11<=i<=14 else (REFERENCE,NOISY if i in (10,17) else IMPACT)
            rows.append(np.where(d,potential[...,k],potential[...,j]))
        return np.stack(rows,-1)
    with np.load(V53/'session53.npz') as f:
        names=list(f['methods'].astype(str)); col=[names.index(n) for n in SOURCES]
        real=evaluate(f['realized'][...,col]); replay=evaluate(f['replay'][...,col])
    ultra=evaluate(ultra)
    real_e=[evaluate(exact[:,res]) for res in (0,1)]
    means=np.stack([mix_event(q).mean((1,2,3,4)) for q in real_e],1)
    quant=quantiles(real,GM); repeat_quant=quantiles(replay,GM); ultra_quant=quantiles(ultra,GM)
    boot=bootstrap(real[...,:3]); ultra_boot=bootstrap(ultra[...,:3])
    population=means.mean(0)
    def primary_contrast(i):
        qdif=boot[:,0]-boot[:,i]; udif=ultra_boot[:,0]-ultra_boot[:,i]
        bounds=[.025/4,1-.025/4]
        return dict(mean=pair_ci(means[:,0,0]-means[:,0,i],4),
                    p90=[quant[0]-quant[i],*np.quantile(qdif,bounds)],
                    ultra_mean=pair_ci(means[:,1,0]-means[:,1,i],4),
                    ultra_p90=[ultra_quant[0]-ultra_quant[i],*np.quantile(udif,bounds)],
                    second_seed=float(repeat_quant[0]-repeat_quant[i]),
                    catalog_wins=int((means[:,0,0]<means[:,0,i]-1e-15).sum()))
    primary={METHODS[i]:primary_contrast(i) for i in (1,2)}
    archived=json.loads((V53/'score53.json').read_text())
    for i,name in [(1,'reference4_joint'),(2,'impact_joint')]:
        ix=archived['methods'].index(name)
        np.testing.assert_allclose([population[0,i],quant[i]],[archived['mean'][ix],archived['p90'][ix]],atol=2e-9,rtol=0)
    # A non-deployable condition average trained on other catalogs should not be called an absolute ceiling.
    ceilings={}
    for m in [7,8,9]:
        ceilings[METHODS[m]]={METHODS[i]:pair_ci(means[:,0,m]-means[:,0,i],1) for i in [0,1,2]}
    rates=[mix_event(d).mean() for d in decisions]
    assert np.all(real_e[0][...,9] <=np.minimum(real_e[0][...,1],real_e[0][...,2])+1e-15)
    cells=np.stack([(1-g)*real_e[0][:,:,0]+g*real_e[0][:,:,1] for g in G],axis=2)
    cell_mean=cells.mean((0,3,4,5)); kmeans=cells.mean((0,1,2,4,5))
    margins=route['delta'];dv=real_e[0][...,2]-real_e[0][...,1]
    qs=np.quantile(np.stack([x[2]['delta'] for x in data]),[0,.25,.5,.75,1])
    bins4=np.digitize(margins,qs[1:-1])
    calib=[]
    for q in range(4):
        mask=bins4==q
        w=mix_event(mask).mean()
        if w>0:
            gain=mix_event(mask*dv).mean((1,2,3,4))/mix_event(mask).mean((1,2,3,4))
            calib.append(dict(quartile=q+1,mass=w,predicted_delta=mix_event(mask*margins).mean()/w,observed_delta=pair_ci(gain,1)))
    rng=np.random.default_rng(540392)
    fm=[];fp=[]
    for r in range(R):
        ix=rng.permutation(np.tile(np.arange(12),4)).reshape(12,4)
        e,g=ix//4,G[ix%4]
        ev=(rng.random((12,4))<g).astype(int)
        k=rng.permutation(np.r_[np.tile(np.arange(5),9),rng.choice(5,3,replace=False)]).reshape(12,4)
        s=real[np.arange(12)[:,None],e,ev,k,np.arange(4)[None,:],r]
        fm.append(s.mean((0,1))); fp.append(np.quantile(s,.9,axis=(0,1)))
    fm=np.array(fm);fp=np.array(fp)
    summary=dict(methods=METHODS,e=E,g=G,k=KAPPA,mean=population[0],ultra_mean=population[1],session_p90=quant,
                 second_seed_p90=repeat_quant,ultra_p90=ultra_quant,primary=primary,impact_rate=rates,
                 impact_rate_by_postk=np.stack([x[2]['impact'] for x in data]).mean((0,2,3)),
                 mean_by_e_g=cell_mean,mean_by_prek=kmeans,quartiles=calib,
                 fleet_worse_mean=(fm[:,0,None]>fm).mean(0),fleet_worse_p90=(fp[:,0,None]>fp).mean(0),
                 diagnostics=ceilings,secondary_mean_vs_adaptive=[pair_ci(means[:,0,0]-means[:,0,j],1) for j in range(len(METHODS))])
    np.savez_compressed(ROOT/'catalog54.npz',means=means,methods=METHODS,cells=cells.mean((4,5)),fleet=fm,fleet_p90=fp)
    np.savez_compressed(ROOT/'session54.npz',loss=real[...,:3],replay=replay[...,:3],ultra=ultra[...,:3])
    (ROOT/'score54.json').write_text(json.dumps(summary,default=lambda x:np.asarray(x).tolist(),indent=1))
    print('Mean/tail',population[0,:3],quant[:3],'rate',rate,'primary',primary,flush=True)

if __name__=='__main__': summarize()
