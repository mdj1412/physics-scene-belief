"""Two path-specific counterfactuals for why contour edits change a binary impact event."""
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np

import physics_audit42 as audit

ROOT=Path(__file__).resolve().parent
SHAPES=audit.SHAPES
HEIGHTS=audit.HEIGHT_KEYS
LAWS=audit.LAWS
OFFSETS=audit.OFFSETS

def components(theta,actual_height,family,variant,law):
    # Express required impulse as a coefficient of the normal impulse.
    g=theta[...,2]+theta[...,3]*(actual_height/.025)+audit.independent.geom(actual_height,family,variant)[0]
    v=.425
    demand=np.abs(v+g)/((1+theta[...,5])*(1-v*g))
    tangent=(v+g)/np.hypot(1,g)
    s=abs(tangent)
    if law=='saturating':
        part=.18*np.tanh(tangent/.62)**3
        gain=1.015+.025*np.tanh(s/.58)**2
    elif law=='crossover':
        part=.20*np.tanh(tangent/.32)*np.tanh((s-.70)/.16)
        gain=1.035-.02*np.exp(-((s-.68)/.24)**2)
    else:raise ValueError(law)
    avail=np.empty((2,3,*g.shape))
    q=np.sin(.5*np.pi*min(abs(actual_height)/.023,1))**2
    for si,sign in enumerate(audit.SIGNS):
        for ri,strength in enumerate((1.,.3,0.)):
            avail[si,ri]=theta[...,4]*gain*(1+sign*part*(1-(1-strength)*q))
    return demand,avail


def panel(idx):
    paths=[ROOT/'results'/'_'.join((family,h,var,law))/f'catalog_{idx:02d}.npz' for
           family,h,var,law in itertools.product(SHAPES,HEIGHTS,audit.VARS,LAWS)]
    while not all(p.with_suffix('.json').exists() for p in paths): time.sleep(9)
    meta=json.loads(paths[0].with_suffix('.json').read_text())
    with np.load(paths[0]) as f: alpha=f['alpha']
    pool=[]
    for e,aa in enumerate(alpha):
        di=audit.independent.rng(meta['deployment_seeds'][e],4120).integers(2048,size=128)
        context=(SHAPES[0],'base',.019,LAWS[0],1,meta['new_panel'])
        pool.append(audit.w.latent_support(aa,context,20).cpu().numpy()[di])
    th=np.stack(pool)
    counts=np.zeros((2,2,2,9,7),int)
    for fi,fam in enumerate(SHAPES):
        for hi,h in enumerate(HEIGHTS):
            nominal=int(h[1:])/1000
            for li,law in enumerate(LAWS):
                observed={}
                for var in audit.VARS:
                    key='_'.join((fam,h,var,law))
                    with np.load(ROOT/'results'/key/f'catalog_{idx:02d}.npz') as f:observed[var]=f['labels']
                for oi,mm in enumerate(OFFSETS):
                    actual=nominal+mm*.001
                    rb,xb=components(th,actual,fam,'base',law)
                    yb=rb<=xb
                    np.testing.assert_array_equal(yb,observed['base'][oi])
                    for var in ('hollow','bulge'):
                        rv,xv=components(th,actual,fam,var,law)
                        y=rv<=xv
                        np.testing.assert_array_equal(y,observed[var][oi])
                        dflip=(rv<=xb)!=yb
                        mflip=(rb<=xv)!=yb
                        flip=y!=yb
                        counts[fi,hi,li,oi]+=np.array([flip.size,flip.sum(),(flip&dflip).sum(),
                            (flip&mflip).sum(),(flip&~dflip&~mflip).sum(),(~flip&dflip).sum(),(~flip&mflip).sum()])
    return counts


def metrics(x):
    # This length-weighted fraction describes matched flips; intervals across panels use a
    # paired nonparametric bootstrap to retain denominators when the event is rare.
    counts=x.sum(0); obs=counts[:,1]
    share=lambda u,den:np.divide(u,den,where=den>0,out=np.zeros_like(u,dtype=float))
    demand_share=share(counts[:,2],obs)
    mask=np.array(OFFSETS)!=0
    boots=np.random.default_rng(421104).integers(len(x),size=(12000,len(x)))
    tally=x[:,mask].sum(1); sums=tally[boots].sum(1)
    ratio=sums[:,2]/sums[:,1]
    pooled=tally.sum(0)
    return dict(total_count=counts.tolist(),actual_flip_rate=share(obs,counts[:,0]).tolist(),
                demand_only_reproduces_actual= np.where(obs>0,demand_share,np.nan).tolist(),
                frozen_demand_reproduces_actual=np.where(obs>0,share(counts[:,3],obs),np.nan).tolist(),
                neither_alone_reproduces_actual=np.where(obs>0,share(counts[:,4],obs),np.nan).tolist(),
                nonzero_demand_only_share=float(pooled[2]/pooled[1]),
                nonzero_demand_only_ci=np.quantile(ratio,[.025,.975]).tolist(),
                nonzero_law_only_share=float(pooled[3]/pooled[1]),
                neither_only_share=float(pooled[4]/pooled[1]))


def jsonsafe(v):
    if isinstance(v,dict): return {k:jsonsafe(x) for k,x in v.items()}
    if isinstance(v,list): return [jsonsafe(x) for x in v]
    if isinstance(v,float) and not np.isfinite(v): return None
    return v

def main():
    assert os.environ['BELIEF_DEVICE']=='cpu'
    audit.w.torch.set_num_threads(1)
    with audit.w.torch.inference_mode():
        raw=np.stack([panel(i) for i in range(12)])
    groups={'all':metrics(raw.sum((1,2,3)))}
    for i,law in enumerate(LAWS):groups[law]=metrics(raw[:,:,:,i].sum((1,2)))
    out=dict(groups=groups,offsets=OFFSETS,counts_columns=['draws','flip','flip_and_demand','flip_and_law',
             'flip_and_neither','unchanged_and_demand','unchanged_and_law'],panels=raw.tolist())
    (ROOT/'friction_path42.json').write_text(json.dumps(jsonsafe(out),indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:(v['nonzero_demand_only_share'],v['nonzero_law_only_share']) for k,v in groups.items()}))
if __name__=='__main__':main()
