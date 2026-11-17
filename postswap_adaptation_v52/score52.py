"""Catalog-paired mean Brier and pooled absolute session Brier p90."""
import json
import os
from pathlib import Path
import numpy as np
from scipy.stats import t
from physics_risk52 import ROOT, EPS, DRIFT, PRE, BUDGET, NREP, N

METHODS=['selector','vote_selector','post_vote_soft','keep_old_soft','refit_post_soft','unsigned','continuous_joint','continuous_vote_joint']
P=BUDGET.index(4)
G=np.array(DRIFT)
GMEAN=G.mean()


def ci(x,n=1):
    mean=x.mean(0); half=t.ppf(1-.025/n,len(x)-1)*x.std(0,ddof=1)/np.sqrt(len(x))
    return np.stack((mean,mean-half,mean+half),-1)


def loss(risk,alpha,unsigned):
    plus,minus,blind,gap=np.moveaxis(risk,-1,0)
    return np.where(unsigned,blind,minus+alpha*(plus-minus)-alpha*(1.-alpha)*gap)


def panel(i):
    name=f'panel_{i:02d}.npz'
    with np.load(ROOT/'risk52'/name) as r:
        exact,raw,surface=(r[k] for k in ('exact','realized','surface'))
    with np.load(ROOT/'decisions52'/name) as f:
        a,chosen,old,new,target=(f[k] for k in ('alpha','action','old','new','target_alpha'))
    assert a.shape==(5,5,5,4,NREP,2)
    values=np.empty((2,3,2,5,4,5,len(METHODS)))
    real=np.empty((3,2,5,4,NREP,5,len(METHODS)),np.float32)
    for event in (0,1):
        for c in range(5):
            post=4-c if event else c
            for b in range(5):
                ix=(c,post,b)
                al0,al1=a[ix].transpose(2,0,1)
                ch0,ch1=chosen[ix].transpose(2,0,1)
                of,ov=old[ix].transpose(2,0,1); nf,nv=new[ix].transpose(2,0,1)
                tf,tv=target[ix].transpose(2,0,1)
                policies=((al0,ch0==2),(al1,ch1==2),(nv,False),(of,False),(nf,False),(0.,True),(tf,False),(tv,False))
                for m,(alpha,unsigned) in enumerate(policies):
                    # exact: two resolutions and three impact errors are broadcast over independent prefix replicates
                    values[:,:,event,c,:,b,m]=loss(exact[:,:,event,c,:,None,:],alpha,unsigned).mean(-1)
                    real[:,event,c,:,:,b,m]=loss(raw[:,event,c],alpha,unsigned)
    assert ((real>=0)&(real<=1)).all()
    return values,real,surface


def weighted_cdf_inverse(groups,weights,prob=.90):
    """Inverse empirical CDF, with per-session weights inside each conditional event."""
    if weights[0]==0: return np.quantile(groups[1],prob,method='inverted_cdf')
    if weights[1]==0: return np.quantile(groups[0],prob,method='inverted_cdf')
    lo=min(x[0] for x in groups)-1e-8; hi=max(x[-1] for x in groups)
    for _ in range(32):
        mid=(lo+hi)/2
        cdf=sum(w*np.searchsorted(x,mid,side='right')/len(x) for x,w in zip(groups,weights))
        if cdf>=prob:hi=mid
        else:lo=mid
    return hi


def quantile_grid(x):
    # x: catalog,impact error,drift event,kappa,env,rep,budget,method
    result=np.empty((3,4,5,len(METHODS)))
    absolute=np.empty((5,len(METHODS)))
    for b,m in np.ndindex(5,len(METHODS)):
        def groups(values):
            return [np.sort(v.ravel().astype(float)) for v in values]
        pooled=groups((x[:,:,0,...,b,m],x[:,:,1,...,b,m]))
        absolute[b,m]=weighted_cdf_inverse(pooled,(1-GMEAN,GMEAN))
        for e in range(3):
            split=groups((x[:,e,0,...,b,m],x[:,e,1,...,b,m]))
            for j,g in enumerate(G):
                result[e,j,b,m]=weighted_cdf_inverse(split,(1-g,g))
    return result,absolute


def bootstrap_primary_p90(real):
    """Exact weighted empirical CDF inversion for paired catalog-bootstrap samples."""
    boot=np.random.default_rng(5216630).integers(N,size=(5000,N))
    mult=(boot[:,:,None]==np.arange(N)).sum(1)/N
    draws=[]
    for m in (0,1,2):
        groups=[[np.sort(real[i,:,ev,...,P,m].ravel().astype(float)) for ev in (0,1)] for i in range(N)]
        lo=np.full(len(boot),min(min(a[0],b[0]) for a,b in groups)-1e-7);hi=np.full(len(boot),max(max(a[-1],b[-1]) for a,b in groups))
        for _ in range(26):
            mid=(hi+lo)/2
            probs=np.array([((1-GMEAN)*np.searchsorted(a,mid,'right')+GMEAN*np.searchsorted(b,mid,'right'))/len(a) for a,b in groups]).T
            below=(mult*probs).sum(1)<.90
            lo=np.where(below,mid,lo);hi=np.where(below,hi,mid)
        draws.append(hi)
    draws=np.stack(draws,1)
    return np.quantile(draws[:,0,None]-draws[:,1:3],[.025/4,1-.025/4],axis=0).T


def main():
    assert os.getenv('SLURM_JOB_ID')
    out=[panel(i) for i in range(N)]
    exact=np.stack([q[0] for q in out]) # catalogs,resolution,eps,event,kappa,env,K,method
    realized=np.stack([q[1] for q in out])
    surfaces=np.stack([q[2] for q in out])
    weighted=np.stack([(1-g)*exact[:,:,:,0]+g*exact[:,:,:,1] for g in G],axis=3)
    # catalog,resolution,epsilon,drift rate,kappa,env,budget,method
    cells=weighted.mean((4,5))
    mean=cells.mean((2,3))
    diff=mean[:,:,:,0,None]-mean[:,:,:,[1,2]]
    k4=cells[:,0,:,:,P,:]
    p90,pooled=quantile_grid(realized)
    ciq=bootstrap_primary_p90(realized)
    truth=realized.mean(axis=(3,4,5),dtype=float)
    mis=np.max(abs(truth-exact[:,0].mean((3,4))))
    discrepancy=(truth-exact[:,0].mean((3,4))).mean(0)
    sx=pooled[P,0]-pooled[P,1:3]
    summary=dict(methods=METHODS,K=BUDGET,e=EPS,g=DRIFT,kappa_pre=PRE,catalog_count=N,repeats=NREP,
                 mean=mean.mean(0),mean_axes='resolution (primary, ultra),post_budget,method',cell_mean=cells.mean(0),
                 absolute_p90=pooled,cell_absolute_p90=p90,
                 primary_mean_differences=ci(diff[:,0,P],4),ultra_mean_differences=ci(diff[:,1,P],4),
                 primary_q90_differences=np.column_stack((sx,ciq)),primary_catalog_counts=(diff[:,0,P]<0).sum(0),
                 primary_cell_mean_diff=ci(k4[...,0,None]-k4[...,[1,2]],24),
                 sensitivity_kappa=weighted[:,0,:,:,:,:,P,:].mean((0,1,2,4)),
                 realized_minus_exact_max_catalog_conditional=float(mis),realized_minus_exact_max_mean=float(abs(discrepancy).max()),
                 primary_q90_statement='Pooled absolute empirical p90 over equal 3 impact laws, 4 drift probabilities, 5 source polarities, environments/cat/rep. A drift event shared by all 128 objects. Two endpoint x two comparator intervals Bonferroni; paired catalog t for means, inverse CDF paired 5000-draw bootstrap for p90. Tail differences are differences of marginal p90s. Sensors cannot see e.')
    for m in (0,4):
        np.testing.assert_allclose(exact[:,0,:,:,2,:,:,m][:,:,0],exact[:,0,:,:,2,:,:,m][:,:,1],rtol=0,atol=2e-12)
    np.savez_compressed(ROOT/'catalog52.npz',exact=exact,selected_surface=surfaces,methods=METHODS,e=EPS,g=DRIFT,budgets=BUDGET,kappa=PRE)
    np.savez_compressed(ROOT/'session52.npz',realized=realized,methods=METHODS,e=EPS,budgets=BUDGET)
    (ROOT/'score52.json').write_text(json.dumps(summary,default=lambda x:np.asarray(x).tolist(),indent=1))
    print('K4 mean',summary['mean'][0,P], 'p90',pooled[P], 'primary intervals',summary['primary_mean_differences'],summary['primary_q90_differences'],flush=True)


if __name__=='__main__': main()
