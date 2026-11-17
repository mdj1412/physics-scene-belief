"""Paired evaluation of separately acquired surface pilots and held-out deployment."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from switch56 import BUDGETS,CROSSING,PRIMARY,W,choose,estimate

ROOT=Path(__file__).resolve().parent
CAT=24
DRIFT=np.mean([0.,.25,.5,1.])
EVENT=np.array([1-DRIFT,DRIFT])
EW=np.stack(((1-W)/2,(1-W)/2,W),axis=1)
B=BUDGETS.index(8)
METHODS=['switch','frozen_gate','fixed_I','known_w_threshold','cell_expected_oracle']

def read_all():
    fine=[]; ultra=[]; reals=[]; rates=[]; summaries=[]; gate_imp=[]; pilot_routes=[]
    for c in range(CAT):
        path=ROOT/'scored56'/f'panel_{c:02d}.npz'
        with np.load(path) as f:
            # e,pre-k,environment,arm -> material,e,arm. Material labels rotate by catalog.
            material=((np.arange(4)+c)%4)//2
            def material_mean(name):
                return np.stack([f[name][:,:,material==group,:2].mean((1,2)) for group in (0,1)])
            fine.append(material_mean('exact_cells')); ultra.append(material_mean('ultra_cells'))
            gate_imp.append([f['gate'][:,material==g].mean() for g in (0,1)])
            # seed×material×e×event×deployment draws×arm.
            raw=f['real'][...,:2]
            assert raw.shape==(2,3,2,5,4,256,2)
            reals.append(np.stack([np.take(raw,np.flatnonzero(material==g),axis=4).reshape(2,3,2,-1,2) for g in (0,1)],axis=1))
        with np.load(ROOT/'pilots'/f'panel_{c:02d}.npz') as f:
            ref,imp=f['pilot_ref'],f['pilot_imp']
            assert ref.shape==imp.shape==(2,len(W),512,16)
            assert np.isin(ref,[-1,0,1]).all() and np.isin(imp,[-1,0,1]).all()
            rg=[];sg=[];rr=[]
            for k in BUDGETS:
                n,flips,q,what=estimate(ref[...,:k],imp[...,:k])
                choice=choose(ref[...,:k],imp[...,:k])
                rg.append(choice.mean(axis=-1));rr.append(choice)
                # RMSE and bias conditional on any reading. Store masked sums and denominator.
                finite=np.isfinite(what)
                err=np.where(finite,what-W[None,:,None],0.)
                sg.append(np.stack([n.mean(-1),(n==0).mean(-1),finite.mean(-1),err.mean(-1),(err**2).mean(-1),
                                    flips.mean(-1)],axis=-1))
            rates.append(np.stack(rg,axis=-1));summaries.append(np.stack(sg,axis=-2));pilot_routes.append(np.stack(rr,axis=-1))
    return tuple(map(np.array,(fine,ultra,reals,rates,summaries,gate_imp,pilot_routes)))

def choices(loss,rates):
    # true conditional costs, given w and material
    risks=np.einsum('cfea,we->cfwa',loss,EW)
    oracle=(risks[...,1]<risks[...,0])
    true_w=W>CROSSING
    shape=rates.shape
    return np.stack([rates,np.zeros(shape),np.ones(shape),np.broadcast_to(true_w[None,None,:,None],shape),
                     np.broadcast_to(oracle[...,None],shape)],axis=-1)

def evaluate(loss,route):
    risk=np.einsum('cfea,we->cfwa',loss,EW)
    return risk[...,0,None,None]+route*(risk[...,1]-risk[...,0])[...,None,None]

def mixweights(route,worlds,b):
    # C,F,W,B,A -> M,C,F,E,event,endpoint. Average selected world levels and both materials.
    s=route[:,:,worlds,b].transpose(3,0,1,2)
    endpoints=np.stack((1-s,s),axis=-1)
    return np.einsum('mcfwa,we,d->mcfeda',endpoints,EW[worlds],EVENT)/(2*len(worlds))

def qsingle(sorted_real, weights):
    lo,hi=0.,1.
    for _ in range(35):
        mid=(lo+hi)/2
        cdf=(weights*(sorted_real<=mid).sum(-1)/sorted_real.shape[-1]).sum()/CAT
        if cdf>=.90:hi=mid
        else:lo=mid
    return hi

def cdf_boot(sorted_real,weights,multiplicity):
    # Precompute CDFs in the union of intervals allowed by each catalog's 90th percentile.
    # All quantiles of a mixture across catalogs are bracketed by min/max catalog quantiles.
    flat=sorted_real.reshape(CAT,-1,sorted_real.shape[-1])
    fractions=weights.reshape(len(weights),CAT,-1)
    lo=0.;hi=1.
    # Coarse grid for locating tight bounds of the bootstrap distribution.
    def cdf_by_catalog(grid):
        out=np.zeros((len(weights),CAT,len(grid)))
        for c in range(CAT):
            fs=np.stack([np.searchsorted(row,grid,'right') for row in flat[c]],axis=0)/flat.shape[-1]
            out[:,c]=fractions[:,c]@fs
        return out
    grid=np.linspace(lo,hi,1001)
    coarse=cdf_by_catalog(grid)
    def bounds():
        il=len(grid); ih=0
        for cdf in coarse:
            v=multiplicity@cdf
            ix=(v<.9).sum(axis=1)
            il=min(il,ix.min()); ih=max(ih,ix.max())
        return grid[max(il-1,0)],grid[min(ih,len(grid)-1)]
    lo,hi=bounds()
    # Pooled endpoints are float32, so using their exact union gives the discrete weighted quantile.
    candidates=np.unique(flat[(flat>=lo)&(flat<=hi)])
    assert len(candidates)
    cdf=cdf_by_catalog(candidates)
    ret=np.empty((len(multiplicity),len(weights)))
    for m,c in enumerate(cdf):
        left=np.zeros(len(ret),int);right=np.full(len(ret),len(candidates)-1,int)
        while (right>left).any():
            center=(left+right)//2
            v=(multiplicity*c[:,center].T).sum(axis=1)
            low=v<.90
            left=np.where(low,center+1,left);right=np.where(low,right,center)
        ret[:,m]=candidates[left]
    return ret

def pack(x):return np.asarray(x).tolist()

def paired(values,weights):
    means=values.mean(axis=0)
    bs=weights@values
    return [contrast(means,bs,i) for i in (1,2)]

def contrast(mean, draws, i):
    return [float(mean[0]-mean[i]),*np.quantile(draws[:,0]-draws[:,i],[.0125,.9875]).tolist()]

def main():
    assert os.getenv('SLURM_JOB_ID')
    frozen=lambda p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
    assert frozen('switch56.py')=='07b860f4edd72eb773eb5d6c56ae6dfc285b958533194c436a29ac488ac24034'
    exact,ultra,real,rate,summaries,gate_imp,pilot_routes=read_all()
    route=choices(exact,rate)
    R=evaluate(exact,route);UR=evaluate(ultra,route)
    bmean=real.mean(axis=-2) # c,seed,material,e,event,arm
    snapshots=np.einsum('csfeda,d->scfea',bmean,EVENT)
    actual=np.stack([evaluate(sd,route) for sd in snapshots])
    # Regroup complete scored worlds/pilot replicates within each independent catalog.
    rg=np.random.default_rng(560730)
    idx=rg.integers(CAT,size=(5000,CAT))
    mult=np.stack([(idx==c).mean(axis=1) for c in range(CAT)],axis=1)
    primary=list(PRIMARY); all_methods=np.arange(len(METHODS))
    def panel_primary(arr):return arr[:,:,primary,B,:].mean((1,2))
    pm=panel_primary(R);um=panel_primary(UR);pa=np.stack([panel_primary(x) for x in actual])
    raw=real.transpose(1,0,2,3,4,6,5)
    quantiles=[];primary_q=[]
    weights=mixweights(route,primary,B)
    full=np.vstack([np.full(CAT,1/CAT),mult])
    for seed in range(len(raw)):
        sr=np.sort(raw[seed],axis=-1)
        vals=cdf_boot(sr,weights[:3],full if seed==0 else full[:1])
        quantiles.append(vals[0])
        if seed==0:primary_q=vals[1:]
    rate_cf=rate.mean((0,1))
    budget_result=[]
    for j,k in enumerate(BUDGETS):
        vals=R[:,:,primary,j,:].mean((1,2))
        budget_result.append(dict(k=k,mean=pack(vals.mean(0)),mean_contrast=paired(vals,mult)))
    qworld=[]
    sr=np.sort(raw[0],axis=-1)
    for iw in range(len(W)):
        wt=mixweights(route,[iw],B)
        qworld.append(cdf_boot(sr,wt[:3],np.full((1,CAT),1/CAT))[0])
    # Exact permutation expectation: redistribute observed switch quota across all w and
    # materials in the primary mixture. A matched choice rate is a secondary control.
    s=rate[:,:,primary,B];rr=np.einsum('cfea,we->cfwa',exact,EW[primary])
    def random_contrast(mult):
        st=mult@s.mean((1,2))
        low=mult@rr[...,0].mean((1,2));hi=mult@rr[...,1].mean((1,2))
        return mult@pm[:,0]-(low+st*(hi-low))
    cv=random_contrast(full)
    result=dict(K=list(BUDGETS),W=pack(W),primary_w=pack(W[primary]),methods=METHODS,
                mean=pack(pm.mean(0)),mean_contrast=paired(pm,mult),
                catalog_wins=pack((pm[:,0,None]<pm[:,1:3]).sum(0)),
                ultra_mean=pack(um.mean(0)),ultra_contrast=paired(um,mult),
                realized_mean=pack(pa.mean(1)),realized_contrast=[paired(a,mult) for a in pa],
                session_p90=pack(quantiles),p90_contrast=[contrast(quantiles[0],primary_q,i) for i in (1,2)],
                switch_by_w_k=pack(rate_cf),mean_by_w=pack(R[:,:,:,B,:].mean((0,1))),
                p90_by_w=pack(qworld),mean_by_k=budget_result,
                k8_pilot_stats=pack(summaries[:,:,:,B,:].mean((0,1))),
                oracle_known_w_mean=pm[:,3].mean(),oracle_material_hindsight_mean=pm[:,4].mean(),
                random_switch_rate=s.mean(),random_difference=[cv[0],*np.quantile(cv[1:],[.025,.975])],
                n_catalog=CAT)
    switches=route[:,:,:,B,:]
    imp_rate=gate_imp[:,:,None,None]+(1-gate_imp[:,:,None,None])*switches
    result['impact_rate']=pack(imp_rate[:,:,primary].mean((0,1,2)))
    # Inactive, sign, definition must be independent of experimental outcome.
    np.savez_compressed(ROOT/'score56_cells.npz',by_e=exact,ultra_by_e=ultra,route_rates=rate,means_by_w_k=R,
                        base_gate_impact=gate_imp,pilot_stats=summaries,boot_mean=mult@pm,boot_p90=primary_q,
                        realized_by_e=snapshots,W=W,budgets=BUDGETS)
    (ROOT/'score56.json').write_text(json.dumps(result,indent=2,default=pack)+'\n')
    print('Means',result['mean'],'P90',result['session_p90'],'contrasts',result['mean_contrast'],flush=True)
if __name__=='__main__': main()
