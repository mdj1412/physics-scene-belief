"""Paired catalog inference under four interventions on the pilot-generating process."""
from pathlib import Path
import hashlib
import json
import os
import sys
import numpy as np

from route58 import METHODS,route,TABLE,SINGLE,NAIVE,log_odds,SOURCE_LOG_CUTOFF
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'mismatch_switch_v56'))
from score56 import cdf_boot,EVENT
MIX=np.array([[.42,0,.58],[0,.84,.16]])
C=24
M=len(METHODS)

def read_panels():
    fine=[];ultra=[];real=[];rate=[];indices=[];gates=[]
    for i in range(C):
        m=((np.arange(4)+i)%4)//2
        scored=ROOT/'scored58'/f'panel_{i:02}.npz'
        with np.load(scored) as f:
            for target,key in ((fine,'exact_cells'),(ultra,'ultra_cells')):
                target.append(np.stack([f[key][:,:,m==g,:2].mean((1,2)) for g in range(2)]))
            gates.append([f['gate'][:,m==g].mean() for g in range(2)])
            x=f['real'][...,:2]
            assert x.shape==(2,3,2,5,4,256,2)
            real.append(np.stack([np.take(x,np.flatnonzero(m==g),axis=4).reshape(2,3,2,-1,2) for g in (0,1)],axis=1))
        pilot=ROOT/'pilots'/f'panel_{i:02}.npz'
        with np.load(pilot) as f:
            code=f['codes']
            assert code.dtype==np.uint8 and code.shape==(2,2,2,2,2,2048,4) and code.max()<=24
            # Route only sees grouped observation codes and the experimental grouping arm.
            grouped=code[1]
            dl=log_odds(grouped)
            single=log_odds(grouped,SINGLE)
            oracle=np.broadcast_to(np.array([1,0],bool)[None,None,:,None,None],dl.shape)
            r=np.stack((dl>1e-12,route(code[0],'dual8_map'),dl>SOURCE_LOG_CUTOFF,
                        route(grouped,'single4_naive'),route(grouped,'gate'),route(grouped,'fixed'),single>1e-12,oracle),axis=-1)
            assert METHODS[6]=='single4_adjusted'
            # rho,h,Z,F,N,M -> F,rho,h,Z,M
            rate.append(r.mean(-2).transpose(3,0,1,2,4))
        indices.append([hashlib.sha256(x.read_bytes()).hexdigest() for x in (scored,pilot)])
    return tuple(map(np.asarray,(fine,ultra,real,rate,gates,indices)))

def cluster_brier(data,r):
    by_regime=np.einsum('cfea,ze->cfza',data,MIX)
    return by_regime[...,0,None]+r*(by_regime[...,1]-by_regime[...,0])[...,None]

def compare(points,draws,i,others,adjust=1):
    j=np.array(others)
    dif=draws[:,i,None]-draws[:,j]
    lo,hi=np.quantile(dif,[.025/adjust,1-.025/adjust],axis=0)
    return np.stack((points[i]-points[j],lo,hi),axis=-1)

def qweights(r):
    # methods x catalog x material x e x event x action
    return np.einsum('cfzma,ze,d->mcfeda',np.stack((1-r,r),axis=-1),MIX,EVENT)/4

def report(fine,raw,rate,wt,ultra,snap,aggregate):
    losses=cluster_brier(fine,rate); by_catalog=losses.mean((1,2)); means=wt@by_catalog
    acc=(rate[:,:,0]+1-rate[:,:,1]).mean(1)/2
    accboot=wt@acc
    weights=qweights(rate)
    tails=cdf_boot(raw[0],weights,wt)
    base2=cdf_boot(raw[1],weights,wt[:1])[0]
    comparisons=[1,3]
    pairing=lambda d:compare(d[0],d[1:],0,comparisons,2 if aggregate else 1)
    out=dict(means=means[0],session_p90=tails[0],balanced_accuracy=accboot[0],route_A_B=rate.mean((0,1)),
             map_minus_8_naive_accuracy=pairing(accboot),map_minus_8_naive_mean=pairing(means),map_minus_8_naive_p90=pairing(tails),
             means_by_regime=losses.mean((0,1)),second_seed_p90=base2)
    if aggregate:
        def variants(matrix):
            return {METHODS[i]:{METHODS[j]:compare(matrix[0],matrix[1:],i,[j])[0].tolist() for j in [1,3,4,5,6] if j!=i} for i in [0,2,3,6]}
        ultra_mean=wt@cluster_brier(ultra,rate).mean((1,2))
        snapshot=[wt@cluster_brier(s,rate).mean((1,2)) for s in snap]
        out.update(exploratory_mean=variants(means),exploratory_accuracy=variants(accboot),exploratory_p90=variants(tails),ultra_mean=ultra_mean[0],
                   realized_means=[s[0] for s in snapshot],realized_primary_contrast=[pairing(s) for s in snapshot],
                   catalog_wins_vs_gate=np.sum(by_catalog<by_catalog[:,4,None],axis=0))
        # Independent inverse-CDF check on the realized 5120 session scores per material/e/event.
        arr=raw[0].ravel()
        order=np.argsort(arr,kind='stable')
        for i in (0,1,2,3):
            w=np.broadcast_to(weights[i][...,None],raw[0].shape).ravel()[order]/(C*raw.shape[-1])
            ix=np.searchsorted(np.cumsum(w),.9)
            np.testing.assert_allclose(arr[order[ix]],tails[0,i],rtol=0,atol=2e-8)
    return out

def main():
    assert os.environ.get('SLURM_JOB_ID')
    assert hashlib.sha256((ROOT/'route58.py').read_bytes()).hexdigest()=="41f3531e97bbedc92360f249dc09155eda33bdbe1d2c23d76939542008b67f68"
    fine,ultra,real,rate,gates,hashes=read_panels()
    raw=np.sort(real.transpose(1,0,2,3,4,6,5),axis=-1)
    snap=np.einsum('csfeda,d->scfea',real.mean(-2),EVENT)
    picks=np.random.default_rng(58096657).integers(C,size=(5000,C))
    wt=np.vstack([np.full((1,C),1/C),np.stack([(picks==i).mean(1) for i in range(C)],axis=1)])
    route_avg=rate.mean((2,3))
    result=dict(methods=METHODS,primary=report(fine,raw,route_avg,wt,ultra,snap,True),cells={})
    for r,h in np.ndindex(2,2):
        result['cells'][f'rho{r}_h{h}']=report(fine,raw,rate[:,:,r,h],wt,ultra,snap,False)
    rr=np.einsum('cfea,ze->cfza',fine,MIX)
    result['source_mapping_correct_materials']=(rr[...,0]-rr[...,1]>0).sum((0,1))
    result['gap_gate_minus_fixed_A_B']=(rr[...,0]-rr[...,1]).mean((0,1))
    result['deployment_final_impact_rate']=(gates[:,:,None,None]+(1-gates[:,:,None,None])*route_avg).mean((0,1,2))
    np.savez_compressed(ROOT/'score58_cells.npz',exact_by_e=fine,ultra_by_e=ultra,route=rate,gate_purchase=gates,
                        realized_by_e=snap,methods=METHODS,input_sha256=hashes)
    (ROOT/'score58.json').write_text(json.dumps(result,indent=2,default=lambda x:np.asarray(x).tolist())+'\n')
    print('score computed',result['primary']['means'],'p90',result['primary']['session_p90'],flush=True)
if __name__=='__main__':main()
