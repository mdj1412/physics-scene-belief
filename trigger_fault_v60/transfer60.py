"""Preplanned surface-level fault randomization and hazard-marginalized decision audit."""
from pathlib import Path
import json
import os
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'sensor_reset_v59'))
from route59 import TABLE, SOURCE_LOG_CUTOFF, log_odds

C, M, E = 24, 2, 512
MIX = np.array([[.42,0,.58],[0,.84,.16]])
ASSIGNMENTS=['matched','half_mapped','reversed','continuous_surface']
ESTIMATORS=['source','hazard_marginalized']


def actions(code, reset, neutral):
    # One unobserved state for the complete four-session pilot.
    if not neutral:
        z, k = log_odds(code,reset,True)
    else:
        l=TABLE[reset,1][:,:,code].sum(-1)
        shared=np.logaddexp(l[1,0],l[1,1])-np.log(2.)
        z=np.logaddexp(l[0,0],shared)-np.logaddexp(l[0,1],shared)
        k=np.logaddexp(l[0,0],l[0,1])-np.logaddexp(l[1,0],l[1,1])
    return np.stack((z>SOURCE_LOG_CUTOFF,k>1e-12,z>1e-12),axis=-1)


def main():
    assert os.getenv('CUDA_VISIBLE_DEVICES')=='' and os.getenv('SLURM_JOB_ID')
    w=np.random.default_rng(60071901).permutation((np.arange(C*M)+.5)/(C*M)).reshape(C,M)
    # A surface's probability is fixed. One random draw selects an entire session pair.
    # dimensions: catalog, delay, reset, estimator, assignment, material, cause, target, metrics
    rates=np.zeros((C,3,2,2,4,M,2,2,3))
    target_risks=[]
    realized=[]
    for c in range(C):
        with np.load(ROOT/'pilots'/f'panel_{c:02}.npz') as f:
            waves=f['codes'][:,1]  # registered optical correction
        with np.load(ROOT/'scored60'/f'panel_{c:02}.npz') as f:
            material=((np.arange(4)+c)%4)//2
            by_e=np.stack([f['exact_cells'][:,:,material==j,:2].mean((1,2)) for j in range(M)])
            target_risks.append(np.einsum('mea,ze->mza',by_e,MIX))
        coin=np.random.default_rng([60071902,c]).random((M,E,4))<w[c,:,None,None]
        realized.append(coin.mean((1,2)))
        for d,r,n in np.ndindex(3,2,2):
            # cause, source regime, material, episode, group
            x=waves[d,r]
            val=actions(x,r,n).mean(axis=-2).transpose(2,0,1,3)
            for k in range(4):
                a=val.copy()
                if k==1:
                    a[:,1,:,:]=a[:,1].mean(axis=1,keepdims=True)
                elif k==2:
                    a[:,1]=val[:,1,::-1]
                elif k==3:
                    # Same independently drawn fault hazard and paid readings under both deployment targets.
                    xc=np.where(coin,x[1,0],x[1,1])
                    a[:,1]=actions(xc,r,n).mean(axis=-2)[:,None,:]
                rates[c,d,r,n,k]=a
    risk=np.stack(target_risks)
    # Compute each catalog's expected loss relative to the same catalog's always-gate risk.
    gaps=risk[:,:,:,1]-risk[:,:,:,0]
    moves=rates[...,0]
    delta=(moves*gaps[:,None,None,None,None,:,None,:]).mean(axis=(-3,-2,-1))
    gate=risk[:,:,:,0].mean((1,2))
    correct_action=(moves[...,0]+1-moves[...,1]).mean(axis=(-1,-2))/2
    cause=rates[...,1]
    cause_ba=(cause[...,0,:]+1-cause[...,1,:]).mean(axis=(-1,-2))/2
    map_rate=rates[...,2]
    map_ba=(map_rate[...,0]+1-map_rate[...,1]).mean(axis=(-1,-2))/2
    draws=np.random.default_rng(60071903).integers(C,size=(6000,C))
    def describe(values):
        return [values.mean(),*np.quantile(values[draws].mean(1),[.025,.975])]
    def crossing(left,right):
        def safe(x,y):
            denom=x-y
            return np.divide(x,denom,out=np.full_like(x,np.nan),where=abs(denom)>1e-16)
        point=safe(np.asarray(left.mean()),np.asarray(right.mean())).item()
        roots=safe(left[draws].mean(1),right[draws].mean(1))
        inside=roots[(roots>=0)&(roots<=1)]
        return dict(reversed=left.mean(),matched=right.mean(),cross=point,
                    bootstrap_fraction_crossing=len(inside)/len(draws),
                    conditional_crossing_CI=np.quantile(inside,[.025,.975]).tolist() if len(inside) else None)
    cells={}
    for n,ename in enumerate(ESTIMATORS):
        cells[ename]={}
        for k,key in enumerate(ASSIGNMENTS):
            a=delta[:,:,1,n,k].mean(1)
            b=delta[:,:,0,n,k].mean(1)
            cells[ename][key]=dict(mean_Brier=(gate+a).mean(),minus_gate=describe(a),minus_sham=describe(a-b),
                                    actual_policy_BA=correct_action[:,:,1,n,k].mean(),cause_BA=cause_ba[:,:,1,n,k].mean(),
                                    MAP_BA=map_ba[:,:,1,n,k].mean(),
                                    fixed_purchase=moves[:,:,1,n,k].mean(),
                                    reset_minus_gate_by_delay=[describe(delta[:,d,1,n,k]) for d in range(3)])
    crossings={}
    for comparator in ['gate','sham']:
        endpoints=[]
        for k in [2,0]:
            a=delta[:,:,1,0,k].mean(1)
            if comparator=='sham': a=a-delta[:,:,0,0,k].mean(1)
            endpoints.append(a)
        crossings[comparator]=crossing(*endpoints)
    target=w*(.29577464788732394-.040796503156872)+.040796503156872
    output=dict(cells=cells,crossings=crossings,delay_weights=[1/3]*3,
                surface_hazard_percentiles=np.quantile(target,[0,.25,.5,.75,1]).tolist(),
                max_empirical_mix_error=float(np.abs(np.stack(realized)-w).max()))
    (ROOT/'transfer60.json').write_text(json.dumps(output,indent=2,default=float)+'\n')
    np.savez_compressed(ROOT/'transfer60_cells.npz',rates=rates,conditional_risk=risk,mixture_weights=w,delta=delta)
if __name__=='__main__':
    main()
