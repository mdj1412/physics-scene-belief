"""Integrate the decision loss of random readout subsets at exactly matched per-pilot coverage."""
import json
import math
import os
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'sensor_reset_v59'))
from route59 import log_odds, SOURCE_LOG_CUTOFF
from score59 import MIX

METHODS=['v61_reset','selective_reset','matched_random_reset','electronics_reset','v54_gate']
B=8
POPCOUNT=np.array([bin(i).count("1") for i in range(256)],dtype=int)
BITS=1<<np.arange(8)
SUBSETS=(np.arange(256)[:,None]&BITS)>0

def pairs_to_codes(symbol):
    pair=symbol.reshape(*symbol.shape[:-1],4,2)
    return (13*pair[...,0]+pair[...,1]).astype('u1')

def split_codes(codes):
    return np.stack([codes//13,codes%13],axis=-1).reshape(*codes.shape[:-1],8)

def classify(s):
    odds, cause=log_odds(pairs_to_codes(s),1,True)
    return np.stack([odds>SOURCE_LOG_CUTOFF,cause>1.e-12],-1)

def randomized(old,blank,valid,keep):
    n=valid.sum(1)
    k=keep.sum(1)
    availability=valid.astype(int)@BITS
    count=np.zeros(len(old),dtype=int)
    totals=np.zeros((len(old),2))
    for mask,subset in enumerate(SUBSETS):
        match=(POPCOUNT[mask]==k)&((availability&mask)==mask)
        if not match.any():
            continue
        masked=np.where(subset,old[match],blank[match])
        totals[match]+=classify(masked)
        count[match]+=1
    expected=np.array([math.comb(int(ni),int(ki)) for ni,ki in zip(n,k)])
    np.testing.assert_array_equal(count,expected)
    return totals/count[:,None]

def aggregate_risk(c):
    with np.load(ROOT/'pilots'/f'panel_{c:02d}.npz') as f:
        old=split_codes(f['codes'][:,1,1]);sel=split_codes(f['codes'][:,2,1])
    shape=old.shape[:-1]
    assert shape==(4,2,2,2,512)
    orig=old.reshape(-1,8)
    selective=sel.reshape(-1,8)
    blank=np.where(orig!=0,1+(orig.astype(int)-1)%4,0).astype('u1')
    valid=orig>4
    keep=selective>4
    assert np.all(~keep | valid)
    np.testing.assert_array_equal(selective,np.where(keep,orig,blank))
    base=classify(orig)
    selected=classify(selective)
    blinded=classify(blank)
    random=randomized(orig,blank,valid,keep)
    gate=np.zeros_like(base)
    routes=np.stack([base,selected,random,blinded,gate],axis=1).reshape(*shape,5,2)
    # Two potential readings for C are distributed equally into target regimes A/B.
    def independent(x):
        x=x.copy()
        x[:,1]=x[:,1].mean(1,keepdims=True)
        return x
    def byboard(x):
        return independent(x).reshape(4,2,2,2,64,B,-1).mean(4)
    a=byboard(routes[...,0]); diag=byboard(routes[...,1])
    moves=routes[...,0]
    changes=byboard(moves+moves[...,:1]-2*moves*moves[...,:1])
    with np.load(ROOT/'scored62'/f'panel_{c:02d}.npz') as f:
        m=((np.arange(4)+c)%4)//2
        fine=np.stack([f['exact_cells'][:,:,m==j,:2].mean((1,2)) for j in range(2)])
        bank=np.einsum('mea,ze->mza',fine,MIX)
    delta=bank[:,:,1]-bank[:,:,0]
    assert np.isfinite(delta).all()
    dr=(a*delta.T[None,None,:,:,None,None]).mean((1,2,3))
    policy=(a[:,:,0]+1-a[:,:,1]).mean((1,2))/2
    cause=(diag[:,0]+1-diag[:,1]).mean((1,2))/2
    metrics=np.stack([dr+bank[:,:,0].mean(),dr,a.mean((1,2,3)),policy,cause,
                      changes.mean((1,2,3))],axis=-1)
    n=valid.sum(1)
    k=keep.sum(1)
    with np.load(ROOT/'vision_audit'/f'panel_{c:02d}.npz') as f:
        t=f['truth']
    truth=(t[...,0]!=t[...,1]).reshape(2,2,2,512,8)
    wrong=((old>=9)!=truth[None]).reshape(-1,8)&valid
    gain=np.stack([n,k,wrong.sum(1),(wrong&keep).sum(1),
                   k*wrong.sum(1)/np.maximum(n,1),n>0,(n>0)&(k==0),n!=k,
                   selected[:,0]!=base[:,0]],axis=-1).reshape(*shape,-1)
    return metrics,byboard(gain).mean((1,2,3))

def main():
    assert os.getenv('CUDA_VISIBLE_DEVICES')=='' and os.getenv('SLURM_JOB_ID')
    results=[aggregate_risk(c) for c in range(24)]
    x,reads=map(np.stack,zip(*results))
    rng=np.random.default_rng(6209533)
    wc=rng.multinomial(24,np.full(24,1/24),size=8000)/24
    wb=rng.multinomial(B,np.full(B,1/B),size=8000)/B
    ave=x.mean(1)
    boot=np.einsum('dc,db,cbmi->dmi',wc,wb,ave,optimize=True)
    def summarize(point,v):
        return [float(point),*np.quantile(v,[.025,.975]).tolist()]
    out={'methods':{},'cells':{},'selective_minus':{}}
    keys=['mean_Brier','delta_gate','fixed_rate','AB_agreement','cause_BA','flips_vs_v61_reset']
    def optical(v):
        n,k,eorig,esel,erandom,any_old,last_lost,differ,flip=v
        return dict(v61_pairs_per_pilot=n,selective_pairs_per_pilot=k,retained_fraction=k/n if n else None,
            v61_error=eorig/n if n else None,selective_error=esel/k if k else None,
            same_number_random_error=erandom/k if k else None,
            dropped_error=(eorig-esel)/(n-k) if n>k else None,
            v61_has_any_read=any_old,last_read_lost=last_lost,at_least_one_dropped=differ,
            selective_action_flip=flip)
    for j,m in enumerate(METHODS):
        out['methods'][m]={key:summarize(ave[:,:,j,i].mean(),boot[:,j,i]) for i,key in enumerate(keys)
                           if m!='v54_gate' or key!='cause_BA'}
        if j!=1:
            out['selective_minus'][m]=summarize((ave[:,:,1,1]-ave[:,:,j,1]).mean(),boot[:,1,1]-boot[:,j,1])
    out['readout']=optical(reads.mean((0,1,2)))
    for i in range(4):
        out['cells'][i]=dict(readout=optical(reads[:,i].mean((0,1))),
                              deltas_gate=x[:,i,:,:,1].mean((0,1)).tolist())
    np.savez_compressed(ROOT/'thinning62_cells.npz',catalog_cell_board_method_metrics=x,
                        readout_counts=reads,methods=METHODS,metrics=keys)
    (ROOT/'thinning62.json').write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__':
    main()
