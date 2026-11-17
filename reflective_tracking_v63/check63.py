"""Independently validate analytic policy marginalization on known v62 optical pilots."""
from itertools import combinations
from pathlib import Path
import json
import os
import numpy as np
from score63 import ROOT, split_codes, classify, randomized, digest
from route59 import log_odds, SOURCE_LOG_CUTOFF

def brute(symbol,k):
    candidates=np.flatnonzero(symbol>=5)
    hidden=symbol.copy().astype(int)
    hidden[hidden>0]=1+(hidden[hidden>0]-1)%4
    res=[]
    for loc in combinations(candidates,k):
        r=hidden.copy()
        r[list(loc)]=symbol[list(loc)]
        odds,c=log_odds((r[::2]*13+r[1::2]).astype('u1'),1,True)
        res.append([odds>SOURCE_LOG_CUTOFF,c>1e-12])
    return np.mean(res,axis=0)

def main():
    assert os.getenv('SLURM_JOB_ID') and os.getenv('CUDA_VISIBLE_DEVICES')==''
    src=ROOT.parent/'exposure_abstention_v62'
    with np.load(src/'pilots'/'panel_01.npz') as f:
        old=split_codes(f['codes'][:,1,1]).reshape(-1,8)
        new=split_codes(f['codes'][:,2,1]).reshape(-1,8)
    n=(old>=5).sum(1);k=(new>=5).sum(1)
    ix=np.r_[np.flatnonzero((k>0)&(k<n))[:50],np.arange(120)]
    a,b=old[ix],new[ix]
    blank=np.where(a!=0,1+(a.astype(int)-1)%4,0).astype('u1')
    reference=np.stack([brute(s,int(ki)) for s,ki in zip(a,k[ix])])
    est=randomized(a,blank,a>=5,b>=5)
    np.testing.assert_array_equal(reference,est)
    np.testing.assert_array_equal(classify(b[k[ix]==n[ix]]),est[k[ix]==n[ix]])
    assert np.isfinite(est).all()
    print('v62 archival pilots;',len(ix),'rows, flexible',((k[ix]>0)&(k[ix]<n[ix])).sum(),
          'maximum error',np.max(abs(reference-est)),flush=True)
    if (ROOT/'score63.json').exists():
        q=json.loads((ROOT/'score63.json').read_text())
        with np.load(ROOT/'score63_cells.npz') as f:
            rates,fine=f['rates'],f['fine']
            from score63 import MIX
            p=rates.copy()
            p[:,:,:,:,1,:,:]=p[:,:,:,:,1,:,:].mean(-2,keepdims=True)
            ref=np.zeros(len(q['methods']))
            for regime,w in enumerate(MIX):
                for mat in (0,1):
                    for mech in (0,1):
                        l0=(fine[:,mat,:,0]@w);l1=(fine[:,mat,:,1]@w)
                        action=p[:,:,:,mat,mech,regime,:].mean((1,2))
                        ref+=(l0[:,None]+action*(l1-l0)[:,None]).mean(0)/8
            err=np.max(abs(ref-q['uniform']['mean_Brier']))
            assert err<1e-13
            sel=q['uniform']['selection']
            assert abs(sel['conditional_mean_Brier_diff_on_flexible'][0]*sel['flexible_pilot_share'][0]-
                       np.subtract(*q['uniform']['mean_Brier'][:2]))<1e-14
            print('v63 direct integration, maximum error',err,flush=True)
if __name__=='__main__': main()
