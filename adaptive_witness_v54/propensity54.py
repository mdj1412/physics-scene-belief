"""Loss for an independent lottery at the robust gate's marginal purchase rate."""
import json
import os
import numpy as np
from score54 import ROOT,V53,GM,METHODS,pair_ci

def main():
    assert os.getenv('SLURM_JOB_ID')
    s=json.loads((ROOT/'score54.json').read_text()); i=METHODS.index('robust_gate')
    rate=s['impact_rate'][i]
    with np.load(ROOT/'catalog54.npz') as f: means=f['means'][:,0]
    random_mean=(1-rate)*means[:,1]+rate*means[:,2]
    with np.load(V53/'session53.npz') as f:
        cols=[list(f['methods'].astype(str)).index(n) for n in ('reference4_joint','impact_joint')]
        scores=f['realized'][...,cols]
    groups=[(np.sort(scores[:,:,ev,...,m].ravel()),ew*mw) for ev,ew in enumerate((1-GM,GM)) for m,mw in enumerate((1-rate,rate))]
    lo,hi=0.,1.
    for _ in range(35):
        mid=(lo+hi)/2
        cdf=sum(w*np.searchsorted(a,mid,'right')/a.size for a,w in groups)
        if cdf>=.9:hi=mid
        else:lo=mid
    res=dict(impact_fraction=rate,random_mean=random_mean.mean(),gate_mean=means[:,i].mean(),
             gain_mean=pair_ci(means[:,i]-random_mean,1).tolist(),random_p90=hi,gate_p90=s['session_p90'][i],diff_p90=s['session_p90'][i]-hi)
    (ROOT/'propensity54.json').write_text(json.dumps(res,indent=1))
    print(res)
if __name__=='__main__':main()
