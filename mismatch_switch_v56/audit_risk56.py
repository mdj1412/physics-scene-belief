"""Independent paid-budget and weighted-session endpoint audit on fresh server panels."""
from pathlib import Path
import json
import numpy as np
from switch56 import choose, W, PRIMARY, CROSSING

ROOT=Path(__file__).resolve().parent
def main():
    v=np.load(ROOT/'score56_cells.npz')
    cells=v['by_e']; fr=v['route_rates'][...,3]
    result=json.loads((ROOT/'score56.json').read_text())
    vectors=[]; weights=[];cells55=[]
    # Direct mixture all session losses; one action is chosen before its outcome,
    # and its mass includes catalog, material, w, e and surface-change event.
    for c in range(24):
        material=((np.arange(4)+c)%4)//2
        with np.load(ROOT/'scored56'/f'panel_{c:02d}.npz') as f: raw=f['real'][0]
        for m in [0,1]:
            for e in range(3):
                for event,pe in enumerate((.5625,.4375)):
                    for arm in (0,1):
                        vals=np.take(raw[e,event,...,arm],np.flatnonzero(material==m),axis=1).ravel()
                        mass=np.zeros(3)
                        for iw in PRIMARY:
                            eprob=W[iw] if e==2 else (1-W[iw])/2
                            a=[fr[c,m,iw],0.,1.]
                            mass+=eprob*np.array([p if arm else 1-p for p in a])
                        mass*=pe/(24*2*len(PRIMARY)*len(vals))
                        vectors.append(vals);weights.append(np.tile(mass,(len(vals),1)))
    vec=np.concatenate(vectors);wt=np.concatenate(weights);ix=np.argsort(vec,kind='stable')
    sums=np.cumsum(wt[ix],axis=0);q=[float(vec[ix[np.searchsorted(row,.90)]]) for row in sums.T]
    np.testing.assert_allclose(sums[-1],1,atol=2e-10)
    np.testing.assert_allclose(q,result['session_p90'][0],atol=2e-7)
    e_gap=(cells[...,0]-cells[...,1]).mean((0,1))
    wzero=-e_gap[:2].mean()/(e_gap[2]-e_gap[:2].mean())
    pm=np.array(result['mean'])
    recovery=(pm[1]-pm[0])/(pm[1]-pm[3])
    # Paired bootstrap of gain-recovery ratio can be unstable; only report point estimate.
    pure_ends=np.array(result['mean_by_w'])[[0,-1],:].mean(0)
    far=v['means_by_w_k'][:,:,[0,6],3,:].mean((1,2))
    draws=np.random.default_rng(561402).integers(len(far),size=(5000,len(far)))
    bs=far[draws].mean(1); delta=bs[:,0,None]-bs[:,1:3]
    ci=np.quantile(delta,[.0125,.9875],axis=0).T
    res=dict(far_secondary95=ci.tolist(),quantiles=q,solved_crossing=float(wzero),headroom_known_w=float(pm[1]-pm[3]),
             headroom_recovered=float(recovery),far_controls_equal_mean=pure_ends.tolist())
    (ROOT/'audit_risk56.json').write_text(json.dumps(res,indent=2)+'\n')
if __name__=='__main__':main()
