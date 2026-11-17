"""Secondary estimator comparison and randomized linkage diagnostic, using the same paid traces."""
import json
import os
import numpy as np
from estimator50 import DEAD_ZONE, infer
from score50 import ROOT, PRED, RHO, KAPPA, BUDGET, interval, components, integrate, blend

MODES=('fitted_hard','vote_hard','fitted_soft','vote_soft','shuffled_hard','shuffled_soft','unsigned_sym')


def estimate(weak,full,k,unpaired):
    x,y=weak[...,:k],full[...,:k]
    obs=(abs(x)>=DEAD_ZONE[0])&(abs(y)>=DEAD_ZONE[1])
    same=obs&((x>0)==(y>0))
    vote=(1+same.sum(-1))/(2+obs.sum(-1))
    cal=infer(x,y)
    shifted=infer(x,unpaired[...,:k])
    return (cal>=.5,vote>=.5,cal,vote,shifted>=.5,shifted,np.full_like(cal,.5))


def main():
    values=np.empty((12,5,4,len(MODES)))
    rmse=np.zeros((12,5,4,3))
    agreement=np.empty((12,5,4))
    for i in range(12):
        n=f'panel_{i:02d}.npz'
        with np.load(ROOT/'cal50'/n) as d:
            weak=d['weak_mm']; full=d['full_mm']; khat=d['khat']
        # Independent fixed nonzero cyclic permutation within each environment and condition.
        unpaired=np.roll(full,1+i*7,axis=-2)
        with np.load(ROOT/'forecast50'/n) as d,np.load(ROOT/'labels50'/n) as t:
            names=list(d['methods'].astype(str))
            p=d['p'][[names.index(j) for j in PRED]]
            comp,gap=components(p,t['labels'])
        for j,k in enumerate(BUDGET):
            policies=estimate(weak,full,k,unpaired)
            np.testing.assert_allclose(policies[2],khat[:,j],rtol=0,atol=1e-12)
            agreement[i,:,j]=(policies[0]==policies[1]).mean((1,2))
            for c,kappa in enumerate(KAPPA):
                risk,g=integrate(comp,gap,.5,kappa) # local only => invariant to rho.
                for a,policy in enumerate(policies):
                    alpha=policy[c].astype(float)
                    values[i,c,j,a]=blend(risk[1],risk[2],g[0],alpha.mean(-1),(alpha*alpha).mean(-1)).mean()
                rmse[i,c,j]=[((policies[m][c]-kappa)**2).mean() for m in (2,3,5)]
    canon=np.load(ROOT/'catalog_scores50.npz')['matched'].mean(axis=1)
    np.testing.assert_allclose(values[:,:,:,0],canon[:,:,:,0],atol=2e-12)
    np.testing.assert_allclose(values[:,:,:,2],canon[:,:,:,6],atol=2e-12)
    np.testing.assert_allclose(values[:,:,:,6],canon[:,:,:,5],atol=2e-12)
    equal=values.mean(1)
    ix=(0,2,0,2); jx=(1,3,4,5)
    d=equal[:,:,list(ix)]-equal[:,:,list(jx)]
    out=dict(modes=MODES,loss=values.mean(0),uniform=equal.mean(0),fit_minus_vote_and_shuffle=interval(d,16),
             hard_vote_fit_decision_agreement=agreement.mean(0),rmse=np.sqrt(rmse.mean(0)),estimator_order=['likelihood','vote','shuffled_likelihood'],
             scope='Secondary negative control; shuffle full trace across calibration replicates within same environment and same generating kappa, severing specimen linkage. Simple vote gets identical dead zones and paid channels.')
    (ROOT/'controls50.json').write_text(json.dumps(out,default=lambda x:np.asarray(x).tolist(),indent=1))


if __name__=='__main__':
    assert os.getenv('SLURM_JOB_ID')
    main()
