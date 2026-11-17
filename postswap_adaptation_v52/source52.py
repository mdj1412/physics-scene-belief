"""Derive expected Brier polynomial for signed blends and unsigned from v47 source only."""
import os
import json
import sys
import numpy as np
from policy52 import ROOT,V50
sys.path.insert(0,str(ROOT.parent/'contact_transfer_v49'))
from policy49 import update
from unsigned50 import predict
from sensor48 import readings,STATES,HEIGHTS
V47=ROOT.parent/'factorial_projection_v47'
V48=ROOT.parent/'preload_projection_v48'
def main():
    signed=np.asarray(json.loads((V48/'direct_source48.json').read_text())['coefficients'])
    unsigned=np.asarray(json.loads((V50/'unsigned_source50.json').read_text())['coefficients'])
    co=np.zeros((12,2,4))
    for i in range(12):
        n=f'panel_{i:02d}.npz'
        with np.load(V47/'worlds'/n) as pop,np.load(V47/'forecast'/n) as pred,np.load(V47/'labels'/n) as lab:
            seed=int(pop['seed']); fams=pred['families']
            ps=pred['p_original']; arm=np.isfinite(pred['query_paid'][...,0]); truth=lab['labels']
        for f,fam in enumerate(fams):
            for h,height in enumerate(HEIGHTS):
                p=ps[f,:,h];a=arm[f,:,h].astype(int)
                for s,prob in zip(STATES,(.5,.25,.25)):
                    d=readings({'seed':seed},str(fam),height,s,height)
                    plus=update(p,d,signed[1,h][a]);minus=update(p,-d,signed[1,h][a])
                    un=predict(p,d,unsigned[h][a]);delta=plus-minus
                    for j,full in enumerate((-s,s)):
                        y=truth[f,h,int(full==1),int(full!=0)].mean((0,1))[...,None]
                        expected=lambda x: (x*x+(1-2*x)*y).mean()
                        coeff=[expected(minus),2*(delta*(minus-y)).mean(),(delta*delta).mean(),expected(un)]
                        weight=prob/(len(fams)*len(HEIGHTS))
                        co[i,j]+=weight*np.asarray(coeff)
                        np.testing.assert_allclose(expected(plus),sum(coeff[:3]),atol=1e-12)
    out=dict(coefficients=co.mean(0).tolist(),per_catalog=co.tolist(),basis=['constant','alpha','alpha squared','unsigned Brier'],
             endpoints=[0,1],scope='v47 only; v47 also trained frozen predictors. Source risk extrapolation to target is diagnostic assumption.')
    (ROOT/'source_risk52.json').write_text(json.dumps(out,indent=1))
if __name__=='__main__':
    assert os.getenv('SLURM_JOB_ID')
    main()
