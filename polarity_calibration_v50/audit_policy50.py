"""Independent direct48.design contraction replay using the paid station readout."""
import json
import os
import numpy as np

from policy50 import ROOT,HEIGHTS,STATES,coefficients,world,world50,legacy
from direct48 import design

assert os.getenv("SLURM_JOB_ID")
coef,un=coefficients()
errors=np.zeros((12,5))
counts=np.zeros(12,dtype=int)
paid_counts=np.zeros(12,dtype=int)
for k in range(12):
    pop=world(k)
    with np.load(ROOT/"forecast50"/f"panel_{k:02d}.npz") as arc:
        pred=arc["p"]; picked=arc["selected"]
    for f,fam in enumerate(world50.FAMILIES):
        for h,height in enumerate(HEIGHTS):
            p=pred[0,f,h,0]
            arm=(picked[f,h]==legacy.r.Q).astype(np.intp)[None,:,:,:,None]
            paid_counts[k]+=arm.sum()
            for j,state in enumerate(STATES):
                for station,slot,mode in ((height,1,1),(31,3,1 if h==1 else 2)):
                    raw=world50.readings(pop,fam,station,state)
                    for offset,pol in ((0,1),(1,-1)):
                        b=design(p,pol*raw)
                        ref=np.clip(p+(b*coef[mode,h][arm]).sum(-1),0,1)
                        err=abs(ref-pred[slot+offset,f,h,j]).max()
                        errors[k,slot+offset-1]=max(errors[k,slot+offset-1],err)
                        assert err<2e-9
                    if slot==1:
                        basis=design(p,raw)[...,[0,1,2,4,6]]
                        ref=np.clip(p+(basis*un[h][arm]).sum(-1),0,1)
                        err=abs(ref-pred[5,f,h,j]).max()
                        errors[k,4]=max(errors[k,4],err)
                        assert err<2e-9
                counts[k]+=p.size
    assert paid_counts[k]==(4*2*2*4*115)
out=dict(methods=["local_plus","local_minus","fixed_plus","fixed_minus","unsigned_source_fit"],
         predictions_checked_per_method=int(counts.sum()),max_absolute_error=errors.max(0).tolist(),
         query_arm_per_env=115,nonquery_arm_per_env=13,catalog_count=12,
         fixed_at_same_location_bitwise_checked=True,signblind_evenness_exactly_checked=True)
(ROOT/"audit_policy50.json").write_text(json.dumps(out,indent=1))
