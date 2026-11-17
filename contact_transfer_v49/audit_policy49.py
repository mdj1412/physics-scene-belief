"""Check all archived station-state predictions against independent v48 predictor paths."""
import json
import os
import numpy as np
import torch
from policy49 import METHODS, ROOT, SOURCE
from direct48 import design
from runner48 import sensor48

NAMES=("original","direct_none","direct_local","gate_local","direct_fixed_v48_mapped","gate_fixed_v48_mapped")
W=(.5,.25,.25)

def main():
    assert os.environ.get("SLURM_JOB_ID") and not torch.cuda.is_available()
    coef=np.array(json.loads((SOURCE/"direct_source48.json").read_text())["coefficients"])
    errs=np.zeros((12,len(NAMES)))
    total=np.zeros(len(NAMES),np.int64)
    for k in range(12):
        name=f"panel_{k:02d}.npz"
        with np.load(ROOT/"forecast49"/name) as archive:
            np.testing.assert_array_equal(archive["methods"],METHODS)
            p=archive["p"]
        with np.load(SOURCE/"forecast"/name) as arc, np.load(SOURCE/"worlds"/name) as w:
            p0=arc["p_original"]; p1=arc["p_projected"]; q=arc["query_paid"]
            gate,fgate=arc["gate_preload"],arc["gate_fixed"]
            d=arc["readings_mm"]
            seed=int(w["seed"])
        for f,h,j in np.ndindex(4,2,3):
            o=p0[f,:,h]; b=p1[f,:,h]; a=np.isfinite(q[f,:,h,...,0]).astype(int)
            init=np.clip(o+(design(o,0.)*coef[0,h][a]).sum(-1),0.,1.)
            direct=np.clip(o+(design(o,d[f,h,j,h])*coef[1,h][a]).sum(-1),0.,1.)
            fd=np.clip(o+(design(o,d[f,h,j,1])*coef[1 if h else 2,h][a]).sum(-1),0.,1.)
            signed=sensor48.states_at({"seed":seed},sensor48.HEIGHTS[h],sensor48.STATES[j],31)
            lookup=np.where(signed==0,0,np.where(signed<0,1,2))
            weak=np.broadcast_to(lookup[...,None],o.shape)[None]
            mapped=lambda idx: np.take_along_axis(p[idx,f,h],weak,axis=0)[0]
            ref=(o,init,direct,np.where(gate[f,:,h,:,j],b,o),fd,np.where(fgate[f,:,h,:,j],b,o))
            values=(p[0,f,h,j],p[1,f,h,j],p[2,f,h,j],p[4,f,h,j],mapped(3),mapped(6))
            for n,(x,y) in enumerate(zip(values,ref)):
                err=np.max(abs(x-y))
                assert err < 2e-9,(k,f,h,j,n,err)
                total[n]+=x.size
                errs[k,n]=max(err,errs[k,n])
    data=dict(methods=NAMES,n_per_method=total.tolist(),max_error=errs.max(0).tolist(),
              per_catalog_max=errs.tolist(),threshold=2e-9,
              scope="No v48 or v49 full-load labels accessed. Fixed predictions checked by restoring v48 station transfer map.")
    (ROOT/"audit_policy49.json").write_text(json.dumps(data,indent=2)+"\n")
if __name__=="__main__":main()
