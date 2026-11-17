"""Score sealed compliant full-history forecasts against external rigid terminal contacts."""
import json
import os
from pathlib import Path
import numpy as np
from crossover68 import H,FAMS,LAWS,ROUTES,archive,routed,ci


def main():
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")==""
    assert (H/"FORECASTS_SEALED").exists()
    raw_sum=np.zeros((12,2,2,5)); raw_n=np.zeros((12,2,2))
    valid_n=total=dis_dt=valid_dt=changed=valid_comp=0
    loaded_n=0
    errors=np.zeros((12,2,2))
    old_diff=np.zeros_like(raw_sum)
    for k in range(12):
        sub=f"panel_{k:02d}"
        rec=archive(H/"forecasts"/"compliant"/(sub+".npz"))
        for name,idx in rec["ix"].items():
            f,s,l,v,height=name.removesuffix(".npz").split("_")
            a,b=FAMS.index(f),LAWS.index(l)
            with np.load(ROUTES/sub/f"{f}_{v}_{height}.npz") as z:gate=z["route"]
            with np.load(H/"bullet_labels"/sub/name) as z:
                valid=z["valid"]; y=z["speed"]<=.02; loaded_n+=z["loaded_end"].sum()
                theta=z["theta"]
            with np.load(H.parent/"invariance_audit_v66"/"locked"/"results"/"worlds"/(sub+".npz")) as z:
                np.testing.assert_array_equal(theta,z["theta"][:,:8])
            with np.load(H/"bullet_fine"/sub/name) as z:
                pairs=valid[:,:,:1,:2]&z["valid"]
                fl=int(np.sum(pairs&((z["speed"]<=.02)!=y[:,:,:1,:2])))
                valid_dt+=pairs.sum();dis_dt+=fl
            with np.load(H/"labels_compliant"/sub/name) as z: yc=z["labels"][...,:8].astype(float)
            p=routed(rec["p"][idx],rec["shift"][idx],gate)[:,:,:,:,:8]
            pred=p.mean((0,5))[:,None]
            sq=(p*p).mean((0,5))[:,None]
            # Score both engines on the same valid rigid-contact attempts.
            sr=[]
            for lab in (y,yc):
                r=(sq+(1-2*pred)*lab[None])*valid[None]
                sr.append(r.sum(axis=(1,2,3,4)))
            raw_sum[k,a,b]+=sr[0];old_diff[k,a,b]+=sr[1]
            raw_n[k,a,b]+=valid.sum();valid_n+=valid.sum();total+=valid.size
            errors[k,a,b]+=np.sum(valid&(y!=yc))
    assert raw_n.min()>0
    risk=raw_sum/raw_n[...,None]; same=old_diff/raw_n[...,None]
    results=dict(families=FAMS,laws=LAWS,primary_subset_score=ci(risk),
                 soft_full=ci(risk[...,1]-risk[...,0]),rescued_pooled=ci((risk[...,4]-risk[...,3]).mean(1)),
                 compared_compliant_soft_full=ci(same[...,1]-same[...,0]),compared_compliant_rescue=ci((same[...,4]-same[...,3]).mean(1)),
                 valid=int(valid_n),total=int(total),loaded_at_end=int(loaded_n),
                 disagreement_with_compliant=ci(errors/raw_n),half_step_compared=int(valid_dt),half_step_changed=int(dis_dt),
                 per_catalog=risk.tolist(),job=os.environ["SLURM_JOB_ID"])
    (H/"bullet_score68.json").write_text(json.dumps(results,indent=2)+"\n")
if __name__=="__main__":main()
