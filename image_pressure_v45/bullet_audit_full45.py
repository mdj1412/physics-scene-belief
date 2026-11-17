"""Summarize actual onset coordinates across all held-out independent Bullet targets."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
    groups=[]
    for scope,sl in [("all",slice(None)),("first8",slice(None,8)),("remaining120",slice(8,None))]:
        errs=[];spans=[];changes=[];preval=[];total=0;valid=0;live=0;maxspeed=0.;dv=[]
        for p in sorted((ROOT/"labels45_bullet_full").rglob("*.npz")):
            rel=p.relative_to(ROOT/"labels45_bullet_full")
            with np.load(p) as a,np.load(ROOT/"labels45"/rel) as b,np.load(ROOT/"labels45_bullet"/rel) as q:
                for k in q:
                    if k=='desired_centers_mm':continue
                    np.testing.assert_array_equal(a[k][:,:,:8],q[k])
                center=a["desired_centers_mm"][:,None,None];x=a["first_contact_x_mm"][...,sl]
                v=a["valid"][...,sl]; y=a["labels"][...,sl]; live+=np.sum(a["loaded_end"][...,sl]&v)
                np.testing.assert_allclose(x-center,a["first_site_delta_mm"][...,sl],atol=1e-12)
                total+=v.size;valid+=int(v.sum())
                errs.append((x-center)[v]);preval.append(y[v])
                changes.append((b["labels"][0,...,sl]!=y)[v])
                dv.append(abs(a["slip_speed"][...,sl]-b["slip_speed"][0,...,sl])[v])
        err=np.concatenate(errs);absx=abs(err)
        groups.append(dict(scope=scope,slots=total,valid=valid,loaded_end=int(live),mean_signed_error_mm=err.mean(),mae_mm=absx.mean(),
             p90_p95_p99_max_abs_mm=np.quantile(absx,[.90,.95,.99,1.]), frac_abs_over_025mm=(absx>.25).mean(),
             stick_fraction=np.concatenate(preval).mean(),point_vs_bullet_label_flip_fraction=np.concatenate(changes).mean(),
             residual_speed_mae_vs_point=np.concatenate(dv).mean()))
    assert groups[0]["slots"]==884736
    (ROOT/"bullet_full_audit45.json").write_text(json.dumps(groups,indent=2,default=lambda x:x.tolist())+"\n")
    print(groups)
if __name__=="__main__":main()
