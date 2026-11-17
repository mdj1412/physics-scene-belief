"""Analytical collision bound, independently checked against explicit binary squared errors."""
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parent
ARCH=ROOT.parent/"factorial_projection_v47"


def main():
    assert os.getenv("SLURM_JOB_ID") and os.getenv("CUDA_VISIBLE_DEVICES")==""
    axes=json.loads((ROOT.parent/"image_pressure_v45/source_axis45.json").read_text())
    by_panel=[]
    counts=np.zeros((2,2),dtype=int)
    max_err=max_midpoint_error=0.
    gaps=[]
    for p in range(12):
        name=f"panel_{p:02d}.npz"
        with np.load(ARCH/"forecast"/name) as raw:
            q = raw["query_paid"].astype(float)*1000.
            selected = raw["selected"]
            np.testing.assert_array_equal(selected[:,:,0],selected[:,:,1])
            active = selected[:,:,0]==raw["query_id"]
        # f,alignment,height,variant,env,object,noise,channel
        mid=q[:,:,:,0]*.5+q[:,:,:,1]*.5
        max_midpoint_error=max(max_midpoint_error,np.nanmax(abs(mid[:,0]-mid[:,1])))
        for j,h in enumerate((19,31)):
            u=np.asarray(axes[str(h)]["axis"])
            u=u/np.linalg.norm(u)
            perp=np.eye(7)-np.outer(u,u)
            base=mid[:,1,j][active[:,j]]
            assert np.isfinite(base).all()
            shared=base @ perp
            a,b=base-1.20*u,base+1.20*u
            for obs in (a,b):
                max_err=max(max_err,np.max(abs(obs@perp-shared)))
            np.testing.assert_allclose((b-a)@u,2.4,atol=1e-12,rtol=0.)
            gaps.append(float(np.min((b-a)@u)))
        with np.load(ARCH/"labels"/name) as data:
            y=data["labels"].astype(float)
        # [family,height,coupling,radius,offset,env,object]
        y0,y1=y[:,:,0],y[:,:,1]
        mask=np.broadcast_to(active[:,:,None,None,None],y0.shape)
        flip=y0!=y1
        optimal=(y0+y1)/2
        explicit=.5*((optimal-y0)**2+(optimal-y1)**2)
        np.testing.assert_array_equal(explicit,flip*.25)
        row=[]
        for coupling in range(2):
            m=mask[:,:,coupling]
            counts[coupling]+=[int(flip[:,:,coupling][m].sum()),int(m.sum())]
            row.append([flip[:,:,coupling][m].mean(),explicit[:,:,coupling][m].mean()])
        by_panel.append(row)
    vals=np.array(by_panel)
    mean=vals.mean(0)
    half=t.ppf(.975,len(vals)-1)*vals.std(0,ddof=1)/np.sqrt(len(vals))
    assert max_err < 1e-10 and mean[0,0]==0
    result=dict(couplings=["inert","strong"],fields=["flip_rate","paired_Brier_floor"],
        mean_lo_hi=np.stack((mean,mean-half,mean+half),-1).tolist(),counts_flip_total=counts.tolist(),
        max_projected_residual_mm=float(max_err),min_full_separation_mm=min(gaps),
        archived_low_high_midpoint_residual_mm=float(max_midpoint_error),panel_rates=vals.tolist(),
        meaning="Analytic pair-identity privileged risk and lower bound on absolute compressed risk; not a bound on operational excess risk.",
        source="v47 pressure labels, newly imposed exact alignment 1; purchased query arm only, scenario repeats clustered",
        job=os.environ["SLURM_JOB_ID"])
    (ROOT/"exp_twins.json").write_text(json.dumps(result,indent=2)+"\n")
    print(mean,counts,flush=True)


if __name__=="__main__":
    main()
