"""Verify all target impacts, matched contact coordinates and endpoint sensitivity."""
import argparse
from itertools import product
import json
from pathlib import Path
import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parent
from geometry46 import SHAPES, VARIANTS

def ci(v,m):
    avg=v.mean(0)
    rad=t.ppf(1.-.025/m,len(v)-1)*v.std(axis=0,ddof=1)/np.sqrt(len(v))
    return np.stack((avg,avg-rad,avg+rad),-1)

def audit():
    area=np.zeros((12,2,3))
    prevalence=np.zeros_like(area)
    switching=np.zeros((12,2))
    d_area=np.zeros_like(area)
    # physical settings, comparisons visual joint vs orig / v45 / refit x
    sensitivity=np.zeros((12,2,4,3))
    flips=np.zeros(3,np.int64)
    point_bullet=np.zeros((12,2))
    err=[]
    load=0
    n=0
    fine_num=0
    mismatch=0
    last_forced=0
    switch_sign=0
    diffv=0.
    for panel,fam,variant,h in product(range(12),range(2),VARIANTS,(19,31)):
        sub=f"panel_{panel:02d}"
        with np.load(ROOT/"route46"/sub/f"{SHAPES[fam]}_{variant}_h{h}.npz") as q:
            route=q["joint"].astype(float);baseline=q["v45"].astype(float);refit=q["position"].astype(float)
        mix=np.stack([route,route-baseline,route-refit],-1)
        for sf,law in product(("inlay","woven"),("rational","gaussian_rate")):
            path=Path(sub)/f"{SHAPES[fam]}_{sf}_{law}_{variant}_h{h}.npz"
            with np.load(ROOT/"results"/path) as p:
                orig,proj=p["pred"][:,0]
            dsq=(proj**2-orig**2).mean(-1)
            lin=-2*(proj-orig).mean(-1)
            with np.load(ROOT/"labels46"/path) as a,np.load(ROOT/"labels46_fine"/path) as b,np.load(ROOT/"labels46_bullet"/path) as c:
                y=a["labels"]
                speed=a["slip_speed"]
                yf=b["labels"]
                np.testing.assert_array_equal(a["normal_impulse_Ns"],b["normal_impulse_Ns"])
                np.testing.assert_array_equal(y,speed<=.02)
                assert c["valid"].all()
                oc=c["first_contact_x_mm"]
                requested=a["centers_mm"][:,None,None]
                err.append(abs(oc-requested).ravel())
                change=abs(oc-requested)>.25
                mismatch+=change.sum()
                last_forced+=c["loaded_end"].sum()
                n+=oc.size;fine_num+=y.size
                diffv=max(diffv,float(np.max(abs(b['slip_speed']-speed))))
                flips+=(y!=yf).sum((1,2,3))
                area[panel,fam]+=a["surface_effective_area_mm2"].mean((1,2,3))/24.
                prevalence[panel,fam]+=y.mean((1,2,3))/24.
                switching[panel,fam]+=(y[1]!=y[2]).mean()/24.
                point_bullet[panel,fam]+=(y[0]!=c["labels"]).mean()/24.
                all_d=dsq+lin*y
                d_area[panel,fam]+=all_d.mean((1,2,3))/24.
                for i,ys in enumerate((y,yf,speed<=.01,speed<=.04)):
                    dy=(dsq+lin*ys[1:].mean(0))[...,None]
                    sensitivity[panel,fam,i]+=(dy*mix).mean((0,1,2))/24.
    e=np.concatenate(err)
    out=dict(shapes=SHAPES, settings=["primary","fine","threshold .010","threshold .040"],
             area_peak_mm2=ci(area,4),stick_rate=ci(prevalence,6),area_flip=ci(switching,2),
             d_project_minus_original_by_area=ci(d_area,6),wide_minus_narrow=ci(d_area[:,:,2]-d_area[:,:,1],2),
             point_bullet_label_flip=ci(point_bullet,2),pressure_outcomes=int(fine_num),rigid_outcomes=int(n),
             rigid_all_valid=True,rigid_at_horizon_loaded=int(last_forced),first_coordinate_abs_mean_p90_max=[float(e.mean()),float(np.quantile(e,.9)),float(e.max())],
             number_over_quarter_mm=int(mismatch),finer_solver_flips_by_radius=flips,fine_max_speed_error=diffv,
             sensitivity_joint_contrasts=ci(sensitivity.mean(1),3),sensitivity_family_joint_contrasts=ci(sensitivity,6))
    (ROOT/"mechanics_audit46.json").write_text(json.dumps(out,indent=2,default=lambda z:np.asarray(z).tolist(),allow_nan=False)+"\n")
    return out
if __name__=="__main__":
    stats=audit()
    for key in ("pressure_outcomes","rigid_outcomes","first_coordinate_abs_mean_p90_max","finer_solver_flips_by_radius","sensitivity_joint_contrasts"):
        print(key,stats[key])
