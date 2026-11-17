"""Read independent pressure artifacts and decompose the D response by event direction."""
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.stats import t

root=Path(__file__).resolve().parent
def ci(a, m=1):
    a=np.asarray(a)
    mid=a.mean(axis=0)
    hw=t.ppf(1-.025/m,len(a)-1)*a.std(axis=0,ddof=1)/np.sqrt(len(a))
    return np.stack((mid,mid-hw,mid+hw),axis=-1)

def audit():
    effects=np.empty((12,2,2,2))
    event=np.empty((12,2,2));labelrate=np.empty((12,2,3))
    span=np.empty((12,2,3));surface_range=[np.inf,-np.inf]
    numeric=[]; total=0; btotal=0; valid=0; sids=[]; geom=[]; bflip=[]; bend=0; changes=[]; drt=[]
    exact_area_error=0.; fine_max_speed=0.; norm_min=np.inf; norm_max=-np.inf
    for p in range(12):
        for s,shape in enumerate(("fold","ripple")):
            per=[];wat=[];evt=[];areas=[]
            fdir=root/"labels45"/f"panel_{p:02d}"
            paths=sorted(fdir.glob(f"{shape}_*.npz"))
            assert len(paths)==24
            for path in paths:
                rel=path.relative_to(root/"labels45")
                with np.load(path) as a,np.load(root/"labels45_pressure_fine"/rel) as b,np.load(root/"labels45_bullet"/rel) as c,np.load(root/"results"/rel) as d:
                    y=a["labels"];speed=a["slip_speed"];fine=b["labels"];fspeed=b["slip_speed"]
                    J=a["normal_impulse_Ns"]
                    assert J.shape==(3,4,128) and np.array_equal(J,b["normal_impulse_Ns"])
                    norm_min=min(norm_min,float(J.min()));norm_max=max(norm_max,float(J.max()))
                    area=a["surface_effective_area_mm2"]
                    exact_area_error=max(exact_area_error,float(abs(area-b["surface_effective_area_mm2"]).max()))
                    assert np.array_equal(y,speed<=.020) and np.isfinite(speed).all() and np.all(area[2]>area[1]) and np.all(area[1]>area[0]) and np.all(area[0]==0)
                    ratio=area[2]/area[1]
                    surface_range=[min(surface_range[0],float(ratio.min())),max(surface_range[1],float(ratio.max()))]
                    nx,ny=y[1],y[2]
                    f=nx&~ny; j=~nx&ny
                    evt.append([f.mean(),j.mean()])
                    areas.append(area.mean(axis=(1,2,3)));wat.append(y.mean(axis=(1,2,3)))
                    pp = d["pred"][:,0]
                    prob = pp.mean(axis=-1)
                    dprob=prob[1:]-prob[0]
                    # D=(p_proj²-p_orig²)−2y(p_proj-p_orig); only y changes with area.
                    dc=np.stack([(2*dprob[:,None]*f).mean(axis=(1,2,3)),(-2*dprob[:,None]*j).mean(axis=(1,2,3))],axis=-1)
                    per.append(dc)
                    derived=np.stack([((p[None,None]-y[...,None])**2).mean(axis=(1,2,3,4)) for p in pp])
                    paired=derived[1:]-derived[0]
                    np.testing.assert_allclose(paired[:,2]-paired[:,1],dc.sum(1),rtol=0,atol=1e-12)
                    base=y[0,:,:,:8]
                    cy=c["labels"]; cv=c["valid"]; err=c["first_site_delta_mm"][cv]
                    assert np.array_equal(cv, np.isfinite(c["first_contact_x_mm"]))
                    np.testing.assert_allclose(c["first_contact_x_mm"],c["desired_centers_mm"][:,None,None]+c["first_site_delta_mm"],rtol=0,atol=1e-11)
                    bflip.append(np.sum((cy!=base)&cv));bend+=np.sum(c["loaded_end"])
                    sids.append(err);geom.append(abs(err));valid+=cv.sum();btotal+=cv.size
                    changes.append((np.sum(fine!=y,axis=(1,2,3)),(fspeed!=speed).size))
                    drt.append(np.mean(abs(fspeed-speed),axis=(1,2,3))); fine_max_speed=max(fine_max_speed,float(abs(fspeed-speed).max()))
                    numeric.append(np.sum(fine!=y));total+=y.size
            effects[p,s]=np.mean(per,axis=0);event[p,s]=np.mean(evt,axis=0);span[p,s]=np.mean(areas,axis=0);labelrate[p,s]=np.mean(wat,axis=0)
    abs_err=np.concatenate(geom); signed=np.concatenate(sids)
    output=dict(dimensions="shape fold/ripple; decoder source-axis/privileged; switch stick-to-slip/slip-to-stick",
             effect_broad_minus_narrow_by_switch=ci(effects,8),
             effect_broad_minus_narrow_by_decoder=ci(effects.sum(-1),4),
             transition_broad_vs_narrow=ci(event,4),transition_total=ci(event.sum(-1),2),
             prevalence=ci(labelrate,6),mean_effective_surface_area_mm2=ci(span,4),
             effective_area_ratio_range=surface_range,quadrature_area_max_error_mm2=exact_area_error,
             scalar_normal_impulse_range_Ns=[norm_min,norm_max],primary_labels=total,
             quadrature_labels_changed=int(np.sum(numeric)),
             quadrature_labels_changed_per_radius=np.sum([n for n,_ in changes],axis=0),
             quadrature_speed_mae_per_radius=np.mean(drt,axis=0),quadrature_max_speed_error=fine_max_speed,
             bullet_total_slots=btotal,bullet_valid=int(valid),bullet_loaded_end=int(bend),
             bullet_center_signed_mean_mm=signed.mean(),bullet_center_mae_mm=abs_err.mean(),
             bullet_center_abs_quantiles=np.quantile(abs_err,[.5,.9,.95,.99,1.]),bullet_v_point_flip_fraction=np.sum(bflip)/valid,
             bullet_nearest_bin_disagreement=(abs_err>.25).mean())
    (root/"pressure_audit45.json").write_text(json.dumps(output,indent=2,allow_nan=False,default=lambda x:x.tolist())+"\n")
    print(json.dumps({k:str(output[k]) for k in ("transition_total","primary_labels","quadrature_labels_changed","bullet_valid","bullet_center_mae_mm","bullet_center_abs_quantiles","bullet_v_point_flip_fraction","effect_broad_minus_narrow_by_decoder")}))
if __name__=="__main__":audit()
