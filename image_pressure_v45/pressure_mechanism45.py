"""Exploratory spatial ablation of surface normals and the pressure-sampled material law."""
import itertools
import json
from pathlib import Path
import numpy as np
import pressure45 as w
from pressure_audit45 import ci

ROOT=Path(__file__).resolve().parent

def run():
    # [panel, shape, geom field, material field] for the source-axis D effect.
    effects=np.zeros((12,2,2,2))
    sense=np.zeros((12,2,3))   # fine standard and primary at lower/upper thresholds
    rate=np.zeros_like(effects)
    count=np.zeros((12,2))
    for panel in range(12):
        with np.load(ROOT/"results"/"worlds"/f"panel_{panel:02d}.npz") as f:
            th=f["theta"].reshape(-1,6)
            signs=np.repeat(f["signs"],128); regs=np.repeat(f["regimes"],128)
        for s,sh in enumerate(w.SHAPES):
            for v,H in itertools.product(w.VARIANTS,w.HEIGHTS):
                obs={};check={};alts={}
                for sf,law in itertools.product(w.SURFACES,w.LAWS):
                    rel=Path(f"panel_{panel:02d}")/f"{sh}_{sf}_{law}_{v}_h{H}.npz"
                    with np.load(ROOT/"results"/rel) as p,np.load(ROOT/"labels45"/rel) as a,np.load(ROOT/"labels45_pressure_fine"/rel) as b:
                        ps=p["pred"][:2,0]
                        obs[sf,law]=(ps[1]-ps[0]).mean(-1).ravel()
                        check[sf,law]=a["labels"][1:]
                        alts[sf,law]=(b["labels"][1:],a["slip_speed"][1:]<=.01,a["slip_speed"][1:]<=.04)
                for d,off in enumerate(w.OFFSETS):
                    center=H+off
                    g0=th[:,2]+th[:,3]*center/25.+w.slope(center,sh,v)
                    vt0=(w.INCOMING+g0)/np.hypot(1.,g0)
                    outputs={}
                    for a,r in enumerate(w.RADII[1:]):
                        h,add,weights=w.slope_field(center,r,sh,v,12,6)
                        g=th[:,2,None]+th[:,3,None]*h/25.+add
                        for gf in range(2):
                            local_g=g if gf else np.broadcast_to(g0[:,None],g.shape)
                            driving,resist,_,vt=w.impulse_geometry(local_g,g0,th[:,5],weights)
                            for sf,law in itertools.product(w.SURFACES,w.LAWS):
                                mu_const=w.material(th[:,4],vt0[:,None],np.array([center]),sf,law,signs,regs)
                                mu_var=w.material(th[:,4],vt,h,sf,law,signs,regs)
                                for mf,mu in enumerate((mu_const,mu_var)):
                                    label=w.residual(driving,resist,mu).astype(np.float32)<=w.THRESHOLD
                                    if gf==1 and mf==1:
                                        np.testing.assert_array_equal(label,check[sf,law][a,d].ravel())
                                    outputs[a,gf,mf,sf,law]=label
                    for sf,law in itertools.product(w.SURFACES,w.LAWS):
                        ps=obs[sf,law]
                        for gf,mf in itertools.product(range(2),repeat=2):
                            narrow=outputs[0,gf,mf,sf,law]
                            broad=outputs[1,gf,mf,sf,law]
                            effects[panel,s,gf,mf]+=(-2*ps*(broad.astype(float)-narrow)).mean()
                            rate[panel,s,gf,mf]+=(broad!=narrow).mean()
                        for k,arr in enumerate(alts[sf,law]):
                            sense[panel,s,k]+=(-2*ps*(arr[1,d].ravel().astype(float)-arr[0,d].ravel())).mean()
                        count[panel,s]+=1
        print("finished",panel,flush=True)
    effects/=count[:,:,None,None];rate/=count[:,:,None,None];sense/=count[:,:,None]
    interaction=effects[:,:,1,1]-effects[:,:,1,0]-effects[:,:,0,1]+effects[:,:,0,0]
    out=dict(axes="shape fold/ripple, normals homogeneous/local, material homogeneous/local, mean and family-adjusted CI",
             broad_minus_narrow_D=ci(effects,8),interaction=ci(interaction,2),event_flip=ci(rate,8),
             sensitivity_order=["24x12 .020","12x6 .010","12x6 .040"],sensitivity=ci(sense,6))
    (ROOT/"pressure_mechanism45.json").write_text(json.dumps(out,indent=2,default=lambda x:x.tolist())+"\n")
    print(np.asarray(out["broad_minus_narrow_D"])[...,0], np.asarray(out["sensitivity"])[...,0])
if __name__=="__main__":run()
