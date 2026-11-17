"""Separate a sample-composition change from an engine change on identical draws."""
from pathlib import Path
import itertools
import numpy as np
from score45 import ROOT, SHAPES, TRANSFER, gate, risk, interval, write_results
import json

def main():
    gates = np.stack([TRANSFER,json.loads((ROOT/"matched_source45.json").read_text())["project"]])
    # panel, source gate, slice of object draws, engine, policy
    vals = np.zeros((12,2,3,3,4))
    n=0
    dist=[]
    disagree=np.zeros(12)
    shape_effect=np.zeros((12,2))
    route_diff=0
    match_error=0.
    valid_total=0
    for pa,i,h,var,sf,law in itertools.product(range(12),range(2),(19,31),("base","hollow","bulge"),
                                             ("inlay","woven"),("rational","gaussian_rate")):
        sub=f"panel_{pa:02d}"; stem=f"{SHAPES[i]}_{sf}_{law}_{var}_h{h}.npz"
        with np.load(ROOT/"camera45"/sub/f"{SHAPES[i]}_{var}_h{h}.npz") as a:
            x=a["x_hat_mm"]
            target=a["oracle_x_mm"]
        with np.load(ROOT/"results"/sub/stem) as p, np.load(ROOT/"labels45"/sub/stem) as f, np.load(ROOT/"labels45_bullet_full"/sub/stem) as b, np.load(ROOT/"labels45_bullet"/sub/stem) as small:
            assert b["valid"].shape==(3,4,128) and b["valid"].all()
            np.testing.assert_array_equal(small["labels"],b["labels"][:,:,:8])
            match_error=max(match_error,np.max(abs(small["first_contact_x_mm"]-b["first_contact_x_mm"][:,:,:8])))
            first=b["first_contact_x_mm"]
            dist.append(abs(first-target).ravel())
            route_diff+=np.sum(gate(first,TRANSFER)!=gate(target,TRANSFER))
            valid_total+=int(b["valid"].sum())
            y=np.stack([b["labels"],f["labels"][0],f["labels"][1:].mean(0)])
            p0,p1=p["pred"][:2,0]
            base=risk(p0,y); delta=risk(p1,y)-base
            disagree[pa]+=(b["labels"]!=f["labels"][0]).mean()/48
            for gi,sw in enumerate(gates):
                acts=np.stack([np.zeros_like(x,bool),np.full(x.shape,gate(np.array(h),sw)),
                             gate(first,sw),gate(x,sw)],-1)
                mix=base[...,None]+delta[...,None]*acts
                mix[1:,...,2]=base[1:]+delta[1:]*gate(target,sw)
                for sl_idx,sl in enumerate((slice(None),slice(None,8),slice(8,None))):
                    vals[pa,gi,sl_idx]+=mix[:,:,:,sl,:].mean(axis=(1,2,3))/48
                if gi==0:
                    shape_effect[pa,i]+=(mix[0,...,3]-mix[0,...,0]).mean()/24
    out=dict(engines=["Bullet","prescribed point","prescribed finite mean"], methods=["original","nominal","oracle","visual"],
             sample=["full 128/env","first 8/env","held-back 120/env"],summary=[],
             primary_source_bullet_visual_net_by_shape=interval(shape_effect,2),
             matched_first8_max_position_error_mm=match_error,n_valid=valid_total,
             rigid_vs_point_disagreement=interval(disagree),true_x_changed_gate=int(route_diff),
             contact_abs_error_mean_p90_max=np.r_[np.concatenate(dist).mean(),np.quantile(np.concatenate(dist),[.90,1])])
    for score in np.moveaxis(vals,1,0):
        pairs=np.stack([score[...,3]-score[...,k] for k in (0,2,1)],-1)
        out["summary"].append(dict(absolute=interval(score),visual_vs_original_oracle_nominal=interval(pairs,3),
                 engine_delta_visual_net=interval(pairs[:,:,0,0]-pairs[:,:,1,0]),
                 first_vs_rest_rigid_visual_net=interval(pairs[:,1,0,0]-pairs[:,2,0,0])))
    write_results(ROOT/"external_full_score45.json",out)

if __name__=="__main__":
    main()
