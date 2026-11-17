"""Apply the frozen visual and true-location routing to a separate rigid Bullet impact."""
from pathlib import Path
import itertools
import numpy as np
from score45 import ROOT, SHAPES, TRANSFER, gate, risk, interval, write_results
import json

def main():
    gates = np.stack([TRANSFER,json.loads((ROOT/"matched_source45.json").read_text())["project"]])
    # gates x panels x solver x methods
    out=np.zeros((2,12,3,5))
    agreement=np.zeros((12,3))
    errors=[]
    for pa,i,h,var,sf,law in itertools.product(range(12),range(2),(19,31),("base","hollow","bulge"),
                                             ("inlay","woven"),("rational","gaussian_rate")):
        sub=f"panel_{pa:02d}"; stem=f"{SHAPES[i]}_{sf}_{law}_{var}_h{h}.npz"
        with np.load(ROOT/"camera45"/sub/f"{SHAPES[i]}_{var}_h{h}.npz") as a:
            x=a["x_hat_mm"][:,:,:8]
            target=a["oracle_x_mm"][:,:,:8]
        with np.load(ROOT/"results"/sub/stem) as p, np.load(ROOT/"labels45"/sub/stem) as f, np.load(ROOT/"labels45_bullet"/sub/stem) as b:
            assert b["valid"].all()
            first=b["first_contact_x_mm"]
            errors.append(abs(first-target).ravel())
            ys=np.stack([b["labels"], f["labels"][0,:,:,:8], f["labels"][1:,:,:,:8].mean(0)])
            pred=p["pred"][:2,0,:,:8]; base=risk(pred[0],ys); change=risk(pred[1],ys)-base
            agreement[pa]+=np.array([(b["labels"]!=f["labels"][0,:,:,:8]).mean(),
               (gate(first,TRANSFER)!=gate(target,TRANSFER)).mean(),b["valid"].mean()])/48
            for n,sw in enumerate(gates):
                # Measured oracle only in Bullet. For point and finite mean oracle is exact requested x.
                acts=np.stack([np.zeros_like(x,bool),np.full(x.shape,gate(np.array(h),sw)),
                         gate(target,sw),gate(first,sw),gate(x,sw)],-1)
                mix=base[...,None]+change[...,None]*acts
                mix[1:,...,3]=mix[1:,...,2]
                out[n,pa]+=mix.mean((1,2,3))/48
    e=np.concatenate(errors)
    results={"methods":["original","nominal","requested_oracle","measured_oracle","visual"],
             "solvers":["Bullet","point","finite_mean"],"n_bullet":e.size,"first_error_mean_p90_max":[e.mean(),*np.quantile(e,[.90,1.])],
             "agreement":interval(agreement),"routes":[]}
    for z in out:
        diff=np.stack([z[...,4]-z[...,k] for k in (0,3,1)],-1)
        results["routes"].append(dict(absolute=interval(z),
                  visual_comparisons=interval(diff,3),
                  bullet_minus_point_visual_net=interval(diff[:,0,0]-diff[:,1,0])))
    write_results(ROOT/"external_score45.json",results)

if __name__=="__main__":
    main()
