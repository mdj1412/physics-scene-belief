"""Post-hoc mechanistic intervention: replace only the decoder axis, retain stored routes."""
import itertools
import json
import numpy as np
from score46 import ROOT, SHAPES, VARS, risk, ci

def main():
    # panel, shape, solver, method (same camera decisions wherever used).
    data=np.zeros((12,2,2,7))
    for pa,f,v,h in itertools.product(range(12),range(2),VARS,(19,31)):
        rel=f"panel_{pa:02d}";rt=f"{SHAPES[f]}_{v}_h{h}.npz"
        with np.load(ROOT/"route46"/rel/rt) as aaa:
            joint=aaa["joint"];pos=aaa["position"]
        for sf,law in itertools.product(("inlay","woven"),("rational","gaussian_rate")):
            stem=f"{SHAPES[f]}_{sf}_{law}_{v}_h{h}.npz"
            with np.load(ROOT/"results"/rel/stem) as a,np.load(ROOT/"diag_axis46"/rel/stem) as ad,np.load(ROOT/"labels46"/rel/stem) as l,np.load(ROOT/"labels46_bullet"/rel/stem) as b:
                assert b["valid"].all()
                y=np.stack([l["labels"][1:].mean(0),b["labels"]])
                orig,old=a["pred"][:,0];new=ad["pred"]
                assert new.shape==orig.shape and np.isfinite(new).all()
                o=risk(orig,y);d=risk(old,y)-o;n=risk(new,y)-o
                aa=np.stack([o,o+d*joint,o+n*joint,o+d*pos,o+n*pos,o+d,o+n],-1)
                data[pa,f]+=aa.mean((1,2,3))/24.
    p=data.mean(1)
    changes=np.stack([p[:,:,2]-p[:,:,0],p[:,:,2]-p[:,:,1],p[:,:,2]-p[:,:,4]],-1)
    out=dict(methods=["original","source_axis_joint","target_axis_joint","source_axis_position","target_axis_position","always_source_axis","always_target_axis"],
             engines=["prescribed finite","rigid"],pooled=ci(p),family=ci(data),aligned_joint_contrasts=ci(changes,3),
             family_aligned_joint_vs_original=ci(data[:,:,:,2]-data[:,:,:,0],4))
    (ROOT/"aligned_axis_score46.json").write_text(json.dumps(out,default=lambda x:np.asarray(x).tolist(),indent=1)+"\n")
if __name__=="__main__":main()
