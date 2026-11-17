"""Paired comparison to the separately published v45 one-bin position rule."""
import itertools
import json
import numpy as np
from score46 import ROOT, SHAPES, VARS, XGRID, ci, risk

def main():
    src=json.loads((ROOT.parent/"image_pressure_v45"/"matched_source45.json").read_text())
    project=np.array(src["project"])
    assert np.array_equal(XGRID,src["centers_mm"])
    assert np.array_equal(project,[0,0,0,1,0,0])
    score=np.zeros((12,2,2,3))
    for p,f,v,h in itertools.product(range(12),range(2),VARS,(19,31)):
        panel=f"panel_{p:02d}"; pref=f"{SHAPES[f]}_{v}_h{h}.npz"
        with np.load(ROOT/"camera46"/panel/pref) as c,np.load(ROOT/"route46"/panel/pref) as r:
            bins=abs(c["x_hat_mm"][...,None]-XGRID).argmin(-1)
            acts=np.stack([np.zeros_like(r["joint"]),project[bins],r["joint"]],axis=-1)
        for sf,law in itertools.product(("inlay","woven"),("rational","gaussian_rate")):
            stem=f"{SHAPES[f]}_{sf}_{law}_{v}_h{h}.npz"
            with np.load(ROOT/"results"/panel/stem) as a,np.load(ROOT/"labels46"/panel/stem) as y,np.load(ROOT/"labels46_bullet"/panel/stem) as b:
                assert b["valid"].all()
                lab=np.stack([y["labels"][1:].mean(0),b["labels"]])
                o,g=a["pred"][:,0]
                base=risk(o,lab);diff=risk(g,lab)-base
                score[p,f]+=(base[...,None]+diff[...,None]*acts).mean(axis=(1,2,3))/24
    x=score.mean(1)
    out=dict(engines=["finite","Bullet"],methods=["always-original","v45-matched-one-bin","joint"],
             absolute=ci(x),joint_minus_conservative=ci(x[:,:,2]-x[:,:,1],2),
             per_family=ci(score[:,:,:,2]-score[:,:,:,1],4))
    (ROOT/"conservative46.json").write_text(json.dumps(out,default=lambda z:np.asarray(z).tolist(),indent=1)+"\n")
if __name__=="__main__":main()
