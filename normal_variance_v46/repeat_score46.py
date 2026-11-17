"""Held-out repeated-camera score on the same first32 draws, with frozen physics probabilities."""
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parent
def main():
    out=np.zeros((12,2,2,3))  # family, shot, (x-original, joint-original, joint-x)
    for p,s,v,h in itertools.product(range(12),("terrace","dimple"),("base","hollow","bulge"),(19,31)):
        sub=f"panel_{p:02d}"; stem=f"{s}_{v}_h{h}.npz"; family=int(s=="dimple")
        with np.load(ROOT/"repeat_camera46"/sub/stem) as a:
            second=a["gates"].astype(float)
        with np.load(ROOT/"route46"/sub/stem) as a:
            first=np.stack((a["position"][:,:,:32],a["joint"][:,:,:32]),axis=-1).astype(float)
        d=np.zeros(first.shape[:-1])
        for sf,law in itertools.product(("inlay","woven"),("rational","gaussian_rate")):
            key=f"{s}_{sf}_{law}_{v}_h{h}.npz"
            with np.load(ROOT/"results"/sub/key) as a,np.load(ROOT/"labels46"/sub/key) as b:
                pred=a["pred"]
                if pred.ndim==5: pred=pred[:,0]
                o,g=pred[:2,:,:32]
                yy=b["labels"][1:,:,:,:32].mean(0)
                d+=((g*g-o*o).mean(-1)-2*(g-o).mean(-1)*yy)*.25
        for j,actions in enumerate((first,second)):
            coord,joint=(d[...,None]*actions).mean((0,1,2))
            out[p,family,j]+=(np.array([coord,joint,joint-coord])/6)
    def ci(a):
        m=a.mean(0);w=t.ppf(.975,11)*a.std(0,ddof=1)/np.sqrt(12)
        return np.stack((m,m-w,m+w),-1).tolist()
    (ROOT/"repeat_score46.json").write_text(json.dumps(dict(
        axis="shoot 1/2, x-only minus original/joint minus original/joint minus x-only, mean/lo/hi",
        pooled=ci(out.mean(1)),by_family=ci(out),shot_delta=ci(out[:,:,1,:].mean(1)-out[:,:,0,:].mean(1))),indent=2)+"\n")
if __name__=="__main__":main()
