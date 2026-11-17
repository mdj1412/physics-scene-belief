"""Does the targeted deformation change outcome at fixed latent object, initial x and surface?"""
import itertools
import json
import numpy as np
from score46 import ROOT,SHAPES,ci
def main():
    s=np.zeros((12,2,2))
    for pa,i,h,sf,law in itertools.product(range(12),range(2),(19,31),("inlay","woven"),("rational","gaussian_rate")):
        sub=f"panel_{pa:02d}";ys=[]
        for v in ("hollow","bulge"):
            stem=f"{SHAPES[i]}_{sf}_{law}_{v}_h{h}.npz"
            with np.load(ROOT/"labels46"/sub/stem) as f,np.load(ROOT/"labels46_bullet"/sub/stem) as b:
                assert b["valid"].all()
                ys.append((f["labels"][1:],b["labels"]))
        s[pa,i]+=[(a!=b).mean()/8 for a,b in zip(*ys)]
    (ROOT/"paired_variants46.json").write_text(json.dumps({"shapes":SHAPES,"engines":["finite","rigid"],
      "fraction_labels_changed":ci(s,4).tolist()},indent=2)+"\n")
if __name__=="__main__":main()
