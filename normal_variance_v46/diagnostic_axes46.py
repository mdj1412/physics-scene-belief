"""Privileged, label-free paired-variant axis at the two new nominal heights."""
import json
from pathlib import Path
import numpy as np
from geometry46 import SHAPES, shape_profile

def make_axis(shape,h):
    x=h+np.arange(-3,4)*2.
    signal=shape_profile(x,shape,"bulge")[0]-shape_profile(x,shape,"hollow")[0]
    return signal/np.linalg.norm(signal)

if __name__=="__main__":
    d={s:{str(h):make_axis(s,h).tolist() for h in (19,31)} for s in SHAPES}
    (Path(__file__).resolve().parent/"diagnostic_axes46.json").write_text(json.dumps(d,indent=2)+"\n")
