"""Apply the unchanged router to archived features; no target outcomes are required."""
from pathlib import Path
import itertools
import os
import numpy as np
import torch
from camera46 import ROOT, gate46, load_lookup, source_gate

def main():
    visual=load_lookup(); oracle=load_lookup(kind="oracle")
    for p,s,v,h in itertools.product(range(12),("terrace","dimple"),("base","hollow","bulge"),(19,31)):
        rel=Path(f"panel_{p:02d}")/f"{s}_{v}_h{h}.npz"
        dest=ROOT/"route46"/rel
        with np.load(ROOT/"camera46"/rel) as a:
            x,vv,tx,tv=[torch.from_numpy(a[k]) for k in ("x_hat_mm","variance_hat","true_x_mm","true_variance")]
        routes=dict(v45=source_gate(x),position=gate46(x,None,visual),joint=gate46(x,vv,visual),
                    oracle_joint=gate46(tx,tv,oracle),oracle_position=gate46(tx,None,oracle))
        dest.parent.mkdir(parents=True,exist_ok=True)
        tmp=dest.with_suffix(".tmp.npz")
        np.savez_compressed(tmp,**{k:a.numpy() for k,a in routes.items()})
        os.replace(tmp,dest)

if __name__=='__main__':
    torch.set_num_threads(1)
    main()
