"""Check that operating GPU image routes match all archived evaluation observations."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from camera46 import render_clip,infer_x,infer_normal_variance,source_gate,gate46,load_lookup
import os

root=Path(__file__).resolve().parent
assert os.getenv("SLURM_JOB_ID") and torch.cuda.device_count()==1
torch.set_num_threads(1)
gpu=load_lookup("cuda");cpu=load_lookup("cpu");count=0;dmax=np.zeros(2);flips=np.zeros(3,int)
def routes(x,v,lookup):
    return torch.stack((source_gate(x),gate46(x,None,lookup),gate46(x,v,lookup)))
for panel in range(12):
    s=f"panel_{panel:02d}"
    with np.load(root/"results"/"worlds"/(s+".npz")) as raw:
        theta=raw["theta"].reshape(-1,6)
    for shape,var,h in itertools.product(("terrace","dimple"),("base","hollow","bulge"),(19,31)):
        fpath=root/"camera46"/s/f"{shape}_{var}_h{h}.npz"
        with np.load(fpath) as a:cx=a["x_hat_mm"].reshape(3,512);cv=a["variance_hat"].reshape(3,512);sha=a["raster_sha256"]
        for d,off in enumerate((-.5,0,.5)):
            clips=torch.cat([render_clip(theta[k:k+128],shape,var,h,off,46032026+panel*128+k//128) for k in range(0,512,128)])
            assert hashlib.sha256(clips.numpy().tobytes()).hexdigest()==sha[d]
            clips=clips.to("cuda")
            x=infer_x(clips)
            v=infer_normal_variance(clips,x)
            actual=routes(x,v,gpu).cpu().numpy()
            ref=routes(torch.from_numpy(cx[d]),torch.from_numpy(cv[d]),cpu).numpy()
            flips+=(actual!=ref).sum(axis=1)
            dmax=np.maximum(dmax,[abs(cx[d]-x.cpu().numpy()).max(),abs(cv[d]-v.cpu().numpy()).max()])
            count+=len(x)
    print("panel",panel,flush=True)
out=dict(checked=count,methods=["legacy_x","refit_x","x_normal"],mismatches=flips.tolist(),
         max_x_error_mm=dmax[0],max_normal_variance_error_rad2=dmax[1])
(root/"route_replay46.json").write_text(json.dumps(out,indent=2)+"\n")
print(out)
