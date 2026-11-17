"""Post-hoc replicated camera capture on a fixed subset. Selection uses the frozen LUT."""
from pathlib import Path
import argparse
import itertools
import json
import numpy as np
import torch
from scipy.stats import t,spearmanr
from camera46 import ROOT, render_clip, infer_x, infer_normal_variance, gate46, load_lookup

def run(p):
    table=load_lookup()
    sub=f"panel_{p:02d}"
    with np.load(ROOT/"results"/"worlds"/f"{sub}.npz") as a:theta=a["theta"][:,:32]
    dest=ROOT/"repeat_camera46"/sub
    dest.mkdir(exist_ok=True,parents=True)
    for s,v,h in itertools.product(("terrace","dimple"),("base","hollow","bulge"),(19,31)):
        xx=[]; vv=[]; gates=[]
        for off in (-.5,0.,.5):
            frame=render_clip(theta.reshape(-1,6),s,v,h,off,46032026+91_000_000+p)
            x=infer_x(frame); z=infer_normal_variance(frame,x)
            xx.append(x.numpy());vv.append(z.numpy())
            gates.append(np.stack([gate46(x,None,table).numpy(),gate46(x,z,table).numpy()],-1))
        path=dest/f"{s}_{v}_h{h}.npz"
        np.savez_compressed(path,x=np.asarray(xx).reshape(3,4,32),v=np.asarray(vv).reshape(3,4,32),
                            gates=np.asarray(gates).reshape(3,4,32,2))
    print(p,flush=True)

def audit():
    out=np.zeros((12,2,6)); values=[[],[]]
    for p,s,v,h in itertools.product(range(12),("terrace","dimple"),("base","hollow","bulge"),(19,31)):
        sub=f"panel_{p:02d}";name=f"{s}_{v}_h{h}.npz";i=int(s=="dimple")
        with np.load(ROOT/"repeat_camera46"/sub/name) as a:
            x2=a["x"];v2=a["v"];xv,jv=a["gates"][...,0],a["gates"][...,1]
        with np.load(ROOT/"camera46"/sub/name) as a:
            x=a["x_hat_mm"][:,:,:32];v1=a["variance_hat"][:,:,:32]
        with np.load(ROOT/"route46"/sub/name) as a:
            xp=a["position"][:,:,:32];jp=a["joint"][:,:,:32]
        values[i].append(np.stack((v1,v2)).reshape(2,-1))
        out[p,i]+=np.array([abs(x2-x).mean(),abs(v2-v1).mean(),(xv==xp).mean(),(jv==jp).mean(),
                          (jv!=xv).mean(),(jp!=xp).mean()])/6
    m=out.mean(0);w=t.ppf(.975,11)*out.std(0,ddof=1)/np.sqrt(12)
    result=dict(axes=["terrace,dimple","x repeat difference(mm), variance repeat difference(rad²), x-only agreement, joint agreement, secondary joint-vs-x disagreement, original joint-vs-x disagreement","mean,lo,hi"],
                metrics=np.stack((m,m-w,m+w),-1).tolist(),
                rho=[spearmanr(*np.concatenate(parts,axis=-1)).statistic for parts in values],
                number_new_clips=12*4*32*2*3*2*3,
                portion="First 32 objects/environment, independent synthetic optical nuisances, same physical outcomes.")
    (ROOT/"repeat_camera46.json").write_text(json.dumps(result,indent=2)+"\n")

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--panels",type=int,nargs="+");ap.add_argument("--audit",action="store_true")
    a=ap.parse_args();torch.set_num_threads(1)
    if a.audit:audit()
    else:
        for p in a.panels:run(p)
