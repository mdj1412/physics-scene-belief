"""Secondary control: feed the same measured scalar to a calibrated original decoder."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from score48 import ROOT, WEIGHT, interval
from sensor48 import HEIGHTS, STATES, readings

SOURCE=ROOT.parent/"factorial_projection_v47"
MODES=("no_readout","local","remote")
FIXED=int(json.loads((ROOT/"policy48.json").read_text())["fixed_height_mm"])


def design(prob, signal):
    # Fixed units, hinges and regularization before target scoring.
    z=np.clip(signal/.084,-2.,2.)
    x=2*(prob-.5)
    return np.stack(np.broadcast_arrays(np.ones_like(prob),x,x*x,z,abs(z),z*x,abs(z)*x),axis=-1)


def source_fit():
    gram=np.zeros((len(MODES),2,2,7,7))
    rhs=np.zeros(gram.shape[:-1])
    for k in range(12):
        tag=f"panel_{k:02d}.npz"
        with np.load(SOURCE/"forecast"/tag) as a, np.load(SOURCE/"labels"/tag) as b, np.load(SOURCE/"worlds"/tag) as w:
            o=a["p_original"]; qs=np.isfinite(a["query_paid"][...,0]); families=a["families"]; y=b["labels"]
            pop=dict(seed=int(w["seed"]))
        for f,family in enumerate(families):
            for h,height in enumerate(HEIGHTS):
                po=o[f,:,h] # alignment, optical variant, environment, object, noise
                mask=qs[f,:,h]
                for j,state in enumerate(STATES):
                    yy=y[f,h,int(state==1),int(state!=0)].mean(axis=(0,1))[...,None]
                    err=(yy-po)
                    for mode,station in enumerate((None,height,50-height)):
                        sig=0. if station is None else readings(pop,str(family),height,state,station)
                        features=design(po,sig)
                        for arm in (0,1):
                            keep=mask==arm
                            xx=features[keep]
                            gram[mode,h,arm]+=WEIGHT[j]*(xx.T@xx)
                            rhs[mode,h,arm]+=WEIGHT[j]*(xx.T@err[keep])
    coef=np.zeros_like(rhs)
    for mode,h,arm in np.ndindex(3,2,2):
        g=gram[mode,h,arm]
        penalty=np.eye(7)*.001*g[0,0]
        penalty[0,0]=1e-9
        coef[mode,h,arm]=np.linalg.solve(g+penalty,rhs[mode,h,arm])
    (ROOT/"direct_source48.json").write_text(json.dumps(dict(coefficients=coef.tolist(),ridge=.001,source="v47 only",fixed_site=FIXED))+"\n")


def evaluate():
    coef=np.array(json.loads((ROOT/"direct_source48.json").read_text())["coefficients"])
    # Direct no-readout, direct fixed, direct preload, gated preload, original.
    sums=np.zeros((12,5,3))
    for k in range(12):
        tag=f"panel_{k:02d}.npz"
        with np.load(ROOT/"forecast"/tag) as a,np.load(ROOT/"labels"/tag) as b:
            oo=a["p_original"]; pp=a["p_projected"]; qs=np.isfinite(a["query_paid"][...,0])
            gate=a["gate_preload"]; disp=a["readings_mm"]; yy=b["labels"]
        for f,h,j in np.ndindex(4,2,3):
            po=oo[f,:,h]; arm=qs[f,:,h].astype(int)
            y=yy[f,h,j].mean((0,1))[...,None]
            preds=[]
            for mode,z in ((0,None),(1 if HEIGHTS[h]==FIXED else 2,HEIGHTS.index(FIXED)),(1,h)):
                sig=0. if z is None else disp[f,h,j,z]
                adjust=np.sum(design(po,sig)*coef[mode,h][arm],axis=-1)
                preds.append(np.clip(po+adjust,0,1))
            preds.extend([np.where(gate[f,:,h,:,j],pp[f,:,h],po),po])
            for i,pred in enumerate(preds):
                loss=pred*pred+y*(1-2*pred)
                sums[k,i]+=WEIGHT[j]*np.array([loss.mean(),loss[arm==0].mean(),loss[arm==1].mean()])/8
    out=dict(methods=["direct_none","direct_fixed","direct_local","gate_local","original"],
             axes="all; inherited physical-probe arm; inherited query-view arm",loss=sums.mean(0),
             direct_minus_gate=interval(sums[:,2]-sums[:,3],3),
             direct_local_minus_fixed=interval(sums[:,2]-sums[:,1],3))
    (ROOT/"direct48.json").write_text(json.dumps(out,default=lambda x:np.asarray(x).tolist(),indent=1)+"\n")
    print(out["loss"],flush=True)


if __name__=="__main__":
    assert os.getenv("SLURM_JOB_ID")
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["fit","evaluate"])
    if p.parse_args().mode=="fit":source_fit()
    else:evaluate()
