"""Frozen, sign-invariant comparator fit only on v47 source outcomes."""
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/"factorial_projection_v47"
sys.path.insert(0,str(ROOT.parent/"preload_projection_v48"))
from sensor48 import readings, STATES, HEIGHTS


def features(p,d):
    x=2*(p-.5)
    z=np.minimum(abs(d)/.084,2.)
    return np.stack(np.broadcast_arrays(np.ones_like(p),x,x*x,z,z*x),axis=-1)


def predict(p,d,coef):
    return np.clip(p+np.sum(features(p,d)*coef,axis=-1),0.,1.)


def main():
    assert os.getenv("SLURM_JOB_ID")
    gram=np.zeros((2,2,5,5))
    rhs=np.zeros((2,2,5))
    for k in range(12):
        tag=f"panel_{k:02d}.npz"
        with np.load(SOURCE/"worlds"/tag) as pop, np.load(SOURCE/"forecast"/tag) as pred,np.load(SOURCE/"labels"/tag) as lab:
            seed=int(pop["seed"]); families=pred["families"]
            ps=pred["p_original"]; arm=np.isfinite(pred["query_paid"][...,0]); y=lab["labels"]
        for f,fam in enumerate(families):
            for h,height in enumerate(HEIGHTS):
                p=ps[f,:,h]; selected=arm[f,:,h]
                for s,w in zip(STATES,(.5,.25,.25)):
                    truth=y[f,h,int(s==1),int(s!=0)].mean((0,1))[...,None]
                    d=readings({"seed":seed},str(fam),height,s,height)
                    b=features(p,d)
                    for a in (0,1):
                        keep=selected==a
                        g=b[keep]
                        gram[h,a]+=w*g.T@g
                        rhs[h,a]+=w*g.T@(truth-p)[keep]
    coef=np.empty_like(rhs)
    for h,a in np.ndindex(2,2):
        g=gram[h,a]
        ridge=np.eye(5)*.001*g[0,0]
        ridge[0,0]=1.e-9
        coef[h,a]=np.linalg.solve(g+ridge,rhs[h,a])
    (ROOT/"unsigned_source50.json").write_text(json.dumps({"coefficients":coef.tolist(),"source":"v47 only","ridge":.001,"basis":["1","x","x^2","|d|","x|d|"]},indent=1))

if __name__=="__main__": main()
