"""Source-only fit and validation, exclusively using frozen v47 catalogs and simulated source readings."""
import json
import os
from pathlib import Path

import numpy as np
import torch

import route48 as model
import sensor48

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent/"factorial_projection_v47"
STATES = (0,-1,1)
WEIGHTS = (.5,.25,.25)
RIDGE = .001
DIM = 36


def initialize():
    axis = json.loads((ROOT.parent/"image_pressure_v45/source_axis45.json").read_text())
    axes = np.array([axis[str(h)]["axis"] for h in model.H])
    x = np.arange(-3.,4.)
    design = np.column_stack((np.ones(7),x/3,(x/3)**2))
    stat = np.linalg.pinv(design).T
    residual = np.eye(7)-design@stat.T
    residual_axes=axes@residual
    residual_axes/=np.linalg.norm(residual_axes,axis=1)[:,None]
    return dict(residual=residual.tolist(),axes=axes.tolist(),residual_axes=residual_axes.tolist(),profile_map=stat.tolist())


def rows(panel,artifact):
    name=f"panel_{panel:02d}.npz"
    with np.load(SRC/"forecast"/name) as a, np.load(SRC/"labels"/name) as l, np.load(SRC/"worlds"/name) as w:
        queries=a["query_paid"];p=a["p_original"];g=a["p_projected"];ys=l["labels"]
        families = a["families"].tolist()
        pop=dict(seed=int(w["seed"]))
    for f,family in enumerate(families):
        for hi,height in enumerate(model.H):
            material = {}
            for state in STATES:
                vi,ci = int(state==1),int(state!=0)
                y=ys[f,hi,vi,ci].mean(axis=(0,1))[...,None]
                read=[sensor48.readings(pop,family,height,state,s) for s in (height,50-height)]
                material[state] = (y,read)
            for alignment in range(2):
                for variant in range(2):
                    config=(f,alignment,hi,variant)
                    features, eligible = model.profile_features(queries[config],height,artifact)
                    mask = eligible.numpy()
                    b = features.numpy()[mask]
                    dp = g[config]-p[config]
                    psum = g[config]+p[config]
                    for state_idx,state in enumerate(STATES):
                        y,read=material[state]
                        target = (dp*(psum-2*y))[mask]
                        base = (p[config]**2 + y*(1.-2*p[config])).mean() # retains probe-arm trials
                        loc,rem = (x[mask] for x in read)
                        idx = (np.searchsorted(model.SIGNED_CUTS,loc),np.searchsorted(model.SIGNED_CUTS,rem),
                               np.searchsorted(model.UNSIGNED_CUTS,abs(loc)),np.zeros(len(b),int))
                        yield hi,b,target,idx,WEIGHTS[state_idx],base,mask.size


def stats():
    return [[[(np.zeros((DIM,DIM)),np.zeros(DIM)) for _ in range(nb)] for _ in model.H] for nb in model.BINS]


def fit(panel_ids,a):
    acc=stats()
    for panel in panel_ids:
        for h,b,target,ids,weight,_,_ in rows(panel,a):
            for k,bins in enumerate(ids):
                for j in range(model.BINS[k]):
                    m = bins==j
                    if not m.any(): continue
                    b1=b[m]; gram,rhs=acc[k][h][j]
                    gram+=weight*(b1.T@b1)
                    rhs+=weight*(b1.T@target[m])
    coef={}
    for k,mode in enumerate(model.MODES):
        cc=np.zeros((2,model.BINS[k],DIM))
        for h in range(2):
            for j,(gram,rhs) in enumerate(acc[k][h]):
                # Penalize feature directions; keep the bin intercept unpenalized.
                reg=np.eye(DIM)*max(1.,RIDGE*gram[0,0]);reg[0,0]=1e-10
                cc[h,j]=np.linalg.solve(gram+reg,rhs)
        coef[mode]=cc
    return coef


def validate(panel_ids,a,coef):
    metrics=[]
    for panel in panel_ids:
        record=np.zeros((6,2))
        for hi,b,dy,idx,wt,base,n in rows(panel,a):
            decisions=[]
            for k,mode in enumerate(model.MODES):
                scores=b@coef[mode][hi].T
                decisions.append(scores[np.arange(len(b)),idx[k]]<0)
            gates=(np.zeros(len(b),bool),decisions[0],decisions[0] if hi==0 else decisions[1],
                   decisions[1] if hi==0 else decisions[0],decisions[2],decisions[3])
            for ii,g in enumerate(gates):
                record[ii]+=wt*np.array([base+(dy*g).sum()/n,g.sum()/n])
        metrics.append((record/(4*2*2*2)).tolist())
    return metrics


if __name__=="__main__":
    assert os.getenv("SLURM_JOB_ID")
    torch.set_num_threads(1)
    r=initialize();a=model.setup(r)
    train=fit(range(8),a); val=validate(range(8,12),a,train)
    ave=np.asarray(val).mean(0)[:,0]
    fixed_height = 19 if ave[2]<=ave[3] else 31
    print("validation [original, local, fixed19, fixed31, unsigned, none]",ave,"fixed",fixed_height,flush=True)
    final=fit(range(12),a)
    r.update(coefficients={k:x.tolist() for k,x in final.items()},fixed_height_mm=fixed_height,
             score="weighted ridge estimated projected-minus-original Brier",ridge=RIDGE,
             modes=model.MODES,state_weights=WEIGHTS,states=STATES,
             signed_cutpoints_mm=model.SIGNED_CUTS,unsigned_cutpoints_mm=model.UNSIGNED_CUTS,
             source_catalogs=list(range(700,712)),global_site_fit_catalogs=list(range(700,708)),
             global_site_validation_catalogs=list(range(708,712)),validation_metrics=val,
             validation_methods=["original","local","fixed19","fixed31","unsigned","none"],
             validation_fields=["absolute_brier","projection_rate"],no_target_outcomes=True)
    tmp=ROOT/"policy48.tmp"
    tmp.write_text(json.dumps(r,allow_nan=False,indent=1)+"\n")
    tmp.replace(ROOT/"policy48.json")
    print("frozen",flush=True)
