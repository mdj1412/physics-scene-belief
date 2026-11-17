"""CUDA evaluation of nested paid prefixes on whole-product holdouts."""
import csv
import hashlib
import json
import math
import os
import platform
import sys

import numpy as np
import torch
from common import HERE,V70,GROUPS,BUDGETS,DRAWS,load_state,draw_indices
from adapt import adapt,features

sys.path.insert(0,str(V70/"solver"))
from benchmark import empirical

CANDIDATES=(.25,1.,4.,16.)

def t(x):
    return torch.as_tensor(x,device="cuda",dtype=torch.float64)

def inner_tune(outer,inner_data,inner_fits):
    risk=np.zeros((len(GROUPS)-1,len(BUDGETS)-1,len(CANDIDATES)))
    for j,name in enumerate(g for g in GROUPS if g!=outer):
        x,y=inner_data[name]
        ep=inner_fits[tuple(sorted((outer,name)))]
        n=len(x)
        pool=np.flatnonzero(np.arange(n)%5!=0)
        ev=np.flatnonzero(np.arange(n)%5==0)
        gen=np.random.default_rng(np.random.SeedSequence([71200,min(GROUPS[outer]),min(GROUPS[name])]))
        idx=np.stack([gen.choice(pool,size=32,replace=False) for _ in range(32)])
        for bidx,k in enumerate(BUDGETS[1:]):
            for h,lam in enumerate(CANDIDATES):
                yp,_=adapt(ep,x[idx[:,:k]],y[idx[:,:k]],x[ev],lam)
                risk[j,bidx,h]=float((yp["gate_expert"]-y[ev]).square().mean())
    aggregate=risk.mean(axis=(0,1))
    return CANDIDATES[int(np.argmin(aggregate))],aggregate.tolist(),risk.tolist()

def binomial_cut(k):
    for d in range(k+1):
        if sum(math.comb(k,i) for i in range(d,k+1))/2**k <= .05:
            return d
    return k+1

def main():
    frozen_file=V70/"solver"/"metrics.json"
    fits=json.loads(frozen_file.read_text())["folds"]
    train={g:tuple(t(a) for a in load_state(c,"train")) for g,c in GROUPS.items()}
    raw_draws={g:draw_indices(len(train[g][0]),c) for g,c in GROUPS.items()}
    np.savez_compressed(HERE/"draw_indices.npz",**raw_draws)
    inner_fits={}
    for a in GROUPS:
        for b in GROUPS:
            key=tuple(sorted((a,b)))
            if a==b or key in inner_fits:continue
            x=torch.cat([train[g][0] for g in GROUPS if g not in key])
            y=torch.cat([train[g][1] for g in GROUPS if g not in key])
            _,inner_fits[key]=empirical(x,y,x[:2])
    # Fitting applied to source/train data alone; serialize before scoring any outer test.
    fit_report={"host":platform.node(),"gpu":torch.cuda.get_device_name(),"torch":torch.__version__,
                "job":os.environ.get("SLURM_JOB_ID"),"v70_coefficients_sha256":hashlib.sha256(frozen_file.read_bytes()).hexdigest(),
                "candidate_penalties":CANDIDATES,"folds":{}}
    for g in GROUPS:
        lam,mean,grid=inner_tune(g,train,inner_fits)
        fit_report["folds"][g]={"penalty":lam,"mean_inner_risk":mean,"inner_risk_by_product_budget":grid}
    (HERE/"fit.json").write_text(json.dumps(fit_report,indent=2)+"\n")
    with (HERE/"draw_metrics.csv").open("w") as mf,(HERE/"detection.csv").open("w") as df,(HERE/"draw0_predictions.csv").open("w") as pf,(HERE/"shot_mean_errors.csv").open("w") as sf:
        m=csv.writer(mf);d=csv.writer(df);p=csv.writer(pf);s=csv.writer(sf)
        m.writerow(["rubber","draw","budget","method","velocity","spin"])
        d.writerow(["rubber","draw","budget","mismatch","flip","certified","held_sign_accuracy"])
        p.writerow(["rubber","budget","method","config","row","vx_pred","vz_pred","wy_pred"])
        s.writerow(["rubber","budget","method","config","row","mean_velocity","mean_spin"])
        for g,configs in GROUPS.items():
            ep=fits[g]["empirical"]; lam=fit_report["folds"][g]["penalty"]
            x,y=train[g]
            xt,yt=map(t,load_state(configs,"test"))
            ids=[(cfg,row) for cfg in sorted(configs) for row in range(len(load_state((cfg,),"test")[0]))]
            source_route=features(xt,ep)@t(ep["logistic"])>0
            test_truth=yt[:,0]-yt[:,2]>0
            idx=raw_draws[g]
            for k in BUDGETS:
                yp,info=adapt(ep,x[idx[:,:k]],y[idx[:,:k]],xt,lam)
                flips=info["flip"]
                ac=((flips[:,None]^source_route)==test_truth).double().mean(1)
                det=torch.column_stack([info["mismatch"],flips,info["mismatch"]>=binomial_cut(k),ac]).cpu().numpy()
                for j,(v,flip,cert,acc) in enumerate(det):d.writerow([g,j,k,int(v),int(flip),int(cert),acc])
                for name,pred in yp.items():
                    diff=pred-yt
                    v=torch.linalg.vector_norm(diff[:,:,:2],dim=-1)
                    w=diff[:,:,2].abs()/.02
                    errs=torch.stack([v.mean(1),w.mean(1)],dim=-1).cpu().numpy()
                    for j,(ve,we) in enumerate(errs):m.writerow([g,j,k,name,ve,we])
                    pp=pred[0].cpu().numpy()
                    shot=torch.column_stack([v.mean(0),w.mean(0)]).cpu().numpy()
                    for i,((vr,vz,rim),(ve,we)) in enumerate(zip(pp,shot)):
                        p.writerow([g,k,name,*ids[i],vr,vz,rim/.02]);s.writerow([g,k,name,*ids[i],ve,we])
                if k==0:
                    ref=fits[g]["metrics"]["two_regime"]["test"]
                    assert abs(float(v.mean())-ref["velocity_mae"])<1.e-9
                    assert abs(float(w.mean())-ref["spin_mae"])<1.e-8

if __name__=="__main__":main()
