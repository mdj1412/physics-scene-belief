"""Source-selected smaller prior penalties; CPU-only Slurm follow-up."""
import json
import os
import platform
import numpy as np
import torch
from common import HERE,V70,GROUPS,BUDGETS,load_state,draw_indices
from adapt import adapt

def numpy_experts(x,y):
    mu=x.mean(0);sd=x.std(0)
    a=np.column_stack([np.ones(len(x)),(x-mu)/sd])
    side=y[:,0]-y[:,2]>0
    beta=np.zeros(4);ridge=np.diag([0,1,1,1])
    for _ in range(35):
        prob=1/(1+np.exp(-np.clip(a@beta,-35,35)))
        step=np.linalg.solve(a.T@((prob*(1-prob))[:,None]*a)+ridge,a.T@(prob-side)+ridge@beta)
        beta-=step
        if np.linalg.norm(step)<1e-9:break
    return {"mean":mu,"std":sd,"logistic":beta,"expert_minus":np.linalg.lstsq(a[~side],y[~side],rcond=None)[0],
            "expert_plus":np.linalg.lstsq(a[side],y[side],rcond=None)[0]}

def main():
    grid=[.015625,.0625,.25]
    primary=json.loads((HERE/"fit.json").read_text())["folds"]
    ep70=json.loads((V70/"solver"/"metrics.json").read_text())['folds']
    source={g:load_state(configs,"train") for g,configs in GROUPS.items()}
    pars={}
    out={}
    for outer in GROUPS:
        risk=np.zeros(len(grid))
        for inner in GROUPS:
            if inner==outer:continue
            key=tuple(sorted((inner,outer)))
            if key not in pars:
                x=np.concatenate([state[0] for g,state in source.items() if g not in key])
                y=np.concatenate([state[1] for g,state in source.items() if g not in key])
                pars[key]=numpy_experts(x,y)
            x,y=map(torch.from_numpy,source[inner]);n=len(x)
            pool=np.flatnonzero(np.arange(n)%5!=0);val=np.flatnonzero(np.arange(n)%5==0)
            rng=np.random.default_rng(np.random.SeedSequence([71200,min(GROUPS[outer]),min(GROUPS[inner])]))
            choices=np.stack([rng.choice(pool,size=32,replace=False) for _ in range(32)])
            for j,lam in enumerate(grid):
                for k in BUDGETS[1:]:
                    pp,_=adapt(pars[key],x[choices[:,:k]],y[choices[:,:k]],x[val],lam)
                    risk[j]+=float((pp["gate_expert"]-y[val]).square().mean())/30
        assert abs(risk[-1]-primary[outer]["mean_inner_risk"][0])<1.e-10
        lam=grid[int(risk.argmin())]
        x,y=map(torch.from_numpy,source[outer]);tx,ty=map(torch.from_numpy,load_state(GROUPS[outer],"test"))
        idx=draw_indices(len(x),GROUPS[outer])
        scores={}
        for k in BUDGETS[1:]:
            pp,_=adapt(ep70[outer]["empirical"],x[idx[:,:k]],y[idx[:,:k]],tx,lam)
            scores[k]={}
            for name in ["gate_expert","routed_expert_only"]:
                delta=pp[name]-ty
                scores[k][name]=dict(velocity=float(delta[:,:,:2].norm(dim=-1).mean()),spin=float(delta[:,:,2].abs().mean()/.02))
        out[outer]={"penalty":lam,"source_cv":risk.tolist(),"scores":scores}
    (HERE/"sensitivity.json").write_text(json.dumps(dict(grid=grid,folds=out,host=platform.node(),job=os.getenv("SLURM_JOB_ID")),indent=2)+"\n")
if __name__=="__main__":main()
