"""Secondary experiment, other-site correlation sensitivity with frozen source-trained routers."""
import json
import os
from itertools import product
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t

import route48
import sensor48
from world48 import FAMILIES, HEIGHTS, STATES, STATE_WEIGHTS

ROOT=Path(__file__).resolve().parent
TRANSFERS=(0.,.30,.60,.85,1.)
METHODS=("query_local","frozen_fixed","fixed_assume_perfect_transfer")


def panel(i,policy):
    name=f"panel_{i:02d}.npz"
    with np.load(ROOT/"worlds"/name) as w, np.load(ROOT/"forecast"/name) as f, np.load(ROOT/"labels"/name) as l:
        seed=int(w["seed"]); y=l["labels"]; q=f["query_paid"]
        orig=f["p_original"]; proj=f["p_projected"]
        local, deployed=f["gate_preload"], f["gate_fixed"]
    fixed=policy["fixed_height_mm"]
    losses=np.zeros((len(TRANSFERS),len(METHODS),2))
    mismatches=0
    for fi,hi,si in product(range(4),range(2),range(3)):
        family=FAMILIES[fi];h=HEIGHTS[hi];s=STATES[si]
        observed=q[fi,:,hi] # alignment,variant,environment,object,noise,channel
        yy=y[fi,hi,si].mean((0,1))[...,None]
        p0=orig[fi,:,hi];p1=proj[fi,:,hi]
        d=(p1-p0)*(p0+p1-2*yy)
        risk=(p0*p0+yy*(1-2*p0)).mean()
        rate=STATE_WEIGHTS[si]/4
        for j,v in enumerate(TRANSFERS):
            rd=sensor48.readings(dict(seed=seed),family,h,s,fixed,transfer=v)
            selected=route48.decision(observed,rd,h,"fixed",policy).numpy()
            informed=route48.decision(observed,rd,h,"preload",policy).numpy()
            if v==.60:
                mismatches+=int(np.sum(selected!=deployed[fi,:,hi,:,si]))
            for m,g in enumerate((local[fi,:,hi,:,si],selected,informed)):
                losses[j,m,hi]+=rate*(risk+(d*g).mean())
    return losses, mismatches


def interval(v):
    m=v.mean(0); e=v.std(0,ddof=1)*t.ppf(.975,len(v)-1)/np.sqrt(len(v))
    return np.stack((m,m-e,m+e),axis=-1)


if __name__=="__main__":
    assert os.getenv("SLURM_JOB_ID") and (ROOT/"TARGET_GO").is_file()
    torch.set_num_threads(1)
    policy=route48.load_policy("cpu")
    outcomes=[panel(i,policy) for i in range(12)]
    loss=np.stack([x[0] for x in outcomes])
    errors=sum(x[1] for x in outcomes)
    assert not errors
    mean=loss.mean(-1)
    effect=mean[:,:,0,None]-mean[:,:,1:]
    d=dict(transfer_probability=TRANSFERS,methods=METHODS,heights_mm=HEIGHTS,average=mean.mean(0),
           by_height=loss.mean(0),local_minus_fixed_pointwise_95=interval(effect),production_replay_errors=errors,
           scope="secondary. Only other-site dependence is varied. At perfect transfer, site-specific sensor noise, gain and shape response stay intact.")
    (ROOT/"transfer48.json").write_text(json.dumps(d,default=lambda x:np.asarray(x).tolist(),indent=1)+"\n")
    print("transfer p, fixed minus local, by mode:",-effect.mean(0))
