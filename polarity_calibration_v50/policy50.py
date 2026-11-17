"""Observable-only sign calibration. No target outcomes or environment labels enter."""
import argparse
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"contact_transfer_v49"))
from policy49 import update
import runner48 as legacy
import world50
from unsigned50 import predict as unsigned_predict

METHODS=("original","local_plus","local_minus","fixed_plus","fixed_minus","local_unsigned_fit")
HEIGHTS=(19,31)
STATES=(0,-1,1)


def coefficients():
    f=ROOT.parent/"preload_projection_v48"/"direct_source48.json"
    signed=json.loads(f.read_text())
    assert signed["fixed_site"]==31
    return (np.asarray(signed["coefficients"]),
            np.asarray(json.loads((ROOT/"unsigned_source50.json").read_text())["coefficients"]))


def world(panel):
    with np.load(ROOT/"worlds"/f"panel_{panel:02d}.npz") as arc:
        return {key:arc[key] for key in arc}


def observe(sensors,fam,height,align,variant):
    offset=.001*world50.sensor_delta_mm(fam,height,align,variant)
    return [{legacy.r.Q:row[legacy.r.Q]+offset,legacy.r.P:row[legacy.r.P]} for row in sensors]


def panel(k,engine,templates,coef,uncoef):
    pop=world(k)
    names=tuple(world50.FAMILIES)
    out=np.empty((len(METHODS),len(names),2,3,2,2,4,128,4),np.float64)
    selected=np.empty((len(names),2,2,4,128),np.int8)
    cache={}
    for f,fam in enumerate(names):
        for h,height in enumerate(HEIGHTS):
            readings={station:np.stack([world50.readings(pop,fam,station,s) for s in STATES]) for station in set((31,height))}
            for v,variant in enumerate(legacy.VARIANTS):
                xx,obs=legacy.apparatus(pop,fam,height,variant)
                key=hashlib.sha256(xx.tobytes()).digest()
                if key not in cache:
                    cache[key]=legacy.infer.actions43(xx,engine)
                act=cache[key]
                selected[f,h,v]=act
                arm=(act==legacy.r.Q).astype(np.intp)
                for a,align in enumerate(legacy.ALIGN):
                    paid=legacy.infer.paid_sensors(observe(obs,fam,height,align,variant),act)
                    p=legacy.infer.decode43(engine,templates[h],xx,act,paid)
                    d=readings[height]
                    fixed=readings[31]
                    cl=coef[1,h][arm][:,:,None,:]
                    cf=coef[1 if height==31 else 2,h][arm][:,:,None,:]
                    cu=uncoef[h][arm][:,:,None,:]
                    obj=out[:,f,h,:,a,v]
                    obj[0]=p
                    obj[1]=update(p,d,cl)
                    obj[2]=update(p,-d,cl)
                    obj[3]=update(p,fixed,cf)
                    obj[4]=update(p,-fixed,cf)
                    obj[5]=unsigned_predict(p,d,cu)
                    np.testing.assert_array_equal(obj[5],unsigned_predict(p,-d,cu))
                    if height==31: np.testing.assert_array_equal(obj[1:3],obj[3:5])
    assert np.isfinite(out).all() and out.min()>=-1e-12 and out.max()<=1+1e-12
    path=ROOT/"forecast50"/f"panel_{k:02d}.npz"
    path.parent.mkdir(exist_ok=True)
    legacy.v38.atomic_npz(path,p=out,methods=METHODS,selected=selected,families=names,pan=k,job=os.environ["SLURM_JOB_ID"],
                          utc=time.time(),node=socket.gethostname(),axes="method,family,height,weak_at_station,alignment,variant,env,object,noise")
    print(f"forecast {k}; {len(cache)} factual groups",flush=True)


def engine50():
    world50.install_world43(legacy.old.world)
    engine=legacy.infer.make_bank()
    return engine,[legacy.infer.make_template(engine,h*.001,None) for h in HEIGHTS]


if __name__=="__main__":
    assert os.getenv("SLURM_JOB_ID") and legacy.torch.cuda.device_count()==1
    a=argparse.ArgumentParser()
    a.add_argument("shard",type=int,choices=(0,1))
    shard=a.parse_args().shard
    legacy.torch.set_num_threads(1)
    c,u=coefficients()
    with legacy.torch.inference_mode():
        eng,temp=engine50()
        for k in range(shard,12,2):panel(k,eng,temp,c,u)
