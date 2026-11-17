"""Blocked crossover on one GPU for queried policies with one paid sensor scalar."""
import json
import os
import socket
import time

import numpy as np
import torch
from policy49 import METHODS as PRED_METHODS, Policy, ROOT, SOURCE, update
import runner48 as r
from timing_repeat45 import snapshot

METHODS=("original","direct_local","direct_fixed","gate_local")
STRATA=((0.,0.),(1.,1.))
NREP=3


def measurement(panel,f,h,which,family,pop,full,retained,copy,independent):
    """Test two mechanical regimes without passing regime/true state to the policy."""
    rho,keep=STRATA[which]
    weak=np.where((full!=0)&(retained>=keep),3-full,full)
    rem=np.where(copy<rho,weak,independent)
    fixed=(r.HEIGHTS[h]==31)
    states=np.stack((weak,weak if fixed else rem))
    readings=[]
    for pos,site in enumerate((r.HEIGHTS[h],31)):
        one=np.stack([r.sensor48.readings(pop,family,site,s,site)[...,0] for s in r.sensor48.STATES])
        readings.append(np.ascontiguousarray(np.take_along_axis(one,states[pos,None],axis=0)[0]))
    return states,readings


def bench(panel,engine,templates,policy):
    name=f"panel_{panel:02d}.npz"
    rng=np.random.default_rng(494928+panel)
    f=panel%4; h=(panel//4+panel//2)%2; a=(panel//4+panel)%2; v=(panel//2+panel//4)%2
    family=r.FAMILIES[f]; height=r.HEIGHTS[h]; variant=r.VARIANTS[v]
    pop=r.world(panel)
    xx,sensors=r.apparatus(pop,family,height,variant)
    obs=r.add_optical(sensors,family,height,r.ALIGN[a],variant)
    one=[{key:row[key][:,:1] for key in (r.r.Q,r.r.P)} for row in obs]
    full=rng.permutation(np.repeat(np.arange(3),[256,128,128])).reshape(4,128)
    retained,copy=rng.random((2,4,128))
    other=rng.choice(3,size=(4,128),p=[.5,.25,.25])
    env,obj=np.indices((4,128))
    with np.load(ROOT/"forecast49"/name) as archive:
        np.testing.assert_array_equal(archive["methods"],PRED_METHODS)
        pred=archive["p"][:,f,h,:,a,v,...,0]
    with np.load(SOURCE/"forecast"/name) as archive:
        sel=archive["selected"][f,h,v]
    raw=np.empty((2,NREP,len(METHODS)))
    orders=np.empty(raw.shape,dtype=int)
    maxerr=np.zeros((2,len(METHODS)))
    before=snapshot()
    for condition in rng.permutation(2):
        states,meas=measurement(panel,f,h,condition,family,pop,full,retained,copy,other)
        for rep in range(-1,NREP):
            seq=rng.permutation(len(METHODS))
            for m in seq:
                method=METHODS[m]
                isfixed=method=="direct_fixed"
                disp=meas[int(isfixed)]
                torch.cuda.synchronize()
                start=time.perf_counter_ns()
                act=r.infer.actions43(xx,engine)
                paid=r.infer.paid_sensors(one,act)
                if method=="gate_local":
                    q=np.stack([z[r.r.Q] for z in paid]).astype(np.float32)
                    gate=r.gates(q,disp[...,None],height,"preload",policy.gate)[...,0]
                else:
                    gate=np.zeros_like(act,bool)
                prob=r.timing45.decode_mixed(engine,templates[h],xx,act,paid,gate)
                if method in ("direct_local","direct_fixed"):
                    mode=2 if isfixed and height!=31 else 1
                    weights=policy.coef[mode,h][(act==r.r.Q).astype(np.intp)]
                    prob=update(prob,disp,weights)
                torch.cuda.synchronize()
                end=time.perf_counter_ns()
                if rep>=0: raw[condition,rep,m]=(end-start)*1.e-6
                ref=pred[PRED_METHODS.index(method),states[int(isfixed)],env,obj]
                np.testing.assert_array_equal(act,sel)
                np.testing.assert_allclose(prob,ref,atol=2e-9,rtol=0)
                maxerr[condition,m]=max(maxerr[condition,m],np.max(abs(prob-ref)))
            if rep>=0: orders[condition,rep]=seq
    dest=ROOT/"timing49"/name
    r.v38.atomic_npz(dest,ms=raw,order=orders,methods=METHODS,maxerr=maxerr,cell=[f,h,a,v],strata=STRATA)
    r.v38.atomic_json(dest.with_suffix(".json"),{"before":before,"after":snapshot()})
    print("bench",panel,flush=True)
    return raw,maxerr


def metrics(clock,bidx):
    p90=np.quantile(clock,.9,axis=(0,1))
    bs=np.quantile(clock[bidx],.9,axis=(1,2))
    diff=p90[1]-p90
    return {"p90_ms":p90.tolist(),"p90_ci_ms":np.quantile(bs,[.025,.975],axis=0).T.tolist(),
            "local_minus_ms":diff.tolist(),"local_minus_ci_ms":np.quantile(bs[:,1,None]-bs,[.025,.975],axis=0).T.tolist()}


def main():
    assert os.getenv("SLURM_JOB_ID") and torch.cuda.device_count()==1
    torch.set_num_threads(1)
    (ROOT/"timing49").mkdir(exist_ok=True)
    r.world48.install_world43(r.old.world)
    start=time.time()
    with torch.inference_mode():
        engine=r.infer.make_bank()
        templates=[[r.infer.make_template(engine,h*.001,t) for t in (None,r.AXES[str(h)]["axis"])] for h in r.HEIGHTS]
        policy=Policy("cuda")
        panels=[bench(k,engine,templates,policy) for k in range(12)]
    clock=np.stack([x[0] for x in panels])
    bidx=np.random.default_rng(491349).integers(0,12,(16000,12))
    results={"methods":METHODS,"strata_rho_retention":STRATA,
             "overall":metrics(clock.reshape(12,2*NREP,len(METHODS)),bidx),
             "strata":[metrics(clock[:,j],bidx) for j in range(2)],"n_catalogs":12,"batch":512,
             "replicates_per_stratum":NREP,"max_replay_error":np.max([x[1] for x in panels],axis=0).tolist(),
             "job":os.environ["SLURM_JOB_ID"],"node":socket.gethostname(),"gpu":torch.cuda.get_device_name(),
             "torch":str(torch.__version__),"cuda":torch.version.cuda,"bootstraps":len(bidx),"utc_start":start,"utc_end":time.time(),
             "scope":"Synchronized wall time for factual action, purchased sensor packing, gate, one selected posterior and optional host calibration. Physical motion and label solver excluded. Intervals are paired, pointwise."}
    r.v38.atomic_json(ROOT/"timing49.json",results)
    print(results,flush=True)


if __name__=="__main__":
    main()
