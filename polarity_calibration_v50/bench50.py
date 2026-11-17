"""K=4 cold-session randomized blocked crossover, synchronized on one GPU."""
import json
import os
import socket
import time

import numpy as np
import torch

import estimator50
from policy50 import (ROOT, STATES,HEIGHTS, METHODS as ARCHIVE_METHODS, coefficients,engine50,observe,
                      unsigned_predict,update,world,world50,legacy as r)
from timing_repeat45 import snapshot

METHODS=("local_hard","local_frozen","fixed_frozen","fixed_hard","local_unsigned_fit","local_symmetric","local_soft")
REPS=4
BUDGET=4


def measurable(k,pop,f,h,states):
    env,obj=np.indices((4,128))
    signals=[]
    for which,site in enumerate((HEIGHTS[h],31)):
        d=np.stack([world50.readings(pop,world50.FAMILIES[f],site,s)[...,0] for s in STATES])
        signals.append(np.ascontiguousarray(d[states[which],env,obj]))
    return signals


def panel(k,eng,templates,signed,unsigned):
    gen=np.random.default_rng(500804+k)
    f=k%4;h=(k//4+k//2)%2;a=(k//4+k)%2;v=(k//2+k//4)%2
    height=HEIGHTS[h]
    name=f"panel_{k:02d}.npz"
    pop=world(k)
    fam=world50.FAMILIES[f]
    xx,sensors=r.apparatus(pop,fam,height,r.VARIANTS[v])
    obs=observe(sensors,fam,height,r.ALIGN[a],r.VARIANTS[v])
    one=[{key:row[key][:,:1] for key in (r.r.Q,r.r.P)} for row in obs]
    local=gen.permutation(np.repeat(np.arange(3),[256,128,128])).reshape(4,128)
    rho=(0.,.5,1.)[k%3]
    remote=np.where(gen.random((4,128))<rho,local,gen.choice(3,size=(4,128),p=(.5,.25,.25)))
    states=np.stack((local,local if h else remote))
    signals=measurable(k,pop,f,h,states)
    env,obj=np.indices((4,128))
    with np.load(ROOT/"cal50"/name) as cal:
        # Each catalog visits two contrasting conditional laws; individual sessions in a batch may differ.
        # Kappa bin and repetition are used by the test fixture to sample measured inputs only.
        kid=(np.arange(4)+k)%5
        rep=gen.integers(256,size=4)
        ix=(kid,np.arange(4),rep)
        wk=cal["weak_mm"][ix]
        full=cal["full_mm"][ix]
        precomputed=cal["khat"][kid,2,np.arange(4),rep]
        assert wk.shape==full.shape==(4,8)
    ref_hat=estimator50.infer(wk[:,:BUDGET],full[:,:BUDGET])[:,None]
    np.testing.assert_allclose(ref_hat[:,0],precomputed,atol=5e-7,rtol=0)
    with np.load(ROOT/"forecast50"/name) as pred:
        assert tuple(pred["methods"])==ARCHIVE_METHODS
        bank=pred["p"][:,f,h,:,a,v,...,0]
        selected=pred["selected"][f,h,v]
    locplus,locminus,fixplus,fixminus,blind=[bank[i,states[int(i in (3,4))],env,obj] for i in range(1,6)]
    ref=[np.where(ref_hat<.5,locminus,locplus),locplus,fixplus,
         np.where(ref_hat<.5,fixminus,fixplus),blind,(locplus+locminus)/2,ref_hat*locplus+(1-ref_hat)*locminus]
    ms=np.zeros((REPS,len(METHODS)))
    orders=np.zeros_like(ms,dtype=int)
    err=np.zeros(len(METHODS))
    before=snapshot()
    for rep_i in range(-1,REPS):
        seq=gen.permutation(len(METHODS))
        for m in seq:
            method=METHODS[m]
            isfixed=method.startswith("fixed")
            d=signals[int(isfixed)]
            torch.cuda.synchronize()
            start=time.perf_counter_ns()
            if method.endswith(("_hard","_soft")):
                hat=estimator50.infer(wk[:,:BUDGET],full[:,:BUDGET])[:,None]
            act=r.infer.actions43(xx,eng)
            paid=r.infer.paid_sensors(one,act)
            p0=r.infer.decode43(eng,templates[h],xx,act,paid)[...,0]
            arm=(act==r.r.Q).astype(np.intp)
            if method=="local_unsigned_fit":
                pred=unsigned_predict(p0,d,unsigned[h][arm])
            else:
                c=signed[2 if isfixed and not h else 1,h][arm]
                if method.endswith("_hard"):
                    pred=update(p0,np.where(hat<.5,-d,d),c)
                elif method.endswith(("_symmetric","_soft")):
                    scale=hat if method=="local_soft" else .5
                    pred=scale*update(p0,d,c)+(1.-scale)*update(p0,-d,c)
                else:
                    pred=update(p0,d,c)
            torch.cuda.synchronize()
            end=time.perf_counter_ns()
            if rep_i>=0:ms[rep_i,m]=(end-start)*1e-6
            np.testing.assert_array_equal(act,selected)
            np.testing.assert_allclose(pred,ref[m],atol=2e-9,rtol=0)
            err[m]=max(err[m],abs(pred-ref[m]).max())
        if rep_i>=0:orders[rep_i]=seq
    # Isolate host estimator for K-sensitivity. Copy-free in-memory readouts; CPU wall time per four sessions.
    budgets=(0,1,4,8)
    sub=np.empty((201,len(budgets)))
    for index in range(-1,len(sub)):
        for j in gen.permutation(len(budgets)):
            num=budgets[j]
            start=time.perf_counter_ns(); estimator50.infer(wk[:,:num],full[:,:num]); stop=time.perf_counter_ns()
            if index>=0:sub[index,j]=(stop-start)*1.e-6
    dest=ROOT/"timing50"/name
    r.v38.atomic_npz(dest,ms=ms,order=orders,methods=METHODS,max_replay_error=err,est_ms=sub,budgets=budgets,
                     test_cell=[f,h,a,v],session_repetition=rep,fixture_regime=kid)
    r.v38.atomic_json(dest.with_suffix(".json"),{"before":before,"after":snapshot()})
    print("timing",k,flush=True)
    return ms,err,sub


if __name__=="__main__":
    assert os.getenv("SLURM_JOB_ID") and torch.cuda.device_count()==1
    torch.set_num_threads(1)
    (ROOT/"timing50").mkdir(exist_ok=True)
    start=time.time()
    with torch.inference_mode():
        eng,templates=engine50()
        c,u=coefficients()
        packets=[panel(k,eng,templates,c,u) for k in range(12)]
    clock=np.stack([p[0] for p in packets])
    point=np.quantile(clock,.9,axis=(0,1))
    boot=np.random.default_rng(5008001).integers(0,12,(16000,12))
    sampled=np.quantile(clock[boot],.9,axis=(1,2))
    err=np.max([p[1] for p in packets],axis=0)
    micro=np.quantile(np.stack([p[2] for p in packets]),.9,axis=(0,1))
    results=dict(methods=METHODS,p90_ms=point.tolist(),p90_ci_ms=np.quantile(sampled,[.025,.975],axis=0).T.tolist(),
                 local_hard_minus_ms=(point[0]-point).tolist(),local_hard_minus_ci_ms=np.quantile(sampled[:,0,None]-sampled,[.025,.975],axis=0).T.tolist(),
                 estimator_p90_for_four_sessions_ms=micro.tolist(),estimator_budgets=[0,1,4,8],max_replay_error=err.tolist(),
                 budgets_per_session=BUDGET,catalogs=12,time_repeats=REPS,warmup_per_method_per_catalog=1,objects_per_batch=512,
                 job=os.environ["SLURM_JOB_ID"],node=socket.gethostname(),gpu=torch.cuda.get_device_name(),
                 cuda=torch.version.cuda,torch=str(torch.__version__),utc_start=start,utc_end=time.time())
    r.v38.atomic_json(ROOT/"timing50.json",results)
    print(json.dumps(results),flush=True)
