"""Post-primary randomized timing crossover for direct calibration on the same GPU."""
import json
import os
import time
import numpy as np
import torch
import runner48 as r
from direct48 import design

METHODS=("always_original","gate_local","direct_local")
ROOT=r.ROOT


def measure(panel,engine,templates,policy,coef):
    rng=np.random.default_rng(481835+panel)
    f=panel%4
    h=(panel//4+panel//2)%2
    a=(panel//4+panel)%2
    v=(panel//2+panel//4)%2
    height=r.HEIGHTS[h]
    xx,sensors=r.apparatus(r.world(panel),r.FAMILIES[f],height,r.VARIANTS[v])
    obs=r.add_optical(sensors,r.FAMILIES[f],height,r.ALIGN[a],r.VARIANTS[v])
    one=[{key:row[key][:,:1] for key in (r.r.Q,r.r.P)} for row in obs]
    state=rng.permutation(np.repeat(np.arange(3),[256,128,128])).reshape(4,128)
    env,obj=np.indices((4,128))
    with np.load(ROOT/"forecast"/f"panel_{panel:02d}.npz") as fa:
        op=fa["p_original"][f,a,h,v,...,0]; pp=fa["p_projected"][f,a,h,v,...,0]
        choose=fa["selected"][f,h,v]; arm=(choose==r.r.Q).astype(int)
        disp=np.ascontiguousarray(fa["readings_mm"][f,h,state,h,env,obj,0])
        given=fa["gate_preload"][f,a,h,v,state,env,obj,0]
    # Independent expansion of the published seven terms on archived full-profile forecasts.
    c=coef[h][arm]
    x=2*(op-.5);z=np.clip(disp/.084,-2.,2.)
    refdirect=np.clip(op+c[...,0]+c[...,1]*x+c[...,2]*x*x+c[...,3]*z+
                      c[...,4]*abs(z)+c[...,5]*z*x+c[...,6]*abs(z)*x,0.,1.)
    refs=(op,np.where(given,pp,op),refdirect)
    ms=np.zeros((4,3));order=np.zeros_like(ms,int);maxerr=np.zeros(3)
    for k in range(-1,len(ms)):
        seq=rng.permutation(3)
        for j in seq:
            torch.cuda.synchronize()
            start=time.perf_counter_ns()
            act=r.infer.actions43(xx,engine)
            paid=r.infer.paid_sensors(one,act)
            if j==1:
                q=np.stack([row[r.r.Q] for row in paid]).astype(np.float32)
                gate=r.gates(q,disp[...,None],height,"preload",policy)[...,0]
            else:
                gate=np.zeros_like(act,bool)
            prob=r.timing45.decode_mixed(engine,templates[h],xx,act,paid,gate)
            if j==2:
                weights=coef[h][(act==r.r.Q).astype(int)]
                adjust=(design(prob,disp)*weights).sum(axis=-1)
                prob=np.clip(prob+adjust,0.,1.)
            torch.cuda.synchronize()
            end=time.perf_counter_ns()
            np.testing.assert_array_equal(act,choose)
            if j==1: np.testing.assert_array_equal(gate,given)
            np.testing.assert_allclose(refs[j],prob,atol=2e-9,rtol=0)
            maxerr[j]=max(maxerr[j],float(abs(refs[j]-prob).max()))
            if k>=0: ms[k,j]=(end-start)*1e-6
        if k>=0: order[k]=seq
    dest=ROOT/"direct_timing"/f"panel_{panel:02d}.npz"
    dest.parent.mkdir(exist_ok=True)
    r.v38.atomic_npz(dest,warm_ms=ms,order=order,methods=METHODS,max_error=maxerr)
    print("panel",panel,flush=True)
    return ms,maxerr


def main():
    assert os.getenv("SLURM_JOB_ID") and torch.cuda.device_count()==1
    r.world48.install_world43(r.old.world)
    torch.set_num_threads(1)
    with torch.inference_mode():
        e=r.infer.make_bank()
        ts=[[r.infer.make_template(e,h*.001,u) for u in (None,r.AXES[str(h)]["axis"])] for h in r.HEIGHTS]
        policy=r.route48.load_policy("cuda")
        coef=np.array(json.loads((ROOT/"direct_source48.json").read_text())["coefficients"])[1]
        start=time.time()
        items=[measure(k,e,ts,policy,coef) for k in range(12)]
    clock=np.array([i[0] for i in items])
    idx=np.random.default_rng(481822).integers(0,12,(16000,12))
    b=np.quantile(clock[idx],.9,axis=(1,2))
    p=np.quantile(clock,.9,axis=(0,1))
    out=dict(methods=METHODS,p90_ms=p.tolist(),p90_ci_ms=np.quantile(b,[.025,.975],axis=0).T.tolist(),
             direct_minus=list(METHODS[:2]),direct_diff_ms=(p[2]-p[:2]).tolist(),
             direct_diff_ci_ms=np.quantile(b[:,2,None]-b[:,:2],[.025,.975],axis=0).T.tolist(),
             n_panels=12,reps=4,batch=512,bootstrap=16000,max_err=np.max([i[1] for i in items],axis=0).tolist(),
             job=os.getenv("SLURM_JOB_ID"),node=os.getenv("SLURMD_NODENAME"),gpu=torch.cuda.get_device_name(),utc_start=start,utc_end=time.time(),
             signature=r.digest(ROOT/"timing_direct_freeze48.sha256"),scope="secondary wall-clock. Original posterior returns to host; direct calibration runs there, with one scalar. Physical motion excluded.")
    r.v38.atomic_json(ROOT/"direct_timing48.json",out)
    print(out,flush=True)


if __name__=="__main__":
    main()
