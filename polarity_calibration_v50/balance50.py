"""Secondary independently ordered timing check, equal positions and balanced predecessors."""
import json
import os
import time
import numpy as np
import torch
from bench50 import ROOT,HEIGHTS,estimator50,coefficients,engine50,world,world50,observe,measurable,r,update,snapshot

METHODS=("local_hard","local_frozen","fixed_hard","local_soft")
BASE=np.array([[0,1,3,2],[1,2,0,3],[2,3,1,0],[3,0,2,1]])

def panel(k,eng,templates,signed):
    rng=np.random.default_rng(500804+k)
    order_rng=np.random.default_rng(501099+k)
    h=(k//4+k//2)%2;f=k%4;a=(k//4+k)%2;v=(k//2+k//4)%2
    height=HEIGHTS[h];fam=world50.FAMILIES[f];name=f"panel_{k:02d}.npz"
    pop=world(k)
    xx,sensors=r.apparatus(pop,fam,height,r.VARIANTS[v])
    obs=observe(sensors,fam,height,r.ALIGN[a],r.VARIANTS[v])
    one=[{key:row[key][:,:1] for key in (r.r.Q,r.r.P)} for row in obs]
    local=rng.permutation(np.repeat(np.arange(3),[256,128,128])).reshape(4,128)
    rho=(0.,.5,1.)[k%3]
    remote=np.where(rng.random((4,128))<rho,local,rng.choice(3,size=(4,128),p=(.5,.25,.25)))
    states=np.stack((local,local if h else remote))
    signals=measurable(k,pop,f,h,states)
    with np.load(ROOT/"cal50"/name) as cal:
        kid=(np.arange(4)+k)%5; rep=rng.integers(256,size=4)
        ix=(kid,np.arange(4),rep)
        wk=cal["weak_mm"][ix][:,:4];full=cal["full_mm"][ix][:,:4]
    perm=order_rng.permutation(4)
    seq=perm[BASE[order_rng.permutation(4)]]
    np.testing.assert_array_equal(np.sort(seq,axis=0),np.tile(np.arange(4)[:,None],(1,4)))
    env,obj=np.indices((4,128))
    with np.load(ROOT/"forecast50"/name) as bank:
        p=bank["p"][:,f,h,:,a,v,...,0]
    hat=estimator50.infer(wk,full)[:,None]
    lp,lm,fp,fm=[p[i,states[int(i>=3)],env,obj] for i in range(1,5)]
    expected=[np.where(hat<.5,lm,lp),lp,np.where(hat<.5,fm,fp),hat*lp+(1.-hat)*lm]
    ms=np.zeros((4,4));err=np.zeros(4)
    before=snapshot()
    for iteration,ordered in enumerate([order_rng.permutation(4),*seq]):
        for m in ordered:
            d=signals[int(m==2)]
            torch.cuda.synchronize()
            start=time.perf_counter_ns()
            if m!=1: weight=estimator50.infer(wk,full)[:,None]
            act=r.infer.actions43(xx,eng)
            paid=r.infer.paid_sensors(one,act)
            p0=r.infer.decode43(eng,templates[h],xx,act,paid)[...,0]
            c=signed[2 if m==2 and not h else 1,h][(act==r.r.Q).astype(np.intp)]
            if m in (0,2): value=update(p0,np.where(weight<.5,-d,d),c)
            elif m==1:value=update(p0,d,c)
            else:value=weight*update(p0,d,c)+(1.-weight)*update(p0,-d,c)
            torch.cuda.synchronize()
            stop=time.perf_counter_ns()
            if iteration: ms[iteration-1,m]=(stop-start)*1e-6
            np.testing.assert_allclose(value,expected[m],atol=2e-9,rtol=0)
            err[m]=max(err[m],abs(value-expected[m]).max())
    dest=ROOT/"timing_balance50"/name
    r.v38.atomic_npz(dest,ms=ms,order=seq,methods=METHODS,maxerr=err)
    r.v38.atomic_json(dest.with_suffix(".json"),{"before":before,"after":snapshot()})
    print("balanced",k,flush=True)
    return ms

if __name__=="__main__":
    assert os.getenv("SLURM_JOB_ID") and torch.cuda.device_count()==1
    torch.set_num_threads(1)
    (ROOT/"timing_balance50").mkdir(exist_ok=True)
    with torch.inference_mode():
        engine,templates=engine50()
        signed,_=coefficients()
        clock=np.stack([panel(k,engine,templates,signed) for k in range(12)])
    pt=np.quantile(clock,.9,axis=(0,1))
    idx=np.random.default_rng(5010051).integers(0,12,(16000,12))
    bs=np.quantile(clock[idx],.9,axis=(1,2))
    r.v38.atomic_json(ROOT/"balance50.json",dict(methods=METHODS,p90_ms=pt.tolist(),p90_ci=np.quantile(bs,[.025,.975],axis=0).T.tolist(),
                     local_hard_minus_ms=(pt[0]-pt).tolist(),local_hard_minus_ci=np.quantile(bs[:,0,None]-bs,[.025,.975],axis=0).T.tolist(),
                     catalogs=12,replicates=4,scheme="Williams 4x4 per catalog, randomized method labels and row order"))
