"""Compare reset vs sham from the acquired images through independent frozen GPU forecasts."""
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
import torch

import bench59 as b
import frame59
from route59 import encode,route

ROOT=Path(__file__).resolve().parent

def paid_camera(pilot,reset,read):
    ref,impact,target,surface=pilot
    # No sign-level camera values are consumed. Attach the tracked optical observations.
    refs=ref[:,reset].copy()
    impacts=impact[:,reset].copy()
    cams=read.reshape(2,8,2)
    valid=np.all(cams!=0,axis=-1) & (refs[...,0]!=0) & (impacts[...,0]!=0)
    refs[...,2]=np.where(valid,cams[...,0],0)
    impacts[...,2]=np.where(valid,cams[...,1],0)
    return encode(refs,impacts,target,True)

def measure(i,engine,templates,coef):
    raw,pilot,sign_codes,params=b.fixture(i)
    regime=(np.arange(2)+i)%2
    cause=(i//2+np.arange(2))%2
    s=np.arange(2)
    with np.load(ROOT/'frame_panels'/f'panel_{i:02}.npz') as f:
        frames=f['frames'][cause,regime,s]
        recorded=f['readout'][cause,regime,s,0]
    assert frames.shape==(2,4,2,2,6,13,27)
    np.testing.assert_array_equal(frame59.read_signs(frames),recorded)
    # Rebuild independent paired baseline from archived pixel readouts and paid
    # electronics-only codes. Verify no primary sign-camera sample survives.
    c=recorded.reshape(2,4,2,2)
    cam=np.where(np.all(c!=0,axis=-1),1+(c[...,0]!=c[...,1]).astype('i2'),0)
    masks=[]
    routes=[]
    serial=b.serial_alpha(raw)
    gate=b.select54(raw[2][:,:3],raw[3][:,:3],mode='robust')
    for k in [0,1]:
        left,right=np.divmod(sign_codes[:,k,0].astype('i2'),13)
        expected=13*(left+4*cam[...,0]*(left>0))+(right+4*cam[...,1]*(right>0))
        np.testing.assert_array_equal(expected,paid_camera(pilot,k,recorded))
        decision=route(expected,('map_r0c1','map_r1c1')[k])
        routes.append(decision);masks.append(decision[pilot[3]]|gate)
    with np.load(ROOT/'forecasts'/f'panel_{i:02}.npz') as f:
        plus,minus,actions=[f[k] for k in ('pplus','pminus','actions')]
    obs=b.fresh.observe(i); times=np.zeros((2,2)); counts=np.zeros(2,int)
    leading=(i%2)
    order=[leading,1-leading,1-leading,leading]
    err=np.zeros(2)
    for t,k in enumerate([1-leading,leading]+order):
        torch.cuda.synchronize()
        start=time.perf_counter_ns()
        image_signs=frame59.read_signs(frames)
        paid=paid_camera(pilot,k,image_signs)
        which=route(paid,('map_r0c1','map_r1c1')[k])
        mask=which[pilot[3]]
        remaining=~mask
        if remaining.any():
            mask[remaining]=b.select54(raw[2][remaining,:3],raw[3][remaining,:3],mode='robust')
        alpha=b.terminal(raw,mask)
        pp,pm,act=b.fresh.v55.forward55(obs,engine,templates,coef)
        prediction=alpha[:,None]*pp+(1-alpha[:,None])*pm
        torch.cuda.synchronize()
        elapsed=(time.perf_counter_ns()-start)/1e6
        if t>=2:
            times[counts[k],k]=elapsed;counts[k]+=1
        np.testing.assert_array_equal(mask,masks[k])
        np.testing.assert_array_equal(act,actions)
        expected=serial[mask.astype(int),np.arange(4)]
        np.testing.assert_allclose(alpha,expected,rtol=0,atol=2e-12)
        error=float(abs(prediction-(expected[:,None]*plus+(1-expected[:,None])*minus)).max())
        assert error<2e-9
        err[k]=max(err[k],error)
    path=ROOT/'frame_timing59'/f'panel_{i:02}.npz'
    path.parent.mkdir(exist_ok=True)
    b.prev.r.v38.atomic_npz(path,order=order,ms=times,outer=np.stack(routes),masks=np.stack(masks),max_error=err)
    return times,err

def main():
    assert os.getenv('SLURM_JOB_ID') and torch.cuda.device_count()==1
    torch.set_num_threads(1)
    before=b.prev.snapshot()
    with torch.inference_mode():
        b.fresh.previous.install()
        engine,templates=b.prev.engine50();coef=b.prev.coefficients()[0]
        data=[measure(i,engine,templates,coef) for i in range(24)]
    times,err=[np.stack(x) for x in zip(*data)]
    p90=np.quantile(times,.9,(0,1))
    idx=np.random.default_rng(5908201).integers(24,size=(16000,24))
    boot=np.quantile(times[idx],.9,(1,2))
    ci=np.quantile(boot[:,1]-boot[:,0],[.025,.975])
    out=dict(reset_order=[0,1],p90_ms=p90.tolist(),reset_minus_sham_ms=p90[1]-p90[0],
             delta_95=ci.tolist(),paid_clips_per_call=32,frames_per_clip=6,objects_per_call=512,
             measured_per_arm=48,catalogs=24,skyline='acquired images to final GPU probabilities',
             max_error=err.max(0).tolist(),before=before,after=b.prev.snapshot(),node=socket.gethostname(),
             job=os.getenv('SLURM_JOB_ID'),gpu=torch.cuda.get_device_name())
    b.prev.r.v38.atomic_json(ROOT/'frame_timing59.json',out)
    print(json.dumps(out),flush=True)
if __name__=='__main__':main()
