"""Synchronized raw-frame-to-terminal bench across crossed trigger and fault conditions."""
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'sensor_reset_v59'))
sys.path.insert(0,str(ROOT.parent/'fresh_surface_v55'))
from bench55 import prev,serial_alpha,terminal
from policy54 import select54
import route59 as router
import generate61 as fresh
import vision61 as vision

METHODS=('dual_risk_reset','dual_risk_sham','gate','led_risk_reset','led_risk_sham','nominal_risk_reset')
# mode 2 uses switch/photodiode, 1 uses optical LED, 0 uses the nominal window.
FLAGS=((2,1),(2,0),None,(1,1),(1,0),(0,1))
MIXTURE=np.array([[.42,0,.58],[0,.84,.16]])


def plan():
    rng=np.random.default_rng(6108443)
    schedule={}
    for i in range(24):
        materials=np.arange(2)
        cause=(i//2+materials)%2
        target=(i+materials)%2
        # Within each cell, common-mode high/low is exactly independent of A/B.
        for b in range(4):
            cell=(b+i)%4
            schedule[i,b]=(cell,cause.copy(),target.copy(),target.copy())
    for d in range(4):
        for target in (0,1):
            subset=[rec for rec in schedule.values() if rec[0]==d and rec[2][rec[1]==1][0]==target]
            states=rng.permutation(np.arange(len(subset))%2)
            assert len(subset)==12 and states.sum()==6
            for rec,state in zip(subset,states):
                rec[3][rec[1]==1]=state
    return schedule


def orders(i):
    # Every three catalogs form two six-sequence Williams crossovers.
    rng=np.random.default_rng([6106073,i//3])
    names=rng.permutation(6)
    rows=rng.permutation(np.tile(np.arange(6),2))[4*(i%3):4*(i%3+1)]
    return names[(np.array([0,1,5,2,4,3])+rows[:,None])%6]


def deployment(i,target):
    rng=np.random.default_rng([6104847,i])
    env=np.arange(4)
    surface=((env+i)%4)//2
    episode=rng.integers(256,size=4)
    pre=rng.integers(5,size=4)
    post=np.where(rng.random(4)<.4375,4-pre,pre)
    eps=np.array([rng.choice(3,p=MIXTURE[j]) for j in target[surface]])
    d=fresh.raw(i)
    raw=(d['ow'][pre,env,episode],d['of'][pre,env,episode],d['nw'][post,env,episode],
         d['nf'][post,env,episode],d['imp'][eps,post,env,episode],None)
    return raw,surface


def choice(frames,wave,electronics,target,arm):
    flags=FLAGS[arm]
    if flags is None: return np.zeros(2,bool),None
    mode,reset=flags
    cam=vision.read_camera(frames,wave,mode=mode).reshape(2,8,2)
    paid=np.zeros((2,8,3,2),dtype=np.int8)
    paid[:,:,:2,:]=electronics[:,reset].reshape(2,8,2,2)
    paid[:,:,2,:]=cam
    encoded=router.encode(paid[...,0],paid[...,1],target,True)
    which=router.route(encoded,f'risk_r{reset}c1')
    return which,encoded


def measure(i,assign,engine,templates,coef):
    m=np.arange(2)
    with np.load(ROOT/'fixtures'/f'panel_{i:02d}.npz') as file:
        pixels=file['frames']
        waveform=file['wave']
        episode=int(file['episode'])
        assert episode==i%8
        electronic=file['electronic']
        targets=file['reset_id']
    with np.load(ROOT/'pilots'/f'panel_{i:02d}.npz') as file:
        codes=file['codes'][:,:,:,:,:,:,episode]
    with np.load(ROOT/'forecasts'/f'panel_{i:02d}.npz') as file:
        plus,minus,actions=(file[key] for key in ('pplus','pminus','actions'))
    obs=fresh.observe(i)
    raw,surface=deployment(i,assign[i,0][2])
    base=serial_alpha(raw)
    gate=select54(raw[2][:,:3],raw[3][:,:3],mode='robust')
    times=np.empty((4,6))
    outer=np.empty((4,6,2),bool)
    bought=np.empty((4,6,4),bool)
    errors=np.zeros(6)
    order=orders(i)
    labels=[]
    warm=np.random.default_rng([6108543,i]).permutation(6)
    for row in range(-1,4):
        d,cause,target,z=assign[i,max(0,row)]
        frame=pixels[d,cause,z,m]
        wave=waveform[d]
        elec=np.stack([electronic[:,cause[j],z[j],j] for j in m])
        assert frame.shape[:4]==(2,4,2,2) and elec.shape==(2,2,4,2,2,2)
        if row>=0: labels.append([d,episode,*cause,*target,*z])
        for arm in (warm if row<0 else order[row]):
            # Only the paid frame data, raw electronics, randomized target, and chosen
            # policy are visible to the selector. No true z/cause/d/hazard enters it.
            torch.cuda.synchronize()
            start=time.perf_counter_ns()
            decision,code=choice(frame,wave,elec,targets,arm)
            mask=decision[surface]
            needed=~mask
            if needed.any():
                mask[needed]=select54(raw[2][needed,:3],raw[3][needed,:3],mode='robust')
            alpha=terminal(raw,mask)
            pp,pm,act=fresh.v55.forward55(obs,engine,templates,coef)
            prediction=alpha[:,None]*pp+(1-alpha[:,None])*pm
            torch.cuda.synchronize()
            elapsed=(time.perf_counter_ns()-start)/1e6
            if FLAGS[arm] is None:
                fixed=np.zeros(2,bool)
            else:
                mode,r=FLAGS[arm]
                expected_code=codes[d,mode,r,cause,z,m]
                np.testing.assert_array_equal(code,expected_code)
                fixed=router.route(expected_code,f'risk_r{r}c1')
            np.testing.assert_array_equal(fixed,decision)
            np.testing.assert_array_equal(mask,fixed[surface]|gate)
            np.testing.assert_array_equal(act,actions)
            alpha_expected=base[mask.astype(int),np.arange(4)]
            np.testing.assert_allclose(alpha,alpha_expected,rtol=0,atol=2e-12)
            err=float(np.abs(prediction-(alpha_expected[:,None]*plus+(1-alpha_expected[:,None])*minus)).max())
            assert err<2e-9
            errors[arm]=max(errors[arm],err)
            if row>=0:
                times[row,arm]=elapsed
                outer[row,arm]=decision
                bought[row,arm]=mask
    dest=ROOT/'timing61'/f'panel_{i:02d}.npz'
    dest.parent.mkdir(exist_ok=True)
    prev.r.v38.atomic_npz(dest,ms=times,outer=outer,bought=bought,errors=errors,order=order,labels=labels,
                          reset_id=targets,methods=np.array(METHODS))
    print('catalog',i,'p90',np.round(np.quantile(times,.9,axis=0),2).tolist(),flush=True)
    return times,outer,bought,order,np.asarray(labels),errors


def main():
    assert os.getenv('SLURM_JOB_ID') and torch.cuda.device_count()==1
    torch.set_num_threads(1)
    before=prev.snapshot()
    schedule=plan()
    with torch.inference_mode():
        fresh.previous.install()
        engine,templates=prev.engine50()
        coef=prev.coefficients()[0]
        rows=[measure(i,schedule,engine,templates,coef) for i in range(24)]
    ms,outer,bought,order,labels,err=map(np.stack,zip(*rows))
    mat=np.array([np.bincount(x,minlength=6) for x in order.reshape(-1,6).T])
    assert (mat==16).all()
    adjacency=np.zeros((6,6),int)
    for line in order.reshape(-1,6):np.add.at(adjacency,(line[:-1],line[1:]),1)
    assert (adjacency[~np.eye(6,dtype=bool)]==16).all()
    p90=np.quantile(ms,.9,(0,1))
    idx=np.random.default_rng(6108491).integers(24,size=(16000,24))
    boot=np.quantile(ms[idx],.9,(1,2))
    delta=boot[:,[0]]-boot[:,1:]
    result=dict(methods=METHODS,p90_ms=p90.tolist(),corrected_reset_minus_others=(p90[0]-p90[1:]).tolist(),
                primary_two_simultaneous95=np.quantile(delta[:,:2],[.0125,.9875],axis=0).T.tolist(),
                pointwise95=np.quantile(delta,[.025,.975],axis=0).T.tolist(),
                cell_p90_ms={f'cell{d}':np.quantile(ms[labels[:,:,0]==d],.9,0).tolist() for d in range(4)},
                board_p90_ms={f'board{b}':np.quantile(ms[labels[:,:,1]==b],.9,0).tolist() for b in range(8)},
                outer_fixed_count=outer.sum((0,1,3)).tolist(),impact_count=bought.sum((0,1,3)).tolist(),
                position_counts=mat.tolist(),adj_counts=adjacency.tolist(),max_replay_error=err.max(0).tolist(),
                measured_per_arm=96,material_pilot_placements=192,per_cell=24,catalogs=24,
                physical_clips_per_call=32,objects_per_call=512,baseline='memory-resident raw camera, switch and photodiode to terminal CUDA synchronization',
                torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
                node=socket.gethostname(),job=os.getenv('SLURM_JOB_ID'),before=before,after=prev.snapshot())
    prev.r.v38.atomic_json(ROOT/'timing61.json',result)

if __name__=='__main__':main()
