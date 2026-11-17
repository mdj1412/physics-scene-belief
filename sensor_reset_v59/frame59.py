"""Image-domain blind sign reader and paired evaluation of a simulated high-speed camera."""
from pathlib import Path
from time import perf_counter
import json
import os
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
N = 64
H, W = 13, 27
X = np.arange(W, dtype=np.float32)
Y = np.arange(H, dtype=np.float32)
TIMES = np.arange(6, dtype=np.float32)


def read_signs(frames):
    """Return sign or zero from the pixels in each six-frame clip."""
    shape = frames.shape[:-3]
    if frames.shape[-3:] != (6, H, W):
        raise ValueError('Expected six 13 by 27 grayscale frames.')
    a = frames.reshape(-1, 6, H, W).astype(np.float32)
    light = np.maximum(a-57., 0.)
    # Upper dot follows the deforming tip; lower dot is a fixture reference.
    def centroid(band):
        columns = light[:, :, band].sum(axis=2)
        mass = columns.sum(axis=-1)
        return (columns*X).sum(axis=-1)/np.maximum(mass, 1.), mass
    top,mt = centroid(slice(0,6))
    bot,mb = centroid(slice(7,13))
    gap = top-bot
    shift = gap[:,3:5].mean(axis=1)-gap[:,:2].mean(axis=1)
    ok = (mt[:,[0,1,3,4]].min(axis=1)>80) & (mb[:,[0,1,3,4]].min(axis=1)>80)
    result = np.where(ok & (abs(shift)>=.5), np.sign(shift), 0.)
    return result.astype('i1').reshape(shape)


def render(sign, impact, rng):
    n = len(sign)
    def rand(lo,hi):
        return rng.uniform(lo,hi,size=(n,1)).astype('f4')
    strength=rand(.25,2.8)
    event_phase=rand(-.45,.45)
    location=rand(10.,16.)
    anchor=rand(10.,16.)
    signal=rand(80.,165.)
    shade = rng.random((n,1)) < .075
    # Each contact has its own optical pose, camera vibration and exposure.
    jitter=rng.normal(0,.42,(n,6)).astype('f4')
    dy=rng.normal(0,.13,(n,6,1,1)).astype('f4')
    topy=np.exp(-.5*((Y[None,None,:,None]-3.-dy)/.95)**2)
    boty=np.exp(-.5*((Y[None,None,:,None]-9.-dy)/.95)**2)
    images=np.zeros((n,6,H,W),dtype='f4')
    for dt in (-.125,.125):
        t=TIMES+dt-event_phase
        ramp=np.clip((t-1.15)/1.9,0.,1.)
        pulse=np.clip((t-1.2)/1.5,0.,1.)*np.exp(-.5*((t-3.15)/1.25)**2)
        dx=sign[:,None]*strength*np.where(impact[:,None],pulse,ramp)
        pos=location+jitter+dx
        topx=np.exp(-.5*((X[None,None,:]-pos[:,:,None])/1.15)**2)
        images += .5*signal[:,:,None,None]*(1-shade[:,:,None,None])*topy*topx[:,:,None,:]
    botx=np.exp(-.5*((X[None,None,:]-(anchor+jitter)[:,:,None])/1.1)**2)
    images += 125.*boty*botx[:,:,None,:]
    images += 24.+rng.normal(0,7,images.shape).astype('f4')
    return np.clip(np.rint(images),0,255).astype('u1')


def make_panel(i):
    with np.load(ROOT/'pilot_audit'/f'panel_{i:02}.npz') as f:
        true=f['truth_signs'][:,:,:,:N]
    # Read-only audit truth drives the physical scene. Only uint8 pixels enter read_signs.
    assert true.shape == (2,2,2,N,4,2,2), true.shape
    tracked=np.zeros_like(true)
    fixtures=np.zeros((2,2,2,4,2,2,6,H,W),dtype='u1')
    image_ms=[]
    for cause,regime,mat in np.ndindex(2,2,2):
        rng=np.random.default_rng([5907331,i,mat])
        truth=true[cause,regime,mat].reshape(-1)
        outputs=[]
        for j in range(0,len(truth),128):
            z=truth[j:j+128]
            frames=render(z,np.arange(j,j+len(z))%2==1,rng)
            # Isolate read time for one 8-object pilot, both contacts per object.
            for start in range(0,len(z),16):
                t=perf_counter()
                signs=read_signs(frames[start:start+16])
                image_ms.append((perf_counter()-t)*1000)
                outputs.append(signs)
            if not j:
                fixtures[cause,regime,mat]=frames[:16].reshape(4,2,2,6,H,W)
        tracked[cause,regime,mat]=np.concatenate(outputs).reshape(N,4,2,2)
    out=ROOT/'frame_panels'; out.mkdir(exist_ok=True)
    tmp=out/f'writing_{i:02}.npz'
    np.savez_compressed(tmp,readout=tracked,frames=fixtures,latency_ms=np.asarray(image_ms))
    tmp.replace(out/f'panel_{i:02}.npz')


def summarize():
    from route59 import log_odds
    with np.load(ROOT/'score59_cells.npz') as f:
        risks=f['exact_by_e']
    mix=np.array([[.42,0,.58],[0,.84,.16]])
    by_regime=np.einsum('cfea,ze->cfza',risks,mix)
    modes=['off','sign_model','pixels']
    points=np.empty((24,3,2,3))
    hist=np.zeros(7,dtype='i8')
    times=[]
    for i in range(24):
        with np.load(ROOT/'pilot_audit'/f'panel_{i:02}.npz') as f:
            t=f['truth_signs'][:,:,:,:N]
        with np.load(ROOT/'pilots'/f'panel_{i:02}.npz') as f:
            codes=f['codes'][:,:,:,:,:,:N].astype('i2')
        with np.load(ROOT/'frame_panels'/f'panel_{i:02}.npz') as f:
            a=f['readout']; times.append(f['latency_ms'])
        active=t[...,0]!=0
        valid=np.all(a!=0,axis=-1)
        err=(t[...,0]!=t[...,1])!=(a[...,0]!=a[...,1])
        hist+=np.array([active.sum(),(~active).sum(),(active&valid).sum(),(~active&valid).sum(),
                        (active&valid&err).sum(),((t!=0)&(a!=0)).sum(),((a!=0)&(t!=0)&(a!=t)).sum()])
        cam=np.where(active&valid,1+(a[...,0]!=a[...,1]).astype('i2'),0)
        for r in (0,1):
            no=codes[r,0]
            first,second=np.divmod(no,13)
            pixel=13*(first+4*cam[...,0])+(second+4*cam[...,1])
            for k,used in enumerate((no,codes[r,1],pixel)):
                pol,cau=log_odds(used,r,k!=0)
                fixed=pol>1e-12; phys=cau>1e-12
                classacc=.5*(fixed[:,0].mean()+1-fixed[:,1].mean())
                causeacc=.5*(phys[0].mean()+1-phys[1].mean())
                rate=fixed.mean(axis=-1).transpose(2,0,1) # material, cause, regime
                risk=by_regime[i,:,None,:,0]+rate*(by_regime[i,:,None,:,1]-by_regime[i,:,None,:,0])
                points[i,k,r]=[causeacc,classacc,risk.mean()]
    boot=np.random.default_rng(5907332).integers(24,size=(5000,24))
    contrasts={'pixel_reset_minus_no_reset':points[:,2,1]-points[:,2,0],
               'pixel_minus_off_with_reset':points[:,2,1]-points[:,0,1],
               'pixel_minus_sign_with_reset':points[:,2,1]-points[:,1,1]}
    out=dict(modes=modes,reset_order=[0,1],metric_order=['cause_BA','policy_BA','mean_Brier'],
             means=points.mean(0).tolist(),counts=hist.tolist(),
             paired_contrasts={k:[v.mean(0).tolist(),*np.quantile(v[boot].mean(1),[.025,.975],axis=0).tolist()] for k,v in contrasts.items()},
             blind_reader_host_ms_p50_p90=np.quantile(np.concatenate(times),[.5,.9]).tolist())
    (ROOT/'frame59.json').write_text(json.dumps(out,indent=2)+'\n')
    np.savez_compressed(ROOT/'frame_cells59.npz',points=points)


if __name__=='__main__':
    assert os.getenv('SLURM_JOB_ID') and os.getenv('CUDA_VISIBLE_DEVICES')==''
    if sys.argv[1]=='score':
        summarize()
    else:
        for index in map(int,sys.argv[1:]):
            make_panel(index)
