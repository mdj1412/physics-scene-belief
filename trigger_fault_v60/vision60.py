"""Observed LED alignment and paid pilot generation, no target state in the camera reader."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
N = 512
Q = .29
H = .26582278481012656
DELAYS = np.array([0., .2, 1.])
TIMES = np.arange(-2, 10, dtype='f4')
X = np.arange(27, dtype='f4')
Y = np.arange(13, dtype='f4')

def optical_features(frames):
    """Read marker position and LED intensity relative to its surrounding background."""
    gray = frames[..., :13, :].astype('f4')
    light = np.maximum(gray - 57., 0.)
    def center(band):
        cols = light[..., band, :].sum(-2)
        mass = cols.sum(-1)
        return (cols * X).sum(-1) / np.maximum(mass,1.), mass
    top, a = center(slice(0,6))
    bottom, b = center(slice(7,13))
    led = (frames[...,13:,1:5].mean(axis=(-1,-2)) -
           frames[...,13:,7:11].mean(axis=(-1,-2)))
    return top-bottom, (a>80.)&(b>80.), led

def track(features,sync):
    gap,valid,led=features
    if sync:
        # Determine the register entirely from a signal in the recording.
        lit=led>38.
        start=lit.argmax(-1)
        ok=(lit.sum(-1)==1)&(start<=2)
        start=np.minimum(start,2)
    else:
        start=np.full(gap.shape[:-1],2,dtype='i8')
        ok=np.ones_like(start,dtype=bool)
    ii=start[...,None]+np.array([0,1,3,4])
    g=np.take_along_axis(gap,ii,axis=-1)
    good=ok & np.take_along_axis(valid,ii,axis=-1).all(-1)
    shift=(g[...,2]+g[...,3]-g[...,0]-g[...,1])*.5
    return np.where(good&(abs(shift)>=.5),np.sign(shift),0).astype('i1'), np.where(ok,start,-1)

def read_camera(frames,sync=True):
    """Read signed deformation from a buffered 8 × 15 × 27 pixel clip."""
    if frames.shape[-3:]!=(8,15,27):
        raise ValueError('An eight-frame buffered camera recording is required.')
    return track(optical_features(frames),sync)[0]

def film_base(rng,n,is_impact):
    def ran(lo,hi):
        return rng.uniform(lo,hi,(n,1)).astype('f4')
    amplitude=ran(.25,2.8)
    phase=ran(-.45,.45)
    top=ran(10.,16.); anchor=ran(10.,16.)
    strength=ran(80.,165.)*(rng.random((n,1))>=.075)
    jitter=rng.normal(0,.42,(n,12)).astype('f4')
    dy=rng.normal(0,.13,(n,12,1,1)).astype('f4')
    topy=np.exp(-.5*((Y[None,None,:,None]-3.-dy)/.95)**2)*strength[:,:,None,None]
    boty=np.exp(-.5*((Y[None,None,:,None]-9.-dy)/.95)**2)
    botx=np.exp(-.5*((X[None,None,:]-(anchor+jitter)[:,:,None])/1.1)**2)
    base=rng.normal(24.,7.,(n,12,15,27)).astype('f4')
    base[:,:,:13,:]+=125.*boty*botx[:,:,None,:]
    led_active=rng.random(n)>=.04
    base[:,2,13:,1:5]+=(rng.uniform(70,170,n)*led_active)[:,None,None]
    paths=[]
    for dt in (-.125,.125):
        t=TIMES+dt-phase
        ramp=np.clip((t-1.15)/1.9,0.,1.)
        pulse=np.clip((t-1.2)/1.5,0.,1.)*np.exp(-.5*((t-3.15)/1.25)**2)
        paths.append(amplitude*np.where(is_impact[:,None],pulse,ramp))
    return base,topy,top+jitter,paths,led_active

def emit(base,sign):
    bg,y,x,paths,clear=base
    image=bg.copy()
    for dx in paths:
        p=x+sign[:,None]*dx
        tx=np.exp(-.5*((X[None,None,:]-p[:,:,None])/1.15)**2)
        image[:,:,:13,:]+=.5*y*tx[:,:,None,:]
    return np.clip(np.rint(image),0,255).astype('u1')

def local_inputs(i):
    rng=np.random.default_rng([60025015,i])
    shape=(2,N,4,2)
    order=rng.random(shape[:-1])<.5
    target=rng.integers(0,2,size=shape[:-1],dtype='u1')
    def perm(x):
        return np.where(order[...,None],x[...,::-1],x)
    active=perm(rng.random(shape)<.5)
    ref=perm(rng.random(shape)<.5)
    independent_t=perm(rng.random(shape)<Q)
    shared=rng.random(shape[:-1]); proxy=rng.random(shape[:-1])<Q
    latched=rng.random(shape[:-1]+(2,))<H
    fake=rng.random(shape[:-1]+(2,))<Q
    er=rng.random(shape+(2,))<.02; ei=rng.random(shape+(2,))<.02
    def sign(bit,ac):
        return np.where(ac,1-2*bit.astype('i1'),0).astype('i1')
    reference=sign(ref[...,None]^er,active[...,None])
    truth=np.empty((2,2)+shape+(2,),dtype='i1')
    electronic=np.empty((2,2,2)+shape+(2,2),dtype='i1')
    for z,m in enumerate((.145,.0925)):
        shared_used=shared<((m-Q*Q)/(Q*(1-Q)))
        upstream=np.where(shared_used[...,None],proxy[...,None],independent_t)
        for cause,t in enumerate((upstream,independent_t)):
            truth[cause,z]=np.stack([sign(ref,active),sign(ref^t,active)],axis=-1)
            for reset in (0,1):
                first=np.where(latched,fake,upstream[...,0,None])
                second_local=latched & ~(np.arange(2)==target[...,None]) if reset else latched
                second=np.where(second_local,fake,upstream[...,1,None])
                imp=sign(ref[...,None]^np.stack([first,second],-2)^ei,active[...,None])
                electronic[reset,cause,z]=np.stack([reference,imp],axis=-1)
    return truth,electronic,target,order

def encode(e,cam,target):
    good=(e[...,0,0]!=0)
    cr=np.where(np.all(cam!=0,axis=-1),1+(cam[...,0]!=cam[...,1]).astype('u1'),0)
    b=e[...,0]!=e[...,1]
    a=np.where(target[...,None]==0,b[...,0],b[...,1])
    b_other=np.where(target[...,None]==0,b[...,1],b[...,0])
    sym=np.where(good,1+a.astype('u1')+2*b_other.astype('u1')+4*cr,0)
    return (sym[...,0]*13+sym[...,1]).astype('u1')

def make_panel(i):
    truth,electronic,target,order=local_inputs(i)
    # Phase 0: nominal, 1: +2 ms, 2: +4 ms. Only impact phase is delayed.
    result=np.empty((3,2)+truth.shape,dtype='i1')
    event0=np.empty((3,2,2,2,4,2,2,8,15,27),dtype='u1')
    summaries=np.zeros((3,4),dtype='i8') # total, within buffer detected, mislocalized, masked detected
    for mat in range(2):
        rng=np.random.default_rng([6007331,i,mat])
        t=truth[:,:,mat].reshape(2,2,-1)
        out=np.empty((3,2,2,2,t.shape[-1]),dtype='i1')
        for j in range(0,t.shape[-1],128):
            stop=j+128
            impact=np.arange(j,stop)%2==1
            base=film_base(rng,len(impact),impact)
            actual=t[:,:,j:stop]
            # Same optical nuisance for paired counterfactual signs; reference is common.
            baseline=actual[0,0]
            alternate=(actual!=baseline).any(axis=(0,1))
            hist=emit(base,baseline)
            feat=optical_features(hist)
            if alternate.any():
                shifted=emit((base[0][alternate],base[1][alternate],base[2][alternate],
                              [a[alternate] for a in base[3]],base[4][alternate]),-baseline[alternate])
                feat2=optical_features(shifted)
            for phase,offset in enumerate((0,2,4)):
                start=offset*impact.astype('i8')
                take=start[:,None]+np.arange(8)
                section=tuple(np.take_along_axis(a,take,axis=1) for a in feat)
                offset_target=2-start
                loc=track(section,True)[1]
                summaries[phase]+=[len(loc),(loc>=0).sum(),((loc>=0)&(loc!=offset_target)).sum(),
                                   (track((section[0],section[1],np.zeros_like(section[2])),True)[1]>=0).sum()]
                for sync in (0,1):
                    b=track(section,sync)[0]
                    a=b.copy()
                    if alternate.any():
                        sec2=tuple(np.take_along_axis(f,take[alternate],axis=1) for f in feat2)
                        a[alternate]=track(sec2,sync)[0]
                    out[phase,sync,:,:,j:stop]=np.where(actual==baseline,b,a)
                if j==0:
                    for c,z in np.ndindex(2,2):
                        fs=hist[:16].copy()
                        if alternate.any():
                            use=np.flatnonzero((actual[c,z]!=baseline)[:16])
                            if len(use):
                                fs[use]=shifted[np.searchsorted(np.flatnonzero(alternate),use)]
                        clips=fs[np.arange(16)[:,None],take[:16]]
                        event0[phase,c,z,mat]=clips.reshape(4,2,2,8,15,27)
        result[:,:,:,:,mat]=out.reshape(3,2,2,2,N,4,2,2)
    delay_rng=np.random.default_rng([6009351,i])
    u=delay_rng.random((2,N,4,2))
    cams=np.empty((3,2)+truth.shape,dtype='i1')
    frames=np.empty_like(event0)
    for d,p in enumerate(DELAYS):
        late=u<p
        cams[d]=np.where(late[None,None,None,...,None],result[1],result[0])
        frames[d]=np.where(late[:,0,...,None,None,None,None],event0[1],event0[0])
    codes=np.empty((3,2,2,2,2,2,N,4),dtype='u1')
    for d,s,r in np.ndindex(3,2,2):
        codes[d,s,r]=encode(electronic[r],cams[d,s],target)
    for d,s in np.ndindex(3,2):
        # Independent replay starts from the raw optical ticket.
        np.testing.assert_array_equal(read_camera(frames[d],s),cams[d,s,:,:,:,0])
    # Negative control on real pixels including background noise: mask the LED ROI.
    erased=event0[0].copy()
    erased[...,13:,1:5]=erased[...,13:,7:11]
    assert not read_camera(erased,True).any()
    name=f'panel_{i:02}.npz'
    def save(folder,**kwargs):
        path=ROOT/folder
        path.mkdir(exist_ok=True)
        tmp=path/f'writing_{i:02}.npz'
        np.savez_compressed(tmp,**kwargs)
        tmp.replace(path/name)
    counts=np.empty((3,2,2,2,4),dtype='i8')
    for d,s,c,z in np.ndindex(3,2,2,2):
        a=cams[d,s,c,z]; t=truth[c,z]
        active=t[...,0]!=0
        paid=np.all(a!=0,axis=-1)
        error=(a[...,0]!=a[...,1])!=(t[...,0]!=t[...,1])
        counts[d,s,c,z]=[active.sum(),(active&paid).sum(),(active&paid&error).sum(),(~active&paid).sum()]
    save('fixtures',frames=frames,electronic=electronic[:,:,:,:,0],reset_id=target[:,0],codes=codes[...,0,:],
         overbuffer=event0[2,0,0,0],clear=event0[0,0,0,0])
    save('pilots',codes=codes)
    save('vision_audit',truth=truth,camera=cams,electronic=electronic,reset_id=target,active_order=order,
         delay_draw=u,source_counts=counts,marker_counts=summaries,overbuffer_signed=result[2])
    print(i,flush=True)

def summarize():
    rows=[]; marker=[]; over=[]; late=[]
    for i in range(24):
        with np.load(ROOT/'vision_audit'/f'panel_{i:02}.npz') as f:
            rows.append(f['source_counts']); marker.append(f['marker_counts'])
            true=f['truth']; x=f['overbuffer_signed']
            over.append([int(np.all(x[s]!=0,axis=-1).sum()) for s in (0,1)])
            late.append(float((f['delay_draw']<.2).mean()))
    counts=np.sum(rows,axis=0)
    # P/C × A/B are paired simulator evaluations, not independent surfaces.
    sc=counts.sum((2,3))
    group={'total_active_read_wrong_invalid':sc.tolist(),'per_cell':counts.tolist(),
           'clock_order':['0%', '20%', '100%'],'strategy_order':['nominal','sync'],
           'marker_total_start_corrected_wrong_masked':np.sum(marker,0).tolist(),
           'actually_late_fraction':float(np.mean(late)),
           'four_ms_read_pairs_uncorrected_corrected':np.sum(over,axis=0).tolist()}
    (ROOT/'vision60.json').write_text(json.dumps(group,indent=2)+'\n')
if __name__=='__main__':
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    ap=argparse.ArgumentParser();ap.add_argument('mode');ap.add_argument('idx',nargs='*',type=int)
    args=ap.parse_args()
    if args.mode=='score': summarize()
    else:
        for index in args.idx: make_panel(index)
