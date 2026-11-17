"""Raw synchronized contact switch and photodiode beside a buffered optical camera."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'trigger_fault_v60'))
from vision60 import X, TIMES, optical_features, film_base, emit, encode
N=512
Q=.29
CELLS=((0,0),(0,2),(2,0),(2,2))
ADC=np.arange(128,dtype='f4')/16-2.0

def boards():
    r=np.random.default_rng(61170917)
    ranges=dict(switch_bias=(.018,.17),diode_bias=(.012,.09),camera_skew=(-.23,.23),
                led_driver=(.06,.32),gain_switch=(85.,150.),gain_diode=(80.,150.),
                adc_noise=(2.,7.),led_gain=(80.,160.),drop_switch=(.003,.03),
                drop_diode=(.008,.05),latch=(.16,.36),sign_error=(.013,.033))
    return {key:r.uniform(lo,hi,8) for key,(lo,hi) in ranges.items()}

BOARDS=boards()

def edge(wave):
    """Interpolated rising-edge time, requiring two consecutive bright ADC samples."""
    baseline=np.median(wave[...,:12],axis=-1)
    level=baseline+40.
    high=wave>level[...,None]
    pair=high[...,:-1]&high[...,1:]
    j=np.argmax(pair,axis=-1)
    ok=pair.any(-1)&(j>0)
    def pick(index):
        return np.take_along_axis(wave,index[...,None],axis=-1)[...,0].astype(float)
    before=pick(np.maximum(0,j-1));after=pick(j)
    ratio=np.clip((level-before)/np.maximum(1.,after-before),0,1)
    return (j-1+ratio)/16-2.,ok

def estimate(features,wave,mode=2):
    displacement,valid,led=features
    shape=displacement.shape[:-1]
    if mode==0:
        return np.full(shape,2,dtype=int),np.ones(shape,bool),np.full(shape,np.nan),np.full(shape,np.nan)
    lit=led>38.
    flash=lit.argmax(-1)
    unique=lit.sum(-1)==1
    if mode==1:
        lag=np.zeros(shape)
        ok=unique
    elif mode==2:
        sw,sv=edge(wave[...,0,:])
        pd,pv=edge(wave[...,1,:])
        lag=pd-sw
        ok=unique & sv & pv
    else:
        raise ValueError('mode must be nominal, optical-only or dual')
    phase=2-flash+lag
    start=np.rint(2-phase).astype(int)
    window=(start>=0)&(start<=2)
    # Timing estimates can be valid even if the complete six-frame window was lost.
    return np.clip(start,0,2),ok&window,np.where(ok,lag,np.nan),np.where(ok,phase,np.nan)

def read_features(features,wave,mode):
    start,ok,_,_=estimate(features,wave,mode)
    idx=start[...,None]+np.array([0,1,3,4])
    g=np.take_along_axis(features[0],idx,-1)
    visible=np.take_along_axis(features[1],idx,-1).all(-1)
    delta=(g[...,2]+g[...,3]-g[...,0]-g[...,1])*.5
    return np.where(ok&visible&(abs(delta)>=.5),np.sign(delta),0).astype('i1')

def read_camera(frames,wave,mode=2):
    if frames.shape[-3:]!=(8,15,27): raise ValueError('8 x 15 x 27 raw camera frames required')
    return read_features(optical_features(frames),wave,mode)

def paid_inputs(i):
    rng=np.random.default_rng([61025015,i])
    shape=(2,N,4,2)
    order=rng.random(shape[:-1])<.5
    target=rng.integers(0,2,size=shape[:-1],dtype='u1')
    def perm(x):return np.where(order[...,None],x[...,::-1],x)
    def sign(bit,active):return np.where(active,1-2*bit.astype('i1'),0).astype('i1')
    ac=perm(rng.random(shape)<.5);ref=perm(rng.random(shape)<.5);ind=perm(rng.random(shape)<Q)
    shared=rng.random(shape[:-1]); apparent=rng.random(shape[:-1])<Q
    h=BOARDS['latch'][np.arange(N)%8][None,:,None,None]
    latched=rng.random(shape[:-1]+(2,))<h; fake=rng.random(shape[:-1]+(2,))<Q
    err=BOARDS['sign_error'][np.arange(N)%8][None,:,None,None,None]
    er=rng.random(shape+(2,))<err; ei=rng.random(shape+(2,))<err
    reference=sign(ref[...,None]^er,ac[...,None])
    truth=np.empty((2,2)+shape+(2,),dtype='i1')
    electronics=np.empty((2,2,2)+shape+(2,2),dtype='i1')
    for z,moment in enumerate((.145,.0925)):
        upstream=np.where((shared<((moment-Q*Q)/(Q*(1-Q))))[...,None],apparent[...,None],ind)
        for cause,state in enumerate((upstream,ind)):
            truth[cause,z]=np.stack([sign(ref,ac),sign(ref^state,ac)],-1)
            for reset in (0,1):
                first=np.where(latched,fake,upstream[...,0,None])
                second_latch=latched&~(np.arange(2)==target[...,None]) if reset else latched
                second=np.where(second_latch,fake,upstream[...,1,None])
                impact=sign(ref[...,None]^np.stack([first,second],-2)^ei,ac[...,None])
                electronics[reset,cause,z]=np.stack([reference,impact],-1)
    return truth,electronics,target,order

def timebase(rng,bid,impact):
    n=len(bid)
    jitter=np.clip(rng.normal(0,.15,n),-.37,.37)
    trigger=BOARDS['camera_skew'][bid]+jitter
    led=np.maximum(0.,BOARDS['led_driver'][bid]+rng.normal(0,.08,n))
    origin=rng.uniform(-.55,.55,n)
    sw=origin+BOARDS['switch_bias'][bid]+rng.normal(0,.028,n)
    diode=origin+led+BOARDS['diode_bias'][bid]+rng.normal(0,.027,n)
    sw_gain=BOARDS['gain_switch'][bid]*(rng.random(n)>BOARDS['drop_switch'][bid])
    pd_gain=BOARDS['gain_diode'][bid]*(rng.random(n)>BOARDS['drop_diode'][bid])
    analog=rng.normal(24.,BOARDS['adc_noise'][bid,None,None],(n,2,128))
    def signal(at,slope):
        return np.clip((ADC[None,:]-at[:,None])/slope,0,1)
    rise=signal(sw,.065)
    # A short bounce after the switch edge need not recur in the optical waveform.
    rebound=((ADC[None,:]>sw[:,None]+.18)&(ADC[None,:]<sw[:,None]+.24))
    analog[:,0]+=(rise-.50*rebound)*sw_gain[:,None]
    waves=[]
    for offset in (0,2):
        event=diode+offset*impact
        pulse=signal(event,.055)-signal(event+.40,.045)
        w=analog.copy();w[:,1]+=pulse*pd_gain[:,None]
        waves.append(np.rint(np.clip(w,0,255)).astype('u1'))
    return trigger,led,waves

def scene(rng,n,impact,trigger_error,board):
    # V60 upper body renderer; replace its temporal paths with actual exposure times.
    bg,y,x,_,visible=film_base(rng,n,impact)
    bg[:,:,13:,:]=rng.normal(24,7,(n,12,2,27))
    amp=rng.uniform(.25,2.8,(n,1))
    phase=rng.uniform(-.45,.45,(n,1))
    paths=[]
    for dt in (-.125,.125):
        t=TIMES+trigger_error[:,None]+dt-phase
        ramp=np.clip((t-1.15)/1.9,0,1)
        pulse=np.clip((t-1.2)/1.5,0,1)*np.exp(-.5*((t-3.15)/1.25)**2)
        paths.append((amp*np.where(impact[:,None],pulse,ramp)).astype('f4'))
    return (bg,y,x,paths,visible), BOARDS['led_gain'][board]*visible*rng.uniform(.8,1.1,n)

def subtract_base(base,mask):
    return (base[0][mask],base[1][mask],base[2][mask],[v[mask] for v in base[3]],base[4][mask])

def save(group,i,**kwargs):
    dst=ROOT/group;dst.mkdir(exist_ok=True)
    tmp=dst/f'_tmp_{i:02}.npz';np.savez_compressed(tmp,**kwargs);tmp.replace(dst/f'panel_{i:02}.npz')

def make(i):
    truth,el,target,order=paid_inputs(i)
    camera=np.empty((4,3)+truth.shape,dtype='i1')
    frames=np.empty((4,2,2,2,4,2,2,8,15,27),dtype='u1')
    wave0=np.empty((4,2,4,2,2,2,128),dtype='u1')
    truth0=np.empty((4,2,4,2,2,2),dtype='f4')
    timing=np.empty((4,2,N,4,2,2,4),dtype='f4') # unique clips: truth and estimates
    fixture_e=i%8
    for mat in range(2):
        rng=np.random.default_rng([6107361,i,mat])
        src=truth[:,:,mat].reshape(2,2,-1)
        out=camera[:,:,:,:,mat].reshape(4,3,2,2,-1)
        for j in range(0,src.shape[-1],128):
            ix=np.arange(j,j+128); pixel_ix=np.arange(128)
            is_i=ix%2==1;board=(ix//16)%8
            trig,lag,waves=timebase(rng,board,is_i)
            base,strength=scene(rng,128,is_i,trig,board)
            actual=src[:,:,j:j+128];s0=actual[0,0];change=(actual!=s0).any((0,1))
            body=emit(base,s0);f0=optical_features(body)
            other=body.copy()
            if change.any():other[change]=emit(subtract_base(base,change),-s0[change])
            f1=optical_features(other)
            for cell,(lo,to) in enumerate(CELLS):
                lt=lag+lo*is_i;tt=trig+to*is_i
                pulse=np.rint(2+lt-trig).astype(int)
                steps=to*is_i[:,None]+np.arange(8)
                rows=body[:,:,13:].copy()
                strength_i=strength[:,None,None]
                rows[pixel_ix,pulse,:,1:5]=np.clip(rows[pixel_ix,pulse,:,1:5]+strength_i,0,255).astype('u1')
                led=rows[:,:,:,1:5].mean((-1,-2))-rows[:,:,:,7:11].mean((-1,-2))
                lbuf=np.take_along_axis(led,steps,1)
                def sliced(f):return (np.take_along_axis(f[0],steps,1),np.take_along_axis(f[1],steps,1),lbuf)
                feat=sliced(f0);alt=sliced(f1);w=waves[lo//2]
                for mode in range(3):
                    a=read_features(feat,w,mode);b=read_features(alt,w,mode)
                    out[cell,mode,:,:,j:j+128]=np.where(actual==s0,a,b)
                _,_,lh,th=estimate(feat,w)
                timing[cell,mat].reshape(-1,4)[j:j+128]=np.stack([lt,tt,lh,th],-1)
                use=(ix//16)==fixture_e
                if use.any():
                    for c,z in np.ndindex(2,2):
                        f=body[use].copy()
                        flip=(actual[c,z]!=s0)[use]
                        f[flip]=other[use][flip]
                        f[:,:,13:]=rows[use]
                        sub=f[np.arange(16)[:,None],steps[use]]
                        frames[cell,c,z,mat]=sub.reshape(4,2,2,8,15,27)
                    wave0[cell,mat]=w[use].reshape(4,2,2,2,128)
                    truth0[cell,mat]=np.stack([lt[use],tt[use]],-1).reshape(4,2,2,2)
    codes=np.empty((4,3,2,2,2,2,N,4),dtype='u1')
    for d,s,r in np.ndindex(4,3,2):codes[d,s,r]=encode(el[r],camera[d,s],target)
    for d,s in np.ndindex(4,3):
        np.testing.assert_array_equal(read_camera(frames[d],wave0[d],s),camera[d,s,:,:,:,fixture_e])
    for ch in [0,1]:
        blind=wave0.copy();blind[...,ch,:]=0
        assert not read_camera(frames[0],blind[0],2).any()
    save('pilots',i,codes=codes,board=np.arange(N)%8)
    save('fixtures',i,frames=frames,wave=wave0,true_delays=truth0,truth=truth[:,:,:,fixture_e],
         electronic=el[:,:,:,:,fixture_e],reset_id=target[:,fixture_e],episode=fixture_e,board=fixture_e)
    save('vision_audit',i,truth=truth,camera=camera,timing=timing,order=order)
    print('catalog',i,flush=True)

if __name__=='__main__':
    assert os.getenv('SLURM_JOB_ID') and os.getenv('CUDA_VISIBLE_DEVICES')==''
    for index in map(int,sys.argv[1:]):make(index)
