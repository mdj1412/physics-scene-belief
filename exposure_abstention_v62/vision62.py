"""Finite exposure imaging of contact and a pre-specified abstaining v61 reader."""
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'switch_diode_v61'))
import vision61 as prev
from vision60 import TIMES,X,film_base,optical_features,encode

N=512
CELLS=((.20,.25),(.20,.80),(.80,.25),(.80,.80))
LAG_CELLS=((0,0),(0,2),(2,0),(2,2))

def boards():
    r=np.random.default_rng(62189327)
    ranges=dict(switch_bias=(.018,.17),diode_bias=(.012,.09),camera_skew=(-.23,.23),
                led_driver=(.06,.32),gain_switch=(85.,150.),gain_diode=(80.,150.),
                adc_noise=(2.,7.),led_gain=(80.,160.),drop_switch=(.003,.03),
                drop_diode=(.008,.05),latch=(.16,.36),sign_error=(.013,.033))
    return {key:r.uniform(lo,hi,8) for key,(lo,hi) in ranges.items()}
BOARDS=boards()

def inputs(c):
    # The contact acquisition model is unchanged; perturb both its draw and sensor serial.
    # Explicitly restore the imported module's table after the call.
    original=prev.BOARDS
    try:
        prev.BOARDS=BOARDS
        return prev.paid_inputs(620000+c)
    finally:
        prev.BOARDS=original

def lags(c):
    rng=np.random.default_rng([62065171,c])
    permutations=np.argsort(rng.random((2,N,8)),axis=-1)
    return np.take(np.repeat(np.arange(4,dtype='u1'),2),permutations).reshape(2,N,4,2)

def read_features(features,wave,mode=2):
    if mode==0: return prev.read_features(features,wave,0)
    if mode==1: return prev.read_features(features,wave,2)
    if mode!=2: raise ValueError('Reader mode must be nominal (0), v61 dual (1), or selective (2)')
    gap,visible,led=features
    start,ok,_,_=prev.estimate(features,wave,2)
    ix=start[...,None]+[0,1,3,4]
    g=np.take_along_axis(gap,ix,-1)
    visible=np.take_along_axis(visible,ix,-1).all(-1)
    delta=(g[...,2]+g[...,3]-g[...,0]-g[...,1])*.5
    sign=np.sign(delta)
    peaks=np.partition(led,-2,axis=-1)[...,-2:]
    # Two independently indexed displacements agree even after the marginal pattern is weakened.
    stable=((g[...,2]-g[...,0])*sign>=.20)&((g[...,3]-g[...,1])*sign>=.20)
    keep=ok & visible & (abs(delta)>=.75)&stable&(peaks[...,1]>=48)&(peaks[...,0]<=28)
    return np.where(keep,sign,0).astype('i1')

def read_camera(frames,wave,mode=2):
    if frames.shape[-3:]!=(8,15,27): raise ValueError('Expected 8 x 15 x 27 frame clip')
    return read_features(optical_features(frames),wave,mode)

def instrument(rng,bid,impact,lag_id):
    n=len(bid); offs=np.asarray(LAG_CELLS)[lag_id]*impact[:,None]
    jitter=np.clip(rng.normal(0,.15,n),-.37,.37)
    camera=BOARDS['camera_skew'][bid]+jitter+offs[:,1]
    led=np.maximum(0,BOARDS['led_driver'][bid]+rng.normal(0,.08,n))+offs[:,0]
    origin=rng.uniform(-.55,.55,n)
    sw=origin+BOARDS['switch_bias'][bid]+rng.normal(0,.028,n)
    pd=origin+led+BOARDS['diode_bias'][bid]+rng.normal(0,.027,n)
    gs=BOARDS['gain_switch'][bid]*(rng.random(n)>BOARDS['drop_switch'][bid])
    gd=BOARDS['gain_diode'][bid]*(rng.random(n)>BOARDS['drop_diode'][bid])
    def rise(at,width):
        return np.clip((prev.ADC[None,:]-at[:,None])/width,0,1)
    base=rng.normal(24,BOARDS['adc_noise'][bid,None,None],(n,2,128))
    bounce=(prev.ADC[None,:]>sw[:,None]+.18)&(prev.ADC[None,:]<sw[:,None]+.24)
    base[:,0]+=(rise(sw,.065)-.5*bounce)*gs[:,None]
    waves={}
    for width in (.20,.80):
        w=base.copy()
        w[:,1]+=(rise(pd,.055)-rise(pd+width,.045))*gd[:,None]
        waves[width]=np.rint(np.clip(w,0,255)).astype('u1')
    return camera,led,waves

def scene(rng,n,impact,board):
    bg,y,x,_,visible=film_base(rng,n,impact)
    bg[:,:,13:,:]=rng.normal(24,7,(n,12,2,27))
    amp=rng.uniform(.25,2.8,(n,1));phase=rng.uniform(-.45,.45,(n,1))
    strength=BOARDS['led_gain'][board]*visible*rng.uniform(.8,1.1,n)
    return bg,y,x,amp,phase,strength

def render(bg,y,x,amp,phase,impact,center,exposure,sign):
    # Four midpoint width-normalized photon integrations of the translated marker PSF.
    # bg and fixture marker are stationary within each frame exposure.
    phot=np.zeros_like(bg[:,:,:13])
    for q in (-.375,-.125,.125,.375):
        t=center+q*exposure-phase
        ramp=np.clip((t-1.15)/1.9,0,1)
        pulse=np.clip((t-1.2)/1.5,0,1)*np.exp(-.5*((t-3.15)/1.25)**2)
        move=amp*np.where(impact[:,None],pulse,ramp)
        tip=x+sign[:,None]*move
        tx=np.exp(-.5*((X[None,None,:]-tip[:,:,None])/1.15)**2)
        phot+=.25*y*tx[:,:,None,:]
    image=bg.copy();image[:,:,:13]+=phot
    return np.rint(np.clip(image,0,255)).astype('u1')

def save(group,i,**data):
    dst=ROOT/group;dst.mkdir(exist_ok=True)
    temp=dst/f'_tmp_{i:02}.npz'
    np.savez_compressed(temp,**data)
    temp.replace(dst/f'panel_{i:02}.npz')

def make(i):
    truth,elec,targets,order=inputs(i)
    schedule=lags(i)
    terms=truth.shape
    camera=np.empty((4,3)+terms,dtype='i1')
    frames=np.empty((4,2,2,2,4,2,2,8,15,27),dtype='u1')
    raw_w=np.empty((4,2,4,2,2,2,128),dtype='u1')
    actual_delay=np.empty((2,N,4,2,2,2),dtype='f4')
    fixture=i%8
    expected=np.broadcast_to(np.arange(4).repeat(2),(2,N,8))
    assert np.array_equal(np.sort(schedule.reshape(2,N,8),axis=-1),expected)
    for mat in range(2):
        rng=np.random.default_rng([6207361,i,mat])
        signs=truth[:,:,mat].reshape(2,2,-1)
        out=camera[:,:,:,:,mat].reshape(4,3,2,2,-1)
        lag=np.repeat(schedule[mat].reshape(-1),2)
        for j in range(0,signs.shape[-1],128):
            ix=np.arange(j,j+128);impact=ix%2==1;board=(ix//16)%8
            t,l,waves=instrument(rng,board,impact,lag[ix])
            actual_delay[mat].reshape(-1,2)[ix]=np.stack([l,t],-1)
            bg,y,x,amp,phase,strength=scene(rng,128,impact,board)
            # Restrict renderer to captured raw buffer; held-out true timestamps only here.
            bg=bg[:,:8];y=y[:,:8];x=x[:,:8]
            center=TIMES[:8]+t[:,None]
            target=signs[:,:,j:j+128];sgn=target[0,0];changed=(sgn!=target).any((0,1))
            for exposure in (.25,.80):
                body=render(bg,y,x,amp,phase,impact,center,exposure,sgn)
                other=body.copy()
                if changed.any():
                    args=(bg,y,x,amp,phase,impact,center)
                    other[changed]=render(*(v[changed] for v in args),exposure,-sgn[changed])
                a=optical_features(body);b=optical_features(other)
                for pulse in (.20,.80):
                    d=CELLS.index((pulse,exposure))
                    dt=np.maximum(0,np.minimum(center+.5*exposure,l[:,None]+pulse)
                                  -np.maximum(center-.5*exposure,l[:,None]))
                    brightness=strength[:,None]*dt/exposure
                    roi=bg[:,:,13:,:].copy()
                    roi[:,:,:,1:5]+=brightness[:,:,None,None]
                    roi=np.rint(np.clip(roi,0,255)).astype('u1')
                    led=roi[:,:,:,1:5].mean((-1,-2))-roi[:,:,:,7:11].mean((-1,-2))
                    feat=a[:2]+(led,);alt=b[:2]+(led,)
                    w=waves[pulse]
                    for mode in range(3):
                        av=read_features(feat,w,mode);bv=read_features(alt,w,mode)
                        out[d,mode,:,:,j:j+128]=np.where(target==sgn,av,bv)
                    use=ix//16==fixture
                    if use.any():
                        for c,z in np.ndindex(2,2):
                            f=body[use].copy();flip=(target[c,z]!=sgn)[use]
                            f[flip]=other[use][flip];f[:,:,13:]=roi[use]
                            frames[d,c,z,mat]=f.reshape(4,2,2,8,15,27)
                        raw_w[d,mat]=w[use].reshape(4,2,2,2,128)
    codes=np.empty((4,3,2,2,2,2,N,4),dtype='u1')
    for d,s,r in np.ndindex(4,3,2): codes[d,s,r]=encode(elec[r],camera[d,s],targets)
    for d,s in np.ndindex(4,3):
        np.testing.assert_array_equal(read_camera(frames[d],raw_w[d],s),camera[d,s,:,:,:,fixture])
    assert np.all((camera[:,2]==0)|(camera[:,2]==camera[:,1]))
    # Filled masks cannot create a valid sign in either dual reader.
    for channel in (0,1):
        blank=raw_w[0].copy();blank[...,channel,:]=24
        assert not read_camera(frames[0],blank,2).any()
    save('pilots',i,codes=codes,board=np.arange(N)%8)
    save('vision_audit',i,truth=truth,camera=camera,order=order,lag_id=schedule,
         electronic=elec,reset_id=targets,delays=actual_delay)
    save('fixtures',i,frames=frames,wave=raw_w,truth=truth[:,:,:,fixture],electronic=elec[:,:,:,:,fixture],
         reset_id=targets[:,fixture],episode=fixture,board=fixture,lag_id=schedule[:,fixture],
         true_delays=np.broadcast_to(actual_delay[:,fixture],(4,2,4,2,2,2)))
    print('catalog',i,flush=True)

if __name__=='__main__':
    assert os.getenv('SLURM_JOB_ID') and os.getenv('CUDA_VISIBLE_DEVICES')==''
    for s in sys.argv[1:]: make(int(s))
