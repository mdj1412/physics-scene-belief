"""Synthetic reflective panel and independent differential strain gauge."""
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'exposure_abstention_v62'))
import vision62 as camera_model
from track_stress62 import move_roi

N = 256
CELLS = ((0., 0.), (.35, 0.), (0., .35), (.35, .35))
PULSE, EXPOSURE = .80, .25
read_camera = camera_model.read_camera
encode = camera_model.encode

def boards():
    ranges = dict(switch_bias=(.018,.17), diode_bias=(.012,.09), camera_skew=(-.23,.23),
                  led_driver=(.06,.32), gain_switch=(85.,150.), gain_diode=(80.,150.),
                  adc_noise=(2.,7.), led_gain=(80.,160.), drop_switch=(.003,.03),
                  drop_diode=(.008,.05), latch=(.16,.36), sign_error=(.013,.033))
    r = np.random.default_rng(63189627)
    return {k:r.uniform(lo,hi,8) for k,(lo,hi) in ranges.items()}

BOARDS = boards()
STRAIN_GAIN = np.random.default_rng(632043).uniform(8.,13.,8)

def surface(i, mat):
    r = np.random.default_rng([630426,i,mat])
    return dict(shine=r.uniform(7.,27.), lateral=r.uniform(6.,21.), ghost=r.uniform(.86,.99),
                coupling=r.uniform(.55,1.35))

def read_strain(wave):
    # Two differential bridge wires, independent ADC. No camera features/metadata.
    diff = wave[...,0,:].astype('f4') - wave[...,1,:].astype('f4')
    disp = diff[...,8:].mean(-1) - diff[...,:4].mean(-1)
    return np.where(abs(disp)>=5., np.sign(disp), 0).astype('i1')

def strain_wave(r, signs, amp, board, surf):
    n=signs.shape[-1]
    noise=r.normal(0,3.2,(n,2,12))
    drift=r.normal(0,1.3,(n,2,1))*np.linspace(-.5,.5,12)
    offset=r.uniform(85,145,(n,2,1))
    ramp=np.clip((np.arange(12)-3)/4.,0,1)
    gain=STRAIN_GAIN[board]*surf['coupling']*amp[:,0]*(r.random(n)>.035)
    # Dropout disconnects both legs to the DC bias with readout noise intact.
    common=r.normal(0,1.8,(n,1,12))
    signed=signs[...,None,None]*gain[:,None,None]*np.array([.5,-.5])[:,None]*ramp
    return np.rint(np.clip(offset+noise+drift+common+signed,0,255)).astype('u1')

def disturb(body, coherent, transient, shift, surf):
    film=body.copy()
    if coherent.any():
        gap, valid, _=camera_model.optical_features(film[coherent])
        mirror=np.where(valid,np.clip(-2*(gap-gap[:,:1]),-8,8),0.)
        patch=film[coherent,:,:6,:]
        ghost=move_roi(patch,mirror)
        film[coherent,:,:6,:]=np.rint((1-surf['ghost'])*patch+surf['ghost']*ghost).astype('u1')
    if transient.any():
        patch=film[transient,:,:6,:]
        film[transient,:,:6,:]=move_roi(patch,shift[transient])
    return film

def save(group,i,**fields):
    directory=ROOT/group
    directory.mkdir(exist_ok=True)
    tmp=directory/f'tmp_{i:02}.npz'
    np.savez_compressed(tmp,**fields)
    tmp.replace(directory/f'panel_{i:02}.npz')

def make(i):
    prev=camera_model.prev
    prev.N=N
    prev.BOARDS=BOARDS
    camera_model.BOARDS=BOARDS
    truth,elec,targets,order=prev.paid_inputs(630000+i)
    fixture=i%8
    schedule=np.argsort(np.random.default_rng([630373,i]).random((2,N,8)),axis=-1)//2
    schedule=schedule.reshape(2,N,4,2)
    camera=np.empty((4,3)+truth.shape,'i1')
    gauge=np.empty_like(truth)
    events=np.zeros((2,N,4,2,2),bool)
    frames=np.empty((4,2,2,2,4,2,2,8,15,27),'u1')
    waves=np.empty((4,2,4,2,2,2,128),'u1')
    strain=np.empty((2,2,2,4,2,2,2,12),'u1')
    for m in range(2):
        r=np.random.default_rng([6307351,i,m])
        rf=np.random.default_rng([6309211,i,m])
        rs=np.random.default_rng([6305911,i,m])
        surf=surface(i,m)
        signs=truth[:,:,m].reshape(2,2,-1)
        lag=np.repeat(schedule[m].reshape(-1),2)
        cam=np.empty((4,3,2,2,signs.shape[-1]),'i1')
        sensor=np.empty_like(signs)
        for j in range(0,signs.shape[-1],128):
            ix=np.arange(j,j+128)
            impact=ix%2==1
            board=ix//16%8
            t,led,w=camera_model.instrument(r,board,impact,lag[ix])
            wave=w[PULSE]
            bg,y,x,amp,phase,power=camera_model.scene(r,len(ix),impact,board)
            bg=bg[:,:8]; y=y[:,:8]; x=x[:,:8]
            # Static surface highlight is independent of marker direction and acquisition cell.
            glow=surf['shine']*np.exp(-.5*((np.arange(27)-surf['lateral'])/2.2)**2)
            bg[:,:,2:5,:]+=glow
            center=camera_model.TIMES[:8]+t[:,None]
            dt=np.maximum(0,np.minimum(center+EXPOSURE/2,led[:,None]+PULSE)
                          -np.maximum(center-EXPOSURE/2,led[:,None]))
            bg[:,:,13:,1:5]+=(power[:,None]*dt/EXPOSURE)[:,:,None,None]
            u=rf.random((len(ix),2))
            event=(u<.35)&impact[:,None]
            events[m].reshape(-1,2)[ix[impact]//2]=event[impact]
            shift=(rf.integers(8,size=(len(ix),1))==np.arange(8))*rf.choice([-3,3],(len(ix),1))
            target=signs[:,:,ix]
            gauge_wave=strain_wave(rs,target,amp,board,surf)
            sensor[:,:,ix]=read_strain(gauge_wave)
            keep=ix//16==fixture
            if keep.any(): strain[:,:,m]=gauge_wave[:,:,keep].reshape(2,2,4,2,2,2,12)
            first=target[0,0]
            changed=(target!=first).any((0,1))
            args=(bg,y,x,amp,phase,impact,center)
            body=camera_model.render(*args, EXPOSURE, first)
            alt=body.copy()
            if changed.any():
                alt[changed]=camera_model.render(*(v[changed] for v in args),EXPOSURE,-first[changed])
            for d,(p,q) in enumerate(CELLS):
                coherent=event[:,1]&(q>0)
                transient=event[:,0]&(p>0)
                film=disturb(body,coherent,transient,shift,surf)
                other=film.copy()
                if changed.any():
                    other[changed]=disturb(alt[changed],coherent[changed],transient[changed],shift[changed],surf)
                a=camera_model.optical_features(film)
                b=camera_model.optical_features(other)
                for mode in range(3):
                    s=camera_model.read_features(a,wave,mode)
                    so=camera_model.read_features(b,wave,mode)
                    cam[d,mode,:,:,j:j+128]=np.where(target==first,s,so)
                if keep.any():
                    for c,z in np.ndindex(2,2):
                        entire=np.where((target[c,z,keep]==first[keep])[:,None,None,None],film[keep],other[keep])
                        frames[d,c,z,m]=entire.reshape(4,2,2,8,15,27)
                    waves[d,m]=wave[keep].reshape(4,2,2,2,128)
        camera[:,:,:,:,m]=cam.reshape((4,3,2,2)+truth.shape[3:])
        gauge[:,:,m]=sensor.reshape((2,2)+truth.shape[3:])
    codes=np.empty((4,3,2,2,2,2,N,4),'u1')
    for d,s,r in np.ndindex(4,3,2): codes[d,s,r]=encode(elec[r],camera[d,s],targets)
    for d,s in np.ndindex(4,3):
        np.testing.assert_array_equal(read_camera(frames[d],waves[d],s),camera[d,s,:,:,:,fixture])
    np.testing.assert_array_equal(read_strain(strain),gauge[:,:,:,fixture])
    assert np.all((camera[:,2]==0)|(camera[:,2]==camera[:,1]))
    save('pilots',i,codes=codes,board=np.arange(N)%8)
    save('vision_audit',i,truth=truth,camera=camera,validation_sign=gauge,events=events,
         electronic=elec,reset_id=targets,lag_id=schedule,order=order)
    save('fixtures',i,frames=frames,wave=waves,electronic=elec[:,:,:,:,fixture],reset_id=targets[:,fixture],
         board=fixture,episode=fixture,truth=truth[:,:,:,fixture],strain_wave=strain,events=events[:,fixture])
    print('catalog',i,flush=True)

if __name__=='__main__':
    assert os.getenv('SLURM_JOB_ID') and os.getenv('CUDA_VISIBLE_DEVICES')==''
    for n in sys.argv[1:]: make(int(n))
