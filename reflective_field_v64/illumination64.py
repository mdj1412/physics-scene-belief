"""Simulated paired directional light on identical paid contacts; no field data."""
import os, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'reflective_tracking_v63'))
import optics63 as prev

N=256
ARMS=[(angle,width) for width in (8.,26.) for angle in (-30.,30.)]
cam=prev.camera_model

def perturb(body,alpha,transient,shift):
    film=body.copy()
    hit=alpha>0
    if hit.any():
        gap,valid,_=cam.optical_features(body[hit])
        mirror=np.where(valid,np.clip(-2*(gap-gap[:,:1]),-8,8),0.)
        patch=body[hit,:,:6,:]
        ghost=prev.move_roi(patch,mirror)
        blend=alpha[hit,None,None,None]
        film[hit,:,:6,:]=np.rint((1-blend)*patch+blend*ghost).astype('u1')
    if transient.any():
        film[transient,:,:6,:]=prev.move_roi(film[transient,:,:6,:],shift[transient])
    return film

def save(group,i,**arrays):
    where=ROOT/group
    where.mkdir(exist_ok=True)
    temp=where/f'tmp_{i:02}.npz'
    np.savez_compressed(temp,**arrays)
    temp.replace(where/f'panel_{i:02}.npz')

def panel(i):
    # Reuse v63 boards/specification; capture and contact draws are new.
    cam.prev.N=N;cam.prev.BOARDS=prev.BOARDS;cam.BOARDS=prev.BOARDS
    truth,elec,targets,order=cam.prev.paid_inputs(640000+i)
    episode=i%8
    schedule=np.argsort(np.random.default_rng([640373,i]).random((2,N,8)),axis=-1)//2
    schedule=schedule.reshape(2,N,4,2)
    camera=np.empty((4,3)+truth.shape,'i1')
    gauge=np.empty_like(truth)
    events=np.zeros((2,N,4,2,2),bool)
    strengths=np.zeros((4,2,N,4,2),'f4')
    frames=np.empty((4,2,2,2,4,2,2,8,15,27),'u1')
    waves=np.empty((4,2,4,2,2,2,128),'u1')
    for m in (0,1):
        rg=np.random.default_rng([6407351,i,m])
        rf=np.random.default_rng([6409211,i,m])
        rs=np.random.default_rng([6405911,i,m])
        surf=prev.surface(i,m)
        mean_tilt=rf.uniform(-18,18)
        signs=truth[:,:,m].reshape(2,2,-1)
        lag=np.repeat(schedule[m].reshape(-1),2)
        reads=np.empty((4,3)+signs.shape,'i1'); sensor=np.empty_like(signs)
        for j in range(0,signs.shape[-1],128):
            ix=np.arange(j,j+128);impact=ix%2==1;board=ix//16%8
            t,led,w=cam.instrument(rg,board,impact,lag[ix]);wave=w[.80]
            bg,y,x,amp,phase,power=cam.scene(rg,len(ix),impact,board)
            bg=bg[:,:8]; y=y[:,:8]; x=x[:,:8]
            bg[:,:,2:5,:]+=surf['shine']*np.exp(-.5*((np.arange(27)-surf['lateral'])/2.2)**2)
            center=cam.TIMES[:8]+t[:,None]; exp=.25
            dt=np.maximum(0,np.minimum(center+exp/2,led[:,None]+.80)-np.maximum(center-exp/2,led[:,None]))
            bg[:,:,13:,1:5]+=(power[:,None]*dt/exp)[:,:,None,None]
            opportunity=(rf.random(len(ix))<.35)&impact
            transient=(rf.random(len(ix))<.15)&impact
            events[m].reshape(-1,2)[ix[impact]//2]=np.stack([opportunity[impact],transient[impact]],-1)
            facet=mean_tilt+rf.normal(0,9,len(ix))
            shift=(rf.integers(8,size=(len(ix),1))==np.arange(8))*rf.choice([-3,3],(len(ix),1))
            target=signs[:,:,ix]; first=target[0,0]; changed=(target!=first).any((0,1))
            sensor[:,:,ix]=prev.read_strain(prev.strain_wave(rs,target,amp,board,surf))
            args=(bg,y,x,amp,phase,impact,center)
            body=cam.render(*args,exp,first); alt=body.copy()
            if changed.any():
                alt[changed]=cam.render(*(a[changed] for a in args),exp,-first[changed])
            keep=ix//16==episode
            for d,(angle,width) in enumerate(ARMS):
                alpha=.96*opportunity*np.exp(-.5*((facet-angle/2)/width)**2)
                strengths[d,m].reshape(-1)[ix[impact]//2]=alpha[impact]
                film=perturb(body,alpha,transient,shift)
                other=film.copy()
                if changed.any():
                    other[changed]=perturb(alt[changed],alpha[changed],transient[changed],shift[changed])
                a=cam.optical_features(film);b=cam.optical_features(other)
                for mode in (0,1,2):
                    s=cam.read_features(a,wave,mode);ss=cam.read_features(b,wave,mode)
                    reads[d,mode,:,:,j:j+128]=np.where(target==first,s,ss)
                if keep.any():
                    for c,z in np.ndindex(2,2):
                        entire=np.where((target[c,z,keep]==first[keep])[:,None,None,None],film[keep],other[keep])
                        frames[d,c,z,m]=entire.reshape(4,2,2,8,15,27)
                    waves[d,m]=wave[keep].reshape(4,2,2,2,128)
        camera[:,:,:,:,m]=reads.reshape((4,3,2,2)+truth.shape[3:])
        gauge[:,:,m]=sensor.reshape((2,2)+truth.shape[3:])
    codes=np.empty((4,3,2,2,2,2,N,4),'u1')
    for d,s,r in np.ndindex(4,3,2):
        codes[d,s,r]=cam.encode(elec[r],camera[d,s],targets)
    for d,s in np.ndindex(4,3):
        np.testing.assert_array_equal(cam.read_camera(frames[d],waves[d],s),camera[d,s,:,:,:,episode])
    assert np.all((camera[:,2]==0)|(camera[:,2]==camera[:,1]))
    save('pilots',i,codes=codes,board=np.arange(N)%8)
    save('fixtures',i,frames=frames,wave=waves,electronic=elec[:,:,:,:,episode],reset_id=targets[:,episode],episode=episode,board=episode)
    save('paired_sim',i,truth=truth,camera=camera,validation_sign=gauge,events=events,alpha=strengths)
    print('catalog',i,flush=True)

if __name__=='__main__':
    assert os.getenv('SLURM_JOB_ID') and os.getenv('CUDA_VISIBLE_DEVICES')==''
    for idx in sys.argv[1:]:panel(int(idx))
