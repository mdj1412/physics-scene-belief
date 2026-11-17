"""Generate future binary outcomes using Bullet, with v50 contours and new material fields."""
import argparse
import os
import sys
import numpy as np
import world50 as world

ROOT=world.ROOT
V49=ROOT.parent/'contact_transfer_v49'
sys.path.insert(0,str(V49))
import solver49 as mechanics
import solver49_ultra_all as finer

REGIME={'strong':1.,'weak':.3,'null':0.,'global_null':0.}


def friction(mu,h,velocity,surface,law,sign,regime):
    scale,orientation=world.surface_field_mm(h,surface)
    v=abs(velocity)
    if law=='rational':
        rate=1.+.095*np.exp(-v/.37)
    elif law=='gaussian_rate':
        rate=.96+.105*np.tanh(v/.45)
    else:
        raise ValueError(law)
    return mu*rate*scale*(1.+.045*orientation*sign*REGIME[regime]*np.tanh(velocity/.34))


def generate(i,resolution):
    tag=f'panel_{i:02d}.npz'
    dest=ROOT/('labels50' if resolution=='fine' else 'labels50_ultra')/tag
    dest.parent.mkdir(exist_ok=True)
    engine=mechanics if resolution=='fine' else finer
    engine.loaded_shape=world.loaded_shape
    sim=engine.Impact(True)
    with np.load(ROOT/'worlds'/tag) as a:
        theta,idx,sf,laws,sign,regime=[a[k] for k in ('theta','object_index','surface','law','signs','regimes')]
    speeds=np.empty((4,2,3,2,3,4,128),dtype=np.float32)
    normal=np.empty_like(speeds)
    curv=np.round(theta[...,3]/.05)*.05
    for fi,family in enumerate(world.FAMILIES):
        for hi,h in enumerate(world.HEIGHTS):
            u,w=engine.velocity(theta[...,2]+theta[...,3]*h/25)
            for j,state in enumerate(world.STATES):
                if fi and state==0:
                    x=np.linspace(h-2.5,h+2.5,401)
                    np.testing.assert_array_equal(world.loaded_shape(x,h,family,0)[0],
                                                  world.loaded_shape(x,h,world.FAMILIES[0],0)[0])
                    speeds[fi,hi,j]=speeds[0,hi,j]
                    normal[fi,hi,j]=normal[0,hi,j]
                    continue
                for k in np.unique(curv):
                    terrain=sim.terrain(family,h,state,float(k))
                    envs,objects=np.nonzero(curv==k)
                    for ri,rad in enumerate(world.RADII):
                        for oi,off in enumerate(world.OFFSETS):
                            site=sim.site(terrain,off,rad)
                            for e,n in zip(envs,objects):
                                coeff=friction(theta[e,n,4],h+off,u[e,n],sf[e],laws[e],sign[e],regime[e])
                                vt,vn=sim.trial(terrain[0],site,rad,u[e,n],w[e,n],coeff,theta[e,n,5])
                                speeds[fi,hi,j,ri,oi,e,n]=vt
                                normal[fi,hi,j,ri,oi,e,n]=vn
    sim.close()
    assert np.isfinite(speeds).all() and np.isfinite(normal).all()
    tmp=dest.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,labels=speeds<=.020,slip_speed=speeds,separation_speed=normal,
                        object_index=idx,axes=engine.AXES,state=world.STATES,radii_mm=world.RADII,
                        heights_mm=world.HEIGHTS,families=world.FAMILIES,offsets_mm=world.OFFSETS,
                        bullet_api=engine.pb.getAPIVersion(),dt_ms=sim.dt,edge_mm=sim.step)
    os.replace(tmp,dest)
    print(i,resolution,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('resolution',choices=['fine','ultra'])
    p.add_argument('panels',nargs='+',type=int)
    opts=p.parse_args()
    assert os.getenv('SLURM_JOB_ID') and (ROOT/'solver50.sha256').exists()
    for panel in opts.panels:
        generate(panel,opts.resolution)
