"""Secondary external engine replay on a fixed balanced subset of the primary worlds."""
from itertools import product
import json
import os
from pathlib import Path
import sys

import numpy as np

H=Path(__file__).resolve().parent
PROJECT=H.parent
V66=PROJECT/'invariance_audit_v66'/'locked'
sys.path[:0]=[str(PROJECT/'contact_patch_v44'),str(V66),str(PROJECT/'law_axis_v67')]
import solver_full44 as simulator
from locked_geometry import profile
from score67 import law as new_mu

FAMILIES=('wendland','dispersive')
SURFACES=('inlay','woven')
LAWS=('strengthening','weakening')
VARIANTS=('base','hollow','bulge')
HEIGHTS=(19,31)
RADII=(.25,2.)
OFFSETS=(-.5,0,.5)
simulator.displacement=lambda h,f,v:profile(h,f,v)[0]
simulator.HORIZON=.0008


def material(theta,h,loc,g,surface,law,signs,reg):
    speed=(simulator.INCOMING+g)/np.hypot(1,g)
    # Each object's height differs; original material() accepts a common height vector.
    # Its diagonal corresponds to each object's own site.
    mu=new_mu(theta[:,4],speed[:,None],h+loc,surface,law,signs,reg)
    return np.diagonal(mu).copy()


def run_panel(k, fine):
    sz=2 if fine else 8
    with np.load(V66/'results'/'worlds'/f'panel_{k:02d}.npz') as z:
        nenv=1 if fine else 4
        theta=z['theta'][:nenv,:sz].reshape(-1,6)
        signs=np.repeat(z['signs'][:nenv],sz); regimes=np.repeat(z['regimes'][:nenv],sz)
    dt=1.e-5 if fine else 2.e-5
    out=H/('bullet_fine' if fine else 'bullet_labels')/f'panel_{k:02d}'
    out.mkdir(parents=True,exist_ok=True)
    for f,v,h in product(FAMILIES,VARIANTS,HEIGHTS):
        files={(s,l):out/f'{f}_{s}_{l}_{v}_h{h}.npz' for s,l in product(SURFACES,LAWS)}
        if all(p.exists() for p in files.values()): continue
        shape=(2,3,nenv,sz)
        res={cell:dict(speed=np.full(shape,np.nan),valid=np.zeros(shape,bool),loaded_end=np.zeros(shape,bool)) for cell in files}
        for off,d in enumerate(OFFSETS):
            obj=simulator.Surface(theta,h+d,f,v,dt,.10)
            for i,r in enumerate(RADII):
                _,site,g=obj.support(r,0.)
                for (s,l),record in res.items():
                    mu=material(theta,h+d,site,g,s,l,signs,regimes)
                    result=obj.simulate(r,0.,mu,theta[:,5])
                    for key,field in [('speed','slip_speed'),('valid','valid'),('loaded_end','loaded_end')]:
                        record[key][i,off]=result[field].reshape(nenv,sz)
        for key,path in files.items():
            tmp=path.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,**res[key],theta=theta.reshape(nenv,sz,6))
            os.replace(tmp,path)


if __name__=='__main__':
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    # Result generator is released only when all primary forecasts have been sealed.
    seal=H/os.environ['V68_SEAL_BASENAME']
    assert seal.is_file() and seal.stat().st_size
    simulator.b.connect(simulator.b.DIRECT)
    for idx in map(int,sys.argv[1:]):
        run_panel(idx,False);run_panel(idx,True)
        print('completed both resolutions',idx,flush=True)
    simulator.b.disconnect()
