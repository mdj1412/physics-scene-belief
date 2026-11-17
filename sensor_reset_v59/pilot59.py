"""Paired factorial instrument experiment; strictly separate signed paid inputs from audit truth."""
import argparse
import os
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
Q=.29
MOMENTS=(.145,.0925)
H=.26582278481012656
N=2048
def sign(bit,active):
    return np.where(active,1-2*bit.astype(np.int8),0).astype(np.int8)

def encode(signs,reset_id,use_camera):
    bits=(signs[...,0]!=signs[...,1]).astype(np.uint8)
    active=signs[...,0,0]!=0
    cam=signs[...,2,0]!=0
    flip=reset_id[...,None]
    first=np.where(flip==0,bits[...,0],bits[...,1])
    second=np.where(flip==0,bits[...,1],bits[...,0])
    sym=np.where(active,1+first+2*second+4*use_camera*cam*(1+bits[...,2]),0)
    return (13*sym[...,0]+sym[...,1]).astype('u1')

def generate(catalog):
    rng=np.random.default_rng([59025015,catalog])
    shape=(2,N,4,2)
    swap=rng.random(shape[:-1])<.5
    selected=rng.integers(0,2,size=shape[:-1],dtype=np.uint8)
    def chron(x):
        return np.where(swap[...,None],x[...,::-1],x)
    active=chron(rng.random(shape)<.5)
    ref=chron(rng.random(shape)<.5)
    independent_t=chron(rng.random(shape)<Q)
    common_coin=rng.random(shape[:-1])
    common_bit=rng.random(shape[:-1])<Q
    local=rng.random(shape[:-1]+(2,))<H
    local_bit=rng.random(shape[:-1]+(2,))<Q
    cam_active=active & (rng.random(shape)<.8)
    readshape=shape+(3,)
    read_error=np.array([.02,.02,.03])
    err_r=rng.random(readshape)<read_error
    err_i=rng.random(readshape)<read_error
    obs_active=np.stack([active,active,cam_active],axis=-1)
    paid_r=sign(ref[...,None]^err_r,obs_active)
    outshape=(2,2,2)+shape+(3,2)
    signs=np.empty(outshape,dtype=np.int8)
    truth_signs=np.empty((2,2)+shape+(2,),dtype=np.int8)
    codes=np.empty((2,2,2,2)+shape[:-1],dtype=np.uint8)
    active_chosen=(np.arange(2)==selected[...,None])
    for z,m in enumerate(MOMENTS):
        c=(m-Q**2)/(Q*(1-Q))
        u=np.where((common_coin<c)[...,None],common_bit[...,None],independent_t)
        for cause,true in enumerate((u,independent_t)):
            truth_signs[cause,z]=np.stack([sign(ref,active),sign(ref^true,active)],axis=-1)
            for reset in (0,1):
                first=np.where(local,local_bit,u[...,0,None])
                latch_second=(local & ~active_chosen) if reset else local
                second=np.where(latch_second,local_bit,u[...,1,None])
                electronic=np.stack([first,second],axis=-2)
                source=np.concatenate([electronic,true[...,None]],axis=-1)
                paid_i=sign(ref[...,None]^source^err_i,obs_active)
                paid=np.stack([paid_r,paid_i],axis=-1)
                signs[reset,cause,z]=paid
                for camera in (0,1):
                    codes[reset,camera,cause,z]=encode(paid,selected,camera)
    np.testing.assert_array_equal(codes[:,0,0],codes[:,0,1])
    np.testing.assert_array_equal(signs[0,...,0,:,:],signs[1,...,0,:,:])
    np.testing.assert_array_equal(signs[0,...,2,:],signs[1,...,2,:])
    name=f'panel_{catalog:02d}.npz'
    for folder in ('pilots','pilot_audit'):
        (ROOT/folder).mkdir(exist_ok=True)
    np.savez_compressed(ROOT/'pilots'/name,codes=codes)
    np.savez_compressed(ROOT/'pilot_audit'/name,signs=signs,truth_signs=truth_signs,reset_id=selected,swap=swap)
    print(catalog,flush=True)

if __name__=='__main__':
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    parser=argparse.ArgumentParser()
    parser.add_argument('catalog',type=int,nargs='*')
    args=parser.parse_args()
    for i in args.catalog or range(24):generate(i)
