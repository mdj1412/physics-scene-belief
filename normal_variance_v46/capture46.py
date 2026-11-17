"""Capture source/held-out features, without access to target outcomes."""
import argparse
import hashlib
import itertools
import os
from pathlib import Path
import time

import numpy as np
import torch
from camera46 import render_clip, infer_x, infer_normal_variance, true_stat

ROOT=Path(__file__).resolve().parent
OFFSETS=(-.5,0.,.5)
CHUNK=128

def capture(source,panel):
    project=ROOT.parent/'image_pressure_v45' if source else ROOT
    path=project/'results'/'worlds'/f'panel_{panel:02d}.npz'
    if not path.exists():
        return False
    with np.load(path) as f:
        th=f['theta'].reshape(-1,6)
    folder=ROOT/('source_camera46' if source else 'camera46')/f'panel_{panel:02d}'
    folder.mkdir(parents=True,exist_ok=True)
    families=('fold','ripple') if source else ('terrace','dimple')
    for shape,var,h in itertools.product(families,('base','hollow','bulge'),(19,31)):
        dst=folder/f'{shape}_{var}_h{h}.npz'
        if dst.exists(): continue
        x=np.zeros((3,512),np.float32)
        v=x.copy()
        vtrue=np.empty_like(x,dtype=np.float64)
        digests=[]
        for d,off in enumerate(OFFSETS):
            dig=hashlib.sha256()
            for k in range(0,len(th),CHUNK):
                seed=(45032026 if source else 46032026)+panel*128+k//CHUNK
                fr=render_clip(th[k:k+CHUNK],shape,var,h,off,seed)
                loc=infer_x(fr)
                var_hat=infer_normal_variance(fr,loc)
                x[d,k:k+CHUNK]=loc.numpy()
                v[d,k:k+CHUNK]=var_hat.numpy()
                dig.update(fr.numpy().tobytes())
                if not source and panel==0 and var=='hollow' and h==31 and d==0 and k==0:
                    np.savez_compressed(folder/f'pixels_{shape}_example.npz',frames=fr[:1].numpy())
            vtrue[d]=true_stat(th,shape,var,np.full(len(th),h+off))
            digests.append(dig.hexdigest())
        if source:
            with np.load(project/'camera45'/f'panel_{panel:02d}'/dst.name) as f:
                np.testing.assert_array_equal(digests,f['raster_sha256'])
                np.testing.assert_allclose(x.reshape(3,4,128),f['x_hat_mm'],rtol=0,atol=1e-5)
        assert np.isfinite(x).all() and np.isfinite(v).all() and np.isfinite(vtrue).all()
        exact=np.broadcast_to(h+np.array(OFFSETS)[:,None,None],(3,4,128))
        tmp=dst.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,x_hat_mm=x.reshape(3,4,128),variance_hat=v.reshape(3,4,128),
                true_x_mm=exact,true_variance=vtrue.reshape(3,4,128),raster_sha256=digests)
        os.replace(tmp,dst)
    print('source' if source else 'target',panel,'complete',flush=True)
    return True

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',action='store_true')
    ap.add_argument('--panels',nargs='+',type=int,required=True)
    a=ap.parse_args(); torch.set_num_threads(1)
    for p in a.panels:
        while not capture(a.source,p): time.sleep(2)
