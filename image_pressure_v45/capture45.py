"""Acquire held-out pre-impact camera clips; restore any clip by seed and draw index."""
import argparse
import hashlib
import itertools
import os
from pathlib import Path
import time

import numpy as np
import torch
from camera45 import render_clip, infer_x, source_gate, damage_clip

ROOT=Path(__file__).resolve().parent
OFFSETS=(-.5,0.,.5)
CHUNK=128


def acquire(panel):
    mpath=ROOT/'results'/'worlds'/f'panel_{panel:02d}.npz'
    if not mpath.exists():
        return False
    with np.load(mpath) as f:
        th=f['theta'].copy()
    assert th.shape==(4,128,6)
    flat=th.reshape(-1,6)
    out=ROOT/'camera45'/f'panel_{panel:02d}'
    out.mkdir(parents=True,exist_ok=True)
    for shape,variant,h in itertools.product(('fold','ripple'),('base','hollow','bulge'),(19,31)):
        path=out/f'{shape}_{variant}_h{h}.npz'
        if path.exists():
            continue
        x=np.empty((3,512),np.float32); drop=x.copy()
        digests=[]
        for d,offset in enumerate(OFFSETS):
            frag=hashlib.sha256()
            for start in range(0,len(flat),CHUNK):
                seed=45032026+panel*128+(start//CHUNK)
                frames=render_clip(flat[start:start+CHUNK],shape,variant,h,offset,seed)
                frag.update(frames.numpy().tobytes())
                x[d,start:start+CHUNK]=infer_x(frames).numpy()
                drop[d,start:start+CHUNK]=infer_x(damage_clip(frames)).numpy()
                if (panel==0 and variant=='hollow' and start==0):
                    np.savez_compressed(out/f'pixels_{shape}_{variant}_h{h}_d{d}.npz',frames=frames[0].numpy())
            digests.append(frag.hexdigest())
        x=x.reshape(3,4,128); drop=drop.reshape(x.shape)
        truth=np.broadcast_to(h+np.array(OFFSETS,dtype=np.float32)[:,None,None],x.shape)
        gate=lambda a: source_gate(torch.tensor(a)).numpy()
        assert np.isfinite(x).all() and np.isfinite(drop).all()
        tmp=path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,x_hat_mm=x,x_drop_mm=drop,oracle_x_mm=truth,
                            visual_project=gate(x),oracle_project=gate(truth),drop_project=gate(drop),
                            raster_sha256=np.array(digests))
        os.replace(tmp,path)
    print(f'panel {panel} camera complete',flush=True)
    return True


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--start',type=int,default=0);p.add_argument('--stride',type=int,default=1)
    args=p.parse_args()
    torch.set_num_threads(1)
    pending=list(range(args.start,12,args.stride))
    while pending:
        pending=[panel for panel in pending if not acquire(panel)]
        if pending:
            time.sleep(2)
