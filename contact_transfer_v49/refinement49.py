"""Numerical refinement diagnostic. Recalculate both primary contrasts with finer labels."""
import json
import os
import argparse
import numpy as np
from score49 import ROOT, N, SPATIAL, RETENTION, interval, joint_state_weights
from transfer_controls49 import risk

parser=argparse.ArgumentParser()
parser.add_argument('resolution',nargs='?',choices=('fine','ultra'),default='fine')
resolution=parser.parse_args().resolution
assert os.getenv('SLURM_JOB_ID')
coarse=np.load(ROOT/'per_catalog49.npz')['loss'][:,:,:,[2,3,4]]
shift=np.zeros_like(coarse)
disagree=[]
for i in range(N):
    tag=f'panel_{i:02d}.npz'
    with np.load(ROOT/f'labels49_{resolution}_all'/tag) as fine,np.load(ROOT/'labels49'/tag) as old:
        a,b=fine['labels'],old['labels']
        assert a.shape==b.shape and np.array_equal(fine['object_index'],old['object_index'])
        delta=a.astype(float)-b
        disagree.append(np.average(abs(delta).mean((0,1,3,4,5,6)),weights=[.5,.25,.25]))
    with np.load(ROOT/'forecast49'/tag) as d:
        p=d['p'][[2,3,4]]
    for f,h in np.ndindex(4,2):
        dy=delta[f,h].mean((1,2)) # full state, environment, object
        for m in range(3):
            meanp=p[m,f,h].mean((1,2,5))
            kernel=np.einsum('seo,teo->st',1.-2*meanp,dy)/(4*128)
            for r,rho in enumerate(SPATIAL):
                for c,keep in enumerate(RETENTION):
                    shift[i,r,c,m]+=np.sum(kernel*joint_state_weights(rho,keep,m==1 and h==0))/8
fine=coarse+shift
contrasts=fine[:,:,:,0,None]-fine[:,:,:,[1,2]]
shift_contrast=shift[:,:,:,0,None]-shift[:,:,:,[1,2]]
out=dict(resolution=resolution,weighted_disagreement=interval(np.array(disagree)),fine_loss=fine.mean(0),
         fine_primary=interval(contrasts,30),fine_minus_coarse_contrast=interval(shift_contrast,30))
output='refinement49.json' if resolution=='fine' else 'ultra_policy49.json'
(ROOT/output).write_text(json.dumps(out,default=lambda x:np.asarray(x).tolist(),indent=1))
