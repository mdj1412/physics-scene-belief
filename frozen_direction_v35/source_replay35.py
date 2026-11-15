"""Recover v34's historical result for the single stored pair deployed in v35."""
import json

import numpy as np
from audit35 import SOURCE, ROOT, SCHEDULES, SIGNS, load_frozen, corrected, read_cue, maxband

def main():
    models,beta=load_frozen()
    vals=np.zeros((8,2,2,3))
    for b in range(8):
        for k,sign in enumerate(SIGNS):
            for s,sc in enumerate(SCHEDULES):
                d=SOURCE/'results_policy'/('rounded_fine_'+sign)/f'block_{b:03d}'/sc
                with np.load(d/'choices.npz') as h, np.load(d/'metrics.npz') as m:
                    ix=(m['actions'][:,:,1]==3).astype(int)
                    p=np.take_along_axis(m['probabilities'],ix[:,:,None,None],axis=2)[:,:,0]
                    y=m['labels'][...,None]
                    z=read_cue(*(h[key][:,:4] for key in ('paid_x','paid_y','paid_direction')),models,sc)[0]
                    p1=corrected(p,z,beta[0])
                    base=((p-y)**2).mean()
                    corr=((p1-y)**2).mean()
                    assert abs(base - m['brier'][:,1].mean())<1e-10
                    vals[b,k,s]=base,corr,corr-base
    rng=np.random.default_rng(350420)
    ii=(rng.integers(4,size=(12000,4,2))*2+np.arange(2)).reshape(-1,8)
    point=vals.mean((0,1))
    draws=vals[ii].mean((1,2))[...,-1]
    out=dict(note='Historical descriptive source reference, rounded_fine only; frozen pair 0.',
             mean_brier=point[:,:2].tolist(),delta=point[:,-1].tolist(),simultaneous_ci95=maxband(point[:,-1],draws).tolist())
    (ROOT/'source_replay35.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':
    main()
