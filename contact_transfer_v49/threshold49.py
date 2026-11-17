"""Secondary robustness at neighboring tangential-speed definitions of stick."""
import json
import os
import numpy as np
from score49 import ROOT, SPATIAL, RETENTION, joint_state_weights, interval, N
from transfer_controls49 import risk

CUTS=(.010,.020,.040)
assert os.getenv('SLURM_JOB_ID')
results=np.zeros((N,len(CUTS),len(SPATIAL),len(RETENTION),3))
for i in range(N):
    name=f'panel_{i:02d}.npz'
    with np.load(ROOT/'labels49'/name) as sim, np.load(ROOT/'forecast49'/name) as archive:
        speed=sim['slip_speed']
        np.testing.assert_array_equal(sim['labels'],speed<=.020)
        bank=archive['p'][[2,3,4]]
    for f,h in np.ndindex(4,2):
        for ci,cut in enumerate(CUTS):
            label=(speed[f,h]<=cut).mean((1,2))
            for m in range(3):
                kernel=risk(bank[m,f,h],label)
                for r,rho in enumerate(SPATIAL):
                    for j,keep in enumerate(RETENTION):
                        law=joint_state_weights(rho,keep,m==1 and h==0)
                        results[i,ci,r,j,m]+=np.sum(kernel*law)/8
primary=np.load(ROOT/'per_catalog49.npz')['loss']
np.testing.assert_allclose(results[:,1],primary[:,:,:,[2,3,4]],atol=2e-12,rtol=0)
contrast=results[:,:,:,:,0,None]-results[:,:,:,:,[1,2]]
out=dict(thresholds_mps=CUTS,losses=results.mean(0),effects_simultaneous_all_cuts=interval(contrast,90))
(ROOT/'threshold49.json').write_text(json.dumps(out,default=lambda x:np.asarray(x).tolist()))
