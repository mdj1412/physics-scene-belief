"""Secondary controls: identical forecast, old outcomes; known sign-reversal calibration.

The known-kappa diagnostic uses only the observable scalar and the specified population
orientation law. It is privileged and is not part of the three primary deployments.
"""
import json
from pathlib import Path

import numpy as np
from policy49 import Policy, update, sensor48, SOURCE, ROOT
from score49 import SPATIAL, RETENTION, interval, joint_state_weights, N, METHODS
from world48 import FAMILIES, HEIGHTS, STATES


def risk(prob, label):
    # prob: sensor state, alignment, optical variant, env, object, noise; label: physical state, env, object
    p=prob.mean((1,2,5))
    return np.square(prob).mean((1,2,3,4,5))[:,None]+np.einsum('seo,teo->st',1.-2*p,label)/(4*128)


def main():
    full=np.load(ROOT/'per_catalog49.npz')['loss']
    source=Policy('cpu')
    old=np.empty((N,3,5,4))
    informed=np.empty((N,5))
    for i in range(N):
        tag=f'panel_{i:02d}.npz'
        with np.load(ROOT/'forecast49'/tag) as data, np.load(SOURCE/'forecast'/tag) as prev:
            probabilities=data['p']
            paid=np.isfinite(prev['query_paid'][...,0])
        with np.load(SOURCE/'worlds'/tag) as pop:
            seed=int(pop['seed'])
        with np.load(ROOT/'labels49'/tag) as d, np.load(SOURCE/'labels'/tag) as o:
            y49=d['labels'].mean((3,4))
            y48=o['labels'].mean((3,4))
        former=np.zeros((3,5,4))
        diagnostic=np.zeros(5)
        for f,h in np.ndindex(4,2):
            p=probabilities[:,f,h]
            kernels=[risk(p[m],y48[f,h]) for m in (2,3,4,1)]
            po=p[0,0]
            arm=paid[f,:,h].astype(int)
            coef=source.coef[1,h][arm]
            sig=np.stack([sensor48.readings({'seed':seed},FAMILIES[f],HEIGHTS[h],state,HEIGHTS[h]) for state in STATES])
            reverse=update(po,-sig[:,None,None],coef)
            normal=update(po,sig[:,None,None],coef)
            np.testing.assert_allclose(normal,p[2],atol=1e-12,rtol=0)
            for k,keep in enumerate(RETENTION):
                # Mixture of two forecasts is a single reported probability, scored after mixing.
                blend=keep*normal+(1.-keep)*reverse
                w=joint_state_weights(1.,keep,False)
                diagnostic[k]+=np.sum(w*risk(blend,y49[f,h]))/8
                for r,rho in enumerate(SPATIAL):
                    for j,kernel in enumerate(kernels):
                        former[r,k,j]+=np.sum(joint_state_weights(rho,keep,j==1 and h==0)*kernel)/8
        old[i]=former
        informed[i]=diagnostic
    # At k=1 there is no orientation correction.
    np.testing.assert_allclose(informed[:,-1],full[:,0,-1,2],rtol=0,atol=1e-12)
    old_effect=old[:,:,:,0,None]-old[:,:,:,[1,2]]
    new_effect=full[:,:,:,2,None]-full[:,:,:,[3,4]]
    out=dict(old_loss=old.mean(0),old_methods=['direct_local','direct_fixed','gate_local','direct_none'],
             old_primary_effect=interval(old_effect,30),new_minus_old_effect=interval(new_effect-old_effect,30),
             known_orientation_brier=informed.mean(0),known_minus_frozen_local=interval(informed-full[:,0,:,2],5),
             known_minus_no_extra=interval(informed-full[:,0,:,1],5),
             scope='Secondary; supplied regime-level kappa is privileged; negates noisy scalar, not simulator state. Old-vs-new paired risk includes new solver plus physical model choices.')
    (ROOT/'transfer_controls49.json').write_text(json.dumps(out,default=lambda x:np.asarray(x).tolist(),indent=1))


if __name__=='__main__':
    import os
    assert os.getenv('SLURM_JOB_ID')
    main()
