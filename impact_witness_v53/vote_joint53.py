"""Secondary ablation: hold continuous blending fixed and discard displacement magnitude."""
import os
import numpy as np
from policy53 import ROOT,mixed53,witnesses53


if __name__=='__main__':
    assert os.environ.get('SLURM_JOB_ID')
    folder=ROOT/'votes53'
    folder.mkdir(exist_ok=True)
    for i in range(12):
        alpha=np.empty((3,2,5,4,256))
        for ei,event,ki in np.ndindex(3,2,5):
            ow,of,nw,nf,s,_=witnesses53(i,ki,4-ki if event else ki,ei)
            alpha[ei,event,ki]=mixed53(ow,of,nw[...,:3],nf[...,:3],nw[...,3],s,method='vote')['alpha_joint']
        np.testing.assert_array_equal(alpha[:,0,2],alpha[:,1,2])
        np.savez_compressed(folder/f'panel_{i:02d}.npz',alpha=alpha)
