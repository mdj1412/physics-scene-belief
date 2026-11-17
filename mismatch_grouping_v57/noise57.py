"""Readout sensitivity for a frozen grouped MAP, integrated over four independent groups."""
import itertools
import json
import os
from pathlib import Path
import numpy as np
from policy57 import LLR

ROOT=Path(__file__).resolve().parent
MIX=np.array([[.42,0,.58],[0,.84,.16]])
E=np.array([0.,.25,.5])

def expected_fixed(noise):
    # Symmetric errors flip the impact readout of each active object independently.
    seen=noise+(1-2*noise)*E
    p=MIX@seen
    both_flips=MIX@(seen**2)
    complete=.5**2
    laws=np.column_stack((np.full(2,1-complete),complete*(1-2*p+both_flips),
                          complete*2*(p-both_flips),complete*both_flips))
    llr=np.r_[0,LLR]
    patterns=np.array(list(itertools.product(range(4),repeat=4)))
    probs=laws[:,patterns].prod(-1)
    np.testing.assert_allclose(probs.sum(1),1)
    return probs[:,llr[patterns].sum(-1)>0].sum(1)

def main():
    assert os.environ.get('SLURM_JOB_ID')
    cells=[]
    for i in range(24):
        with np.load(ROOT/'scored57'/f'panel_{i:02d}.npz') as f:
            cells.append(f['exact_cells'][...,:2].mean((1,2)))
    # catalog x regime x arm, before policy selection
    risk=np.einsum('re,cea->cra',MIX,np.asarray(cells))
    result={}
    for n in (0,.05,.1):
        use=expected_fixed(n)
        switched=((1-use)*risk[:,:,0]+use*risk[:,:,1]).mean(1)
        d=switched[:,None]-risk.mean(1)
        rg=np.random.default_rng(57082023)
        boots=d[rg.integers(24,size=(5000,24))].mean(1)
        result[n]=dict(fixed_rate_by_regime=use.tolist(),
                       balanced_accuracy=float((use[0]+1-use[1])/2), overall_mean=switched.mean(),
                       delta_vs_gate_I=d.mean(0).tolist(),ci=np.quantile(boots,[.0125,.9875],axis=0).T.tolist())
    (ROOT/'pilot_noise57.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__': main()
