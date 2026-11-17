"""Attribute held-out mean gains to correct and mistaken switches, and readout count."""
from pathlib import Path
import os,json
import numpy as np
from policy57 import choose,MIXTURE
ROOT=Path(__file__).resolve().parent
assert os.getenv('SLURM_JOB_ID')
with np.load(ROOT/'score57_cells.npz') as f:
    cells=f['by_e']
    methods=list(f['methods'])
    avg=f['mean_cells'].mean((0,1,2))
loss=np.einsum('cfea,ze->cfza',cells,MIXTURE)
gaps=loss[...,0]-loss[...,1]
groups=[]
for i in range(len(cells)):
    with np.load(ROOT/'pilots'/f'panel_{i:02d}.npz') as f:
        ref,im=f['pilot_ref'][:,:,1],f['pilot_imp'][:,:,1]
    valid=(ref!=0)&(im!=0)
    n=valid.reshape(*valid.shape[:-1],4,2).all(-1).sum(-1)
    entries=[]
    for number in range(5):
        mask=n==number
        entries.append(np.stack([mask.mean(2)]+[(mask&choose(ref,im,'4x2',rule)).mean(2) for rule in ('map','risk')],-1))
    groups.append(np.stack(entries,-2))
hist=np.asarray(groups)
answers={}
for ix,name in enumerate(['4x2_map','4x2_risk'],1):
    value=hist[...,ix]*gaps[...,None]
    per_regime=value.mean((0,1)).sum(1)/2
    np.testing.assert_allclose(per_regime.sum(),avg[methods.index('v54_gate')]-avg[methods.index(name)],atol=1e-14,rtol=0)
    contributes=value.mean((0,1,2))
    answers[name]={'gain_contribution_A_B':per_regime.tolist(),'gain_by_number_double_readable':contributes.tolist(),
        'probability_fixed_given_number_double_readable':(hist[...,ix].mean((0,1,2))/hist[...,0].mean((0,1,2))).tolist()}
answers['number_double_readable_rate']=hist[...,0].mean((0,1,2)).tolist()
(ROOT/'opportunity57.json').write_text(json.dumps(answers,indent=2)+'\n')
