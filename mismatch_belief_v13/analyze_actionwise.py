import os
from pathlib import Path
import numpy as np
import common
common.R=Path('results_actionwise')
from common import *
from analyze import ci,CONDITIONS
assert os.environ.get('SLURM_JOB_ID')
freeze=read(R/'decision_freeze.json');assert sha(R/'decision_predictions.npz')==freeze['predictions']
p=np.load(R/'decision_predictions.npz');truth=np.load(R/'data/test/private.npz');target=truth['target'].reshape(-1,2);mask=truth['theta'][:,1]==1
loss={k:(.5*((v-target[:,None,:])**2).sum(-1)).mean(1).reshape(256,6) for k,v in p.items()};np.savez_compressed(R/'world_losses.npz',**loss)
out=dict(changed_worlds=int(mask.sum()),means={},contrasts={},flags={},matched_harm=ci(loss['actionwise_gate'][:,0],loss['base_native'][:,0]))
flags=np.load(R/'flags.npz')
for j,c in enumerate(CONDITIONS):
    out['means'][c]={k:float(v[:,j].mean()) for k,v in loss.items()}
    out['flags'][c]={k:dict(all=float(v.reshape(256,6,16)[:,j].mean()),changed=float(v.reshape(256,6,16)[mask,j].mean())) for k,v in flags.items()}
    for baseline in ['original_gate','qmc16_probe','expanded_qmc16_probe']:
        out['contrasts'][c+'/'+baseline]=ci(loss['actionwise_gate'][:,j]-loss[baseline][:,j],level=1-.05/6 if j in [3,5] else .95)
out['negative_control']={k:float(abs(v[~mask,1:]-v[~mask,0,None]).max()) for k,v in loss.items()}
out['timing_gpu']={k:{a:b for a,b in v.items() if a!='rows'} for k,v in read(R/'timing_gpu.json').items()}
out['timing_cpu']={k:{a:b for a,b in v.items() if a!='rows'} for k,v in read(R/'timing_cpu.json').items()}
write(R/'summary.json',out);print(out)
