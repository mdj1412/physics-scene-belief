"""Post-primary diagnostic decomposition and GPU working-memory audit."""
import os,gc
import numpy as np
import torch
from common import *
from runtime import Engine,METHODS
from analyze import CONDITIONS
assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(2)
actions=np.load(R/'decision_actions.npz');loss=np.load(R/'world_losses.npz');out={}
for j,c in enumerate(CONDITIONS):
    a=actions['qmc16_pair'].reshape(256,6)[:,j];out[c]={}
    for base in ['qmc16_native','qmc16_nested128','qmc16_probe']:
        b=actions[base].reshape(256,6)[:,j];diff=a!=b;delta=loss['qmc16_pair'][:,j]-loss[base][:,j]
        out[c][base]=dict(agreement=float((~diff).mean()),changed_worlds=int(diff.sum()),mean_loss_difference_when_changed=float(delta[diff].mean()) if diff.any() else 0.)
audit=np.load(R/'ranking_audit.npz');truth=np.load(R/'data/test/private.npz')
out['pair512_flips']=[dict(case=int(i),base_world=int(i//6),condition=CONDITIONS[i%6],frictional_class=int(truth['theta'][i//6,1])) for i in np.flatnonzero(audit['pair512'].argmax(1)!=actions['qmc16_pair'][:96])]
x=public('test');observations=obs('test');stored=np.load(R/'decision_predictions.npz');memory={}
for nm in METHODS:
    e=Engine(nm);torch.cuda.synchronize();rows=[]
    for i in [0,515,1030,1535]:
        before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        state,a=e.start(x[i:i+1]);p=e.finish(state,a,observations[a][i:i+1,0] if a else None);torch.cuda.synchronize()
        assert a==actions[nm][i] and abs(p.cpu().numpy()[0]-stored[nm][i,0]).max()<1e-8
        rows.append(dict(case=i,temporary_peak_bytes=int(torch.cuda.max_memory_allocated()-before)))
        del state,p;gc.collect()
    memory[nm]=dict(rows=rows,max_observed_temporary_peak_bytes=max(r['temporary_peak_bytes'] for r in rows));del e;gc.collect();torch.cuda.empty_cache()
out['memory_audit']=memory;write(R/'diagnostics.json',out);print(out)
