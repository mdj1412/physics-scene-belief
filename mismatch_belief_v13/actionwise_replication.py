"""Independent worlds for a post-primary action-conditioned threshold repair."""
import os,sys,time
from pathlib import Path
import numpy as np
import torch
import common
ORIGINAL=Path('results');common.R=Path('results_actionwise')
from common import *
from detection import upper_threshold
import run_compact

def main():
    assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(2)
    old=read(ORIGINAL/'thresholds.json');cal=np.load(ORIGINAL/'calibration_scores.npz');thresholds=dict(old)
    thresholds['post_actionwise']={}
    for key in ['primary','native','probe']:
        aa=cal['primary_actions'] if key=='primary' else cal['native_actions'] if key=='native' else np.full(len(cal['probe']),3)
        thresholds['post_actionwise'][key]={str(a):upper_threshold(cal[key][aa==a]) if (aa==a).any() else old['post'][key] for a in [1,2,3]}
    thresholds['actionwise_created']=time.time();thresholds['original_calibration_sha']=sha(ORIGINAL/'calibration_scores.npz')
    write(R/'thresholds.json',thresholds);snapshot('source_freeze.json');print('actionwise thresholds',thresholds,flush=True)
    run_compact.generate('test',256,2400101,True);run_compact.test()
    import run_banks
    run_banks.main()
    assemble()

def assemble():
    th=read(R/'thresholds.json');p=np.load(R/'policies/base.npz');base=np.load(R/'evaluations/base.npz');exp=np.load(R/'evaluations/expanded.npz')
    pre=p['pre_score']>th['pre'];a=np.where(pre,3,p['actions']);ix=np.arange(len(a));scores=base['post_score'][ix,a]
    flags={'original_gate':pre[:,None]|(scores>th['post']['primary']),
           'actionwise_gate':pre[:,None]|(scores>np.array([th['post_actionwise']['primary'][str(z)] for z in a])[:,None])}
    predictions={k:np.where(f[:,:,None],exp['predictive'][ix,a],base['predictive'][ix,a]) for k,f in flags.items()}
    predictions.update(base_native=base['predictive'][ix,p['actions']],base_probe=base['predictive'][:,3],expanded_probe=exp['predictive'][:,3],expanded_decoder_same_actions=exp['predictive'][ix,a])
    for path in (R/'evaluations').glob('*qmc*.npz'):
        d=np.load(path)['predictive'];aa=np.load(R/'policies'/f'{path.stem}.npz')['actions'];predictions[path.stem+'_probe']=d[:,3];predictions[path.stem+'_native']=d[ix,aa]
    np.savez_compressed(R/'decision_predictions.npz',**predictions);np.savez_compressed(R/'flags.npz',**flags);np.savez_compressed(R/'actions.npz',actions=a)
    write(R/'decision_freeze.json',dict(time=time.time(),predictions=sha(R/'decision_predictions.npz'),flags=sha(R/'flags.npz'),targets_read=False))
if __name__=='__main__':main()
