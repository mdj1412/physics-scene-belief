"""Frozen policy evaluation. Does not read targets."""
import os,time
import numpy as np
from common import *
def main():
    assert os.environ.get('SLURM_JOB_ID');th=read(R/'thresholds.json');base=np.load(R/'evaluations/base.npz');exp=np.load(R/'evaluations/expanded.npz')
    pol=np.load(R/'policies/base.npz');a=pol['actions'];pre=pol['pre_score']>th['pre'];N=len(a);idx=np.arange(N)
    pred={};acts={};flags={}
    choices={'base_native':a,'base_probe':np.full(N,3),'select_gate':np.where(pre,3,a),'expand_only':a,'primary_gate':np.where(pre,3,a),'probe_gate':np.full(N,3)}
    for name,act in choices.items():
        p=base['predictive'][idx,act];gate=np.zeros((N,16),bool)
        if name in ['expand_only','primary_gate','probe_gate']:
            key={'expand_only':'native','primary_gate':'primary','probe_gate':'probe'}[name]
            score=base['post_score'][idx,act];gate=pre[:,None]|(score>th['post'][key]);p=np.where(gate[:,:,None],exp['predictive'][idx,act],p)
        pred[name]=p;acts[name]=act;flags[name]=gate
    act=np.load(R/'policies/expanded.npz')['actions']
    for nm,aa in [('expanded_native',act),('expanded_probe',np.full(N,3))]:pred[nm]=exp['predictive'][idx,aa];acts[nm]=aa;flags[nm]=np.ones((N,16),bool)
    for p in (R/'evaluations').glob('*qmc*.npz'):
        name=p.stem;d=np.load(p)['predictive'];a=np.load(R/'policies'/f'{name}.npz')['actions']
        for suffix,aa in [('native',a),('probe',np.full(N,3))]:pred[name+'_'+suffix]=d[idx,aa];acts[name+'_'+suffix]=aa
    np.savez_compressed(R/'decision_predictions.npz',**pred);np.savez_compressed(R/'decision_actions.npz',**acts);np.savez_compressed(R/'expansion_flags.npz',**flags)
    write(R/'decision_freeze.json',dict(time=time.time(),predictions=sha(R/'decision_predictions.npz'),actions=sha(R/'decision_actions.npz'),flags=sha(R/'expansion_flags.npz'),thresholds=sha(R/'thresholds.json'),targets_read=False))
if __name__=='__main__':main()
