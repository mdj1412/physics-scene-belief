"""Fixed policies + selected-action predictions, never target truth."""
import os,time
import numpy as np
from common import *
def main():
    assert os.environ.get('SLURM_JOB_ID');pred={};acts={};N=len(public('test'));ix=np.arange(N)
    for p in (R/'policies').glob('*.npz'):
        check(p.stem);decoder=p.stem.split('_')[0];a=np.load(p)['actions'];full=np.load(R/'evaluations'/f'{decoder}.npz')['predictive']
        pred[p.stem]=full[ix,a];acts[p.stem]=a
    for decoder in ['compact','qmc16','qmc64']:
        full=np.load(R/'evaluations'/f'{decoder}.npz')['predictive']
        for suffix,a in [('none',0),('query',1),('probe',3)]:pred[decoder+'_'+suffix]=full[:,a];acts[decoder+'_'+suffix]=np.full(N,a)
    for choices,decoder,label in [('compact_pair','qmc16','compact_pair_qmc16_decoder'),('qmc16_pair','compact','qmc16_pair_compact_decoder')]:
        pred[label]=np.load(R/'evaluations'/f'{decoder}.npz')['predictive'][ix,acts[choices]];acts[label]=acts[choices]
    np.savez_compressed(R/'decision_predictions.npz',**pred);np.savez_compressed(R/'decision_actions.npz',**acts)
    write(R/'decision_freeze.json',dict(time=time.time(),predictions=sha(R/'decision_predictions.npz'),actions=sha(R/'decision_actions.npz'),targets_read=False))
if __name__=='__main__':main()
