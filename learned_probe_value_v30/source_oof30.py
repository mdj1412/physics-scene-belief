"""Descriptive cross-fitted policy replay on the historical source-law data."""
import json
from pathlib import Path

import numpy as np

from analyze30 import ROOT, RESULTS, digest, quota

SRC = ROOT.parent / 'adaptive_stopping_v29' / 'results'


def allocate(scores):
    counts = np.full(len(scores), 2, dtype=int)
    active = np.arange(len(counts))
    for j,n in enumerate([4,8]):
        ranking = np.argsort(-scores[active, j], kind='stable')
        active = np.sort(active[ranking[:len(active)//2]])
        counts[active]=n
    return counts


def main():
    model = json.loads((RESULTS/'model_lock.json').read_text())
    scores = np.stack([r['out_of_fold_predictions'] for r in model['training_diagnostics']],axis=-1)
    assert scores.shape == (24,8,2)
    risks,win_groups=[],[]
    for block,pred in enumerate(scores):
        base=SRC/f'block_{block:03d}'
        assert digest(base/'metrics.npz') == model['development_hashes'][block]
        with np.load(base/'metrics.npz') as z:
            idx=list(z['methods']).index('uncertainty')
            counts=np.stack([allocate(pred),np.full(8,4),z['counts'][:,idx]],axis=-1)
            assert (counts.sum(0)==32).all()
            actions=np.take_along_axis(quota(z['audit_gap']),np.searchsorted([2,4,8],counts)[:,None,:],axis=2)
            assert ((actions==3).sum(1)==13).all()
            cost=z['cost'][:,:,-1,:]
            risk=np.where(actions==3,cost[:,:,1,None],cost[:,:,0,None]).mean((0,1))
            assert np.allclose(risk[1:],z['quota_common_known'][:,:,[1,idx]].mean((0,1)))
            risks.append(risk)
    values=np.array(risks)
    d=values[:,0:1]-values[:,1:]
    grouped=np.stack([d[np.arange(24)%6==i].mean(0) for i in range(6)])
    summary=dict(methods=['learned','fixed4','uncertainty'],risk=values.mean(0).tolist(),
                 differences=d.mean(0).tolist(),outer_fold_differences=grouped.tolist(),
                 note='Policies replay only out-of-fold predictions on v29 blocks. Fits overlap across folds. Descriptive source comparator without independent replication or training-uncertainty interval.')
    (RESULTS/'source_oof_policy.json').write_text(json.dumps(summary,indent=2))
    print(summary)
if __name__=='__main__':
    main()
