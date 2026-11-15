"""Regression checks for the v32 held-out analysis, no GPU or target data needed."""
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
import json

from fit32 import PENALTY, verify_fit
from stats32 import (SEED, METHODS, allocations, audit_block, brier_summary, predict, risk_differences, sha,
                     sample_ids, timing_summary)
from sensitivity32 import ridge_svd


def main():
    rng=np.random.default_rng(SEED)
    # A fixed policy effect mixed with substantial shared law/block variation.
    baseline=rng.uniform(.1,.3,size=(3,16,1))
    shift=np.array([0.,.0005,.001,-.001])
    paired=brier_summary(baseline+shift,sample_ids(16,rng))
    target=risk_differences(shift)
    np.testing.assert_allclose(paired['delta'],target,atol=1e-12)
    np.testing.assert_allclose(paired['simultaneous_ci'],np.stack([target,target],axis=-1),atol=1e-12)
    null=brier_summary(np.repeat(baseline,4,axis=-1),sample_ids(16,rng))
    np.testing.assert_array_equal(null['simultaneous_ci'],np.zeros((5,2)))
    # Across-world policies can scale time while order and block noise stay paired.
    ms=rng.uniform(10,12,size=(3,6,3,1)) * np.array([1.,2.,1.,4.])
    timing=timing_summary(ms,sample_ids(6,rng))
    np.testing.assert_allclose(timing['p90_ratio'],[2.,1.,2.,.25,.5])
    np.testing.assert_allclose(timing['ratio_pointwise_ci'],np.repeat(np.asarray(timing['p90_ratio'])[:,None],2,axis=1))
    # Independent normal-equation comparison and prefix behavior with ties.
    x=rng.normal(size=(2,20,15));x[:,:,0]=2.; y=rng.normal(0,.01,size=(2,20))
    models=[ridge_svd(a,b) for a,b in zip(x,y)]
    verify_fit(models,x,y)
    material=predict(models,x.transpose(1,0,2))
    for i,(xx,yy) in enumerate(zip(x,y)):
        z=(xx-xx.mean(0))/np.maximum(xx.std(0),1e-8)
        beta=np.linalg.solve(z.T@z+len(z)*PENALTY*np.eye(15),z.T@(yy-yy.mean()))
        np.testing.assert_allclose(material[:,i],z@beta+yy.mean())
    np.testing.assert_array_equal(allocations(np.ones((8,2))),[8,8,4,4,2,2,2,2])
    with TemporaryDirectory() as td:
        folder=Path(td); features=rng.normal(size=(8,2,15))
        lock={'models':{'broad':models,'repeated':models[::-1]}}
        scores=np.stack([predict(lock['models'][arm],features) for arm in METHODS[:2]],axis=-1)
        uncertainty=rng.normal(size=(8,2));counts=np.stack([allocations(scores[:,:,0]),allocations(scores[:,:,1]),
                   np.full(8,4),allocations(uncertainty)],axis=1)
        gaps=rng.normal(size=(8,3,128)); select=np.argsort(-gaps,axis=-1)[...,:13]
        all_actions=np.ones((8,3,128),dtype=int);np.put_along_axis(all_actions,select,3,axis=2)
        actions=np.stack([all_actions[np.arange(8),np.searchsorted([2,4,8],c)] for c in counts.T],axis=-1)
        p=rng.uniform(size=(8,128,2,32));y=rng.integers(2,size=(8,128))
        err=((p-y[:,:,None,None])**2).mean(3)
        oracle=np.stack([np.take_along_axis(err,(all_actions[:,k,:]==3)[:,:,None].astype(int),axis=2).mean(axis=(1,2)) for k in range(3)],axis=1)
        brier=np.stack([oracle[np.arange(8),np.searchsorted([2,4,8],c)] for c in counts.T],axis=1)
        np.savez(folder/'choices.npz',gaps=gaps,counts=counts,actions=actions,features=features,scores=scores,uncertainty=uncertainty)
        (folder/'allocation.json').write_text(json.dumps({'methods':METHODS, 'model_sha256':'unit_test','counts':counts.tolist()}))
        np.savez(folder/'metrics.npz',methods=METHODS,counts=counts,actions=actions,labels=y,probabilities=p,
                 brier=brier,oracle_stage_risk=oracle,scores=scores,uncertainty=uncertainty,features=features,
                 full_ms=np.ones((3,4)),warm_ms=np.full((3,4),.5))
        md={f'{k}_sha256':sha(folder/f'{k}.{ext}') for k,ext in [('choices','npz'),('metrics','npz'),('allocation','json')]}
        md.update({'law':'unit','block':0,'model_sha256':'unit_test','bank_pair':[0,1],
                   'calibration_seeds':list(range(2,10)),'deployment_seeds':list(range(10,18))})
        (folder/'meta.json').write_text(json.dumps(md))
        audited=audit_block(folder,lock,'unit_test','unit',0)
        np.testing.assert_allclose(audited['brier'],brier.mean(axis=0))
    print('PASS independent synthetic raw data replay, paired effect, null interval, known latency ratios, constant feature ridge and stable bounds')


if __name__=='__main__':
    main()
