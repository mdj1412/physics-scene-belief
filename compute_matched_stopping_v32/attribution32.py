"""Hold all source prefixes fixed, vary only which cell supervised the stage-wise ridge."""
from pathlib import Path
from itertools import product
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
TRAIN=ROOT/'results_train'
EVAL=ROOT/'results_eval'
LAWS=('split_static','speed_weak','directional')


def fit(x,y):
    mean=x.mean(0);sd=x.std(0).clip(1e-8);z=(x-mean)/sd
    beta=np.linalg.solve(z.T@z+len(x)*3*np.eye(x.shape[1]),z.T@(y-y.mean()))
    return dict(mean=mean.tolist(),sd=sd.tolist(),beta=beta.tolist(),intercept=float(y.mean()))


def quota_stop(scores):
    counts=np.full(8,2)
    survivors=np.arange(8)
    for stage in range(2):
        survivors=np.sort(survivors[np.argsort(-scores[survivors,stage],kind='stable')[:len(survivors)//2]])
        counts[survivors]*=2
    assert counts.sum()==32
    return counts


def predict(m,x):
    return (x-np.asarray(m['mean']))/m['sd']@m['beta']+m['intercept']


def freeze():
    main=json.loads((TRAIN/'model_lock.json').read_text())['models']['repeated']
    xx=[];gg=[]
    for path in sorted((TRAIN/'repeated').glob('block_*/data.npz')):
        with np.load(path) as z:
            xx.append(z['features']);gg.append(z['gain'])
    x=np.concatenate(xx,axis=1);g=np.concatenate(gg,axis=1)
    assert x.shape[:2]==g.shape[:2] and g.shape[2:]==(2,2)
    models=[[fit(x[i],g[i,:,f,b]) for i in range(2)] for f,b in product(range(2),repeat=2)]
    for i in range(2):
        for field in ('beta','mean','sd','intercept'):
            np.testing.assert_allclose(np.mean([m[i][field] for m in models],0),main[i][field],atol=1e-10)
    file=TRAIN/'single_cell_lock.json'
    file.write_text(json.dumps(dict(models=models,training_environments=x.shape[1],positions=list(product(range(2),repeat=2)),
                              note='Same prefix features and penalty. Offline CPU explanatory diagnostic.'),indent=2))


def main():
    mainmodels=json.loads((TRAIN/'model_lock.json').read_text())['models']['repeated']
    models=json.loads((TRAIN/'single_cell_lock.json').read_text())['models']
    blocks=[];counts_checks=[]
    for law in LAWS:
        rows=[];checks=[]
        for p in sorted((EVAL/law).glob('block_*/metrics.npz')):
            with np.load(p) as z:
                x=z['features'];risk=z['oracle_stage_risk']
                rowscores=np.array([np.column_stack([predict(ms[k],x[:,k]) for k in range(2)])
                                    for ms in [mainmodels]+models])
                np.testing.assert_allclose(rowscores[0],z['scores'][:,:,1],atol=1.e-10,rtol=0)
                np.testing.assert_allclose(rowscores[0],rowscores[1:].mean(0),atol=1.e-10,rtol=0)
                alloc=np.stack([quota_stop(sc) for sc in rowscores])
                np.testing.assert_array_equal(alloc[0],z['counts'][:,1])
                losses=risk[np.arange(8),np.searchsorted([2,4,8],alloc)]
                np.testing.assert_allclose(losses[0],z['brier'][:,1],atol=1e-12)
                rows.append(losses.mean(1));checks.append((alloc[1:]!=alloc[0]).mean())
        assert len(rows)==16
        blocks.append(rows);counts_checks.append(checks)
    values=np.array(blocks)
    delta=values[:,:,0]-values[:,:,1:].mean(2)
    draw=np.random.default_rng(32614309).integers(16,size=(10000,3,16))
    boot=delta[np.arange(3)[None,:,None],draw].mean(2)
    doc=dict(laws=LAWS,columns=['2x2 averaged labels','F1B1','F1B2','F2B1','F2B2'],
             by_law=values.mean(1).tolist(),pooled=values.mean((0,1)).tolist(),repeated_minus_single_mean=float(delta.mean()),
             paired_ci=np.quantile(boot.mean(1),[.025,.975]).tolist(),delta_law=delta.mean(1).tolist(),
             delta_law_ci=np.quantile(boot,[.025,.975],axis=0).T.tolist(),allocation_difference=float(np.mean(counts_checks)))
    (EVAL/'attribution32.json').write_text(json.dumps(doc,indent=2))
if __name__=='__main__':
    import sys
    if sys.argv[1:] == ['--freeze']: freeze()
    else: main()
