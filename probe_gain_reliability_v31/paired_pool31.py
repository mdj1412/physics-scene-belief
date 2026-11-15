"""Matched continuation sensitivity and block-paired differences from the fresh-object panel."""
import json
from pathlib import Path

import numpy as np

from stats31 import LAWS, RESULTS, ROOT, N_BOOT, analyze, block_metrics, sha, joint_intervals

POOL=ROOT/'results_pool'


def audit_pool(folder,primary):
    meta=json.loads((folder/'metrics.json').read_text())
    ref=json.loads((primary/'metrics.json').read_text())
    assert (meta['law'],meta['block'])==(ref['law'],ref['block'])
    assert sha(folder/'metrics.npz')==meta['metrics_sha256']
    assert sha(folder/'choices.npz') == meta['choices_sha256']
    assert meta['prefix_sha256'] == ref['prefix_sha256'] == sha(primary/'prefix.npz')
    assert meta['primary_choices_sha256'] == ref['choices_sha256']
    assert meta['primary_metrics_sha256'] == ref['metrics_sha256']
    np.testing.assert_array_equal(ref['bank_pair'],meta['bank_pair'])
    with np.load(folder/'choices.npz') as choices, np.load(primary/'prefix.npz') as prefix:
        gap=choices['gap_after']
        actions=np.ones_like(gap,dtype=np.int8)
        order=np.argsort(-gap,axis=-1,kind='stable')
        np.put_along_axis(actions,order[...,:13],3,axis=-1)
        assert gap.shape == (2,8,4,4,128) and ((actions==3).sum(-1)==13).all()
        np.testing.assert_array_equal(actions,choices['action_after'])
        with np.load(primary/'choices.npz') as base_choices:
            np.testing.assert_array_equal(choices['action_before'],base_choices['action_before'])
        np.testing.assert_array_equal(choices['future_seeds'],meta['future_seeds'])
        for s,k in enumerate((2,4)):
            ids=choices[f'ids_{k}']
            observed=choices[f'responses_{k}']
            assert ids.shape==(8,4,k) and observed.shape==(8,4,k,2)
            assert np.isfinite(observed).all()
            assert ids.min()>=0 and ids.max()<prefix['factual_pool'].shape[1]
            for e in range(8):
                assert all(len(set(row))==k for row in ids[e])
                assert set(ids[e].ravel()).isdisjoint(prefix['paid_ids'][e,:k])
                assert 4*k - len(set(ids[e].ravel())) == choices['overlap_excess'][s,e]
    with np.load(folder/'metrics.npz') as z, np.load(primary/'metrics.npz') as old:
        gains=z['gain'];before=z['before'];after=z['after']
        np.testing.assert_allclose(before,old['before'],rtol=0,atol=1.e-12)
        cost=old['cost']
        via_actions=np.where(actions==3,cost[None,:,None,:,:,1],cost[None,:,None,:,:,0]).mean(-1)
        np.testing.assert_allclose(via_actions,after,rtol=0,atol=1.e-12)
        np.testing.assert_array_equal(z['alpha'],old['alpha'])
        np.testing.assert_allclose(z['score'],old['score'],rtol=0,atol=1.e-12)
        np.testing.assert_allclose(z['uncertainty'],old['uncertainty'],rtol=0,atol=1.e-12)
        np.testing.assert_allclose(gains,before[:,:,None,:]-after,rtol=0,atol=1.e-12)
        assert before.shape==(2,8,4) and gains.shape==after.shape==(2,8,4,4)
    return gains


def leave_one(arr):
    leave=[]
    for j in range(3):
        for i in range(6):
            pooled=[np.delete(a, i,axis=0).mean(0) if j==l else a.mean(0) for l,a in enumerate(arr)]
            leave.append(np.mean(pooled,axis=0))
    return dict(leave_one_block_min=np.min(leave,axis=0).tolist(),leave_one_block_max=np.max(leave,axis=0).tolist(),
                leave_one_law=[np.delete(arr,i,axis=0).mean((0,1)).tolist() for i in range(3)])


def four_contrasts(d):
    c=np.array(d['components'])
    ranks=np.array(d['ranks'])
    return np.stack([d['gain'],c[...,1]-c[...,0],ranks[...,2],ranks[...,5]],axis=-1)


def main():
    fresh=json.loads((RESULTS/'crossed_stats31.json').read_text())
    blocks=[[block_metrics(audit_pool(POOL/law/f'block_{i:03d}',RESULTS/law/f'block_{i:03d}'))
             for i in range(6)] for law in LAWS]
    pooled=analyze(blocks)
    (POOL/'crossed_stats31.json').write_text(json.dumps(pooled,indent=2,allow_nan=False)+'\n')
    p=four_contrasts(pooled['block_rows'])
    f=four_contrasts(fresh['block_rows'])
    d=p-f
    seed=3186101
    ids=np.random.default_rng(seed).integers(6,size=(N_BOOT,3,6))
    boot=d[np.arange(3)[None,:,None],ids].mean(2).mean(1)
    pt=d.mean((0,1))
    bounds,crit=joint_intervals(pt,boot)
    out=dict(contrasts=['mean_gain','batch_minus_future_variance','independent_1x1_concordance','independent_2x2_concordance'],
        pooled_fresh=f.mean((0,1)).tolist(),pooled_pool=p.mean((0,1)).tolist(),paired_difference=pt.tolist(),
        pointwise95=np.moveaxis(np.quantile(boot,[.025,.975],axis=0),0,-1).tolist(),simultaneous95=bounds.tolist(),
        bootstrap_critical=float(crit),bootstrap_seed=seed,bootstrap_draws=N_BOOT,
        per_law_paired_difference=d.mean(1).tolist(),law_order=LAWS,
        leave_one_pool=leave_one(p),leave_one_difference=leave_one(d),
        note='Secondary pooled minus fresh contrasts. Paired within exactly the same environment-prefix, law, seed block and deployment batches. Four endpoints by two stages. Primary analysis remains in results/.')
    (POOL/'paired_pool31.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print('Secondary estimands', p.mean((0,1)))
    print('Paired differences',pt)
    print('Simultaneous intervals',bounds)
if __name__=='__main__':
    main()
