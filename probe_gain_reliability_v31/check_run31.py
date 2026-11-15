"""Independent CPU audit of frozen world splits, quota actions and paired Brier contrasts."""
import json
import hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
SHA=lambda file:hashlib.sha256(file.read_bytes()).hexdigest()


def main():
    cfg=json.loads((ROOT/'design.json').read_text())
    actual_seeds=set()
    old_seeds=set()
    v30_artifacts=list((ROOT.parent/'learned_probe_value_v30/results').glob('*/block_*/metrics.json'))
    for filename in v30_artifacts:
        record=json.loads(filename.read_text())
        old_seeds.update(record['seeds'])
    count=0;comparisons=0;max_gap_error=0.;max_brier_error=0.;max_control_error=0.;seeds_checked=0
    elapsed={}
    for base in sorted(OUT.glob('*/block_*/metrics.json')):
        meta=json.loads(base.read_text())
        folder=base.parent
        assert all(SHA(folder/f)==meta[k+'_sha256'] for f,k in [('prefix.npz','prefix'),('choices.npz','choices'),('metrics.npz','metrics')])
        for key in ['bank_pair','pool_seeds','future_seeds','batch_seeds']:
            seeds=np.asarray(meta[key]).reshape(-1).tolist()
            assert len(set(seeds))==len(seeds)
            assert actual_seeds.isdisjoint(seeds)
            assert old_seeds.isdisjoint(seeds)
            actual_seeds.update(seeds);seeds_checked+=len(seeds)
        with np.load(folder/'metrics.npz') as r,np.load(folder/'choices.npz') as choices,np.load(folder/'prefix.npz') as prefix:
            assert np.array_equal(prefix['score'],r['score'])
            assert np.array_equal(prefix['alpha'],r['alpha'])
            preds=r['probabilities'];labels=r['labels']
            costs=np.square(preds-labels[:,:,:,None,None]).mean(-1)
            err=np.max(abs(costs-r['cost']))
            assert err<1e-12
            before=choices['action_before'];after=choices['action_after']
            all_actions=np.concatenate([before[:,:,None],after],axis=2)
            expected=np.ones_like(all_actions)
            np.put_along_axis(expected,np.argsort(-choices['gap'],kind='stable')[...,:13],3,axis=-1)
            assert np.array_equal(all_actions,expected)
            def cost_for(action):
                return np.where(action==3,costs[None,:,None,:,:,1],costs[None,:,None,:,:,0]).mean(axis=-1)
            risk=cost_for(all_actions)
            replay=risk[:,:,:1]-risk[:,:,1:]
            err=max(err,np.max(abs(r['gain']-replay)))
            assert err<1e-12
            max_brier_error=max(max_brier_error,float(err))
            assert r['gain'].shape==(2,8,4,4)
            if meta['block']==0:
                assert len(meta['reference_max_errors'])==4
                comparisons+=len(meta['reference_max_errors'])*128
                max_gap_error=max(max_gap_error,max(meta['reference_max_errors']))
                cc=cost_for(choices['factual_only_action'])
                err=np.max(abs(r['gain_factual_only']+r['gain_probe_only']-replay))
                assert np.max(abs(cc-r['factual_only_brier']))<1e-12 and err<1e-12
                max_control_error=max(max_control_error,float(err))
        elapsed.setdefault(meta['law'],[]).append(meta['elapsed_s']);count+=1
    assert count==cfg['blocks_per_law']*len(cfg['laws'])==18 and len(v30_artifacts)==32
    validation=dict(verified_blocks=count,unique_v31_seed_count=seeds_checked,
        v30_input_seed_values_checked=len(old_seeds),v30_32_target_block_seeds_disjoint=True,
        scalar_ref_world_gaps_compared=comparisons,scalar_vs_batch_gap_max_error=max_gap_error,
        quota_top13_reference_identical=True,max_brier_error=max_brier_error,
        x_only_control_telescoping_error=max_control_error,guard_exit=json.loads((ROOT/'gpu_worker/results/worker_exit.json').read_text()),
        wall_seconds_by_law=elapsed)
    assert validation['guard_exit']['code']==0 and validation['guard_exit']['foreign']==[]
    (OUT/'runner_validation.json').write_text(json.dumps(validation,indent=2,allow_nan=False))
    print(json.dumps(validation))
if __name__=='__main__':main()
