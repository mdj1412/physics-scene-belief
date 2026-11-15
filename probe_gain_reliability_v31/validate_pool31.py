"""CPU replay of a sampled finite-pool probe at each branch and all locked action/Brier outputs."""
import json
import os
from pathlib import Path
import numpy as np
from runner31 import (ROOT, OUT, STAGES, digest, draw_theta, means, sigmas, stream, actions, torch, t)

POOL=ROOT/'results_pool'


def check(folder):
    meta=json.loads((folder/'metrics.json').read_text())
    source=ROOT/meta['source_folder']
    assert digest(source/'choices.npz')==meta['primary_choices_sha256']
    assert digest(source/'metrics.npz')==meta['primary_metrics_sha256']
    assert digest(source/'prefix.npz')==meta['prefix_sha256']
    assert digest(folder/'choices.npz')==meta['choices_sha256']
    assert digest(folder/'metrics.npz')==meta['metrics_sha256']
    with np.load(source/'prefix.npz') as pre,np.load(source/'metrics.npz') as main:
        alpha=pre['alpha'];pool_seeds=pre['pool_seeds'];pids=pre['paid_ids'];cost=main['cost'];before=main['before']
    noise_errors=[];overlap=np.zeros(2,int)
    with np.load(folder/'choices.npz') as ch,np.load(folder/'metrics.npz') as m:
        assert np.array_equal(m['alpha'],alpha) and np.array_equal(m['before'],before)
        kept=ch['action_after']
        assert np.array_equal(kept,actions(ch['gap_after']))
        after=np.where(kept==3,cost[None,:,None,:,:,1],cost[None,:,None,:,:,0]).mean(-1)
        assert np.max(abs(before[:,:,None]-after-m['gain']))<1e-13
        for e,seed in enumerate(pool_seeds):
            replay_theta=draw_theta(1024,int(seed),10,alpha[e])
            expected_means=means(replay_theta,meta['law'])[3].numpy()
            for s,k in enumerate(STAGES):
                ids=ch[f'ids_{k}'][e]
                available=np.setdiff1d(np.arange(1024),pids[e,:k])
                overlap[s]+=ids.size-len(np.unique(ids))
                for j,cseed in enumerate(ch['future_seeds'][s,e]):
                    expected=stream(int(cseed),81).choice(available,size=k,replace=False)
                    assert np.array_equal(ids[j],expected) and len(set(ids[j]))==k
                    yy=expected_means[expected]+stream(int(cseed),82).normal(size=(k,2))*sigmas()[3].numpy()
                    noise_errors.append(np.max(abs(yy-ch[f'responses_{k}'][e,j])))
        assert np.array_equal(overlap,ch['overlap_excess'].sum(1))
    return max(noise_errors),overlap,meta['elapsed_s']


def main():
    assert not torch.cuda.is_available()
    validation=[]
    for file in sorted(POOL.glob('*/block_*/metrics.json')):
        validation.append(check(file.parent))
    assert len(validation)==18
    max_error=max(x[0] for x in validation)
    assert max_error<1e-12
    guard=json.loads((ROOT/'gpu_worker_pool/results/worker_exit.json').read_text())
    assert guard['code']==0 and guard['foreign']==[]
    data=dict(verified_blocks=len(validation),max_sensor_response_replay_abs_error=float(max_error),
              cross_future_duplicate_ids_by_stage=sum(x[1] for x in validation).tolist(),
              drawn_future_ids_by_stage=[144*4*2,144*4*4],
              total_block_wall_seconds=sum(x[2] for x in validation),guard=guard)
    (POOL/'runner_validation.json').write_text(json.dumps(data,indent=2))
    print(json.dumps(data))
if __name__=='__main__':main()
