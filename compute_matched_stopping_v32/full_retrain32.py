"""Exact full-policy replay on the same deployment worlds for independent 540-second fits."""
from pathlib import Path
import json
import numpy as np
from stats32 import allocations, predict, sha, LAWS, METHODS, PAIRS, risk_differences, simultaneous

ROOT=Path(__file__).resolve().parent
EVAL=ROOT/'results_eval'


def main():
    cfg=json.loads((ROOT/'retrain_roots.json').read_text())
    dirs=[ROOT/n for n in cfg['model_dirs']]
    assert len(dirs)==3
    locks=[json.loads((d/'model_lock.json').read_text()) for d in dirs]
    values=np.zeros((3,3,16,4)); moved=np.zeros((3,3,16,2))
    old_hash=sha(dirs[0]/'model_lock.json')
    for i,law in enumerate(LAWS):
        for b in range(16):
            d=EVAL/law/f'block_{b:03d}'
            md=json.loads((d/'meta.json').read_text())
            assert md['model_sha256']==old_hash and md['metrics_sha256']==sha(d/'metrics.npz')
            with np.load(d/'metrics.npz') as z:
                x=z['features']; ref=z['brier']
                for j,lock in enumerate(locks):
                    out=[]
                    for a,arm in enumerate(METHODS[:2]):
                        sc=predict(lock['models'][arm],x)
                        alloc=allocations(sc)
                        losses=z['oracle_stage_risk'][np.arange(8),np.searchsorted([2,4,8],alloc)]
                        moved[j,i,b,a]=np.mean(alloc!=z['counts'][:,a])
                        if j==0:
                            np.testing.assert_array_equal(alloc,z['counts'][:,a])
                            np.testing.assert_allclose(losses,ref[:,a],atol=1e-12,rtol=0)
                        out.append(losses.mean())
                    values[j,i,b]=out+ref.mean(0)[2:].tolist()
    byseed=values.mean((1,2))
    perlaw=values.mean((0,2))
    delta=risk_differences(byseed)
    seed_avg=delta.mean(0)
    rng=np.random.default_rng(32841120)
    groups=rng.integers(16,size=(10000,3,16))
    # Same sampled target blocks are used for each training seed.
    bootstrap_by_seed=np.stack([risk_differences(v[np.arange(3)[None,:,None],groups].mean((1,2))) for v in values],axis=1)
    fixed=bootstrap_by_seed.mean(1)
    seed_ids=rng.integers(3,size=(10000,3))
    crossed=bootstrap_by_seed[np.arange(10000)[:,None],seed_ids].mean(1)
    charge=[[v['training'][a]['labels_plus_fit_s'] for a in METHODS[:2]] for v in locks]
    assert all(max(pair)/min(pair)<1.02 for pair in charge)
    leave_one=np.array([np.delete(delta,j,axis=0).mean(0) for j in range(3)])
    seeds=[]
    for d,lk in zip(dirs,locks):
        codes=[]
        for arm,info in lk['training'].items():
            for row in info['blocks']:
                path=d/arm/f"block_{row['block']:03d}"
                assert sha(path/'data.npz')==row['data_sha256']
                ids=np.load(path/'data.npz')
                codes.extend(ids['prefix_seed'].tolist())
                codes.extend(ids['deployment_seeds'].ravel().tolist())
                codes.extend(json.loads((path/'meta.json').read_text())['bank_pair'])
        assert len(codes)==len(set(codes))
        seeds.append(set(codes))
    assert all(seeds[i].isdisjoint(seeds[j]) for i in range(3) for j in range(i))
    obj=dict(methods=METHODS,laws=LAWS,model_paths=cfg['model_dirs'],model_hashes=[sha(d/'model_lock.json') for d in dirs],
             training_counts=[[lk['training'][a]['n_environments'] for a in METHODS[:2]] for lk in locks],
             charged_seconds=charge,by_seed=byseed.tolist(),by_law=perlaw.tolist(),
             contrasts=[[METHODS[i],METHODS[j]] for i,j in PAIRS],delta_seed=delta.tolist(),pooled=seed_avg.tolist(),
             target_only_simultaneous_ci=simultaneous(seed_avg,fixed)[0].tolist(),
             crossed_simultaneous_ci=simultaneous(seed_avg,crossed)[0].tolist(),
             crossed_pointwise_ci=np.quantile(crossed,[.025,.975],axis=0).T.tolist(),
             leave_one_min=leave_one.min(0).tolist(),leave_one_max=leave_one.max(0).tolist(),
             allocation_changed_vs_seed_one=moved.mean((1,2)).tolist(),
             note='The online probabilities and latency were checked directly on seed one. Subsequent fits use exact feedback-dependent policy replay and counterfactual k-specific Brier on the same held-out blocks. Three independent training pairs; crossed bootstrap exploratory.')
    (EVAL/'full_retrain32.json').write_text(json.dumps(obj,indent=2))
if __name__=='__main__':main()
