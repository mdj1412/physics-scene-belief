"""CPU-only evaluation and independent policy replay under the v32 statistical plan."""
from pathlib import Path
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parent
LAWS = ('split_static', 'speed_weak', 'directional')
METHODS = ('broad', 'repeated', 'fixed4', 'uncertainty')
PAIRS = ((1, 0), (0, 2), (1, 2), (0, 3), (1, 3))
CONTRASTS = [f'{METHODS[a]}-minus-{METHODS[b]}' for a, b in PAIRS]
SEED = 32071023
BOOT = 10000


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def risk_differences(values):
    return np.stack([values[..., i] - values[..., j] for i, j in PAIRS], axis=-1)


def allocations(scores):
    # v30 stable ranking with ascending entity order as tie-breaker.
    scores = np.asarray(scores)
    if scores.shape != (8, 2):
        raise ValueError('Expected an 8-environment score panel with 2 stopping stages')
    counts = np.full(8, 2)
    active = np.arange(8)
    for k, before in enumerate((2, 4)):
        active = np.sort(active[np.argsort(-scores[active, k], kind='stable')[:len(active)//2]])
        counts[active] = 2 * before
    return counts


def predict(models, features):
    return np.stack([((features[:, s] - m['mean']) / m['sd']) @ m['beta'] + m['intercept']
                     for s, m in enumerate(models)], axis=1)


def audit_block(path, lock, model_hash, law, block):
    meta = json.loads((path / 'meta.json').read_text())
    for file, key in (('metrics.npz','metrics_sha256'),('choices.npz','choices_sha256'),
                      ('allocation.json','allocation_sha256')):
        assert sha(path / file) == meta[key], (path, key)
    assert (law,block,model_hash) == (meta['law'],meta['block'],meta['model_sha256'])
    with np.load(path / 'metrics.npz', allow_pickle=False) as data:
        assert tuple(data['methods'].tolist()) == METHODS
        y, p = data['labels'], data['probabilities']
        actions, counts, stage_risk = data['actions'], data['counts'], data['oracle_stage_risk']
        x, scores, unc = data['features'], data['scores'], data['uncertainty']
        recorded = data['brier']
        full, warm = data['full_ms'], data['warm_ms']
    allocation=json.loads((path/'allocation.json').read_text())
    assert tuple(allocation['methods'])==METHODS and allocation['model_sha256']==model_hash
    np.testing.assert_array_equal(allocation['counts'],counts)
    with np.load(path/'choices.npz',allow_pickle=False) as z:
        gaps=z['gaps']; assert gaps.shape==(8,3,128) and np.isfinite(gaps).all()
        for key,value in [('counts',counts),('actions',actions),('features',x),('scores',scores),('uncertainty',unc)]:
            np.testing.assert_array_equal(z[key],value)
    fallback=np.argsort(-gaps,axis=-1,kind='stable')[...,:13]
    all_actions=np.ones(gaps.shape,dtype=np.int8)
    np.put_along_axis(all_actions,fallback,3,axis=-1)
    selected=all_actions[np.arange(len(x))[:,None],np.searchsorted([2,4,8],counts)].transpose(0,2,1)
    np.testing.assert_array_equal(selected,actions)
    assert y.shape==(8,128) and p.shape==(8,128,2,32) and recorded.shape==(8,4)
    assert actions.shape==(8,128,4) and stage_risk.shape==(8,3) and np.isin(counts,[2,4,8]).all()
    assert x.shape==(8,2,15) and scores.shape==(8,2,2) and unc.shape==(8,2)
    assert np.isin(y,[0,1]).all() and np.isfinite(p).all() and ((p>=-1e-12)&(p<=1+1e-12)).all()
    assert np.isin(actions,[1,3]).all() and np.all((actions==3).sum(axis=1)==13)
    assert counts.shape==(8,4) and (counts.sum(axis=0)==32).all()
    est=np.stack([predict(lock['models'][arm], x) for arm in METHODS[:2]],axis=2)
    np.testing.assert_allclose(est, scores, rtol=1e-8, atol=1e-11)
    expected=np.stack([allocations(est[:,:,0]), allocations(est[:,:,1]), np.full(8,4),allocations(unc)],axis=1)
    np.testing.assert_array_equal(counts,expected)
    potential_loss = ((p - y[:,:,None,None])**2).mean(axis=3)
    all_losses=np.where(all_actions==3,potential_loss[:,None,:,1],potential_loss[:,None,:,0]).mean(axis=-1)
    np.testing.assert_allclose(all_losses,stage_risk,rtol=1e-7,atol=3e-8)
    recomputed = np.where(actions==3, potential_loss[:,:,1,None],potential_loss[:,:,0,None]).mean(axis=1)
    np.testing.assert_allclose(recomputed,recorded,rtol=1e-7,atol=3e-8)
    np.testing.assert_allclose(stage_risk[np.arange(len(x))[:,None],np.searchsorted([2,4,8],counts)],
                               recorded, rtol=1e-7,atol=3e-8)
    assert full.shape==warm.shape==((3,4) if block<6 else (0,4))
    assert np.isfinite(full).all() and np.all((warm>0)&(full>=warm))
    return {'brier':recorded.mean(axis=0),'full': full, 'warm': warm,
            'max_brier_error': np.max(np.abs(recomputed - recorded)),
            'max_score_error': np.max(np.abs(est - scores)), 'scores':scores, 'risk':stage_risk,
            'x':x,'counts':counts, 'hash':meta['metrics_sha256'],
            'seeds':meta['bank_pair']+meta['calibration_seeds']+meta['deployment_seeds']}


def intervals(samples):
    return np.quantile(samples, [.025,.975], axis=0).T


def simultaneous(estimate, samples):
    std = samples.std(axis=0, ddof=1)
    standardized=(samples-estimate)/np.where(std>0,std,1)
    critical=float(np.quantile(np.max(np.abs(standardized),axis=1),.95))
    return np.stack([estimate-critical*std,estimate+critical*std],axis=1),critical


def sample_ids(n_blocks, rng):
    return rng.integers(n_blocks, size=(BOOT,len(LAWS),n_blocks))


def brier_summary(values, ids):
    assert values.shape==(len(LAWS),16,len(METHODS))
    by_law = values.mean(axis=1)
    law_star = values[np.arange(len(LAWS))[None,:,None],ids].mean(axis=2)
    delta = risk_differences(by_law.mean(axis=0))
    delta_star = risk_differences(law_star.mean(axis=1))
    ci,crit=simultaneous(delta,delta_star)
    subsets = {}
    for name,indices in [(law,[i]) for i,law in enumerate(LAWS)]+[('historical_two',[0,1])]:
        local = by_law[indices].mean(axis=0)
        draws = law_star[:,indices].mean(axis=1)
        subsets[name]={'brier':local,'delta':risk_differences(local),'pointwise_ci':intervals(risk_differences(draws))}
    return {'mean':by_law.mean(axis=0),'individual_mean_pointwise_ci':intervals(law_star.mean(axis=1)),
            'delta':delta,'simultaneous_ci':ci,'critical':crit,'pointwise_ci':intervals(delta_star),
            'secondary':subsets,'pooled_block_risks':values}


def timing_summary(t,ids):
    assert t.shape==(len(LAWS),6,3,len(METHODS))
    # Each draw retains all methods/repeats of a block. Budget-sized chunks cap memory.
    def metrics(values):
        q=np.quantile(values, .9,axis=-2)
        ratios=np.stack([q[...,i]/q[...,j] for i,j in PAIRS],axis=-1)
        return np.concatenate([q,ratios],axis=-1)
    main=metrics(t.reshape(-1,len(METHODS)))
    samples=[]
    for sub in np.array_split(ids, 50):
        selected=t[np.arange(len(LAWS))[None,:,None],sub]
        samples.append(metrics(selected.reshape(len(sub),-1,len(METHODS))))
    ci=intervals(np.concatenate(samples))
    return {'p90_ms':main[:4], 'p90_ms_pointwise_ci':ci[:4], 'p90_ratio':main[4:],
            'ratio_pointwise_ci':ci[4:],'by_law_p90_ms':np.quantile(t.reshape(3,-1,4),.9,axis=1),
            'n_per_method':int(np.prod(t.shape[:-1])),'independent_timed_blocks':18}


def load_results(root=ROOT):
    lock_path=root/'results_train'/'model_lock.json'
    lock=json.loads(lock_path.read_text()); h=sha(lock_path)
    blocks=[[audit_block(root/'results_eval'/law/f'block_{i:03d}',lock,h,law,i) for i in range(16)]
            for law in LAWS]
    target=[z for row in blocks for item in row for z in item['seeds']]
    assert len(target)==len(set(target))==3*16*18
    source=[]
    for arm,info in lock['training'].items():
        for record in info['blocks']:
            folder=root/'results_train'/arm/f"block_{record['block']:03d}"
            meta=json.loads((folder/'meta.json').read_text())
            assert sha(folder/'data.npz')==record['data_sha256']==meta['data_sha256']
            assert sha(folder/'meta.json')==record['metadata_sha256']
            source+=meta['bank_pair']
            with np.load(folder/'data.npz',allow_pickle=False) as data:
                source+=data['prefix_seed'].ravel().tolist()+data['deployment_seeds'].ravel().tolist()
    assert len(source)==len(set(source)) and not set(source).intersection(target)
    return lock,blocks


def analyze(root=ROOT):
    lock, blocks = load_results(root)
    def collect(key,timed=False):
        return np.asarray([[b[key] for b in (row[:6] if timed else row)] for row in blocks])
    rng=np.random.default_rng(SEED)
    brier=brier_summary(collect('brier'),sample_ids(16,rng))
    times=sample_ids(6,rng)
    out={'methods':METHODS,'contrasts':CONTRASTS,'laws':LAWS,'bootstrap_seed':SEED,'bootstrap_size':BOOT,
         'brier':brier,'full':timing_summary(collect('full',True),times),'warm':timing_summary(collect('warm',True),times),
         'audit':{'max_brier_error':collect('max_brier_error').max(),'max_score_error':collect('max_score_error').max(),
                  'source_sealed':sha(root/'results_train'/'model_lock.json'),'target_artifact_hashes':collect('hash')},
         'training':{a:{k:v for k,v in info.items() if k!='blocks'} for a,info in lock['training'].items()}}
    out['training_relative_difference']={k:abs(lock['training']['broad'][k]-lock['training']['repeated'][k])/
                min(lock['training'][a][k] for a in METHODS[:2]) for k in ('labels_plus_fit_s','labels_plus_read_plus_fit_s')}
    return clean(out)


if __name__=='__main__':
    result=analyze()
    path=ROOT/'results_eval'/'paired_stats32.json'
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(path,result['brier']['delta'],result['brier']['simultaneous_ci'],flush=True)
