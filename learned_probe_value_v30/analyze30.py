"""Paired conditional inference on locked v30 environments. No access from the policy."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
LAWS = ('split_static', 'speed_weak')
METHODS = ['learned', 'fixed4', 'uncertainty']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quota(gaps):
    actions = np.ones_like(gaps, dtype=np.int8)
    order = np.argsort(-gaps, axis=1, kind='stable')
    np.put_along_axis(actions, order[:, :13], 3, axis=1)
    return actions


def selected(cost, actions):
    """Return losses with action replaced by policy axis: E,W,D,M."""
    return np.where(actions[:, :, None, :] == 3, cost[:, :, :, 1, None], cost[:, :, :, 0, None])


def pair_accuracy(score, target):
    i, j = np.triu_indices(len(score), 1)
    order = np.sign(score[i]-score[j])*np.sign(target[i]-target[j])
    return (.5 + .5 * order).mean()


def inspect_block(path, nullmeans):
    with np.load(path) as data:
        z = {k:data[k] for k in data.files}
    meta=json.loads(path.with_name('metrics.json').read_text())
    for nm,k in [('metrics.npz','metrics_sha256'),('allocation_lock.json','allocation_sha256'),('choices.npz','choices_sha256')]:
        assert digest(path.with_name(nm)) == meta[k]
    assert meta['model_sha256'] == digest(RESULTS/'model_lock.json')
    assert np.allclose(((z['probabilities']-z['labels'][:,:,None,None,None])**2).mean(-1), z['cost'], atol=1e-11,rtol=0)
    counts = z['counts']
    assert np.array_equal(counts.sum(0), [32]*3) and np.all(counts[:, 1] == 4)
    for j in (0, 2):
        assert list(sorted(counts[:, j])) == [2, 2, 2, 2, 4, 4, 8, 8]
    for method,score_key in ((0,'scores'),(2,'uncertainty')):
        active=np.arange(8)
        for stage,threshold in enumerate([4,8]):
            expected=active[np.argsort(-z[score_key][active,stage],kind='stable')[:len(active)//2]]
            assert np.array_equal(np.sort(expected),np.flatnonzero(counts[:,method]>=threshold))
            active=np.sort(expected)
    if 'methods' in z:
        assert list(z['methods']) == METHODS
    gaps = z['gap']
    assert gaps.shape == (8, 128, 3)
    cost = z['cost']
    assert cost.shape == (8, 128, 2, 2) and np.all((cost >= 0) & (cost<=1+1.e-10))
    base_actions = [quota(gaps), np.where(gaps>1.e-12, 3, 1)]
    levels = np.searchsorted([2,4,8], counts)
    risks, random_risks, changes = [], [], []
    for acts, key in zip(base_actions, ('quota_action', 'stream_action')):
        actual = np.take_along_axis(acts, levels[:, None, :], axis=2)
        assert np.array_equal(actual,z[key])
        risks.append(selected(cost, actual).mean((0,1)))
        random_risks.append(selected(cost,acts).mean((0,1)) @ np.array([.5,.25,.25]))
        changes.append((actual[:,:,:1]!=actual[:,:,1:]).mean((0,1)))
    assert np.all(np.sum(z['quota_action']==3, axis=1) == 13)
    potential = selected(cost,base_actions[0])[:,:,0,:].mean(1)
    scores = z['scores']
    stage_stats=[]
    for stage,active in enumerate([np.ones(8,dtype=bool), counts[:,0]>=4]):
        pred=scores[active,stage]
        gain=(potential[:,stage]-potential[:,stage+1])[active]
        retained=counts[active,0]>= ([4,8][stage])
        stage_stats.append([pred.mean(),gain.mean(),np.abs(pred-gain).mean(),
                            np.abs(gain).mean(),np.abs(gain-nullmeans[stage]).mean(),
                            pair_accuracy(pred,gain),gain[retained].mean()-gain[~retained].mean(),
                            ((pred-gain)**2).mean(), ((gain-nullmeans[stage])**2).mean()])
    label=z['labels'].astype(bool);source=z['source_labels'].astype(bool)
    physical = [label.mean(),source.mean(),(label!=source).mean(),z['factual_shift_norm'].mean(),z['probe_shift_norm'].mean()]
    return dict(risk=risks, random=random_risks, switches=changes,
                probe=(z['stream_action']==3).mean((0,1)), stages=stage_stats,
                physical=physical, realloc=np.mean(counts[:,0]!=counts[:,2]))


def summarize(bylaw, samples, key):
    law_values=[np.array([b[key] for b in blocks]) for blocks in bylaw]
    values=np.stack(law_values)
    boot=np.stack([v[i].mean(1) for v,i in zip(law_values,samples)],axis=1)
    return values.mean(1),boot


def point_ci(point, boot):
    return dict(mean=point, interval=np.quantile(boot, [.025,.975], axis=0).T)


def clean(o):
    if isinstance(o,np.ndarray):return o.tolist()
    if isinstance(o,np.generic):return o.item()
    raise TypeError(str(type(o)))


def main():
    model=json.loads((RESULTS/'model_lock.json').read_text())
    means=[v['actual_mean'] for v in model['training_diagnostics']]
    bylaw=[[inspect_block(p,means) for p in sorted((RESULTS / law).glob('block_*/metrics.npz'))] for law in LAWS]
    assert [len(x) for x in bylaw] == [16,16]
    rng=np.random.default_rng(300118)
    # Target-law seed and bank partitions, verified against the runner manifest.
    draws=[rng.integers(16,size=(10000,16)) for _ in LAWS]
    risk,boot=summarize(bylaw,draws,'risk')
    delta=risk[:,:,:,:1]-risk[:,:,:,1:]
    bd=boot[:,:,:,:,:1]-boot[:,:,:,:,1:]
    point=delta[:,0,0,:].mean(0)
    sample=bd[:,:,0,0,:].mean(1)
    sd=sample.std(axis=0,ddof=1)
    crit=np.quantile(np.max(np.abs(sample-point)/np.where(sd>0,sd,1),axis=1),.95)
    random,rboot=summarize(bylaw,draws,'random')
    primary=dict(contrasts=point,sim95=np.stack([point-crit*sd,point+crit*sd],axis=1),
                 rank_family='learned minus fixed4,learned minus uncertainty; two-sided familywise 95%')
    output=dict(methods=METHODS,laws=LAWS,stage_names=['2to4 all','4to8 learned survivors'],
        stage_columns=['prediction','realized','mae','zero_mae','trainmean_mae','pair_concordance','retained_minus_stopped_gain','mse','trainmean_mse'],
        physical_columns=['target_stick','source_stick','label_flip','mean_factual_norm','mean_probe_norm'],
        policy_order=['quota','stream'],decoder_order=['source','matched'],risk_by_law=risk,risk_pooled=risk.mean(0),primary=primary,
        law_delta=delta,law_pointwise95=np.moveaxis(np.quantile(bd,[.025,.975],axis=0),0,-1),
        pooled_delta=delta.mean(0),pooled_pointwise95=np.moveaxis(np.quantile(bd.mean(1),[.025,.975],axis=0),0,-1),
        random_by_law=random,random_pooled=random.mean(0),
        vs_random=point_ci((risk[:,0,0,0]-random[:,0,0]).mean(),(boot[:,:,0,0,0]-rboot[:,:,0,0]).mean(1)))
    for k in ('physical','probe','switches','stages','realloc'):
        p,b=summarize(bylaw,draws,k)
        output[k]=dict(law=p,pooled=p.mean(0),pooled95=np.moveaxis(np.quantile(b.mean(1),[.025,.975],axis=0),0,-1))
    (RESULTS/'analysis30.json').write_text(json.dumps(output,default=clean,indent=2,allow_nan=False))

if __name__=='__main__':
    main()
