"""Choose a prefix rule on cohort A and test a four-endpoint family on independent B."""
from pathlib import Path
from itertools import product
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results_prefix'
FAMILIES=('rounded_fine','rounded_coarse','ridged_fine','ridged_coarse')
DIRECTIONS=('plus','minus')
FEATURES = (('posterior_friction_sd',3),('paid_tangential_spread',11),('paid_velocity_covariance',14))
CANDIDATES=list(product(range(3),('upper','lower'),range(2),range(2)))
WORLDS=128
N_BOOT=10_000


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,obj):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def checked_panel(cohort):
    shape=(4,4,2,8,2)
    gain=np.empty(shape+(2,2)); change=np.empty_like(gain)
    x=np.empty(shape+(15,)); hashes=[]; seeds=set()
    for i,family in enumerate(FAMILIES):
        for block in range(4):
            common=None
            for direction,d in enumerate(DIRECTIONS):
                folder=OUT/cohort/f'{family}_{d}'/f'block_{block:03d}'
                m=json.loads((folder/'meta.json').read_text())
                assert sha(folder/'metrics.npz')==m['metrics_sha256'] and sha(folder/'choices.npz')==m['choices_sha256']
                hashes.extend([m['metrics_sha256'],m['choices_sha256']])
                with np.load(folder/'metrics.npz') as values,np.load(folder/'choices.npz') as choices:
                    labels=values['labels']
                    probabilities=values['probabilities']
                    assert probabilities.shape==(8,2,128,2,32) and np.isfinite(probabilities).all()
                    assert probabilities.min()>=-1e-12 and probabilities.max()<=1+1e-12
                    costs=((probabilities-labels[:,:,:,None,None])**2).mean(-1)
                    np.testing.assert_allclose(costs,values['cost'],atol=2e-14,rtol=0)
                    selected=choices['action']==3
                    np.testing.assert_array_equal(selected.sum(-1),np.full(selected.shape[:-1],13))
                    expected_top=np.argsort(-choices['gap'],axis=-1,kind='stable')[...,:13]
                    expected=np.zeros_like(selected); np.put_along_axis(expected,expected_top,True,axis=-1)
                    np.testing.assert_array_equal(expected,selected)
                    risk=np.where(selected,costs[:,None,None,:,:,1],costs[:,None,None,:,:,0]).mean(-1)
                    actual=risk[:,:,:1]-risk[:,:,1:]
                    np.testing.assert_allclose(actual,values['gain'],atol=2e-14,rtol=0)
                    gain[i,block,direction]=actual
                    changed=(selected[:,:,:1]!=selected[:,:,1:]).mean(-1)
                    np.testing.assert_array_equal(changed,values['switch'])
                    change[i,block,direction]=changed
                    x[i,block,direction]=choices['features']
                    for st,k in enumerate((2,4)):
                        tan=choices['paid_y'][:,:k,1]
                        vel=choices['paid_x'][:,:k,-2]
                        np.testing.assert_allclose(tan.std(1),choices['features'][:,st,11],atol=2e-14,rtol=0)
                        cov=((tan-tan.mean(1,keepdims=True))*(vel-vel.mean(1,keepdims=True))).mean(1)
                        np.testing.assert_allclose(cov,choices['features'][:,st,14],atol=2e-14,rtol=0)
                        extra=choices[f'additional_{k}']
                        for ids,futures in zip(choices['original_ids'][:,:k],extra):
                            for extra_ids in futures:
                                assert len(np.unique(np.r_[ids,extra_ids]))==2*k
                    paired=[choices[k] for k in ('x_hash','original_ids','additional_2','additional_4','x','alpha','prefix_seed','batch_seed','future_seed','bank_pair')]
                    if direction==0:
                        common=paired
                        ids=np.concatenate([choices[k].ravel() for k in ('prefix_seed','batch_seed','future_seed','bank_pair')])
                        assert len(ids)==len(set(ids)) and seeds.isdisjoint(ids);seeds.update(ids)
                    else:
                        for a,b in zip(common,paired):
                            np.testing.assert_array_equal(a,b)
    return dict(gain=gain,x=x,change=change,hashes=hashes,seeds=seeds)


def strata(panel,candidate):
    feature,side,stage,direction=candidate
    index=FEATURES[feature][1]
    observed=panel['x'][:,:,direction,:,stage,index]
    sorted_ids=np.argsort(-observed,axis=-1,kind='stable')
    high=sorted_ids[:,:,:4]
    low=sorted_ids[:,:,4:]
    if side=='lower':
        high,low=low,high
    gain=panel['gain'][:,:,direction,:,stage]
    switched=panel['change'][:,:,direction,:,stage]
    def split(array):
        high_mean=np.take_along_axis(array,high[:,:,:,None,None],axis=2).mean(2)
        low_mean=np.take_along_axis(array,low[:,:,:,None,None],axis=2).mean(2)
        return np.stack((high_mean,low_mean,high_mean-low_mean),axis=-1)
    return split(gain),split(switched)


def encoded(c):
    f,side,stage,d=c
    return dict(feature=FEATURES[f][0],half=side,stage=(2,4)[stage],direction=DIRECTIONS[d])


def enumerate_candidates(panel):
    output=[]
    for c in CANDIDATES:
        subgroup,_=strata(panel,c)
        target,other,excess=subgroup.mean((0,1,2,3))
        output.append(dict(rule=encoded(c),gain=float(target),other=float(other),excess=float(excess),
                           utility=float(min(target,excess))))
    return output


def freeze_winner():
    data=checked_panel('A')
    rows=enumerate_candidates(data)
    winner=int(np.argmax([x['utility'] for x in rows]))
    f,side,s,d=CANDIDATES[winner]
    threshold=float(np.median(data['x'][:,:,d,:,s,FEATURES[f][1]]))
    key=OUT/'discovery_lock.json'
    obj=dict(winner=winner,rule=rows[winner]['rule'],single_prefix_threshold=threshold,candidates=rows,source_hashes=data['hashes'],
             plan_sha256=sha(ROOT/'PREFIX_PLAN_KO.md'))
    if key.is_file():
        assert json.loads(key.read_text())==obj
    else:
        assert not list((OUT/'B').glob('*/block_*/meta.json'))
        save(key,obj)
    return obj


def reduce(f):
    return f.mean(axis=(0,1))


def simultaneous(values,draws):
    point=reduce(values)
    sampled=values[np.arange(4)[None,:,None],draws].mean((1,2))
    sd=sampled.std(0,ddof=1)
    z=np.divide(sampled-point,sd,out=np.zeros_like(sampled),where=sd>0)
    critical=np.quantile(np.max(np.abs(z),axis=1),.95)
    return point,np.stack((point-critical*sd,point+critical*sd),axis=1),sampled


def replicates(panel,winner):
    fixed=[strata(panel,(0,'upper',s,d))[0] for s in range(2) for d in range(2)]
    fixed=np.mean(fixed,axis=0)
    selected,switch=strata(panel,CANDIDATES[winner])
    paired=np.stack([fixed[...,0],fixed[...,2],selected[...,0],selected[...,2]],axis=-1)
    return paired,switch


def fixed_threshold(panel, candidate, threshold, draws):
    f,side,s,d=candidate
    observed=panel['x'][:,:,d,:,s,FEATURES[f][1]]
    use=observed>=threshold if side=='upper' else observed<threshold
    gain=panel['gain'][:,:,d,:,s].mean((-1,-2))
    masks=np.stack((use,~use),axis=-1)
    sums=np.sum(gain[:,:,:,None]*masks,axis=2)
    count=np.sum(masks,axis=2)
    def ratio(total,n):
        return np.divide(total,n,out=np.full(total.shape,np.nan),where=n>0)
    pt=ratio(sums.sum(1),count.sum(1)).mean(0)
    family_ids=np.arange(len(FAMILIES))[None,:,None]
    boot=ratio(sums[family_ids,draws].sum(2),count[family_ids,draws].sum(2)).mean(1)
    valid=np.isfinite(boot).all(1)
    with_excess=lambda n:np.stack((n[...,0],n[...,1],n[...,0]-n[...,1]),axis=-1)
    return dict(threshold=threshold,features_range=[float(observed.min()),float(observed.max())],
                group_counts_by_family=count.sum(1).tolist(),total_counts=count.sum((0,1)).tolist(),
                valid_resamples=int(valid.sum()),invalid_resamples=int((~valid).sum()),
                estimates=with_excess(pt).tolist() if np.isfinite(pt).all() else None,
                pointwise95=np.quantile(with_excess(boot[valid]),(.025,.975),axis=0).T.tolist()
                if valid.any() and np.isfinite(pt).all() else None)


def finalize():
    lock=json.loads((OUT/'discovery_lock.json').read_text())
    winner=lock['winner']
    a=checked_panel('A'); b=checked_panel('B')
    assert a['seeds'].isdisjoint(b['seeds']) and lock['source_hashes']==a['hashes']
    assert max(a['seeds'])<min(b['seeds'])
    policy_seeds=range(510_000_000,514_000_000)
    assert not any(s in policy_seeds for s in a['seeds']|b['seeds'])
    confirm=enumerate_candidates(b)
    assert confirm[winner]['rule']==lock['rule']
    draws=np.random.default_rng(330701).integers(4,size=(N_BOOT,4,4))
    summaries=[]
    for panel in (a,b):
        repeated,switch=replicates(panel,winner)
        raw=repeated.mean((2,3))
        point,ci,sampled=simultaneous(raw,draws)
        pointwise=np.quantile(sampled,(.025,.975),axis=0).T
        summaries.append(dict(paired_means=point.tolist(),simultaneous95=ci.tolist(),pointwise95=pointwise.tolist(),
                    diagonal_00=reduce(repeated[:,:,0,0]).tolist(),diagonal_11=reduce(repeated[:,:,1,1]).tolist(),
                    selected_switch_rates_target_other_excess=switch.mean((0,1,2,3)).tolist()))
    strat=[{'cohort':tag,'stage':k,'direction':d,
            'high_posterior_sd_gain_other_excess':strata(p,(0,'upper',s,i))[0].mean((0,1,2,3)).tolist()}
        for tag,p in zip(('A','B'),(a,b)) for s,k in enumerate((2,4)) for i,d in enumerate(DIRECTIONS)]
    f,half,stage,direction=CANDIDATES[winner]
    values=b['x'][:,:,direction,:,stage,FEATURES[f][1]]
    order=np.argsort(-values,axis=-1,kind='stable')
    masks={'upper':order[:,:,:4],'lower':order[:,:,4:]}
    gbar=b['gain'][:,:,direction,:,stage].mean((-1,-2))
    dist={}
    for label,side in [('target',half),('complement','lower' if half=='upper' else 'upper')]:
        signed=np.take_along_axis(gbar,masks[side],axis=2).ravel()
        dist[label]=dict(n=len(signed),positive=int((signed>0).sum()),fraction_positive=float(np.mean(signed>0)),
                         quartiles=np.quantile(signed,[.25,.5,.75]).tolist())
    result=dict(observed_gain_distribution=dist,endpoints=['fixed_high_absolute_gain','fixed_high_excess_over_low','selected_absolute_gain','selected_excess'],
                A=summaries[0],B=summaries[1],selected=lock['rule'],discovery_hash=sha(OUT/'discovery_lock.json'),
                confirmed_rules=[bool(np.min(np.asarray(summaries[1]['simultaneous95'])[idx:idx+2,0])>0) for idx in (0,2)],
                exceeds_003=[bool(np.min(np.asarray(summaries[1]['simultaneous95'])[idx:idx+2,0])>.003) for idx in (0,2)],
                prespecified_by_stage_direction=strat,all_B_rules=confirm,B_hashes=b['hashes'],
                frozen_numeric_threshold_B=fixed_threshold(b,CANDIDATES[winner],lock['single_prefix_threshold'],draws),
                seeds_A_count=len(a['seeds']),seeds_B_count=len(b['seeds']))
    save(OUT/'prefix_gain33.json',result)
    return result

if __name__=='__main__':
    finalize()
