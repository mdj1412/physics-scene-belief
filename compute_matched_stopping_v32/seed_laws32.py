"""Independent CPU policy/Brier replay, crossed bootstrap audit and seed × law attribution."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
LAWS=('split_static','speed_weak','directional')
ARMS=('broad','repeated')
PAIRS=((1,0),(0,2),(1,2),(0,3),(1,3))
RUNS=10000
SEED=32073511


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose(x, models):
    # Python stable sort independent of the online NumPy implementation.
    counts=np.full(8,2)
    active=list(range(8))
    for s,(k,model) in enumerate(zip((2,4),models)):
        scores={}
        for e in active:
            centered=(x[e,s]-model['mean'])/model['sd']
            scores[e]=float(np.dot(centered,model['beta'])+model['intercept'])
        active=sorted(sorted(active,key=lambda e:(-scores[e],e))[:len(active)//2])
        counts[active]=2*k
    return counts


def verify_source_dirs(paths):
    locks=[]
    for d in paths:
        model_path=d/'model_lock.json'
        lock=json.loads(model_path.read_text())
        for a in ARMS:
            ledger=lock['training'][a]
            manifest=json.loads((d/a/'manifest.json').read_text())
            assert manifest['complete'] and manifest['n_environments']==ledger['n_environments']
            np.testing.assert_allclose(manifest['elapsed_s'],ledger['labels_s'])
        cost=[lock['training'][a]['labels_plus_fit_s'] for a in ARMS]
        assert max(cost)/min(cost)<1.02, (d,cost)
        locks.append(lock)
    return locks


def replay(locks):
    # axes training seed, law, deployment block, outcome (policy loss or stage effect).
    values=np.zeros((len(locks),3,16,4));stage=np.zeros((len(locks),3,16,2))
    for law_idx,law in enumerate(LAWS):
        for b in range(16):
            path=ROOT/'results_eval'/law/f'block_{b:03d}'; md=json.loads((path/'meta.json').read_text())
            assert md['choices_sha256']==sha(path/'choices.npz') and md['metrics_sha256']==sha(path/'metrics.npz')
            with np.load(path/'choices.npz') as z:
                x, gaps, ref_counts=z['features'], z['gaps'], z['counts']
            with np.load(path/'metrics.npz') as z:
                labels,p=z['labels'],z['probabilities']; online=z['brier']; recorded=z['oracle_stage_risk']
            risk=np.empty((8,3))
            potential=((p-labels[:,:,None,None])**2).mean(axis=-1)
            for e in range(8):
                for k in range(3):
                    use=sorted(range(128),key=lambda j:(-gaps[e,k,j],j))[:13]
                    cost=potential[e,:,0].copy();cost[use]=potential[e,use,1]
                    risk[e,k]=cost.mean()
            np.testing.assert_allclose(risk,recorded,atol=3e-12,rtol=0)
            for s,lock in enumerate(locks):
                decomposition=[]
                for j,a in enumerate(ARMS):
                    counts=choose(x,lock['models'][a]);level=np.searchsorted((2,4,8),counts)
                    chosen=risk[np.arange(8),level]
                    if s==0:
                        np.testing.assert_array_equal(counts,ref_counts[:,j]);np.testing.assert_allclose(chosen,online[:,j],atol=3e-12,rtol=0)
                    values[s,law_idx,b,j]=chosen.mean()
                    terms=[np.mean((counts>=4)*(risk[:,1]-risk[:,0])),np.mean((counts==8)*(risk[:,2]-risk[:,1]))]
                    np.testing.assert_allclose(risk[:,0].mean()+sum(terms),chosen.mean(),atol=3e-14,rtol=0)
                    decomposition.append(terms)
                for j in (2,3):
                    lv=np.searchsorted([2,4,8],ref_counts[:,j]);vals=risk[np.arange(8),lv]
                    np.testing.assert_allclose(vals,online[:,j],atol=3e-12,rtol=0)
                    values[s,law_idx,b,j]=vals.mean()
                stage[s,law_idx,b]=np.subtract(*decomposition[::-1])
    np.testing.assert_allclose(stage.sum(axis=-1),values[...,1]-values[...,0],rtol=0,atol=2e-14)
    return values,stage


def bands(x):
    return np.moveaxis(np.quantile(x,[.025,.975],axis=0),0,-1)


def interactions(samples):
    return samples[...,2]-samples[...,:2].mean(axis=-1)


def cross_draws(sources, rng, block_indices):
    # Common target block multiplicities across all source fits, independent by law.
    ns,nl,nb,p=sources.shape
    counts=np.eye(nb,dtype=np.int32)[block_indices].sum(axis=2)
    raw=np.einsum('rlb,slbe->rsle',counts,sources)/nb
    seed_ids=rng.integers(ns,size=(RUNS,ns))
    weights=np.eye(ns,dtype=np.int32)[seed_ids].sum(axis=1)
    crossed=np.einsum('rs,rsle->rle',weights,raw)/ns
    return raw,crossed


def main():
    paths=[ROOT/d for d in json.loads((ROOT/'retrain_roots.json').read_text())['model_dirs']]
    locks=verify_source_dirs(paths); vals,stage=replay(locks)
    estimated=np.concatenate((vals[...,1,None]-vals[...,0,None],stage),axis=-1)
    assert estimated.shape==(3,3,16,3)
    point=estimated.mean(axis=2)
    rng=np.random.default_rng(SEED);groups=rng.integers(16,size=(RUNS,3,16))
    targets,crossed=cross_draws(estimated,rng,groups)
    # Interaction for signed repeated−broad difference; historical laws weighted equally.
    h=interactions(point[...,0]);h_target=interactions(targets[...,0]);h_cross=interactions(crossed[...,0])
    # Reproduce the separate full-retrain analysis including its bootstrap RNG sequence.
    ref=json.loads((ROOT/'results_eval/full_retrain32.json').read_text())
    np.testing.assert_allclose(ref['by_seed'],vals.mean(axis=(1,2)),atol=2e-14,rtol=0)
    np.testing.assert_allclose(ref['by_law'],vals.mean(axis=(0,2)),atol=2e-14,rtol=0)
    all5=np.stack([vals[...,i]-vals[...,j] for i,j in PAIRS],axis=-1)
    pooled=all5.mean((0,1,2));np.testing.assert_allclose(ref['pooled'],pooled,atol=2e-14,rtol=0)
    np.testing.assert_allclose(ref['delta_seed'],all5.mean((1,2)),atol=2e-14,rtol=0)
    rng=np.random.default_rng(32841120);groups=rng.integers(16,size=(RUNS,3,16))
    fixed_draw,retrain_draw=cross_draws(all5,rng,groups)
    for draws,key in ((fixed_draw.mean((1,2)),'target_only_simultaneous_ci'),(retrain_draw.mean(axis=1),'crossed_simultaneous_ci')):
        sd=draws.std(axis=0,ddof=1);c=np.quantile(np.max(np.abs((draws-pooled)/sd),axis=-1),.95)
        computed=np.stack((pooled-c*sd,pooled+c*sd),axis=-1)
        np.testing.assert_allclose(ref[key],computed,atol=2e-14,rtol=0)
    np.testing.assert_allclose(ref['crossed_pointwise_ci'],bands(retrain_draw.mean(axis=1)),atol=2e-14,rtol=0)
    report={'laws':LAWS,'outcomes':['final_brier_repeated_minus_broad','2_to_4_component','4_to_8_component'],
         'model_sha256':[sha(d/'model_lock.json') for d in paths], 'n_independent_target_blocks':48,
         'training_seed_pairs':len(locks),'bootstrap_seed':SEED,'bootstrap_draws':RUNS,
         'by_seed_law':point.tolist(),'by_seed_law_conditional_ci':bands(targets).tolist(),
         'by_seed':point.mean(axis=1).tolist(),'by_seed_conditional_ci':bands(targets.mean(axis=2)).tolist(),
         'by_law':point.mean(axis=0).tolist(),'by_law_crossed_ci':bands(crossed).tolist(),
         'pooled':point.mean(axis=(0,1)).tolist(),'pooled_crossed_ci':bands(crossed.mean(axis=1)).tolist(),
         'directional_minus_historical_by_seed':h.tolist(),'interaction_by_seed_conditional_ci':bands(h_target).tolist(),
         'directional_minus_historical_average':h.mean(),'interaction_crossed_ci':bands(h_cross).tolist(),
         'max_abs_stage_sum_error':float(np.abs(stage.sum(-1)-(vals[...,1]-vals[...,0])).max()),
         'full_retrain_reference_sha256':sha(ROOT/'results_eval/full_retrain32.json'),'full_retrain_check':'PASS, <=2e-14'}
    (ROOT/'results_eval/seed_laws32.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    render(report)


def render(d):
    def effect(value,interval):
        return f'{value:+.6f} [{interval[0]:+.6f}, {interval[1]:+.6f}]'
    p=np.asarray(d['by_seed_law']);ci=np.asarray(d['by_seed_law_conditional_ci']); pooled=np.asarray(d['by_law']);pc=np.asarray(d['by_law_crossed_ci'])
    out=['# Training seed별 contact law와 stopping stage 효과','',
      'Loss 차이는 repeated(2×2)에서 broad(1×1)를 뺀 값이다. 양수는 repeated의 Brier가 더 높음을 뜻한다. Primary seed 1의 directional 결과를 확인한 후 이 분석을 정했고, 나머지 두 seed를 새로 학습했다. 세 seed 모두 같은 48개 target blocks에서 평가했다.', '',
      '| Training seed | split_static | speed_weak | directional | directional − historical 평균 |', '|---|---:|---:|---:|---:|']
    for s in range(3):
        row=[effect(p[s,l,0],ci[s,l,0]) for l in range(3)]+[effect(d['directional_minus_historical_by_seed'][s],d['interaction_by_seed_conditional_ci'][s])]
        out+=['| '+str(s+1)+' | '+' | '.join(row)+' |']
    row=[effect(pooled[l,0],pc[l,0]) for l in range(3)]+[effect(d['directional_minus_historical_average'],d['interaction_crossed_ci'])]
    out+=['| 3 seed 평균 | '+' | '.join(row)+' |','','각 seed의 괄호는 같은 law 안에서 16개 target blocks를 복원추출한 pointwise 95% interval이다. 세 seed 평균에는 독립 training seed pair도 같이 재표집했다. 각 draw에서 target block index는 세 seed에 공통으로 적용했다. Historical 평균에는 split_static과 speed_weak의 가중치를 절반씩 주었다. Independent training seed가 3개이고, 개별 law·stage 비교의 multiplicity를 조정하지 않았으므로 crossed intervals는 탐색적으로 해석한다.','','## Stage별 ΔBrier 기여','','각 policy에 대해 4회까지 진행한 환경에서 R4−R2, 8회까지 진행한 환경에서 R8−R4를 계산하고, 각 합을 전체 환경 8개로 나눈다. 아래에서는 두 policy의 stage별 손실 변화를 뺀다. 4→8 기여에는 앞 stage에서 살아남은 환경의 구성이 반영된다.','',
       '| Training seed / law | 2→4 기여 | 4→8 기여 | 전체 ΔBrier |','|---|---:|---:|---:|']
    for s in range(3):
        for l,law in enumerate(LAWS):
            a=p[s,l];out.append(f'| {s+1} / {law} | {a[1]:+.6f} | {a[2]:+.6f} | {a[0]:+.6f} |')
        a=p[s].mean(axis=0);out.append(f'| {s+1} / 세 law 평균 | {a[1]:+.6f} | {a[2]:+.6f} | {a[0]:+.6f} |')
    c=np.asarray(d['pooled_crossed_ci']);v=d['pooled']
    out+=['| 세 seed·세 law 평균 | '+' | '.join(effect(v[i],c[i]) for i in (1,2,0))+' |','',
         f"각 block의 실제 ΔBrier와 두 stage의 합 사이 최대 잔차는 {d['max_abs_stage_sum_error']:.2e}였다. 전체 48개 block의 probability로 각 observation의 Brier를 다시 계산하고 stable top-13과 순차적 stopping을 재생했다. 독립 코드가 full-budget 재학습 분석의 policy별 Brier, 다섯 contrasts, 세 seed pooled target-only 및 crossed intervals를 절대 오차 2×10⁻¹⁴ 이내로 재현했다.",'',
         '[분석 계획](SEED_LAWS_PLAN_KO.md) · [전체 추정치와 stage별 interval](results_eval/seed_laws32.json) · [독립 검증 코드](seed_laws32.py)','']
    (ROOT/'SEED_LAWS_KO.md').write_text('\n'.join(out))


if __name__=='__main__':
    main()
