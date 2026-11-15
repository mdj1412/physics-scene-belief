"""Evaluate regime-dependent gate routing with the environment, sign and noise paired."""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
CELLS=('sinusoid_atan','sinusoid_algebraic','quartic_atan','quartic_algebraic')
REGIMES=('strong','weak','null','global_null')
SPEEDS=(.50,.75,1.10)
BETA=.8289760825845313
FIELDS=('gate','always_minus_off','shrink_minus_off','variance','strength','direction_correct',
        'score','gate_minus_shrink','sign_correct_and_gate','correct_sign_sample')

def bootstrap(v,indices):
    return v[np.arange(4)[None,:,None],indices].mean((1,2))

def simultaneous(point,rep):
    std=rep.std(axis=0,ddof=1)
    z=(rep-point)/np.where(std<1.e-14,1.,std)
    critical=np.quantile(np.max(np.abs(z),axis=tuple(range(1,z.ndim))),.95)
    return np.stack((point-critical*std,point+critical*std),axis=-1)

def aggregate(v,indices):
    p=v.mean((0,1)); bt=bootstrap(v,indices)
    return dict(point=p.tolist(),simultaneous95=simultaneous(p,bt).tolist())

def read_record(path,sign):
    with np.load(path,allow_pickle=False) as f:
        z=f['z']
        p=f['p_base']
        ratio=np.exp(BETA*z)[None,:,None,None]
        pf=p*ratio/(1-p+p*ratio)
        y=f['labels'][None,:,:,None]
        risk=f['risk']
        np.testing.assert_allclose(risk[:,1]-risk[:,0],((pf-y)**2-(p-y)**2).mean((2,3)),
                                   atol=1.e-10,rtol=0)
        g=f['accept4'].astype(float)
        delta=risk[:,1]-risk[:,0]
        np.testing.assert_allclose(risk[:,3]-risk[:,0],g*delta,atol=1.e-12,rtol=0)
        correct=np.broadcast_to((z*sign)>0,g.shape)
        return np.stack((g,delta,risk[:,2]-risk[:,0],((pf-p)**2).mean((2,3)),f['near_strength4'],
                        correct,f['score4'],risk[:,3]-risk[:,2],correct*g,correct),axis=-1)

def route_scope(panel,regions):
    part=panel[:,:,regions]
    gates,delta,shrink,variance=tuple(part[...,i] for i in range(4))
    group=gates.mean(2,keepdims=True)
    exact=((gates-group)*delta).mean((2,3,5))
    price=(group*(1-group)*variance).mean((2,3,5))
    blend=(group*delta-group*(1-group)*variance-shrink).mean((2,3,5))
    total=part[...,7].mean((2,3,5))
    np.testing.assert_allclose(exact+price+blend,total,atol=1e-12,rtol=0)
    same=(gates.max(2)==gates.min(2)).mean((2,4))
    return np.stack((exact,price,blend,total,same),axis=-1)

def analyze():
    # cell x block x regime x orientation x speed x calibration environment x feature.
    panel=np.empty((4,8,4,2,3,8,len(FIELDS)))
    for c,cell in enumerate(CELLS):
        for b in range(8):
            folder=ROOT/'results'/cell/f'block_{b:03d}'
            for regid,regime in enumerate(REGIMES):
                for oi,suffix in enumerate(('plus','minus')):
                    if regime=='global_null' and oi:
                        panel[c,b,regid,oi] = panel[c,b,regid,0]
                        continue
                    panel[c,b,regid,oi]=read_record(folder/(regime+'_'+suffix+'.npz'),1 if oi==0 else -1)
    rng=np.random.default_rng(370912)
    ids=(2*rng.integers(4,size=(12000,4,4,2))+np.arange(2)).reshape(-1,4,8)
    routing={}
    groups=[('strong_weak_null',(0,1,2)),('weak_null',(1,2)),('strong_weak',(0,1)),('strong_null',(0,2))]
    for name,regions in groups:
        arr=route_scope(panel,regions)
        stats={k:aggregate(arr[:,:,:,j],ids) for j,k in enumerate(('routing','hard_random_penalty','blend_minus_fixed','actual_minus_fixed','identical_decisions'))}
        stats['routing_change_vs_0p50']=aggregate(arr[:,:,1:,0]-arr[:,:,:1,0],ids)
        routing[name]=stats
    local=panel[:,:,:3]
    by_orientation={}
    for oi,side in enumerate(('positive','negative')):
        regdata=local[:,:,:,oi].mean((2,4))
        by_orientation[side]=dict(
           sign_accuracy=float(regdata[...,5].mean()),
           gate_application=regdata[...,0].mean((0,1)).tolist(),
           gate_minus_shrink=aggregate(regdata[...,7],ids))
    correct=local[...,5].astype(bool)
    by_signquality={}
    for name,keep in [('correct',correct),('incorrect',~correct)]:
        # A single shared proposal for all 3 speeds and local regimes.
        den=keep.sum(axis=(0,1,2,3,5))
        avg=lambda x:np.sum(x*keep,axis=(0,1,2,3,5))/den
        by_signquality[name]=dict(fraction=keep.mean(),gateway_on=avg(local[...,0]).tolist(),
                                 original_to_full_gain=avg(-local[...,1]).tolist(),
                                 gate_minus_shrink=avg(local[...,7]).tolist())
    rank_results={}
    for name,pair in [('strong_vs_weak',(0,1)),('weak_vs_null',(1,2)),('strong_vs_null',(0,2))]:
        x=panel[:,:,pair[0],:,:,:,4]; y=panel[:,:,pair[1],:,:,:,4]
        diff=x-y
        # Exact density equalities can have harmless roundoff.
        tol=1.e-12
        score=(diff>tol)+.5*(np.abs(diff)<=tol)
        rank_results[name]=dict(counterfactual_rank=aggregate(score.mean((2,4)),ids),
                                 tie_fraction=(np.abs(diff)<=tol).mean((0,1,2,4)).tolist())
    # Per-environment sign of realized Brier gain for applying full correction vs no correction.
    gates=local[...,0].astype(bool); good=local[...,1]<0
    npos=good.sum(axis=(0,1,2,3,5)); nneg=(~good).sum(axis=(0,1,2,3,5))
    tpr=(gates&good).sum(axis=(0,1,2,3,5))/npos
    tnr=(~gates&~good).sum(axis=(0,1,2,3,5))/nneg
    utility=dict(true_positive_rate=tpr.tolist(),true_negative_rate=tnr.tolist(),
                 balanced_accuracy=(.5*(tpr+tnr)).tolist(),observed_beneficial_rate=good.mean((0,1,2,3,5)).tolist())
    return dict(speeds=SPEEDS,regime_permutation=routing,by_orientation=by_orientation,
                by_quality_of_first3_sign=by_signquality,paired_strength=rank_results,utility=utility)

def fmt(x):return f'{x:+.5f}'
def interval(v):return '['+', '.join(fmt(x) for x in v)+']'
def pc(x):return f'{100*x:.1f}%'

def report(o):
    lines=['## 같은 환경에서 보정 배정을 바꾼 결과','',
           '각 calibration 환경에서 물체, sensor noise, 첫 세 접촉, signed proposal을 고정했다. 세 local regime에 실제 배정한 gate decision을 같은 세 regime 사이에서 무작위로 교환한 경우와 실제 배정을 비교했다. 각 표면은 원래 관측과 deployment 확률, 실제 query outcome을 유지한다. 따라서 물체나 방향의 차이를 이용해 얻은 이득을 표면 강도를 구별한 이득과 분리할 수 있다.','',
           '| 마지막 speed | 세 regime 사이 실제 − 재배정 Brier [95%] | Weak와 null 사이 실제 − 재배정 Brier [95%] | 세 regime에서 같은 결정 |','|---|---:|---:|---:|']
    a=o['regime_permutation']['strong_weak_null']
    w=o['regime_permutation']['weak_null']
    for j,s in enumerate(SPEEDS):
        z,a0,w0=a['identical_decisions']['point'][j],a['routing'],w['routing']
        lines.append(f'| {s:.2f} | {fmt(a0["point"][j])} {interval(a0["simultaneous95"][j])} | {fmt(w0["point"][j])} {interval(w0["simultaneous95"][j])} | {pc(z)} |')
    changed=a['routing_change_vs_0p50']
    low_changed=w['routing_change_vs_0p50']
    lines+=['',f'+0.50에서 +0.75로 속도를 바꿀 때 세 표면의 조건부 배정 효과 변화는 {fmt(changed["point"][0])} {interval(changed["simultaneous95"][0])}였으며, weak와 local-null 두 표면만 비교하면 {fmt(low_changed["point"][0])} {interval(low_changed["simultaneous95"][0])}였다. +1.10에서 기준 속도 대비 세 표면의 변화는 {fmt(changed["point"][1])} {interval(changed["simultaneous95"][1])}였다. 이 구간은 두 속도의 기준 대비 변화에 대해 동시에 보정했다.']
    lines+=['','Δ가 음수이면 같은 빈도로 보정을 적용해도 실제 배정이 더 좋다. 여기서 보정을 켤지는 첫 세 번의 signed proposal z와 실제 해당 속도의 네 번째 velocity로 결정한다. 95% interval은 object–surface와 첫 방향으로 층화한 paired block bootstrap 12,000회로 각 지표에서 세 속도를 동시에 보정했다.','',
            '| Speed | 표면별 배정 효과 | Hard randomization penalty | 연속 혼합 − fixed shrinkage | 합계 Gate − fixed shrinkage |','|---|---:|---:|---:|---:|']
    for j,s in enumerate(SPEEDS):
        parts=[a[f]['point'][j] for f in ('routing','hard_random_penalty','blend_minus_fixed','actual_minus_fixed')]
        lines.append('| '+' | '.join([f'{s:.2f}',*[fmt(x) for x in parts]])+' |')
    lines+=['','세 항의 합은 각 paired block에서 관측된 Gate−fixed Brier와 일치한다. 세 표면에서 평균 적용률을 h, 원 probability를 p, full correction probability를 q라고 하자. 같은 h로 켜고 끄는 randomized forecast는 연속 확률 p+h(q−p)보다 h(1−h)(q−p)²만큼 더 큰 기대 Brier를 갖는다. Hard randomization penalty는 이 convexity cost의 평균이다. 이 항등식을 위해 정의한 h에는 같은 환경의 세 반사실적 표면이 모두 필요하며, 배포 가능한 baseline의 성능으로 사용할 수는 없다.','',
       '| Speed | Strong > weak · paired ordering | Weak > null · paired ordering | 보정이 유리한 환경 판별 balanced accuracy |','|---|---:|---:|---:|']
    for j,s in enumerate(SPEEDS):
        lines.append('| '+' | '.join([f'{s:.2f}',*[pc(o['paired_strength'][key]['counterfactual_rank']['point'][j]) for key in ('strong_vs_weak','weak_vs_null')],pc(o['utility']['balanced_accuracy'][j])])+' |')
    lines+=['','Paired ordering은 같은 물체·orientation·noise에서 더 강한 표면이 더 큰 posterior strength를 받는 비율이다. 같은 값에는 0.5를 준다. 서로 독립인 두 환경을 비교하는 AUROC와 이 진단은 다른 질문을 다룬다. Balanced accuracy의 positive label은 같은 환경의 독립 deployment 물체 128개에서 full correction Brier가 원래 확률보다 낮았던 경우다. 참측정량의 부호는 유한 표본에서 흔들릴 수 있다.','',
        '| 첫 세 접촉의 방향 판독 | 환경 비중 | 0.50 / 0.75 / 1.10의 gate 적용률 | 세 속도의 Gate − fixed shrinkage |','|---|---:|---|---|']
    for key,lab in [('correct','방향 정답'),('incorrect','방향 오답')]:
        r=o['by_quality_of_first3_sign'][key]
        lines.append(f'| {lab} | {pc(r["fraction"])} | '+', '.join(pc(x) for x in r['gateway_on'])+' | '+', '.join(fmt(x) for x in r['gate_minus_shrink'])+' |')
    lines.extend(['','세 local regime에서는 방향 판독에 쓰는 첫 세 고속 관측이 같아 판독의 정오가 속도와 무관하다. Global-null에서는 실제 방향이 정의되지 않아 이 표에 포함하지 않았다. [배정 분석 수치](routing37.json) · [코드](routing37.py)'])
    (ROOT/'ROUTING_KO.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    results=analyze()
    (ROOT/'routing37.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    report(results)
    print(results['regime_permutation']['strong_weak_null']['routing'])
