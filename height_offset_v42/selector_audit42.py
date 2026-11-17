"""Three-action selection and value of learning the true offset under identical observations."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
from selector42 import BETA, FIXED, ROOT, load_selector, select

K=3
ENTRIES=('fixed','optimizer','selector','mixed','known_empirical','pooled_empirical',
         'nominal_empirical','known_crossfit','pooled_crossfit','oracle_changes_vs_nominal',
         'choice_zero','choice_fixed','choice_optimizer','contrib_fixed','contrib_optimizer')


def interval(x):
    x=np.asarray(x)
    m=x.mean(0); h=t.ppf(.975,len(x)-1)*x.std(0,ddof=1)/np.sqrt(len(x))
    return dict(mean=m.tolist(),lower=(m-h).tolist(),upper=(m+h).tolist())


def splits(seeds):
    # Use precisely the v41/v42 deployment stream; random split of unique IDs, not draws.
    all_masks=[]
    for seed in seeds:
        ids=np.random.default_rng(np.random.SeedSequence([seed,270000,4120])).integers(2048,size=128)
        u,inv=np.unique(ids,return_inverse=True)
        rng=np.random.default_rng(np.random.SeedSequence([seed,420881]))
        group=[]
        for _ in range(4):
            mask=np.zeros(len(u),bool);mask[rng.permutation(len(u))[:len(u)//2]]=True
            row=mask[inv]
            assert not np.intersect1d(ids[row],ids[~row]).size
            group.append(row)
        all_masks.append(group)
    return np.asarray(all_masks).transpose(1,0,2)


def sums(y,base,cc,z,mask=None):
    """Delta Brier summed across objects. Axis order offset, environment, noise, action."""
    use=np.ones(base.shape[:2]) if mask is None else mask
    values=[]
    for scale in (FIXED,cc):
        odds=np.exp(BETA*z*scale)[:,None,None]
        step=base*(1-base)*(odds-1)/(1+base*(odds-1))
        const=np.einsum('en,enr->er',use,step*(2*base+step))
        linear=np.einsum('oen,enr->oer',y*use[None],step,optimize=True)
        values.append(const[None]-2*linear)
    return np.stack([np.zeros_like(values[0]),*values],-1)


def take(cost,index):
    return np.take_along_axis(cost,np.broadcast_to(index,cost.shape[:3])[...,None],axis=-1)[...,0]


def read_block(path,model):
    m=json.loads(path.with_suffix('.json').read_text())
    split=splits(m['deployment_seeds'])
    with np.load(path) as f:
        a={k:f[k] for k in ('offsets_mm','labels','risk','p_base','choice','z','weights','actions','first3_y','fourth_y','dirs')}
    offs=a['offsets_mm']; nonzero=offs!=0; zero=int(np.flatnonzero(~nonzero)[0])
    stats=[]; global_err=0.
    for sign in range(2):
        base=a['p_base'][2,sign];z=a['z'][sign]
        for reg in range(3):
            w=a['weights'][sign,reg];y=a['labels'][:,sign,reg].astype(float)
            predicted=select(model,base,z,w,a['actions'][sign],a['first3_y'][sign],a['fourth_y'][sign,reg],a['dirs'][sign])
            np.testing.assert_array_equal(predicted,a['choice'][2,sign,reg])
            total=sums(y,base,w,z)
            risk=total/128
            ref=a['risk'][:,2,:,sign,reg]
            used=take(risk,predicted[None])
            global_err=max(global_err,np.max(abs(risk.mean(2).transpose(0,2,1)-(ref[:,:3]-ref[:,0:1]))),
                           np.max(abs(used.mean(2)-(ref[:,3]-ref[:,0]))))
            known=risk.argmin(-1)
            pooled=risk[nonzero].mean(0).argmin(-1)[None]
            nominal=known[zero][None]
            cf=[];bf=[]
            for mask in split:
                half=sums(y,base,w,z,mask)
                other=total-half
                off_cf=(take(other,half.argmin(-1))+take(half,other.argmin(-1)))/128
                blind_cf=(take(other,half[nonzero].mean(0).argmin(-1)[None])+
                          take(half,other[nonzero].mean(0).argmin(-1)[None]))/128
                cf.append(off_cf.mean((1,2)));bf.append(blind_cf.mean((1,2)))
            v=np.column_stack([risk[...,1].mean((1,2)),risk[...,2].mean((1,2)),used.mean((1,2)),
               np.zeros(len(offs)),take(risk,known).mean((1,2)),take(risk,pooled).mean((1,2)),
               take(risk,nominal).mean((1,2)),np.mean(cf,axis=0),np.mean(bf,axis=0),
               (known!=nominal).mean((1,2)),
               *[np.full(len(offs),(predicted==k).mean()) for k in range(3)],
               (used*(predicted[None]==1)).mean((1,2)),(used*(predicted[None]==2)).mean((1,2))])
            stats.append(v)
    assert global_err<2.e-12
    avg=np.mean(stats,axis=0)
    avg[:,3]=avg[:,11]*avg[:,0]+avg[:,12]*avg[:,1]
    return avg,global_err


def main():
    meta=json.loads((ROOT/'results/completion.json').read_text()); assert meta['blocks']==288
    model=load_selector(); off=np.array(meta['offsets_mm']); nonzero=off!=0
    panels=[]; maxima=[]
    for p in range(12):
        out=[]
        for c in meta['cells']:
            v,e=read_block(ROOT/'results'/c/f'catalog_{p:02d}.npz',model)
            out.append(v);maxima.append(e)
        panels.append(np.mean(out,axis=0))
    a=np.array(panels)
    v=a[:,nonzero].mean(1)
    delta={
        'selector_minus_zero':v[:,2],
        'selector_minus_fixed':v[:,2]-v[:,0],
        'selector_minus_optimizer':v[:,2]-v[:,1],
        'selector_minus_matched_frequency':v[:,2]-v[:,3],
        'known_minus_pooled_empirical':v[:,4]-v[:,5],
        'known_minus_pooled_crossfit':v[:,7]-v[:,8],
        'selector_minus_pooled_crossfit':v[:,2]-v[:,8],
        'known_crossfit_minus_zero':v[:,7],
        'pooled_crossfit_minus_zero':v[:,8],
        'pooled_minus_nominal_empirical':v[:,5]-v[:,6],
    }
    payload=dict(columns=ENTRIES,offsets=off.tolist(),panel_delta={k:interval(x) for k,x in delta.items()},
                 signed=a.mean(0).tolist(),nonzero=v.mean(0).tolist(),max_error=float(max(maxima)))
    (ROOT/'selector_audit42.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    lines=['# v42 보정 선택의 독립 검산','',
           '실제 contact offset마다 세 정책 중 가장 작은 Brier를 얻는 선택이 어떻게 바뀌는지 estimated decoder에서 다시 계산했다. 모든 값은 12개 독립 catalog에서 같은 물체·같은 noise를 대응시킨 결과다. 표의 음수는 0% 보정보다 낮은 Brier를 뜻한다.','',
           '| 실제 offset (mm) | Fixed−zero | Optimizer−zero | Selector−zero | 같은 빈도의 무작위 정책−zero | Nominal 최선과 실제 offset 최선이 다른 비율 |',
           '|---:|---:|---:|---:|---:|---:|']
    for o,row in zip(off,a.mean(0)):
        lines.append(f'| {o:+.3f} | {row[0]:+.6f} | {row[1]:+.6f} | {row[2]:+.6f} | {row[3]:+.6f} | {row[9]:.1%} |')
    m=v.mean(0)
    lines+=['',f'실제 selector의 0% / fixed / optimizer 비율은 {m[10]:.1%} / {m[11]:.1%} / {m[12]:.1%}였다. 같은 구매 관측이 들어와 모든 offset에서 선택은 같았다. Nonzero offset 균등 평균에서 fixed 선택의 전체 Brier 변화 기여는 {m[13]:+.6f}, optimizer 선택의 기여는 {m[14]:+.6f}였다. 각 값은 해당 action을 택한 경우의 loss 변화에 그 사용 빈도를 반영한 것이다.','',
          '| Nonzero 8개 offset의 paired 대조 | 평균 차이 [개별 95% CI] |','|---|---:|']
    names=['Selector−zero','Selector−fixed','Selector−optimizer','Selector−동일 빈도 무작위 정책',
           'Offset별 사후 최선−모든 offset에서 공유하는 사후 최선','Offset별 cross-fitted−모든 offset에서 공유하는 cross-fitted',
           'Selector−모든 offset에서 공유하는 cross-fitted','Offset별 cross-fitted−zero','모든 offset에서 공유하는 cross-fitted−zero',
           '모든 offset에서 공유하는 사후 최선−nominal 사후 최선 재사용']
    for name,(k,x) in zip(names,delta.items()):
        z=interval(x); lines.append(f"| {name} | {z['mean']:+.6f} [{z['lower']:+.6f}, {z['upper']:+.6f}] |")
    lines+=['','사후 최선은 해당 environment·noise마다 전체 128개 deployment label을 본 진단이다. Cross-fitting은 한쪽 half의 label로 정책을 선택하고 독립 prototype으로 구성된 반대쪽 half에서 평가한 뒤 역할을 바꾼 결과다. 실제 offset을 아는 경우와 모르는 경우에 어느 만큼 다른 선택이 가능한지 평가하며, 실제 selector는 어느 쪽의 deployment label도 쓰지 않는다. 무작위 정책은 각 cell·panel의 선택 빈도를 고정해 환경별 배정을 무작위화한 기대 Brier다.','',f'원 확률에서 재계산한 archive risk 차이 최대 절대 오차는 {max(maxima):.2e}였다. [원자료](selector_audit42.json) · [별도 분석 계획](SELECTOR_AUDIT_PLAN_KO.md) · [코드](selector_audit42.py)']
    (ROOT/'SELECTOR_AUDIT_KO.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__': main()
