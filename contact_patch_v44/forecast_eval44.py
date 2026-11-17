"""Score preregistered v44 direction calls on fresh Bullet outcomes; bootstrap catalog units."""
from pathlib import Path
import json
from itertools import product
import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parent
SHAPES=('fold','ripple')
H=(19.,31.)
R=(.5,2.)
L=(0.,2.)
DELTA=(-.5,0.,.5)
SCOPES=('all','deformed')
BOOT=20000
SEED=440517
METHODS=('support','carry','harm')


def panel_risks(panel):
    shape=(2,2,3,2,2,3)
    sums=np.zeros(shape)
    counts=np.zeros(shape,int)
    valid_n=0
    for i,h,v,surf,law in product(range(2),range(2),range(3),('inlay','woven'),('rational','gaussian_rate')):
        var=('base','hollow','bulge')[v]
        path=Path(f'panel_{panel:02d}')/f'{SHAPES[i]}_{surf}_{law}_{var}_h{H[h]:.0f}.npz'
        with np.load(ROOT/'results'/path) as a:
            pred=a['pred'][:,0,:,:8]
            mom=(pred[1]**2-pred[0]**2).mean(-1)
            move=2*(pred[1]-pred[0]).mean(-1)
        with np.load(ROOT/'labels44'/path) as a:
            y=a['labels'].astype(float); valid=a['valid'].astype(bool)
        assert y.shape==(2,2,3,4,8)
        val=mom-move*y
        if not valid.all():
            # Match the physical-evaluator denominator in the parent protocol.
            if np.any(valid.sum(axis=(-2,-1))==0):
                raise ValueError(f'No valid objects in a cell {path}')
            row=np.where(valid,val,0).sum(axis=(-2,-1))/valid.sum(axis=(-2,-1))
        else:
            row=val.mean(axis=(-2,-1))
        sums[i,h,v]+=row
        counts[i,h,v]+=1
        valid_n+=valid.sum()
    unpooled=sums/counts
    return np.stack((unpooled.mean(2),unpooled[:,:,1:].mean(2))),int(valid_n)


def accuracy(target, guess):
    return np.mean(np.sign(target)==np.sign(guess),axis=-1)


def ci(vals):
    return np.quantile(vals,[.025,.975],axis=0).T.tolist()


def calc(x,y, ref):
    flat=x.reshape(len(x),-1)
    scores=flat.mean(0)
    m,a,b=len(scores),np.asarray(y).ravel(),np.asarray(ref).ravel()
    boot_id=np.random.default_rng(SEED).integers(0,len(flat),(BOOT,len(flat)))
    boot=flat[boot_id].mean(axis=1)
    half=t.ppf(1-.025/m,len(x)-1)*flat.std(axis=0,ddof=1)/np.sqrt(len(x))
    established=(np.abs(scores)>half)
    guesses=(a,b,np.ones_like(a))
    accs=[accuracy(scores,z) for z in guesses]
    boots=[accuracy(boot,z) for z in guesses]
    mae=[np.abs(scores-z).mean() for z in (a,b)]
    mae_b=[np.abs(boot-z).mean(1) for z in (a,b)]
    switches=(np.sign(a)!=np.sign(b))
    return dict(n=m,correct=[int(np.sum(np.sign(scores)==np.sign(z))) for z in guesses],
                observed_ties=int((scores==0).sum()),bootstrap_accuracy_ci=[ci(bs) for bs in boots],
                paired_acc_advantage=[accs[0]-accs[1],*ci(boots[0]-boots[1])],
                confidence_resolved=int(established.sum()),resolved_correct=[
                    int(np.sum(established&(np.sign(z)==np.sign(scores)))) for z in guesses],
                mae=mae,mae_diff_ci=[mae[0]-mae[1],*ci(mae_b[0]-mae_b[1])],
                changed_calls=int(switches.sum()), changed_correct=[int(np.sum(switches&(np.sign(z)==np.sign(scores)))) for z in guesses[:2]],
                observed=scores.tolist(),sim_lo=(scores-half).tolist(),sim_hi=(scores+half).tolist())


def main():
    frozen=json.loads((ROOT/'forecast44.json').read_text())
    index={(r['scope'],r['shape'],r['height_mm'],r['r_mm'],r['span_mm'],r['offset_mm']):r for r in frozen['rows']}
    forecasts=np.array([[(index[(scope,sh,h,r,l,d)]['pred'],index[(scope,sh,h,r,l,d)]['carry'])
                          for sh,h,r,l,d in product(SHAPES,H,R,L,DELTA)] for scope in SCOPES]).reshape(2,2,2,2,2,3,2)
    rows=[];valid_n=0
    for panel in range(12):
        p,n=panel_risks(panel);rows.append(p);valid_n+=n
    catalog=np.stack(rows)
    out=dict(valid=valid_n,methods=METHODS,scope_order=SCOPES,bootstrap_seed=SEED,score={},cells=[],aggregates=[])
    for k,scope in enumerate(SCOPES):
        guesses=forecasts[k]
        px=catalog[:,k]
        res=calc(px,guesses[...,0],guesses[...,1])
        # Use explicit per-shape reductions: h is height, not radius.
        coarse=np.stack((px[...,1].mean(axis=2),px[...,[0,2]].mean(axis=(2,-1))),axis=-1)
        coarse_g=np.stack((guesses[...,1,:].mean(axis=1),guesses[...,[0,2],:].mean(axis=(1,-2))),axis=-2)
        coarse_r=calc(coarse,coarse_g[...,0],coarse_g[...,1])
        for idx,(sh,h,r,l,d) in enumerate(product(SHAPES,H,R,L,DELTA)):
            inp=index[scope,sh,h,r,l,d]
            out['cells'].append(dict(**inp,observed=res['observed'][idx],sim95=[res['sim_lo'][idx],res['sim_hi'][idx]]))
        for idx,(sh,r,l,kind) in enumerate(product(SHAPES,R,L,('nominal','shifted'))):
            vals=coarse_g.reshape((-1,2))[idx]
            out['aggregates'].append(dict(scope=scope,shape=sh,r_mm=r,span_mm=l,stratum=kind,pred=vals[0],carry=vals[1],
                                          observed=coarse_r['observed'][idx],sim95=[coarse_r['sim_lo'][idx],coarse_r['sim_hi'][idx]]))
        for obj in (res,coarse_r):
            for name in ('observed','sim_lo','sim_hi'):obj.pop(name)
        out['score'][scope]=dict(fine=res,pooled=coarse_r)
    (ROOT/'forecast_eval44.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    lines=['# 동결된 손익 부호의 held-out 재현성','',
           'Projection−original의 Brier 부호를 먼저 정해 둔 다음, 새 catalog의 PyBullet label로 채점했다. 같은 catalog 안의 radius·span·offset을 함께 보존해 panel을 20,000회 재표집했다. 아래 48개 조건은 독립적인 Bernoulli 시행이 아니다.', '',
           '| 평가 모집단 | Support forecast | V43 그대로 | 항상 손해 | 부호 차이(support−v43), 95% | 동시 CI로 구별된 수 | 그 안에서 support/v43/항상 손해 |',
           '|---|---:|---:|---:|---:|---:|---|']
    for scope in SCOPES:
        u=out['score'][scope]['fine'];c=u['correct'];d=u['paired_acc_advantage']; rc=u['resolved_correct']
        lines.append(f'| {scope} | {c[0]}/48 | {c[1]}/48 | {c[2]}/48 | {100*d[0]:+.1f} pp [{100*d[1]:+.1f}, {100*d[2]:+.1f}] | {u["confidence_resolved"]}/48 | {rc[0]} / {rc[1]} / {rc[2]} |')
    lines+=['','| 평가 모집단 | Support MAE | V43 MAE | ΔMAE [paired bootstrap 95%] | 두 방식의 부호가 갈린 칸에서의 점추정 적중(support/v43) |','|---|---:|---:|---:|---|']
    for scope in SCOPES:
        u=out['score'][scope]['fine'];m=u['mae'];d=u['mae_diff_ci']; n=u['changed_calls'];c=u['changed_correct']
        lines.append(f'| {scope} | {m[0]:.6f} | {m[1]:.6f} | {d[0]:+.6f} [{d[1]:+.6f}, {d[2]:+.6f}] | {c[0]}/{n} · {c[1]}/{n} |')
    changed=[r for r in out['cells'] if r['scope']=='all' and np.sign(r['pred'])!=np.sign(r['carry'])]
    lines+=['','### V43와 방향을 다르게 예측한 사례','',
            '아래 D는 부호 예측을 변경한 all 표본의 전 사례다. 48개 칸 동시 95% 구간은 처음 세 사례에서 0을 제외했다. Bracket은 Brier 단위다.','',
            '| Shape/height, R/L, offset (mm) | Carry | Support | 실측 D [동시 95%] |','|---|---:|---:|---:|']
    for w in changed:
        lo,hi=w['sim95']
        lines.append(f'| {w["shape"]}/{w["height_mm"]:g}, {w["r_mm"]:g}/{w["span_mm"]:g}, {w["offset_mm"]:+g} | {w["carry"]:+.6f} | {w["pred"]:+.6f} | {w["observed"]:+.6f} [{lo:+.6f}, {hi:+.6f}] |')
    pooled=out['score']['all']['pooled']['correct']
    lines+=['',f'두 높이를 평균하고 signed offset을 nominal/shifted로 합치면 all의 16개 집계에서 support는 {pooled[0]}/16, carry는 {pooled[1]}/16, 항상 손해는 {pooled[2]}/16이다. 따라서 pooled sign만 평가하면 항상 손해라는 규칙도 매우 강하다. 다섯 개 새 방향이 모두 fold에 집중됐다. 48개 조건의 10.4 pp 차이가 bootstrap에서 거의 움직이지 않는 이유는 이 다섯 결정을 고정한 채 동일한 catalog만 재표집했기 때문이다. 새 family를 뽑을 때의 불확실성은 이 구간에 포함되지 않는다.']
    lines+=['','아래 수치는 두 nominal height를 같은 가중으로 평균한 all 표본의 D×1,000이다. Shifted는 두 방향 오차의 균등 평균이다.','',
           '| Shape / R / L | Nominal 예상 → 실측 | Shifted 예상 → 실측 |','|---|---:|---:|']
    for sh,r,l in product(SHAPES,R,L):
        pair=[rr for rr in out['aggregates'] if (rr['scope'],rr['shape'],rr['r_mm'],rr['span_mm'])==('all',sh,r,l)]
        texts=[f'{w["pred"]*1000:+.2f} → {w["observed"]*1000:+.2f}' for w in pair]
        lines.append(f'| {sh} / {r:g} / {l:g} | '+' | '.join(texts)+' |')
    lines+=['','예상과 실측의 차이는 static initial support 및 v43 impulse surrogate의 전이 오차와 유한 catalog 추출 오차를 함께 포함한다. 부호를 맞힌 것만으로 해당 관측이 미래 실제 물체에서도 옳다고 단정할 수 없다. [48개 셀의 두 모집단 및 전체 구간](forecast_eval44.json) · [사전 예측과 설계](FORECAST_KO.md)']
    (ROOT/'FORECAST_RESULT_KO.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
