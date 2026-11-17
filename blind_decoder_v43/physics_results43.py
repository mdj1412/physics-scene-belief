"""Post-freeze, matched evaluator audit over stored v43 objects and predictions."""
from itertools import product
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

import physics_check43 as other
import world43 as world

ROOT=Path(__file__).resolve().parent
ORDER=tuple(product(world.TARGET_SHAPES,world.HEIGHTS))
NCELLS=48


def terms(theta,shape,variant,surface,law,sign,regime,height,delta):
    g=theta[:,2]+theta[:,3]*(1000*height+delta)/25+other.slope(1000*height+delta,shape,variant)
    v=.425
    vt=(v+g)/np.sqrt(1+g*g)
    mu=other.friction(theta[:,4],vt,height+.001*delta,surface,law,sign,regime)[0]
    denom=(1+theta[:,5])*np.maximum(1-v*g,0.)
    demand=np.divide(abs(v+g),denom,out=np.full_like(mu,np.inf),where=denom>0)
    return demand,mu


def panel_audit(panel):
    result=np.zeros((len(ORDER),len(other.DELTAS),9))
    checked=0;max_brier=0.;max_identity=0.
    for i,(shape,height) in enumerate(ORDER):
        for surface,law in product(world.TARGET_SURFACES,world.TARGET_LAWS):
            ref=None
            for variant in world.VARIANTS:
                path=ROOT/'results'/f'panel_{panel:02d}'/f'{shape}_{surface}_{law}_{variant}_h{round(1000*height)}.npz'
                with np.load(path) as a:
                    y=a['labels'].astype(bool)
                    p=a['pred'];risk=a['risk']
                    seed=int(a['seed']); alpha=a['alphas']
                    signs=a['signs']; regimes=a['regimes']
                    xx,selected,cal=a['xx'],a['selected'],a['calibration_y']
                dem=np.empty(y.shape);mus=np.empty(y.shape)
                for e,al in enumerate(alpha):
                    context=shape,variant,height,surface,law,int(signs[e]),panel
                    theta=world.draw_object(y.shape[-1],seed+200+e,20,al,context).cpu().numpy()
                    for j,d in enumerate(other.DELTAS):
                        dem[j,e],mus[j,e]=terms(theta,shape,variant,surface,law,int(signs[e]),str(regimes[e]),height,d)
                np.testing.assert_array_equal(dem<=mus,y)
                pe=p[2,0];po=p[0,0]
                rcalc=np.mean((p[None]-y[:,None,None,:,:,None])**2,axis=(-1,-2))
                max_brier=max(max_brier,float(abs(rcalc-risk).max()))
                checked+=y.size
                if variant=='base':
                    ref=y,dem,mus,xx,selected,cal
                    continue
                y0,d0,m0,x0,s0,c0=ref
                np.testing.assert_array_equal(xx,x0)
                np.testing.assert_array_equal(cal,c0)
                np.testing.assert_array_equal(selected,s0)
                np.testing.assert_array_equal(y[4],y0[4])
                flip=y!=y0
                events=flip.mean((1,2))
                target=np.asarray(y,dtype=float)
                base=np.asarray(y0,dtype=float)
                pd=pe-po
                mean=lambda a:a.mean(axis=(1,2,3))
                diffr=mean((pe[None]-target[...,None])**2-(po[None]-target[...,None])**2)
                specific=-2*mean(pd[None]*(target-base)[...,None])
                ordinary=-2*mean(pd[None]*(base-base[4])[...,None])
                max_identity=max(max_identity,float(np.max(abs(diffr-diffr[4]-specific-ordinary))))
                # Each of 2 surfaces x 2 laws x 2 deformed variants.
                result[i] += np.stack((events, diffr, specific, ordinary,
                    ((dem<=m0)==y).__and__(flip).mean((1,2)),
                    ((d0<=mus)==y).__and__(flip).mean((1,2)),
                    ((dem<=m0)!=y).__and__((d0<=mus)!=y).__and__(flip).mean((1,2)),
                    target.mean((1,2)), (target!=target[4]).mean((1,2))),axis=-1)/8
    assert max_brier<1e-9 and max_identity<1e-9
    dest=ROOT/'physics_panels';dest.mkdir(exist_ok=True)
    np.savez(dest/f'panel_{panel:02d}.npz',rows=result,checked=checked,brier=max_brier,identity=max_identity)
    print('checked',panel,checked,flush=True)


def interval(x):
    m=x.mean(axis=0);w=t.ppf(.975,len(x)-1)*x.std(axis=0,ddof=1)/np.sqrt(len(x))
    return np.stack([m,m-w,m+w],axis=-1).tolist()


def finish():
    raws=[]
    for k in range(12):
        with np.load(ROOT/'physics_panels'/f'panel_{k:02d}.npz') as a:
            raws.append((a['rows'],int(a['checked']),float(a['brier']),float(a['identity'])))
    panel_rows=np.array([r[0] for r in raws]); tab=panel_rows.mean(axis=0)
    forecast=json.loads((ROOT/'geometry_forecast43.json').read_text())
    truth=[];geom=[];rows=[]
    for row in forecast:
        i=ORDER.index((row['shape'],row['height_mm']/1000.)); j=list(other.DELTAS).index(row['delta_mm'])
        score=tab[i,j,1]
        if j!=4:
            truth.append(np.sign(score));geom.append(row['geometry_sign'])
        rows.append(dict(**row,observed_projection_minus_original=score,flip_rate=tab[i,j,0]))
    truth=np.asarray(truth)
    nonzero=other.DELTAS!=0
    sums=panel_rows[:,:,nonzero].mean((1,2))
    weighted=sums.mean(0); ratio=(weighted[4:7]/weighted[0]).tolist()
    rng=np.random.default_rng(430519)
    b=sums[rng.integers(0,len(sums),(16000,len(sums)))].mean(1)
    ratio_ci=np.quantile(b[:,4:7]/b[:,:1],[.025,.975],axis=0).T.tolist()
    out=dict(decoder_names=['privileged','original'],policies='zero corrected',
             nonzero_sign_matches=int((truth==geom).sum()),nonzero_n=len(truth),
             nonzero_always_harm=int((truth>0).sum()),forecast_table=rows,checked_labels=sum(r[1] for r in raws),
             max_recomputed_brier_error=max(r[2] for r in raws),max_difference_identity_error=max(r[3] for r in raws),
             signed_mm=list(other.DELTAS),flip_by_offset=interval(panel_rows.mean(1)[:,:,0]),
             arithmetic_by_offset=panel_rows.mean((0,1)).tolist(),columns=['event_flip','D','specific','ordinary','demand_only','law_only','neither','prevalence','placement_flip'],
             conditional_observed_flip_reproduced=ratio,conditional_ci=ratio_ci)
    (ROOT/'physics_results43.json').write_text(json.dumps(out,indent=2)+'\n')
    print(out['checked_labels'],out['nonzero_sign_matches'],out['nonzero_always_harm'],ratio,flush=True)
    lines=['# 물리 및 사전 예측의 독립 검산','',
           f'12개 panel의 576개 archive에서 label {out["checked_labels"]:,}개를 displaced contour의 수치 미분과 독립 NumPy impulse inequality로 검산했다. 불일치가 없었다. 모든 예측으로 재구성한 Brier의 최대 오차는 {out["max_recomputed_brier_error"]:.2e}였다. Base·hollow·bulge 사이의 factual evidence, 선택한 관측, 네 번의 calibration, 명목 label은 물체별로 일치했다.','',
           '| Offset (mm) | 같은 높이에서 contour만 바꿨을 때 flip, % [panel 95%] | Privileged−original Brier (변형 물체만) | Contour label 개입 항 | 기본물체 placement 항 |',
           '|---:|---:|---:|---:|---:|']
    for d,coef,zz in zip(other.DELTAS,out['flip_by_offset'],out['arithmetic_by_offset']):
        lines.append(f'| {d:+g} | {coef[0]*100:.2f} [{coef[1]*100:.2f}, {coef[2]*100:.2f}] | {zz[1]:+.6f} | {zz[2]:+.6f} | {zz[3]:+.6f} |')
    lines += ['', '같은 물체·같은 offset에서 base와 hollow/bulge를 비교했다. 마지막 두 열의 합이 명목 높이 대비 Brier 차이의 변화다. 정확한 식은 −2·mean[(p_priv−p_orig)·(Y_variant,δ−Y_base,δ)]와 −2·mean[(p_priv−p_orig)·(Y_base,δ−Y_base,0)]다. 예측은 offset을 보지 않아 이 분해가 성립한다.',
              '',f'관측된 flip 중 impulse demand만 바꿔 재현한 비율은 {ratio[0]:.2%} [{ratio_ci[0][0]:.2%}, {ratio_ci[0][1]:.2%}], 속도에 따른 마찰 한계만 바꿔 재현한 비율은 {ratio[1]:.2%}다. 두 비율은 같은 충돌을 중복 포함할 수 있다. 각 경로 모두 단독으론 재현하지 못한 비율은 {ratio[2]:.2%}다. 구간은 panel bootstrap이다.','',
              f'Label을 보기 전 7점 profile의 apparent slope와 실제 tangent 변화로 예측한 projection 손익 부호는 비영 조건 {len(truth)}개 중 {out["nonzero_sign_matches"]}개에서 Brier 점추정과 일치했다. 항상 손해로 두면 {out["nonzero_always_harm"]}개가 맞았다. 이 보조 비교는 deformed contour만 포함하며 family·offset은 서로 독립적인 32개의 trial이 아니다. 실제 offset과 simulator 기하를 알아야 하는 예측이어서 blind policy accuracy가 아니다.','',
              '| Shape / 명목 높이 | 8개 offset 부호 일치 | 예측과 달랐던 signed offset (mm) |','|---|---:|---|']
    for shape,h in ORDER:
        chosen=[r for r in rows if r['shape']==shape and r['height_mm']==h*1000. and r['delta_mm']!=0]
        wrong=[r['delta_mm'] for r in chosen if r['geometry_sign']!=np.sign(r['observed_projection_minus_original'])]
        lines.append(f'| {shape} / {h*1000:.0f} | {8-len(wrong)}/8 | '+(', '.join(f'{d:+g}' for d in wrong) or '없음')+' |')
    lines+=['','[동결된 사전 예측](geometry_forecast43.json) · [동결 기록](geometry_freeze43.sha256) · [검산 수치](physics_results43.json) · [검산 코드](physics_results43.py)']
    (ROOT/'PHYSICS_RESULTS_KO.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('panel','finish','all'))
    parser.add_argument('index',type=int,nargs='?')
    args=parser.parse_args()
    if args.mode=='all':
        for k in range(12):
            panel_audit(k)
        finish()
    elif args.mode=='finish':finish()
    else:panel_audit(args.index)
