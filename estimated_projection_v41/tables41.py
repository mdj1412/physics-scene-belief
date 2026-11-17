"""Write Korean result tables directly from verified experimental archives."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def read(name): return json.loads((ROOT/name).read_text())
def contrast(est,index=None,digits=6):
    if isinstance(est,dict):
        vals=[est[k] for k in ('mean','lo','hi')]
    else: vals=est
    if index is not None: vals=[v[index] for v in vals]
    m,lo,hi=vals
    return f'{m:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]'
def band(item):
    m=item['ratio'];l,u=item['ci']
    return f'{m:.3f}× [{l:.3f}, {u:.3f}]'
def main():
    s=read('stats41.json'); c=read('contour_breakdown41.json'); sham=read('sham41.json')
    soft=read('marginal41.json'); win=read('local_window41.json'); time=read('latency_uncorrected41.json')
    methods=['Original','Oracle projection','Estimated projection']
    policies=['Fixed 32.5%','v38 optimizer 상시 적용','환경별 selector']
    rows=['## Held-out Brier와 GPU p90','',
      'Brier는 양성 사건에 대한 한 항 (p−Y)²이다. GPU p90은 동일 GPU에서 8개 환경×128개 물체를 한 번 처리한 동기화 wall-clock 시간이다. Warm은 query template을 재사용한다. Cold는 group profile이 메모리에 들어온 뒤 방향을 다시 적합하고 template을 만드는 비용까지 포함한다. 두 시간 모두 센서 촬영·충돌·reset, source bank 생성, 모델 학습 시간을 제외한다.','',
      '| Decoder | Correction | Held-out Brier ↓ | Warm p90 (ms) | Cold p90 (ms) |','|---|---|---:|---:|---:|']
    for i,m in enumerate(methods):
        for j,p in enumerate(policies):
            rows.append(f"| {m} | {p} | {s['mean_brier'][i][j]:.6f} | {s['timing']['p90'][i][j]:.2f} | {s['cold']['p90'][i][j]:.2f} |")
    rows+=['','각 panel에서 9개 방법을 섞은 순서로 1회 warm-up 후 3회 측정했다. 방법당 36개 latency를 합친 p90이다. 12개 timing cluster를 함께 재표집한 percentile bootstrap 95% 구간으로 ratio의 불확실성을 평가했다. Warm p90의 estimated selector / estimated fixed는 **'+band(s['timing']['selected_vs_estimated_fixed'])+'**, estimated selector / original fixed는 '+band(s['timing']['selected_vs_original_fixed'])+'였다. Cold에서 같은 두 ratio는 각각 '+band(s['cold']['selected_vs_estimated_fixed'])+' 및 '+band(s['cold']['selected_vs_original_fixed'])+'였다. [원 통계](stats41.json)','',
      '**원래 decoder를 보정 없이 사용한 경우.** 같은 576개 block에서 w=0으로 산출한 Brier와, 주 실험 종료 후 같은 GPU 및 같은 catalog panel에서 측정한 세 방법의 timing이다. 이 표 안에서 비교해야 timing session 차이를 통제할 수 있다.','',
      '| 방법 | 전체 held-out Brier | 별도 warm p90 (ms) | 별도 cold p90 (ms) |','|---|---:|---:|---:|']
    for i,(name,loss) in enumerate([('Original, 보정 없음',c['uncorrected']['mean'][0]),('Original + fixed',s['mean_brier'][0][0]),('Estimated + fixed',s['mean_brier'][2][0])]):
        rows.append(f"| {name} | {loss:.6f} | {time['full']['p90'][i]:.2f} | {time['cold']['p90'][i]:.2f} |")
    rows+=['',f"보정을 끄면 oracle과 estimated의 Brier는 각각 {c['uncorrected']['mean'][1]:.6f}, {c['uncorrected']['mean'][2]:.6f}였다. [추가 crossover timing](latency_uncorrected41.json) · [원 확률의 재채점](contour_breakdown41.json)",'',
      '### 같은 decoder에서 정책을 고르는 이득','',
      '| Decoder | Selector−fixed, 동시 95% CI | Selector−상시 optimizer, 동시 95% CI | Optimizer 선택 비율 |','|---|---:|---:|---:|']
    for i,m in enumerate(methods):
        rows.append(f"| {m} | {contrast(s['selector_minus_fixed'],i)} | {contrast(s['selector_minus_optimizer'],i)} | {s['selection_rate'][i]:.1%} |")
    rows+=['','Estimated가 주 검정이다. Original 및 oracle은 같은 방식으로 계산한 decoder별 보조 대조다. 각 대조의 음수는 앞의 방법이 유리하다는 뜻이다.','',
      '### 보정을 고정했을 때 projection의 효과','',
      f"Estimated−original은 **{contrast(s['estimated_minus_original_fixed'])}**, estimated−oracle은 **{contrast(s['estimated_minus_oracle_fixed'])}**였다. Oracle−original의 개별 95% 구간은 {contrast(s['oracle_minus_original_fixed'])}였다."]
    if s['recovery']:
        r=s['recovery']
        rows.append(f"원래 decoder 대비 oracle의 Brier 감소 중 estimated가 회수한 비율은 {r['value']:.1%} [paired bootstrap 95% CI {r['ci'][0]:.1%}, {r['ci'][1]:.1%}]였다. Denominator는 이 catalog에서 관측된 oracle 이득이다.")
    rows+=['','| Shape·height | Estimated + fixed | Estimated−original, fixed | Estimated−oracle, fixed | Estimated selector−fixed |','|---|---:|---:|---:|---:|']
    for k,x in s['by_family_height'].items():
        rows.append(f"| {k.replace('_h',' / ')} mm | {x['brier'][2][0]:.6f} | {contrast(x['est_vs_original'])} | {contrast(x['est_vs_oracle'])} | {contrast(x['selection_vs_fixed'])} |")
    rows+=['','각 열에서 네 shape·height 대조를 함께 보정했다. 이 표의 4개 행은 같은 catalog panel을 공유한다. 모든 24개 개별 cell과 panel별 원 값은 [전체 결과](stats41.json)에 있다.']
    time0=read('latency_estimated_zero41.json')
    rows+=['','### 보정 없음과 직접 비교한 별도 GPU 실행','',
           'Estimated decoder를 고정하고 보정 없음·fixed·selector의 순서를 섞어 같은 12개 panel에서 다시 측정했다. 보정을 끈 baseline이 강한 것을 확인한 뒤 실시한 보조 비교다. 위 주 timing에서 규정한 +2% 판정은 그대로 유지한다.','',
           '| Estimated decoder의 correction | Held-out Brier | Warm p90 (ms) | Cold p90 (ms) |','|---|---:|---:|---:|']
    for i,(name,result) in enumerate([('없음',c['uncorrected']['mean'][2]),('Fixed 32.5%',s['mean_brier'][2][0]),('Selector',s['mean_brier'][2][2])]):
        rows.append(f"| {name} | {result:.6f} | {time0['full']['p90'][i]:.2f} | {time0['cold']['p90'][i]:.2f} |")
    low,high=time0['full']['ratio_to_uncorrected_ci'][2]
    rows+=['',f"Selector / 보정 없음의 warm p90 ratio는 {time0['full']['ratio_to_uncorrected'][2]:.3f}× [{low:.3f}, {high:.3f}]였다. 서로 다른 timing session의 절대값 차이를 방법 간 효과로 환산하지 않았다. [원 timing](latency_estimated_zero41.json) · [동일 panel Brier 차이](no_correction41.json)"]
    (ROOT/'TABLES_KO.md').write_text('\n'.join(rows)+'\n')
    x=c['pooled']; cm=x['means']; eff=x['estimated_minus_original']
    out=['## 방향 제거가 무엇을 바꿨는가','',
      '같은 물체에서 base·hollow·bulge의 실제 stick event가 같기 때문에, 세 contour에서의 평균 Brier가 두 항으로 정확하게 나뉜다. 첫 항은 세 예측 확률의 분산이고, 둘째 항은 세 예측의 평균과 실제 event 사이의 제곱 오차다. 이는 contour에 따른 흔들림을 얼마나 줄였는지와, 흔들림을 없앤 확률의 중심이 얼마나 맞았는지를 구분한다. Fixed 32.5%에서 계산했다.','',
      '| 항목 | Original | Oracle | Estimated | Estimated−original, 개별 95% CI |','|---|---:|---:|---:|---:|']
    for i,label in enumerate(['Contour 간 예측 분산','Contour 평균 예측의 Brier','Base만 사용한 Brier','Hollow·bulge 평균 Brier']):
        out.append('| '+label+' | '+' | '.join(f'{v:.6f}' for v in cm[i])+f' | {contrast(eff,i)} |')
    out+=['',f"세 contour의 비율은 1:1:1이다. 같은 query view 가지에서 base와 두 변형 사이의 보정 전 절대 확률 변화는 original {s['view_abs_change'][0]:.6f}, oracle {s['view_abs_change'][1]:.2e}, estimated {s['view_abs_change'][2]:.6f}였다. 같은 물체의 probe 가지는 세 decoder에서 동일했다. [재계산](contour_breakdown41.json)"]
    if x['break_even_point'] is not None:
        out.append(f"Base 확률을 1−q로, 나머지 두 변형 확률을 각각 q/2로 놓고 이 실험의 strata를 그대로 재가중하면 estimated와 original의 평균 Brier 순위가 바뀌는 q는 {x['break_even_point']:.1%}다. 실세계 prevalence와 그 값의 신뢰구간은 측정하지 않았다.")
    out+=['','### 물체 대응 관계와 저렴한 대안','',
      '이 보조 비교는 각 panel마다 홀짝으로 지정한 한 block, 32회 중 고정된 noise 0·16만 사용했다. 24개 cell과 12개 panel, 구매한 관측, source bank와 contact label은 모두 같고 추론은 서버 CPU에서 계산했다. GPU p90 비교에는 위 주 실험의 9개 방법과 별도 crossover만 포함한다.','',
      '| 같은 물체와 noise에서의 방법 | Group당 추가 profile | Fixed 32.5% Brier |','|---|---:|---:|']
    for i,(label,n) in enumerate([('Original',0),('Oracle projection','oracle 정보'),('Estimated projection (matched)',64),('Unpaired projection (sham)',64)]):
        out.append(f"| {label} | {n} | {sham['mean'][i]:.6f} |")
    for a,b in ((soft['means'][0][0],sham['mean'][2]),(win['mean'][2],sham['mean'][2])):
        np.testing.assert_allclose(a,b,atol=2.e-12,rtol=0.)
    out += [f"| Local window (중앙 ±2 mm 세 좌표) | 0 | {win['mean'][3]:.6f} |",
        f"| Finite-variance marginalization (matched) | 64 | {soft['means'][1][0]:.6f} |",'',
        f"Estimated−unpaired는 **{contrast(sham['paired_minus_sham'])}**였다. 같은 profile을 유지하고 두 번째 관측의 물체 대응만 한 칸 순환 이동했다. Estimated−local window는 **{contrast(win['estimated_minus_local'])}**였다. Local window는 contact height와 가까운 세 좌표만 likelihood에 넣는다. [Sham 결과](sham41.json) · [Local window 결과](local_window41.json)",'',
        f"Finite-variance 방법은 pair difference의 첫 eigenvalue에서 sensor noise를 뺀 뒤, 해당 방향의 residual variance를 유한한 값으로 적분했다. 같은 û를 사용해 정보를 부분적으로 남기는 방법이다. Fixed에서 finite-variance−hard projection은 **{contrast(soft['soft_minus_hard'][0])}**였다. Finite-variance 안에서 selector−fixed는 {contrast(soft['soft_select_minus_fixed'])}였다. Local window와 finite-variance는 주 실험의 평가 도중 별도로 명세한 보조 분석이며 실용 후보 간 순위를 넓혀 보기 위한 것이다. [방법과 결과](marginal41.json)",'',
        '이 보조 분석의 구간들은 12개 panel의 개별 95% 구간이다. 보조 표본의 평균 Brier를 576개 block의 주 표와 직접 빼지 않았다.']
    (ROOT/'MECHANISM_KO.md').write_text('\n'.join(out)+'\n')

if __name__=='__main__': main()
