"""Korean report section from the prospectively locked paid-prefix experiment."""
from pathlib import Path
import json
import numpy as np
from prefix33_stats import OUT, ROOT, FAMILIES, DIRECTIONS, CANDIDATES, FEATURES, checked_panel, strata, replicates

NAMES={'posterior_friction_sd':'friction coupling posterior SD',
       'paid_tangential_spread':'관측된 tangential velocity spread',
       'paid_velocity_covariance':'factual velocity–probe tangential velocity covariance'}
def fmt(x):
    return f'{x:+.5f}'
def ci(x,interval):
    return f'{fmt(x)} [{fmt(interval[0])}, {fmt(interval[1])}]'
def name(rule):
    return f"{rule['direction']} orientation, {rule['stage']}→{2*rule['stage']}회, {NAMES[rule['feature']]} {'상위' if rule['half']=='upper' else '하위'} 절반"
def main():
    result=json.loads((OUT/'prefix_gain33.json').read_text())
    lock=json.loads((OUT/'discovery_lock.json').read_text())
    rule=name(result['selected'])
    cand=CANDIDATES[lock['winner']]
    row=lock['candidates'][lock['winner']]
    a,b=result['A'],result['B']
    figures=[('Friction coupling posterior SD 상위 절반',0),('A에서 선택한 조건',2)]
    lines=['## 어느 paid prefix에서 다음 probe를 살 가치가 있었는가?',
        '서로 다른 두 cohorts에서 각각 네 object–surface 조합마다 네 독립 blocks를 생성했다. 각 block의 여덟 환경에 대해 동일한 물체·noise를 사용해 friction orientation을 양쪽으로 설정했다. 합계는 독립 blocks 32개, 서로 다른 calibration pools 256개, orientation을 포함한 환경 512개, 2회와 4회를 모두 센 paid prefixes 1,024개다. 각 prefix에 독립적인 두 futures와 두 deployment batches를 교차해 총 4,096개의 Brier gain을 계산했다. 각 cohort에는 반대 orientation에서도 같은 난수로 생성한 32,768개 deployment 물체가 있다. 두 stage, 방향, sensor noise, 또는 같은 행·열의 gain cells로 독립 block 수를 늘리지 않았다.',
        '이미 지불한 접촉 다음에 probe를 추가해도 128개 deployment 물체 중 probe를 받는 물체는 정확히 13개다. 양의 gain은 13곳을 새로 골랐을 때 평균 Brier가 줄었다는 뜻이다. 두 stage에서 효과는 추가 접촉을 2회 또는 4회 구매한 전체 가치다. 4→8회는 실제 첫 stage의 survivor로 제한하기 전 모든 환경에서 측정했다.',
        f'사전에 정한 규칙은 같은 block의 friction coupling posterior SD가 상위 절반인 환경이었다. 별도로 24개 규칙 중 **{rule}** 조건이 탐색용 cohort A에서 선택됐다. A에서 이 조건의 실제 평균 gain은 {fmt(row["gain"])}, 반대 절반과의 차이는 {fmt(row["excess"])}였다. 이를 파일로 고정한 뒤 독립 cohort B를 생성했다. B에서는 추가 선택이나 threshold 변경을 하지 않았다. [사전 계획](PREFIX_PLAN_KO.md) · [선택된 규칙과 A의 모든 후보](results_prefix/discovery_lock.json)',
        '| Cohort B의 prefix | 평균 gain [simultaneous 95%] | 반대 절반 대비 gain 차이 [simultaneous 95%] | 두 하한이 모두 0 초과 |',
        '|---|---:|---:|---:|']
    for n,(label,i) in enumerate(figures):
        vals=b['paired_means']; bounds=b['simultaneous95']
        lines.append(f"| {label} | {ci(vals[i],bounds[i])} | {ci(vals[i+1],bounds[i+1])} | {'예' if result['confirmed_rules'][n] else '아니오'} |")
    lines+=['',
        ('두 조건 모두에서 절대 gain과 반대 절반 대비 gain 차이의 하한이 동시에 0을 넘지 않았다. 주어진 관측 특징만으로 추가 probe를 우선할 집단을 독립적으로 확인하지 못했다.' if not any(result['confirmed_rules']) else
         '두 하한이 0을 넘은 조건에서 추가 probe는 같은 시점의 다른 환경보다 더 큰 평균 Brier 개선을 보였다.'),
        (f'두 하한이 모두 0.003을 넘은 조건은 {sum(result["exceeds_003"])}개였다. 같은 observed prefix의 realized gain은 미래 응답과 deployment worlds에 따라 변한다. 따라서 실험 단위는 개별 접촉이나 전체 gain cell 수가 아닌 16개 독립 confirmation blocks다.'),
        '### 이미 얻은 접촉 정보와 orientation',
        '| Cohort B stage | Friction orientation | 높은 posterior SD의 gain | 낮은 posterior SD의 gain | 차이 |',
        '|---|---|---:|---:|---:|']
    for item in result['prespecified_by_stage_direction']:
        if item['cohort']=='B':
            k=item['stage']; v=item['high_posterior_sd_gain_other_excess']
            lines.append(f"| {k}→{2*k} | {item['direction']} | {fmt(v[0])} | {fmt(v[1])} | {fmt(v[2])} |")
    lines+=['','Stage와 orientation별 추정치는 고정 규칙을 구분해 보고한 보조 결과다. 해당 네 행에서 효과가 큰 행을 새 가설의 독립 검증으로 채택하지 않았다.',
            '| A에서 선택한 규칙의 독립 측정 | 대상 gain | 반대 절반 대비 차이 |','|---|---:|---:|']
    for k,label in [('diagonal_00','Cohort B, future 1 × batch 1'),('diagonal_11','Cohort B, future 2 × batch 2')]:
        v=b[k]
        lines.append(f"| {label} | {fmt(v[2])} | {fmt(v[3])} |")
    switch=b['selected_switch_rates_target_other_excess']
    lines+=['',f'선택된 규칙에서 추가 probe로 query view와 probe 사이의 선택이 바뀐 deployment 물체의 비율은 cohort B 대상 절반에서 {100*switch[0]:.2f}%, 나머지에서 {100*switch[1]:.2f}%였다. 이때 변화율은 128개 전체를 분모로 한다. 정책이 계속 같은 관측을 고르면 동일한 common decoder의 그 물체에 대한 gain은 0이다. 집단 간 gain 차이는 바뀐 물체 수와 그 물체에서 고른 sensor의 예측 성능 양쪽에 의존한다.',
            'Cohort B에서 네 endpoints에 부여한 simultaneous 95% interval은 각각의 object–surface 안에서 양 방향과 여덟 환경을 하나의 묶음으로 재표집했다. B에서 처음까지 모두 16개 independent blocks를 사용했다. 32회의 sensor noise는 같은 물체를 반복 측정한 것이다. 같은 prefix에서 얻은 두 diagonal 측정도 prefix를 공유하므로 독립된 두 연구의 효과처럼 해석할 수 없다. 이 실험은 surface orientation을 사전에 지정한 합성 조건에서의 probe 효과를 측정한다. 개별 특징과 gain의 관계를 실제 장비에 옮기려면 해당 특징과 양 방향의 접촉을 함께 기록해야 한다. [Cohort B의 모든 추정치](results_prefix/prefix_gain33.json) · [사전 설계대로 gain을 측정한 코드](prefix33.py) · [독립 검산과 cluster bootstrap](prefix33_stats.py)']
    dist=result['observed_gain_distribution']
    lines.extend(['### 같은 집단 안에서 관측된 gain의 분포',
        '아래 수치는 Cohort B에서 각 prefix마다 2×2 cells를 평균한 실제 관측 gain의 분포다. 각 환경의 conditional expected gain을 정확히 알 때의 양수 비율은 이 설계로 직접 추정할 수 없다.',
        '| A에서 선택한 조건 | 양의 2×2 gain 관측 | Median [Q1, Q3] |','|---|---:|---:|'])
    for group,label in [('target','대상 절반'),('complement','나머지 절반')]:
        z=dist[group]; q1,m,q3=z['quartiles']
        lines.append(f"| {label} | {z['positive']}/{z['n']} ({100*z['fraction_positive']:.1f}%) | {fmt(m)} [{fmt(q1)}, {fmt(q3)}] |")
    numeric=result['frozen_numeric_threshold_B']
    estimate=numeric['estimates']
    counts=numeric['total_counts']
    cutoff=numeric['threshold']
    lines.extend(['### 한 환경에서 적용할 수 있는 고정 threshold',
        f"다른 환경의 순위가 없어도 같은 특징을 사용할 수 있는지 별도로 확인했다. Cohort A에서 선택된 특징의 median은 {cutoff:.7f}였고, 이 수치를 고정해 B의 대상 {counts[0]}개 환경과 반대쪽 {counts[1]}개 환경을 나눴다."])
    if result['selected']['feature']=='paid_tangential_spread' and result['selected']['stage']==2:
        direction_text='작은' if result['selected']['half']=='lower' else '큰'
        lines.append(f"두 관측값에 ddof=0을 적용한 spread는 두 tangential velocity 차이의 절댓값을 2로 나눈 값이다. 따라서 이 threshold는 두 값의 차이가 {2*cutoff:.5f}보다 {direction_text} prefix를 가리킨다.")
    if estimate is None:
        lines.append('네 물체·표면 조합 중 한 곳 이상에서 한 그룹이 비어 전체 조합에 같은 비중을 주는 추정량을 계산할 수 없었다.')
    else:
        bounds=numeric['pointwise95']
        lines.append(f"대상 gain은 {ci(estimate[0],bounds[0])}, 반대 그룹 대비 gain 차이는 {ci(estimate[2],bounds[2])}였다. 네 조합에 동일한 가중치를 주었다. 유효한 bootstrap draws는 {numeric['valid_resamples']:,}/10,000회였다. 괄호는 탐색적 pointwise 95% interval이다. 단일 prefix용 threshold에 대한 이 보조 결과는 위 primary simultaneous 판정과 구분한다.")
    # Preserve contiguous table rows and separate Markdown prose paragraphs.
    text=''
    for i,line in enumerate(lines):
        if not line:
            continue
        newline='\n' if text and text.splitlines()[-1].startswith('|') and line.startswith('|') else '\n\n'
        text+=('' if not text else newline)+line
    (ROOT/'PREFIX_RESULTS_KO.md').write_text(text+'\n')

if __name__=='__main__':
    main()
