"""Summarize policy, confirmatory cohort and single-world transfer, without treating discovery as confirmation."""
from pathlib import Path
import json,numpy as np
ROOT=Path(__file__).resolve().parent
read=lambda n:json.loads((ROOT/n).read_text())
def iv(v,b):return f'{v:+.5f} [{b[0]:+.5f}, {b[1]:+.5f}]'
def main():
    s=read('policy_stats33.json'); g=read('results_prefix/prefix_gain33.json');t=read('transfer33.json');q=read('shadow33.json')
    pre=g['frozen_numeric_threshold_B'];th=pre['threshold'];est=pre['estimates'];num=pre['total_counts']
    b=g['B']['paired_means'];ci=g['B']['simultaneous95']
    singleton=(f'같은 규칙을 비교할 다른 환경 없이 사용할 수 있는지도 확인했다. A에서만 정한 절대 threshold {th:.7f}를 B로 옮겼을 때 대상 {num[0]}개, 나머지 {num[1]}개였다. '+(f'대상 gain은 {iv(est[0],pre["pointwise95"][0])}, 반대 집단 대비 차이는 {iv(est[2],pre["pointwise95"][2])}였다.' if est is not None else '하나 이상의 물체·표면 조합에 한쪽 표본이 없어 전체 조합 추정치는 구하지 않았다.')+' 이 보조 분석의 구간은 pointwise이며 primary 판정을 대체하지 않는다. [전체 threshold 검증](results_prefix/prefix_gain33.json)')
    if g['selected']['feature']=='paid_tangential_spread' and g['selected']['stage']==2:
        singleton+=' 두 측정에서 이 population SD는 |v₁−v₂|/2다. 따라서 threshold에 해당하는 두 tangential velocity의 간격은 '+f'{2*th:.5f}'+'이다. 단위는 simulator의 velocity 좌표계다.'
    observed=[]
    for k,label in [('target','선택 집단'),('complement','반대 절반')]:
        z=g['observed_gain_distribution'][k];q1,median,q3=z['quartiles']
        observed.append(f"{label}에서 관측한 2×2 평균 gain은 {z['positive']}/{z['n']}개 환경에서 양수였고, median {median:+.5f} (IQR {q1:+.5f}~{q3:+.5f})였다.")
    singleton+='\n\n'+' '.join(observed)+' 이는 두 futures·두 batches의 관측 통계이며, 각 환경의 true expected gain이 양수일 확률의 추정치가 아니다.'
    direction=g['selected']['direction']
    rr=f'**발견된 규칙으로 실제 예산을 배정하면?** 실험자가 {direction} 방향임을 알고 있을 때, 선택된 한 단계에서만 해당 규칙을 적용하고 나머지는 uncertainty-only로 배정했다. 같은 32-probe budget에서 이 정책의 Brier는 {q["absolute"]:.6f}, fixed 4 대비 {iv(q["delta"][0],q["simultaneous_ci"][0])}, uncertainty-only 대비 {iv(q["delta"][1],q["simultaneous_ci"][1])}였다. 여덟 환경 중 같은 횟수의 접촉을 받을 곳이 바뀐 비율은 uncertainty-only 대비 {100*q["fraction_counts_changed_vs_uncertainty"]:.1f}%였다. 이 진단 정책은 표면 방향이라는 추가 정보를 받으므로 네 primary policies의 공정한 대체 baseline은 아니다. 별도 GPU p90은 측정하지 않았다. [규칙을 임의로 바꾸지 않기 위한 사전 설계](SHADOW_PLAN_KO.md) · [실제 trajectory의 재계산](shadow33.json)'
    (ROOT/'AUXILIARY_KO.md').write_text(singleton+'\n\n'+rr+'\n')
    plus=s['orientation_ci'][0];inter=s['orientation_ci'][2]
    desc=('v32에서 보였던 양의 방향 차이(plus에서 2×2의 손실이 큼)는 세 대비를 함께 보정한 뒤에도 양수였다.' if plus[0]>0 else 'v32에서 보였던 plus 방향에서의 1×1 우위는 이번 세 대비를 함께 보정한 기준으로 재현되지 않았다.')
    effect=('방향 간 차이의 구간도 0을 제외했다.' if inter[0]*inter[1]>0 else '방향 간 차이 자체의 구간도 0을 포함했다.')
    subgroup=('추가 probe의 효과가 집중되는 집단을 별도 cohort에서 확인했다. 이를 policy 개선으로 연결하려면 중간 순위뿐 아니라 전체 32개 probe의 배분이 fixed 4를 개선해야 한다.' if g['confirmed_rules'][1] else '독립 cohort에서 후보 집단의 absolute gain과 나머지 대비 gain 차이가 동시에 양수임을 확인하지 못했다. A에서 가장 큰 숫자만 골라 선택 가능한 prefix의 증거로 제시해서는 안 된다.')
    ranges=np.asarray(s['ci'])
    margin=('다섯 primary policy 차이의 simultaneous interval이 모두 ±0.003 안에 있었다.' if np.max(abs(ranges))<.003 else '일부 primary policy 차이에서 ±0.003의 실용적 차이를 배제하지 못했다.')
    evidence=f'**연구 판단.** 이 설정에서는 비용을 우선하면 가장 빠른 fixed 4를 선택하는 것이 합리적이다. 평균 Brier만 보면 uncertainty-only가 가장 낮았지만 fixed 4 대비 차이는 −0.00050(pointwise 95% −0.00115~+0.00018)으로, 어느 쪽이 예측에서도 더 낫다고 확정할 수 없다. [두 baseline의 직접 대조](headroom33.json)  {desc} {effect} {margin} {subgroup} 두 비교 모두 새 simulation data에서 측정했지만, 여전히 두 유형의 물체와 두 유형의 표면·고정된 여덟 coupling 및 source-law common decoder에 대한 결론이다. 이후 데이터에서도 항상 동일하다는 뜻은 아니다.'
    contribution='**논문 기여 가능성.** 방어 가능한 기여는 같은 물체에서 방향만 바꾼 paired test에, 실제로 지불한 prefix의 gain을 독립 outcome으로 재측정하고, 같은 예산의 전체 stopping loss까지 추적한 검증 절차다. 마찰 방향에 따른 차이, uncertainty 기반 물리 탐색, 학습된 error reduction 자체에는 직접 선례가 있다. 독립 B와 전체 policy의 근거 없이 새로운 acquisition 방법이 우월하다고 주장하기는 어렵다. [실물 active exploration의 직접 선례](https://arxiv.org/html/2411.09020v1) · [group별 방향이 반전된 pulling 실험](https://pmc.ncbi.nlm.nih.gov/articles/PMC5805965/)'
    nextstep='**다음 실험에서 먼저 판단할 것.** 실제 표면 session을 통째로 discovery와 confirmation으로 나누고 pulling 양방향을 따로 측정해 sign을 고정한다. 그 뒤 calibration에 쓴 물체와 별도 deployment 물체에서 probe·query view·stick label을 수집한다. 날짜·reset·입사 속도를 함께 기록한다. 현재처럼 factual과 calibration에 한쪽 속도만 줄 때와 양쪽 속도를 번갈아 줄 때를 같은 접촉 횟수로 비교하면, signed evidence 자체가 부족한지와 scorer가 기존 evidence를 쓰지 못하는지를 분리할 수 있다. 실제 contact 시간으로 값을 매긴 실용 기준을 데이터 공개 전에 정하고, 새 session에서도 absolute gain과 excess gain의 두 하한이 기준을 넘으며 전체 policy가 fixed 4보다 나을 때에만 방법 우위를 주장하는 것이 적절하다.'
    (ROOT/'ASSESSMENT_KO.md').write_text('\n\n'.join((evidence,contribution,nextstep))+'\n')
if __name__=='__main__':main()
