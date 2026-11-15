"""Render the statistical report from the two immutable v32 CPU analysis outputs."""
import json
from pathlib import Path

from stats32 import ROOT


def ci(x, digits=6):
    return f'[{x[0]:.{digits}f}, {x[1]:.{digits}f}]'


def render(root=ROOT):
    d=json.loads((root/'results_eval/paired_stats32.json').read_text())
    s=json.loads((root/'results_eval/source_subset_sensitivity32.json').read_text())
    b=d['brier']; m=d['methods']; full=d['full']; warm=d['warm']; contrasts=d['contrasts']
    out=['# v32 통계 결과','', '## Training budget과 실험 단위','',
       '두 scorer에서 동일한 15개 features와 λ=3인 standardized ridge를 각 stage에 학습했다. Brier gain은 paid prefix와 추가 probe 뒤의 Brier 차이로 정의했다. Repeated는 각 prefix에서 2개 calibration futures와 2개 deployment batches가 교차하는 네 gain의 평균을 사용했다. Broad는 한 개를 사용했다.','',
       '| Source arm | Prefix 수 | Group 수 | Gain cells (두 stages) | Label time (s) | Ridge fit (s) | Label+fit (s) | Read·hash 포함 (s) |',
       '|---|---:|---:|---:|---:|---:|---:|---:|']
    for a in m[:2]:
        t=d['training'][a]; out.append(f"| {a} | {t['n_environments']} | {(t['n_environments']+7)//8} | {t['labeled_cells']} | {t['labels_s']:.4f} | {t['fit_s']:.4f} | {t['labels_plus_fit_s']:.4f} | {t['labels_plus_read_plus_fit_s']:.4f} |")
    gap=d['training_relative_difference']['labels_plus_fit_s']
    out+=['',f'작은 arm 시간에 대한 label+fit 차이는 {100*gap:.2f}%였다. 사전 기준은 2%다. 두 arm은 independent source bank와 paid prefix를 사용하며, 각 block은 최대 8개 환경의 calibration을 포함한다.', '',
          '## Held-out Brier와 GPU latency','',
          '3개 target laws에 같은 비중을 두었다. 각 law의 16개 independent blocks가 8개 environments를 공유 단위로 묶는다. 총 384 environments, 49,152 deployment worlds를 측정했다. 32회 sensor noise는 같은 world의 반복 측정이다. GPU p90은 law별 첫 6개 blocks에서 같은 네 policy를 interleave해 3회씩 실행한 결과다. Full-call 하나는 8개 environments·1,024개 worlds의 allocation, acquisition과 선택한 observation의 decoding을 포함한다.', '',
          '| Method | Held-out Brier | 95% pointwise CI | Full GPU p90 (ms) | 95% pointwise CI (ms) | Warm GPU p90 (ms) |','|---|---:|---|---:|---|---:|']
    for j,a in enumerate(m):
        out.append(f"| {a} | {b['mean'][j]:.6f} | {ci(b['individual_mean_pointwise_ci'][j])} | {full['p90_ms'][j]:.2f} | {ci(full['p90_ms_pointwise_ci'][j],2)} | {warm['p90_ms'][j]:.2f} |")
    out+=['','| Contrast A−B | Δ Brier | Simultaneous 95% CI | 해석 (0 기준) | Full p90 A/B | Pointwise 95% CI |','|---|---:|---|---|---:|---|']
    for j,a in enumerate(contrasts):
        interval=b['simultaneous_ci'][j]
        direction='A 우위' if interval[1]<0 else ('B 우위' if interval[0]>0 else '우열 미결정')
        out.append(f"| {a.replace('-minus-',' − ')} | {b['delta'][j]:+.6f} | {ci(interval)} | {direction} | {full['p90_ratio'][j]:.3f} | {ci(full['ratio_pointwise_ci'][j],3)} |")
    out+=['',f"다섯 primary contrasts에 law-stratified paired block bootstrap 10,000회와 simultaneous critical multiplier {b['critical']:.3f}을 적용했다. ΔBrier는 음수일수록 앞 policy에 유리하다. ±0.003은 absolute Brier의 실용적 기준이다. P90 interval은 18개 timed blocks를 resample한 pointwise interval이며 preprocessing·sensing·reset·공통 detector 및 decoder bank 생성 시간을 제외한다. Warm은 동일 호출에서 deployment selection·decoding만 잰 값이다.", '',
          '## Laws에 따른 차이','', '| Held-out law | Broad Brier | Repeated Brier | Fixed4 Brier | Uncertainty Brier | Repeated−broad | Pointwise 95% CI |','|---|---:|---:|---:|---:|---:|---|']
    for name,v in b['secondary'].items():
        risk=' | '.join(f'{a:.6f}' for a in v['brier'])
        out.append(f"| {name} | {risk} | {v['delta'][0]:+.6f} | {ci(v['pointwise_ci'][0])} |")
    out+=['','`historical_two`는 v30·v31 시점에 이미 조사한 두 contact laws에 같은 비중을 준 값이다. `directional`은 v32에서 사전에 지정한 signed tangential velocity 의존 law다. 세 law 모두 scorer 학습에서 제외했다.','', '## Source subset sensitivity','',
          '각 arm의 생성 순서에서 block index를 2 또는 3으로 나눈 나머지를 이용해 disjoint groups를 만들었다. 각 group에서 동일한 ridge를 새로 fit하고 전체 target block의 frozen stage별 risk로 policy를 replay했다. 각 행의 interval은 해당 두 reduced-budget fits에 조건을 둔다. Modulo 2와 3은 source observations를 서로 공유한다.', '',
          '| Groups·residue | Broad / repeated prefixes | Label time (s), B / R | Δ repeated−broad | Pointwise 95% CI | Full fit과 allocation이 바뀐 환경 비율 B / R |','|---|---|---|---:|---|---|']
    for v in s['subsets']:
        a,r=(v['source'][j] for j in m[:2]);c=v['allocation_change_from_full']
        out.append(f"| {v['modulus']}·{v['residue']} | {a['n_environments']} / {r['n_environments']} | {a['label_seconds']:.1f} / {r['label_seconds']:.1f} | {v['repeated_minus_broad']:+.6f} | {ci(v['paired_conditional_ci'])} | {c[0]:.1%} / {c[1]:.1%} |")
    out+=['','## 검증 범위','',f"각 artifact의 SHA-256, 학습·평가 seed 분리, 두 ridge fit의 CPU SVD reference, common decoder probabilities, quota, stage별 rank, 실제 paid trajectory를 독립적으로 재현했다. Recorded Brier와 independent replay의 최대 절대 차이는 {d['audit']['max_brier_error']:.2e}, scorer output 차이는 {d['audit']['max_score_error']:.2e}였다. 이 primary bootstrap은 첫 full-budget seed pair에 조건을 둔다. 학습 seed 간 차이는 별도 full-budget 재학습 분석에서 평가한다.",'']
    (root/'STATS_KO.md').write_text('\n'.join(out))


if __name__=='__main__':
    render()
