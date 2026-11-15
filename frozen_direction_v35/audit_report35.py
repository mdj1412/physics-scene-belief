"""Format the independent audit result for readers of the Korean report."""
import json
from pathlib import Path

R=Path(__file__).resolve().parent


def f(x):
    return format(x,'.5f')


def band(x):
    return '['+', '.join(map(f,x))+']'


def main():
    d=json.loads((R/'audit35.json').read_text()); p=json.loads((R/'physics_audit35.json').read_text())
    rr=d['risks']
    t=d['timing']
    ctrl=d['isotropic_control']
    text=['# v35 독립 계산 기록','',
          '양 방법은 동일한 4회의 paid calibration, 같은 deployment action, 같은 sensor draw를 사용한다. '
          '이 표에서 Δ는 corrected − original이며 음수일 때 개선이다. 전체 평균은 각 object × surface cell, '
          'friction orientation, schedule에 동일 가중치를 준다. 32개의 독립 seed block 안에서 mirror 및 반복 측정을 묶었다.','',
          '| 평가 | Original Brier | Corrected Brier | Δ [95% CI] |',
          '|---|---:|---:|---:|']
    for name in ('pooled','same','alternating'):
        v=rr[name]; overall=v['overall']; interval=v.get('ci95') or v['simultaneous_ci95']
        text.append(f"| {name} | {f(overall['original'])} | {f(overall['corrected'])} | {f(overall['delta'])} {band(interval)} |")
    text.extend(['',f"Isotropic control (16개 독립 seed block)의 평균 Δ는 {f(ctrl['pooled_delta'])} {band(ctrl['pooled_ci95'])}이다. "
                 '이 control에서는 true orientation이 정의되지 않아 direction accuracy를 계산하지 않았다.',
                 '', '## 효과가 발생한 조건','',
                 '| 조건 | Same Δ | Alternating Δ |','|---|---:|---:|'])
    for k in rr['same']['groups']:
        text.append(f"| {k} | {f(rr['same']['groups'][k]['delta'])} | {f(rr['alternating']['groups'][k]['delta'])} |")
    text.extend(['','각 조건의 값은 descriptive estimate다. Simultaneous interval은 두 schedule의 primary comparison에 적용한다.',
                 '', '## Direction readout과 Brier 변화','',
                 '| Schedule | Direction accuracy | Direction Brier | Mean squared correction | Residual alignment term | Reversed cue Δ | True cue Δ |',
                 '|---|---:|---:|---:|---:|---:|---:|'])
    for sch in ('same','alternating'):
        v=rr[sch]['overall']
        text.append('| '+sch+' | '+' | '.join(f(v[k]) for k in ('direction_accuracy','direction_brier','quadratic','alignment','wrong_cue','true_cue'))+' |')
    text.extend(['','Mean squared correction = E[(p′−p)²]이고 residual alignment term = 2E[(p′−p)(p−y)]다. '
                 '두 항의 합은 Δ와 같다. 방향을 뒤집은 cue와 true cue에는 v34의 동일 estimated β를 사용했다. '
                 '두 값은 추정한 부호가 실제 예측 오차와 정렬되는지를 진단한다.',
                 '', '## Compute 비용과 불확실성','',
                 '| Schedule | Original p90 (ms) | Corrected p90 (ms) | p90 ratio [동시 95% CI] |',
                 '|---|---:|---:|---:|'])
    for j,sch in enumerate(('same','alternating')):
        a,b=t['p90_ms'][j]
        text.append(f"| {sch} | {a:.2f} | {b:.2f} | {t['ratio'][j]:.5f} {band(t['ratio_simultaneous_ci95'][j])} |")
    text.extend(['', f"각 방법·schedule당 {t['timed_calls_per_method_schedule']}회, {t['timed_blocks']}개 paired timing block을 사용했다. "
                 '한 번의 call은 8 environments × 128 worlds에 대해 실제 관측 하나를 사용한다. '
                 '서로 다른 family에서는 seed block을 독립적으로 복원 추출하되, 네 개의 고정된 cell 자체는 다시 뽑지 않았다. '
                 '한 장치, 한 실행 session의 p90이므로 다른 GPU나 물리적 probing 대기 시간까지 일반화할 수 없다.',
                 '', '## Physics 및 information flow 검증','',
                 f"- {p['archives_verified']}개 raw archive의 checksum과 {p['input_files_verified']}개 실행 source의 checksum이 일치했다. "
                 f"CPU가 독립 구현한 reader와 GPU worker가 기록한 cue의 최대 편차는 {d['errors_cue_prob_max'][0]:.2e}, "
                 f"corrected probability의 최대 편차는 {d['errors_cue_prob_max'][1]:.2e}였다.",
                 '- 저장된 latent geometry와 contact law로 stick label을 별도 계산해 모든 archive에서 일치했다. '
                 '각 calibration의 네 paid object가 서로 다르고, 각 환경에서 128개 중 13개에 deployment probe를 배정한다. '
                 '각 양방향 schedule과 mirror에 같은 seed를 배정했고, 독립 block 사이에서 calibration/deployment/bank seeds가 겹치지 않았다.',
                 '- v34 reader와 β는 ridged_coarse source 자료에서 정한 저장 계수다. 각 schedule에 따라 reader를 선택하며 '
                 '새 family 이름, target label 또는 target orientation은 correction 계산에 들어가지 않는다. '
                 '두 arm 모두 simulator의 proxy correlation hyperparameter를 공통으로 사용한다.',
                 '- Sinusoid 및 quartic profile을 유한 차분하여 local slope와의 일치를 확인했다. '
                 'Query contact의 factual·probe sticking/sliding 조건과 잔여 normal impulse를 별도 NumPy 식으로 검산했다. '
                 '두 새 friction law가 v34와 공유하는 부호·단조성은 검증 범위의 제약이다.',
                 '', '[Statistical audit](audit35.json) · [Physics audit](physics_audit35.json) · [사전 계획](AUDIT_PLAN_KO.md)'])
    (R/'AUDIT_RESULTS_KO.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':
    main()
