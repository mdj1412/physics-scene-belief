"""Write the primary Korean results directly from sealed scored outputs."""
import json
from pathlib import Path
H=Path(__file__).resolve().parent
r=json.loads((H/'score_main68.json').read_text())
q=json.loads((H/'gpu_run.json').read_text())
s=r['summaries']
def interval(x):
    return f'{x[0]:+.6f} [{x[1]:+.6f}, {x[2]:+.6f}]'
def row(b,l,p):
    return s[f'{b}_{l}_{p}']
def report():
    out=['# v68 · 처음부터 contact law가 달라질 때의 GPU 예측',
        '',
        '### 사전 고정과 측정 경로',
        '',
        'v66의 Wendland·Dispersive object와 공통 noise를 재생하고 처음 factual contact부터 calibration과 deployment의 모든 물리 측정에 같은 contact law와 solver를 사용했다. ROI·height·surface·속도·probe 수는 같았다. v67의 source axis 네 개와 8개 readout weight를 그대로 불러왔고 checkpoint hash를 확인했다. 각 128개 deployment object 중 13개에서 물리 probe를 구매하고 나머지에서 contour query를 샀다. Q/P 중 고르지 않은 observation은 GPU inference 직전에 NaN으로 대체하고, 구현 안에서 모든 행의 mask를 검산했다. 각 법칙에 도착한 관측을 넣어 acquisition은 다시 계산했다.',
        '',
        'Calibration의 물리적 경로를 예측에 실제 연결하기 위해, 사전에 정한 v38 continuous optimizer를 4개 측정 쌍에 적용했다. 높이 0의 초기 세 방향 probe로 원시 부호 cue를 추정하고 23 mm의 마지막 +1.10 m/s probe로 source gain table의 가중치를 선택했다. 네 개의 쌍에서 얻은 log-odds shift를 다섯 predictor 모두에 동일하게 적용했다. 별도의 raw 비교에서는 이 값을 0으로 두었고, prefix3 비교에서는 네 번째 측정 이전의 source posterior를 썼다. 모두 동일한 frozen acquisition과 source head를 쓴다. Calibrated forecast는 v67 predictor에 추가된 변환이다. Solver/law·실제 sign·contact outcome은 predictor 입력에 없다. [사전 계획](FORECAST_PLAN_KO.md) · [테이프 생성](tapes68.py) · [예측 코드](predict68.py)',
        '',
        '### Held-out Brier',
        '',
        '| Solver | Law | Wendland soft−full | Dispersive soft−full | 평균 restored−dropped |',
        '|---|---|---:|---:|---:|']
    for b in r['solver']:
        for l in r['law']:
            out.append('| '+' | '.join([b,l,*map(interval,row(b,l,'cal4')['contrasts'])])+' |')
    out += ['','각 row는 12개 paired catalog에서 네 source axis를 먼저 평균했다. 새 independent compliant solver의 두 law × 세 primary contrast는 Bonferroni 동시 95% 구간이다. Prescribed row는 같은 폭을 적용한 secondary 대조다. 각 contact의 binary Brier는 두 radius와 세 offset에 동일 가중을 두었다. Held-out family는 고정된 두 종류다.','',
            '| Solver | Law | Family | Full | Soft | Hard | Dropped | Restored |','|---|---|---|---:|---:|---:|---:|---:|']
    for b in r['solver']:
        for l in r['law']:
            for family,z in zip(r['family'],row(b,l,'cal4')['risk_by_family']):
                out.append('| '+' | '.join([b,l,family,*[f'{m:.6f}' for m in z]])+' |')
    out += ['','### Calibration 입력과 source axis별 민감도','','| Compliant law | 보정 입력 | Wendland soft−full | Dispersive soft−full | 평균 restored−dropped |','|---|---|---:|---:|---:|']
    for l in r['law']:
        for policy in r['policy']:
            out.append('| '+' | '.join([l,policy,*[f'{m[0]:+.6f}' for m in row('compliant',l,policy)['contrasts']]])+' |')
    out+=['','| Law | Source seed | Wendland soft−full | Dispersive soft−full | 평균 restored−dropped |','|---|---:|---:|---:|---:|']
    for l in r['law']:
        z=row('compliant',l,'cal4')
        for seed,vals in zip(r['source_seeds'],zip(*z['seed_W_D_rescue'])):
            out.append('| '+' | '.join([l,str(seed),*[f'{m:+.6f}' for m in vals]])+' |')
        vals=z['catalog_signs']
        out+=['']
    out+=['','| Solver | Law | 평균 절대 4쌍 shift | 네 번째 관측 추가 후 평균 절대 shift 변화 | v66 원래 acquisition과 다른 물체 비율 |','|---|---|---:|---:|---:|']
    for i,b in enumerate(r['solver']):
        for j,l in enumerate(r['law']):
            z=r['shift_solver_law'][i][j]
            out.append(f"| {b} | {l} | {z[0]:.6f} | {z[1]:.6f} | {100*r['change_vs_archived_acquisition_solver_law'][i][j]:.3f}% |")
    out+=['','두 solver에서 measured calibration probe 2,304행과 deployment probe 73,728행 모두에서 law 교체가 응답을 바꿨다. Compliant solver에서 두 법칙의 쌍별 차이를 성분별 RMS로 보면 calibration probe는 0.13948 m/s, deployment probe는 0.14515 m/s였다. 원점의 0 lateral speed factual contact에서는 0.00000608 m/s에 그쳤다. 같은 물체 표면의 광학 profile은 두 law에서 동일했다. [입력 검산](input_audit68.json)',
        '',
        f"서버 {q['device']}, Slurm allocation {q['job']}에서 소스 검증·GPU inference·저장을 포함해 {q['duration_s']:.1f}초가 걸렸다. v67의 원래 tape와 frozen head를 재생한 최대 probability 차이는 {q['historical_replay_max_probability_error']:.3g}이었다. 독립 직접 Brier 재계산 최대 오차는 {r['brier_replay_absmax']:.3g}이었다. [실행 기록](gpu_run.json) · [24개 forecast shard 봉인](FORECASTS_SEALED) · [catalog별 원자료](score_main68.json) · [채점 코드](score_main68.py)",
        '']
    out+=['별도 release audit에서 일부 terminal trajectory가 엄격한 global separation 조건을 통과하지 않아 33.33 ms의 finite-time velocity를 nominal label로 쓰고 별도 unresolved mask를 저장했다. 이 표는 해당 nominal label의 결과다. 최초 한도 내 완료된 trajectory와 모든 history tape는 유지했다. [수치 종료 규약](SOLVER_STATUS_KO.md) · [미확정 label 민감도 계산](unresolved_bounds68.py)', '']
    (H/'GPU_RESULTS_KO.md').write_text('\n'.join(out))
if __name__=='__main__':report()
