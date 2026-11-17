"""Render the frozen diagnostic comparisons after all panels are complete."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def read(name):
    return json.loads((ROOT/name).read_text())

def ci(v,scale=1,digits=6):
    a,b=v['ci']
    return f"{scale*v['mean']:.{digits}f} [{scale*a:.{digits}f}, {scale*b:.{digits}f}]"

def main():
    metrics=read('selector_audit41.json')['metrics']
    cross=read('crossfit41.json')
    out=['# V41 보정 선택의 환경별 재검증','',
        '평가할 물체 128개의 관측으로 한 번 내린 선택이 어느 environment에서 이득을 내는지 검산했다. 아래 수치는 12개 독립 catalog panel에서 계산한 paired estimate다. 괄호는 개별 대조의 95% t interval이며, 여러 대조를 한꺼번에 보정한 주 분석은 본 보고서에서 다룬다. 모든 Brier 차이는 음수일 때 앞의 방법이 우세하다.','',
        '| Decoder | Optimizer를 택한 환경·noise 비중, % | 선택한 곳의 optimizer−fixed | 선택하지 않은 곳의 optimizer−fixed | Selector−fixed | Selector−동일 비율 blind mixing |','|---|---:|---:|---:|---:|---:|']
    for key,v in metrics.items():
        out.append(f"| {key} | {ci(v['optimizer_fraction_ci'],100,1)} | {v['mean_gap_if_selected']:+.6f} | {v['mean_gap_if_fixed']:+.6f} | {v['route']:+.6f} | {ci(v['vs_blind'])} |")
    out.extend(['','각 panel·cell 내에서 optimizer 선택 비율만 맞추면, 관측을 보지 않고도 `fixed + 선택 비율 × 평균(optimizer−fixed)`의 Brier를 기대할 수 있다. 마지막 열은 실제 selector가 그 기대값보다 낮은 loss의 환경에 optimizer를 배정했는지를 측정한다. 선택한 곳과 선택하지 않은 곳의 두 열에는 각각 그 subset 전체의 실제 평균을 썼다. 이 두 subset은 무작위로 나눈 집단이 아니다.','',
        '| Decoder | 원래 decoder와 선택이 다른 비율, % | Selector−원래 decoder에서 내린 선택 재사용 | Selector−더 높은 stick probability 선택 | Selector−같은 배치에서 사후 선택 |','|---|---:|---:|---:|---:|'])
    for key,v in metrics.items():
        out.append(f"| {key} | {100*v['switch_vs_original']:.1f} | {ci(v['vs_original_choice'])} | {ci(v['vs_v39_choice'])} | {ci(v['regret_vs_oracle'])} |")
    out.extend(['','더 높은 stick probability를 택하는 규칙은 calibration cue의 부호와 두 보정량만으로 실행할 수 있다. 사후 선택은 128개 정답을 확인한 후 그 배치에서 Brier가 작은 방법을 고른 값이다. 이는 같은 배치의 경험적 하한이다.','',
        '| Decoder | Cross-fitted−fixed | Cross-fitted−항상 optimizer | Selector−cross-fitted | 같은 배치의 사후 선택−cross-fitted | 두 half에서 우세한 정책이 일치, % |','|---|---:|---:|---:|---:|---:|'])
    for key,v in cross.items():
        out.append(f"| {key} | {ci(v['crossfit_minus_fixed'])} | {ci(v['crossfit_minus_optimizer'])} | {ci(v['selector_minus_crossfit'])} | {ci(v['same_batch_oracle_minus_crossfit'])} | {ci(v['half_sign_agreement'],100,1)} |")
    out.extend(['','Cross-fitting은 128개의 절반가량에서 실제 label로 방법을 택한 뒤 나머지 물체에 적용하고 역할을 맞바꾼 결과다. 같은 prototype ID의 물체는 항상 같은 half에 둔다. 서로 다른 8개 분할의 평균을 내지만 독립 panel 수는 12개다. 한 방법이 거의 모든 환경에서 우세해도 두 half의 일치율은 높을 수 있다. 실제로 label 정보가 선택에 주는 이득은 cross-fitted 방법과 두 고정 방법의 Brier 대조로 해석한다. 실제 운영에 쓸 수 있는 label-free 선택 방법의 이론적 한계를 추정하지는 않는다.','',
        '사전 봉인한 소스와 계수는 [선택식](SELECTOR_KO.md), 보조 대조 정의는 [선택 진단](SELECTOR_DIAGNOSTICS_PLAN_KO.md)과 [cross-fitting 계획](CROSS_FIT_KO.md), 수치는 [진단 JSON](selector_audit41.json) 및 [cross-fitting JSON](crossfit41.json)에 있다.'])
    (ROOT/'SELECTOR_AUDIT_KO.md').write_text('\n'.join(out)+'\n')

if __name__=='__main__':
    main()
