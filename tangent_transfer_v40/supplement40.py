"""Include paired checks and the cost of withholding a physical law's identity."""
import json
from pathlib import Path
from tables40 import fmt,ci
root=Path(__file__).resolve().parent
def data(file): return json.loads((root/file).read_text())
def main():
    p=data('projection40.json'); l=data('law_blind40.json');e=data('forecast_eval40.json')
    pc=p['projected_minus_original']
    lines=['## Law와 관측 방향에 대한 보조 대조','',
           f"**Law 이름이 없는 경우.** 같은 contour에서 두 law의 사전 예측을 평균했다. 부호는 {l['sign_hits']}/12에서 맞고 MAE는 {l['mae']:.6f}였다. Law를 아는 full forecast의 MAE는 {e['metrics']['delta']['mae']:.6f}이며, law를 가렸을 때의 MAE 변화의 95% 구간은 {ci(l['mae_minus_full_ci'])}였다. 이 규칙이 고른 정책의 Brier는 {l['brier']:.6f}, full forecast가 고른 정책 대비 차이는 {fmt(l['blind_minus_full']['mean'])} {ci(l['blind_minus_full']['ci'])}였다. 이 결과도 contour 이름과 후보 law 두 개의 비율을 안다고 가정한다. 주 실험 시작 뒤 target score 열람 전에 추가한 대조다. [보조 계획](LAW_BLIND_PLAN_KO.md) · [수치](law_blind40.json)",'',
           '**Decoder가 contour 변화에 둔감해지면 어떻게 되는가.** Query view의 7개 좌표 중 변형이 일어나는 한 방향을 미리 알고 있다고 가정했다. 관측과 256개 template을 sensor noise로 표준화한 뒤 그 방향에 수직인 여섯 차원만으로 posterior를 계산했다. 실제 변형의 부호·진폭, law, label은 사용하지 않았다. 각 catalog에서 홀짝으로 정한 한 block만 사용해 두 decoder에 동일하게 적용했다.','',
           '| 동일한 절반 표본 | Original Brier | Projected Brier | Projected − original [pointwise 95%] |','|---|---:|---:|---:|']
    for i,method in enumerate(['Fixed 32.5%','Optimizer']):
        lines.append(f"| {method} | {p['original'][i]:.6f} | {p['projected'][i]:.6f} | {fmt(pc['mean'][i])} {ci([pc['lo'][i],pc['hi'][i]])} |")
    lines+=['','| Contour | Projection이 바꾼 fixed Brier | Projection이 바꾼 optimizer Brier |','|---|---:|---:|']
    for v,x in p['per_variant_change'].items():
        lines.append(f"| {v} | {fmt(x['mean'][0])} | {fmt(x['mean'][1])} |")
    lines+=['',f"같은 물체의 변위가 제거한 방향과 평행하므로 projected probability는 이론적으로 세 contour에서 같다. 실제로 각 batch에서 점검한 paired query의 최대 오차는 {p['max_invariance_error_checked']:.2g}였다. Projected decoder에서 optimizer − fixed는 {fmt(p['projected_optimizer_minus_fixed']['mean'])} {ci([p['projected_optimizer_minus_fixed']['lo'],p['projected_optimizer_minus_fixed']['hi']])}다. 변형 방향을 아는 이 대조는 nuisance 제거로 얻는 이득과 유효한 기하 정보를 같이 잃는 비용을 포함한다. 이 보조 decoder만의 GPU p90은 측정하지 않았다. [사전 계획](PROJECTION_PLAN_KO.md) · [48개 paired world group 결과](projection40.json) · [구현](projection40.py)"]
    (root/'SUPPLEMENT_KO.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
