"""Generate report tables directly from paired estimates and pre-outcome forecasts."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent

def read(file): return json.loads((root/file).read_text())
def fmt(x): return f'{x:+.6f}'.replace('-','−')
def ci(bounds):return '['+', '.join(fmt(x) for x in bounds)+']'

def main():
    s=read('stats40.json'); f=read('forecast40.json')['cells']; e=read('forecast_eval40.json')
    verdict={'confirmed':'예측 확인','refuted':'반대 확인','unresolved':'미확정'}
    lines=['## 사전 예측과 독립 target 결과','',
           'Δ는 **optimizer Brier − fixed Brier**다. 음수이면 optimizer의 확률이 실제 stick 여부에 더 가까웠다. 사전 Δ는 v39 자료의 replay로 계산했다. 새 law나 v40 물체의 정답을 사용하지 않았다. 조건별 구간은 12개 독립 catalog panel, 12개 조건에 대해 보정한 Bonferroni–t 동시 95% 구간이다.','',
           '| Tangent / contour / law | 사전 Δ | Fixed Brier | Optimizer Brier | 실측 Δ [동시 95%] | 판정 | Optimizer 우세 panel |',
           '|---|---:|---:|---:|---:|---|---:|']
    for key,c in s['cells'].items():
        lines.append(f"| {key.replace('_',' / ')} | {fmt(f[key]['delta'])} | {c['loss'][0]:.6f} | {c['loss'][1]:.6f} | {fmt(c['mean'])} {ci(c['ci'])} | {verdict[e['verdicts'][key]]} | {c['wins']}/12 |")
    lines+=['','음수·양수 어느 쪽도 동시 구간으로 확인할 수 없으면 미확정으로 기록했다. 각 panel 안에서 두 시작 방향과 모든 paired 환경을 먼저 평균했다. 개별 물체의 승패 적중률과 다른 단위다. [사전 예측](PREDICTIONS_KO.md) · [Panel별 Brier](stats40.json)','',
            '### 부호를 맞히는 것과 이득 크기를 맞히는 것','',
            '| Forecast에 쓰는 정보 | 12개 cell의 MAE | 부호 적중 | Spearman ρ | Full − 해당 forecast의 MAE 차이 [paired bootstrap 95%] |',
            '|---|---:|---:|---:|---:|']
    full=e['metrics']['delta']
    for key,desc in [('pooled_prior','v39 전체 평균'),('tangent_only','Tangent family만 사용'),('law_replay_no_contour','Tangent와 law를 사용하고 contour 변화를 무시'),('delta','Tangent, law, 관측 contour를 모두 replay')]:
        a=e['metrics'][key];r=a['rho']
        rank='정의되지 않음' if r is None else f'{r:.3f}'
        d=full['mae']-a['mae']
        lines.append(f"| {desc} | {a['mae']:.6f} | {a['point_correct']}/12 | {rank} | {fmt(d)+' '+ci(a['mae_full_minus_this_ci']) if key!='delta' else '기준'} |")
    lines+=['',f"사전에 Δ<0인 조건에서만 optimizer를 고르면 Brier는 {e['selected_loss']:.6f}이다. 항상 fixed와의 차이는 {fmt(e['chosen_vs_fixed']['mean'])} {ci(e['chosen_vs_fixed']['ci'])}, 항상 optimizer와의 차이는 {fmt(e['chosen_vs_optimizer']['mean'])} {ci(e['chosen_vs_optimizer']['ci'])}였다. 두 구간은 각각 pointwise 95%다. 이 선택은 설계자가 아는 shape·law를 사용하므로 미지 표면의 online router 성능과 구분해야 한다. [Forecast 채점](forecast_eval40.json)",'',
            '## 같은 tangent와 물체에서 off-contact contour만 바꾼 효과','',
            '표에서 Δ 변화는 (변형 contour의 optimizer−fixed) − (base의 optimizer−fixed)다. 음수일수록 해당 변형이 optimizer에 상대적으로 유리하다. 아래의 절대 Brier 변화는 각 정책 안에서 base와 비교한 값이다.','',
            '| 조건 | Δ 변화 [8개 동시 95%] | Fixed Brier 변화 | Optimizer Brier 변화 | Query를 받은 물체의 평균 절대 base probability 변화 |',
            '|---|---:|---:|---:|---:|']
    for key,d in s['contour_change'].items():
        u,v=d['fixed_and_optimizer_change']
        lines.append(f"| {key.replace('_',' / ')} | {fmt(d['mean'])} {ci(d['ci'])} | {fmt(u)} | {fmt(v)} | {d['query_only_abs_base_change']:.5f} |")
    lines+=['','형상을 바꾼 모든 짝에서 stick label·sensor 선택·signed cue·보정량이 같았다. Probe branch에서 base probability도 같았다. 두 경로를 교환해 loss를 다시 계산하면 contour 효과 전부가 query view를 통한 base probability 변화로 복원된다. 이 결과는 접촉 위치를 고정한 본 simulator의 설계와 실측을 함께 반영한다. [분해와 negative control](stats40.py) · [검산 수치](stats40.json)','',
            '## GPU p90','',
            '| 방법 | 1,024개 물체의 GPU p90 | Matched-panel bootstrap 95% 구간 |','|---|---:|---:|']
    a=s['timing']
    for i,name in enumerate(['Fixed 32.5%','Frozen v38 optimizer']):
        lo,hi=a['ci'][i]; lines.append(f"| {name} | {a['p90'][i]:,.1f} ms | [{lo:,.1f}, {hi:,.1f}] ms |")
    lo,hi=a['ratio_ci']
    lines+=['',f"비율은 {a['ratio']:.5f} [{lo:.5f}, {hi:.5f}]로, 상한 ≤1.02인 사전 조건을 {'충족했다' if hi<=1.02 else '충족하지 못했다'}. 서버 GPU 한 장에서 24개 timing cluster, 각 방법당 72회 측정했다. 한 번의 warm online 호출에 acquisition, 선택한 센서의 decoder, 보정 선택과 host 반환이 들어간다. 실제 probe·로봇 reset·bank 구축은 제외했다. [GPU 실행](run40.py) · [Timing 원자료와 산출 방식](stats40.json)"]
    (root/'TABLES_KO.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__': main()
