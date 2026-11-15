"""Render the offline feature-subset sensitivity results without hardcoded measurements."""
import json
from pathlib import Path
r=Path('results');data=json.loads((r/'diagnostics.json').read_text())
def ci(z):
    return f"{z['mean']:+.6f} [{z['ci95'][0]:+.6f}, {z['ci95'][1]:+.6f}]"
def tbl(head,rows):
    return '\n'.join(['| '+' | '.join(head)+' |','|'+'|'.join(['---']*len(head))+'|']+
                 ['| '+' | '.join(map(str,z))+' |' for z in rows])
names={'all':'전체 factual features','mean_vx':'Profile mean 및 수평 velocity','delete_mean_vx':'위 두 features 제거','prior_width_only':'훈련 없는 prior width rule'}
rows=[]
for m,l in names.items():
    e=data['paired_effects'][m];model=data['models'].get(m,{})
    rows.append([l,f"{data['mean_loss'][m]:.6f}", f"{100*model['test_direct_fraction']:.1f}%" if model else "100%",
                 ci(e['all']),ci(e['k256']),data['flip_vs_full'][m]])
cal=data['split_calibration']['all'];rr=data['robust_decisions']['all']
parts=['### 생성한 proxy에 얼마나 의존했는가',
"""추가 분석에서는 전체 모델에서 두 가지 제한 모델을 만들었다. 하나는 profile mean과 수평 velocity만 쓰고, 다른 하나는 두 값을 제거한 나머지 네 features를 쓴다. 각 제한 모델을 원래 training/validation worlds에서만 24개 후보로 다시 학습한 뒤, 고정된 test K1024 fallback과 같은 decoder로 평가했다. 별도 prior width rule은 observation의 두 proxy를 generator의 offset 및 restitution 분포에서 유도한 근사 log-odds로 변환해 Q/P를 고른다. 이 rule은 실제 cell이나 label을 보지 않으나 generator 관계식을 사용한다. 이 표는 primary 계획 이후의 CPU replay로, 추가 p90은 측정하지 않았다.""",
tbl(['모델','Held-out Brier','직접 선택률','모델−전체 gate [95% CI]','모델−K256 [95% CI]','전체 gate와 달랐던 고유 worlds'],rows),
tbl(['전체 모델의 score 진단','고유 worlds','Spearman ρ · score와 실현 P−Q','Score 부호와 실현 winner 일치','평균 score / 실현 P−Q'],[[n,z['n_worlds'],f"{z['spearman_unique_worlds']:.3f}",f"{100*z['sign_agreement']:.1f}%",f"{z['mean_score']:+.4f} / {z['mean_actual_probe_minus_query']:+.4f}"] for n,z in cal.items()]),
tbl(['Test score quintile','평균 예측 advantage','평균 실현 advantage','Probe 우세 worlds'],[[j+1,f"{z['score_mean']:+.4f}",f"{z['realized_probe_minus_query_mean']:+.4f}",f"{100*z['realized_probe_win_fraction']:.1f}%"] for j,z in enumerate(cal['test']['quintiles'])]),
f"두 noise halves에서 0.02 이상 안정적으로 Q가 유리했던 worlds 중 직접 선택한 {rr['robust_query']['direct_worlds']}개에서 Q를 놓친 경우는 {rr['robust_query']['direct_errors']}개였다. P가 안정적으로 유리했던 worlds 중 직접 선택한 {rr['robust_probe']['direct_worlds']}개에서는 P를 놓친 경우가 {rr['robust_probe']['direct_errors']}개였다. Held-out score는 선택 전의 예측이고, 여기서 대조한 winner는 실제 hidden state와 반복 observation에 조건화된 realized loss로 정해진다."]
Path('ABLATION_KO.md').write_text('\n\n'.join(parts)+'\n')
