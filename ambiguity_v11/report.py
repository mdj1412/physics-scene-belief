"""Build Korean report from measured summaries and reviewed interpretation."""
import json
from pathlib import Path
P=Path('/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/ambiguity_v11')
R=Path('results');s=json.loads((R/'analysis/summary.json').read_text());t=s['test'];o=s['misspec'];m=t['means']
labels={'none':'추가 observation 없음','rule_fixed':'Best fixed rule + joint decoder','rule_best':'Learned rule + joint decoder','lmmse':'LMMSE + joint decoder','plugin_aware':'Updated hard plug-in','mixture_aware':'Updated mixture plug-in','factorized':'Factorized factual belief','standard_snis':'Standard SNIS + joint decoder','cf_snis':'Cross-fitted SNIS + joint decoder','joint':'Joint selector + joint decoder'}
examples=json.loads((R/'analysis/examples.json').read_text())
def table(rows,head):return '| '+' | '.join(head)+' |\n| '+' | '.join(['---']*len(head))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)
lines=['# Geometry–contact ambiguity v11: 동일 visibility/query에서 observation의 가치\n',
'작성일: 2026-11-12 KST. 서버에서 실제 구현·실행한 controlled diagnostic이다. 설명은 한글, 전문용어는 English로 작성했다.\n',
Path('INTERPRETATION_KO.md').read_text(),
'## Primary test 결과\n',
f'Independent test **256 worlds × 32 additional-observation noise repeats**를 사용했다. 모든 world의 visibility와 query는 같다. Joint의 no-observation 대비 평균 MSE 감소율은 **{100*(1-m["joint"]/m["none"]):.2f}%**다. 이 비율은 pilot에서 의도적으로 만든 ambiguity distribution에 대한 값이다.\n',
table([(labels[k],f'{m[k]:.9f}') for k in labels],['방법','Decision MSE (m²/s²)']),
f'![Decision loss]({P}/results/analysis/decision_loss.png)\n',
'Figure의 CI는 방법별 descriptive95% CI다. Primary 판단은 아래 paired comparisons를 사용한다.\n',
'### Primary paired comparisons\n',
'차이는 Joint − comparator다. 음수이면 Joint가 유리하다. World-cluster bootstrap30,000회와5 comparisons에 대한 Bonferroni99% CI를 사용했다.\n',
table([(labels[k],f'{v["mean"]:+.9f}',f'[{v["low"]:+.9f}, {v["high"]:+.9f}]') for k,v in t['primary_joint_minus'].items()],['Comparator','평균 차이','99% CI']),
'### Observation 선택과 decoder를 분리한 비교\n',
table([(name,f'{m[key]:.9f}') for name,key in [('Joint choices + joint decoder','joint'),('Joint choices + hard plug-in','plugin_at_joint'),('Joint choices + mixture plug-in','mixture_at_joint'),('Joint choices + factorized decoder','factorized_at_joint'),('Hard plug-in choices + common joint decoder','plugin_aware_common'),('Factorized choices + common joint decoder','factorized_common'),('Learned rule choices + mixture plug-in','mixture_at_rule_best')]],['조합','Decision MSE']),
'이 표의 same-action comparisons는 representation/decoding 효과를, common-decoder comparisons는 acquisition 효과를 분리한다. 모든 수치의 paired descriptive95% CI는 `results/analysis/summary.json`에 있다.\n',
'## 같은 visibility/query에서 선택이 달라졌는가\n',
table([(labels.get(k,k),*v,f'{100*t["agreement"][k]:.2f}%') for k,v in t['actions'].items()],['Policy','none','query-view','factual-view','probe','Joint와 일치율']),
f'![Action context]({P}/results/analysis/action_context.png)\n',
'이 figure의 모든 점은 같은 visibility/query를 갖는다. Horizontal axis는 factual profile에서 구한 slope estimate이고 vertical axis는 frictional contact의 posterior probability다. Ground-truth contact class를 selector에 제공하지 않았다.\n',
f'Factual posterior의 geometry–contact mutual information은 평균 **{t["MI_mean"]:.5f} nats**, median **{t["MI_median"]:.5f} nats**였다. |Corr(s,e)| 평균은 **{t["abs_s_e_corr_mean"]:.5f}**다. MI는 geometry=(g,s,a), contact=(c,e) 묶음 사이의 dependence를 측정한다. Correlation의 존재 자체가 decision improvement를 증명하는 것은 아니다.\n',
'### Ranking audit\n',
table([(k,str(v)) for k,v in s['ranking_audit'].items()],['항목','측정값']),
'Order61, independent outer samples1024개씩 두 번으로 first32 test worlds를 다시 평가했다. Main policies를 바꾸지 않았다. 이 reference도 finite outer estimate이며 exact optimum은 아니다. Reference regret는 평균 reference risks 중 선택 action과 가장 작은 값의 차이다.\n',
'### 두 가지 구체적인 factual evidence 사례\n',
table([(f'World {e["world"]:03d}',e['action'],f'{e["p_friction_factual"]:.6f}',f'{e["predicted_risks"][1]:.6f}',f'{e["predicted_risks"][3]:.6f}') for e in examples],['사례','선택','Factual p(frictional)','Query-view expected risk','Probe expected risk']),
'World000에서는 contact class가 거의 결정되어 geometry measurement가 더 유리했다. World023에서는 contact class가 여전히 모호해 probe가 더 유리했다. Visibility와 query는 동일하다. 두 사례는 first32 worlds 안에서 각 선택의 predicted risk margin이 가장 큰 사례이며, actual test loss나 true contact class로 고르지 않았다. 대표 빈도나 평균 효과를 추정하는 표는 아니다.\n',
f'![Posterior examples]({P}/results/analysis/posterior_examples.png)\n',
'Figure는 geometry family와 amplitude를 marginalize한 slope–restitution quadrature posterior mass다. 각 row는 하나의 world, 각 column은 contact class다.\n',
'## Candidate family 밖의 stress test\n',
table([(labels[k],f'{o["means"][k]:.9f}') for k in ['none','rule_best','lmmse','plugin_aware','mixture_aware','factorized','cf_snis','joint']],['방법','Misspecification MSE']),
'별도128 worlds 결과다. In-support test와 합쳐 평균내지 않았다. Third boundary family와 continuous friction을 동시에 바꾼 exploratory stress test이며 real-world validation은 아니다.\n',
Path('METHODS_KO.md').read_text(),
Path('NUMERICAL_REPORT_KO.md').read_text(),
Path('CONTRIBUTION_KO.md').read_text(),
Path('LITERATURE_KO.md').read_text(),
'## 실행 및 재현 기록\n',
'서버 경로: `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/ambiguity_v11`. `datacenter3`의 기존 allocation2680491, compute node `node56`, idle NVIDIA RTX PRO 6000 한 개를 사용했다. 기존 jobs와 v10 결과는 변경하지 않았다.\n',
'Python 환경은 기존 `../continuous_v2/.venv/bin/python`이다. Main simulation/inference는 GPU, rule fitting과 analysis는 compute node의 CPU에서 수행했다. Login node에서는 numerical experiment를 실행하지 않았다.\n',
'LMMSE와 order81/101 numerical audit도 같은 allocation의 CPU에서 CUDA를 비활성화하여 수행했다. GPU primary worker와 충돌하지 않도록 분리했으며, 이 audit은 CPU/GPU execution 차이까지 포함한 cross-check다.\n',
f'Main acquisition의 test world당 평균 시간은 **{t["acquisition_seconds_mean"]:.4f}초**다. Joint, hard/mixture plug-in, factorized, Standard/CF-SNIS를 함께 계산한 시간이며 개별 method latency가 아니다. Rule fitting에는 **{s["rule_fit"]["seconds"]:.3f}초**가 들었고 training risk-label 생성 비용은 별도다. Online compute가 동일한 budget이라고 주장하지 않는다.\n',
f'Validation에서 선택한 learned rule은 **{s["rule_fit"]["selected"]}**이다. RBF width={s["rule_fit"]["rbf_width"]}, ridge={s["rule_fit"]["rbf_ridge"]}; CART depth={s["rule_fit"]["tree_depth"]}, min-leaf={s["rule_fit"]["tree_minleaf"]}. SNIS median ESS는 query-view/factual-view/probe 순서로 **{t["snis_ess_median"]}**다.\n',
'Code와 result files의 SHA256 manifest를 별도로 저장한다. `source_main.json`, `environment.json`, `pipeline_done.json`, worker logs, 각 split의 `freeze.json`, `extra_freeze.json`, `integrity.json`이 실행과 policy provenance를 기록한다.\n',
'주요 files: `model.py`(physics/inference), `run.py`(public-only acquisition·freeze·evaluation), `rules.py`(strong rules), `extra.py`(LMMSE), `numerical.py`(high-order audit), `analyze.py`(paired analysis), `PROTOCOL.md`·`DESIGN_ADDENDUM.md`·`LMMSE_ADDENDUM.md`(설계 기록).\n']
Path('REPORT_KO.md').write_text('\n'.join(lines))
print('REPORT_KO.md written')
