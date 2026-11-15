import json
from pathlib import Path
from common import read,R
s=read(R/'summary.json');d=read(R/'secondary_diagnostics.json')
o={dev:read(R/f'timing_optimized_{dev}.json') for dev in ['gpu','cpu']}
tasks=['smooth_velocity','contact_velocity','contact_event','slip_velocity','slip_event']
cells=['matched_nominal','shifted_nominal','shifted_correct']
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def f(v):return f'{v:.6f}'
def ci(v):return f"{v['mean']:+.6f} [{v['low']:+.6f}, {v['high']:+.6f}]"
p=['# Expanded-prior control, contact/no-contact와 slip/stick transition — v15 최종 보고서\n\n작성일: 2026-11-13 KST. 기존 protocol과 numerical settings를 유지하여 서버 실험을 완료했다. 이 보고서는 2026-11-12의 진행 보고서를 대체하는 최종 결과다.']
p.append(Path('REPORT_CONCLUSION_KO.md').read_text())
p.append(Path('REPORT_METHODS_KO.md').read_text())
p.append('''## 5. Decision loss와 primary comparisons

512 independent worlds × two paired data-generating conditions × five query tasks × actual observation noise 16 repeats를 평가했다. Inference cells는 matched/nominal, shifted/nominal, shifted/correct의 세 개다. World마다 noise repeats를 평균한 뒤 independent world 단위로 paired bootstrap했다. 16 repeats나 두 paired conditions를 independent sample로 세지 않았다.

Velocity tasks는 half squared Euclidean error `(m/s)²`, event tasks는 dimensionless Brier loss다. 서로 다른 task losses를 합쳐 단일 accuracy score로 만들지 않는다. `slip_event`는 exact stick probability를 예측하는 task의 code name이다. `native`는 LMMSE acquisition, `pair`는 disagreement selector, `query`/`probe`는 fixed action이다.
''')
for cell in cells:
    p.append('### '+cell)
    names=['native','pair','nested128','nested512','variance512','query','factual','probe','none','unfiltered','agnostic']+(['qmc64_native','qmc64_pair','qmc64_query','qmc64_probe'] if cell!='shifted_correct' else [])
    p.append(table(['Method']+tasks,[[m]+[f(s['means'][cell][task][m]) for task in tasks] for m in names]))
p.append('''### Primary family와 confidence intervals

Matched contact-event/stick-event에서 pair 대 native/probe/query/nested512, nested512 대 native의 총10 contrasts를 사전에 고정했다. 30,000 paired bootstrap과 Bonferroni-adjusted **99.5% CI**다. Difference가 음수이면 앞에 적힌 방법이 유리하다. CI가 0을 포함하는 것을 equivalence로 해석하지 않는다. 나머지 contrasts는 secondary95% CI로 `results/summary.json`에 저장했다.
''')
p.append(table(['Task / comparison','Loss difference [99.5% CI]','Relative mean change'],[[k,ci(v['delta']),f"{100*v['relative_mean_change']:+.2f}%"] for k,v in s['primary'].items()]))
p.append('LMMSE failure screening은 nested512가 mean loss를 최소10% 낮추고 adjusted CI upper<0인 조건이다. 판정: '+str(s['failure_screen'])+'. Pair가 네 baseline 모두에 우위를 확보하는 screening 판정: '+str(s['pair_screen'])+'. 이는 사전 engineering criteria이며 모든 application에 적용되는 보편적 임계값은 아니다.')
p.append('![Event losses](results/event_losses.png)\n\nBars는 means이며 uncertainty는 위 paired CI table로 판단한다.')
p.append('### Observation choices')
for cell in cells:
    p.append(table([cell,'Task','[none, query_view, factual_view, probe]'],[[m,task,s['actions'][cell][task][m]] for task in ['contact_event','slip_event'] for m in ['native','pair','nested512']]))
p.append('''## 6. Model mismatch와 decoder/selection decomposition

같은 shifted data에서 nominal pipeline loss minus correct-law pipeline loss를 계산했다. Positive difference는 nominal pipeline이 더 나쁘다는 뜻이다. Correct-law는 privileged diagnostic이다. Nominal choices를 correct decoder에 적용하여 decoder effect와 selection effect를 나누었고, 두 항의 합이 total과 일치하는지 검증했다. Finite-bank error가 남으므로 exact continuous-posterior의 causal decomposition으로 확대 해석하지 않는다.
''')
for m in ['native','pair','nested512']:
    p.append(table([m,'Total [95% CI]','Decoder [95% CI]','Selection under correct decoder [95% CI]'],[[task]+[ci(s['mismatch_decomposition'][m][task][key]) for key in ['total','decoder','selection_under_correct_decoder']] for task in tasks]))
p.append(f"실제로 law를 바꾼 flexible-component worlds는 **{s['changed_worlds']}**, unchanged base-law controls는 **{s['control_worlds']}**다. Changed-only means는 다음과 같다. 이 subgroup은 전체 population primary와 구분한다.")
p.append(table(['Cell / method']+tasks,[[cell+'/'+m]+[f(s['changed_means'][cell][task][m]) for task in tasks] for cell in cells for m in ['native','pair','nested512','probe']]))
p.append('''## 7. LMMSE blind spot과 numerical audit

LMMSE는 nonlinear dependence 전체가 아니라 covariance를 통한 linear prediction gain을 계산한다. Symmetric hypotheses `h∈{−2,−0.5,0.5,2}`와 target `1{|h|<1}`의 abstract mechanism check에서는 covariance가0이면서 observation이 target을 거의 완전히 구별했다. No-observation Brier0.25, Gaussian noise SD0.1의 고정 finite samples에서 measured Brier 약1.66e−34였다. 이는 abstract illustration이며 physics population 성능을 대신하지 않는다.

Secondary subgroups는 public factual posterior만으로 정했다. Ambiguous는 event probability `[.2,.8]`, low-linear-gain은 event variance>.05이면서 maximum LMMSE gain/variance<.2다. Hidden targets로 유리한 subset을 고르지 않았다. 작은 subgroup의 descriptive CI는 primary evidence와 구분한다.
''')
for cell in ['matched_nominal','shifted_nominal']:
    rows=[]
    for key,v in d[cell]['subgroups'].items():rows.append([key,v['n'],ci(v['pair-native']) if 'pair-native' in v else '—',ci(v['nested512-native']) if 'nested512-native' in v else '—'])
    p.append(table([cell,'n','Pair − native [95% CI]','Nested512 − native [95% CI]'],rows))
p.append('### Integration sensitivity: first32 worlds')
for cell in cells:
    rows=[]
    for key,v in s['audit'][cell].items():
        if key=='reference_regret':continue
        rows.append([key]+[f"{100*v['agreement_per_task'][k]:.2f}% / {v['loss_change_per_task'][k]:+.6f}" for k in [2,4]])
    p.append(table([cell,'Contact event: agreement / loss change','Stick event: agreement / loss change'],rows))
p.append('Pair K512는 main pairK256과, nested variants는 main nested512와 비교했다. QMC64 nested는 bank size와 scramble도 바뀌므로 pure outer-sample audit와 다르다. First32 worlds는 전체512에 대한 uniform stability 보증이 아니다. Primary choices는 audit를 보고 변경하지 않았다. Independent-seed nested2048 내부 risk에 대한 regret도 `results/summary.json`에 저장했으며 실제 held-out loss와 구분한다.')
p.append('### Posterior ESS와 common-action decoder audit')
p.append(table(['Cell','Factual ESS median','Factual ESS p05','Pair eligible fraction','Pair fallback counts (five tasks)'],[[c,f(s['diagnostics'][c]['factual_ess_median']),f(s['diagnostics'][c]['factual_ess_p05']),f(s['diagnostics'][c]['eligible_mean']),s['diagnostics'][c]['fallback_count']] for c in cells]))
p.append(table(['Cell / task','QMC16 pair-selected ESS median','QMC64 same-action ESS median','QMC16 fraction ESS<20','QMC16 loss − QMC64 decoder loss [95% CI]'],[[c+'/'+tasks[k],f(d[c]['ess']['pair']['qmc16']['median_per_task'][k]),f(d[c]['ess']['pair']['qmc64']['median_per_task'][k]),f"{100*d[c]['ess']['pair']['qmc16']['below20_fraction_per_task'][k]:.2f}%",ci(d[c]['decoder_swaps']['pair'][tasks[k]]['small_minus_large'])] for c in ['matched_nominal','shifted_nominal'] for k in [2,4]]))
p.append('''## 8. Compute 비용

RTX PRO6000 Blackwell Server Edition, float64, batch1에서 측정했다. CPU는 같은 node의2threads다. Fixed20 matched worlds×5tasks×3repetitions, method당300 measurements이며 warmup 후 factual fit, query-specific target generation, acquisition, selected-observation update/prediction을 포함했다. GPU synchronization을 적용했다. Selector에는 unselected actual observations를 전달하지 않고, 선택 후 해당 observation 하나로 update한다. Online action/prediction은 stored evaluation과 대조했다.
''')
names=['native','pair','nested128','nested512','qmc64_native','qmc64_pair','query','probe']
p.append('### 수학적으로 동일한 optimized runtime: 최종 compute 비교\n\n최초 timing에서 skinny matrix contraction에 따른 큰 latency 차이가 관찰되어 사후 implementation audit를 추가했다. LMMSE의 observation moments를 cache하고 nested prediction을 output별 matrix-vector products로 계산했다. Direct-risk decision에 사용하지 않는 posterior-variance 계산도 제거했다. Primary policies/settings/losses는 바꾸지 않았다. 동일100 cases×3에서 actions와 predictions, 비교 가능한 candidate scores가 일치하는지 검증했다. 아래 optimized 비용을 최종 trade-off 평가에 사용한다.')
p.append(table(['Method','GPU median ms','CPU median ms','GPU contact-event ms','GPU stick-event ms'],[[m,f"{o['gpu'][m]['median_s']*1000:.3f}",f"{o['cpu'][m]['median_s']*1000:.3f}"]+[f"{o['gpu'][m]['task_medians_s'][task]*1000:.3f}" for task in ['contact_event','slip_event']] for m in names]))
p.append(table(['Method','Optimized bank/moment cache bytes','Observed temporary bytes','GPU warm setup ms'],[[m,o['gpu'][m]['cache_bytes'],o['gpu'][m]['temporary_peak_bytes'],f"{1000*o['gpu'][m]['setup_s']:.3f}"] for m in names]))
p.append(f"Optimized aggregated median 기준 nested512/pair는 GPU **{o['gpu']['nested512']['median_s']/o['gpu']['pair']['median_s']:.2f}배**, CPU **{o['cpu']['nested512']['median_s']/o['cpu']['pair']['median_s']:.2f}배**다. Pair/native GPU latency ratio는 **{o['gpu']['pair']['median_s']/o['gpu']['native']['median_s']:.2f}**다. Fixed actions와 더 저렴한 LMMSE도 함께 비교해야 한다.")
p.append('![Optimized loss and latency](results/optimized_tradeoff.png)')
p.append('### 최초 runtime 측정값: implementation sensitivity 기록\n\n다음 수치는 원래 code의 결과로 보존한다. 이를 optimized baseline 대비 algorithmic speedup으로 사용하지 않는다.')
p.append(table(['Method','GPU median ms','GPU p90 ms','CPU median ms','GPU contact-event ms','GPU stick-event ms'],[[m,f"{s['timing_gpu'][m]['median_s']*1000:.3f}",f"{s['timing_gpu'][m]['p90_s']*1000:.3f}",f"{s['timing_cpu'][m]['median_s']*1000:.3f}"]+[f"{s['timing_gpu'][m]['task_medians_s'][task]*1000:.3f}" for task in ['contact_event','slip_event']] for m in names]))
for device in ['gpu','cpu']:
    p.append(f"{device.upper()} aggregated median 기준 nested512/pair latency ratio는 **{s['timing_'+device]['nested512']['median_s']/s['timing_'+device]['pair']['median_s']:.2f}**다. Pair/native latency ratio는 **{s['timing_'+device]['pair']['median_s']/s['timing_'+device]['native']['median_s']:.2f}**다.")
p.append(table(['Method','Belief weights bytes','Bank cache bytes','Observed GPU temporary peak bytes','Warm setup ms'],[[m,s['timing_gpu'][m]['state_bytes'],s['timing_gpu'][m]['cache_bytes'],s['timing_gpu'][m]['temporary_peak_bytes'],f"{1000*s['timing_gpu'][m]['setup_s']:.3f}"] for m in names]))
p.append('State bytes는 per-world posterior weight arrays만 센 numerical payload다. Bank cache는 theta/observation/whitened arrays이며 Python objects, CUDA context, allocator reserve, Sobol plan 등 전체 process memory가 아니다. Temporary peak는20 smooth-velocity cases에서 관찰한 추가 CUDA allocated memory다. 모든 tasks/inputs의 global peak로 부르지 않는다. Setup은 warm process bank construction이며 cold startup 전체가 아니다. Physical sensing/reset 비용은 측정하지 않았다. 서로 다른 sample budgets를 같은 compute라고 가정하지 않고 실제 latency를 제시했다.')
p.append('최초 optimized audit에서는 nested128의 no-observation diagnostic score까지 동일하다고 검사하여 assertion이 발생했다. Main nested128은 conditional128 samples와 diagnostic512 samples를 함께 저장하지만 online128은 모두128 samples를 쓴다. 선택 후보는 actions1..3뿐이므로 그 실제 candidate scores를 비교하도록 audit만 수정했다. Main policies, selected actions, predictions 또는 primary results를 변경한 것은 아니다. 실패 로그도 보존했다.')
p.append(Path('REPORT_DISCUSSION_KO.md').read_text())
p.append('''## 10. 실행과 재현성

Server project: `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/transition_control_v15`. 기존 allocation2680491, node56에서 실행했다. 2026-11-12의 세 차례 interruption은 별도 Blender process를 감지한 guard가 자신의 worker만 종료한 것이다. 2026-11-13 GPU가 비워졌다는 사용자 지시 후 process/memory를 확인하고 재개했다. 기존 jobs/allocation은 변경하지 않았다.

`PROTOCOL.md`는 primary settings, `DIAGNOSTICS_SPEC.md`는 secondary subgroup 기준이다. `model.py`와 `acquisition.py`는 candidate forward/acquisition, `generate.py`는 private generator, `run_policy.py`는 public-only policy selection, `evaluate.py`는 policy freeze 이후 actual outcomes 평가를 담당한다. `benchmark.py`는 online timing, `analyze.py`와 `diagnostics.py`는 paired statistics 및 numerical diagnostics다.

Actual outputs는 `results/policies`, `results/audits`, `results/evaluations`, `results/world_losses.npz`, `results/summary.json`, `results/secondary_diagnostics.json`, `results/timing_gpu.json`, `results/timing_cpu.json`에 있다. `source_policy.json`, `policy_freeze.json`, worker logs와 exit records가 실행 evidence다. `INTEGRITY.json`과 `MANIFEST.json`에 source/policy/result 및 이전 v14 artifacts 검증을 남겼다.

실제 simulation/inference/statistics는 모두 allocated compute node에서 수행했다. Local mirror는 준비·검토·전달용이다. 재실행할 때는 sealed outputs를 덮어쓰지 않고 새 directory에 code/data를 복사한다. `run_resume.py` → `benchmark.py --cpu` → `analyze.py` → `diagnostics.py` → `write_report.py` → `seal.py` 순서이며, GPU와 CPU timing을 다른 workload와 겹치지 않게 실행한다.
''')
Path('REPORT_KO.md').write_text('\n\n'.join(p));print('Wrote REPORT_KO.md')
