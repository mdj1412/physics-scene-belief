from common import *
from run_policy import CELLS,ETASKS,ADAPT
from analyze import METHODS
s=read(R/'summary.json')
def f(x):return f'{x:.6f}'
def signed(x):return f'{x:+.3e}' if 0<abs(x)<1e-7 else f'{x:+.6f}'
def ci(x):return signed(x['mean'])+' ['+signed(x['low'])+', '+signed(x['high'])+']'
def pct(x):return f'{100*x:.2f}%'
def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','|'+'|'.join(['---']*len(head))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
p=['# Adaptive decision-region compression — v17\n\n작성일: 2026-11-13 KST. Allocated server에서 구현, fresh-world experiments, ranking audits 및 GPU/CPU timing을 완료했다.']
p.append(Path('REPORT_CONCLUSION_KO.md').read_text());p.append(Path('REPORT_METHODS_KO.md').read_text())
p.append('## 4. Decision loss\n\n각 cell의 512 worlds × 16 observation repeats 결과다. 낮을수록 좋다. V16과 worlds가 다르므로 V16 보고서의 평균과 직접 비교해 개선율을 만들지 않는다.')
for cell,_,_ in CELLS:p.append('### '+cell+'\n\n'+table(['Method']+ETASKS,[[m]+[f(s['means'][cell][t][m]) for t in ETASKS] for m in METHODS]))
p.append('### Primary paired contrasts\n\nAdaptive minus baseline이다. 음수는 adaptive가 유리하다. CI는 12 comparisons를 보정한 99.58333% CI다.')
p.append(table(['Task / comparison','Difference [adjusted CI]'],[[k,ci(v)] for k,v in s['primary'].items()]))
p.append('10⁻¹³ 수준의 차이나 CI endpoint는 numerical precision에 가까우므로 실질적 superiority로 해석하지 않는다. 특히 adaptive와 fixed128의 losses는 이번 sample에서 사실상 같았다.')
p.append('### Approximation 및 adaptivity screens')
p.append(table(['Task / adaptive-to-baseline','Mean-loss ratio [95% CI]'],[[t+'/'+m,f"{v['mean']:.4f} [{v['low']:.4f}, {v['high']:.4f}]"] for t,vals in s['ratios'].items() for m,v in vals.items()]))
p.append('사전 screen 결과: `'+str(s['screens'])+'`. Superiority가 검출되지 않은 것과 equivalence는 다르다.')
p.append('### Observation choices')
p.append(table(['Cell / method','Contact [none,query,factual,probe]','Stick [none,query,factual,probe]'],[[c+'/'+m]+[s['actions'][c][t][m] for t in ETASKS] for c,_,_ in CELLS for m in ['adaptive','adaptive_global','fixed128','global128','native','nested512']]))
p.append('## 5. Ranking 안정성과 error diagnostics')
p.append('### Independent acquisition seed: 512 matched worlds\n\nHypothetical sampling과 compression phases를 함께 바꾼 Top-1 action agreement다. Loss 변화는 seed2 minus primary이며 secondary 95% CI다.')
p.append(table(['Method','Contact agreement','Stick agreement','Contact loss change','Stick loss change'],[[m]+[pct(x) for x in s['ranking']['matched_nominal'][m]['seed2_agreement']]+[ci(x) for x in s['ranking']['matched_nominal'][m]['seed2_loss_delta']] for m in ADAPT+['nested512']]))
p.append('Fixed32 대비 adaptive의 agreement 증가량: '+str({t:{k:round(100*v,3) for k,v in x.items()} for t,x in s['repeatability_improvement'].items()})+' percentage points. 여기의 CI는 secondary95%다.')
p.append('### Independent posterior banks: 첫 64 matched worlds\n\nAcquisition banks만 교체하고 실제 prediction decoder는 primary bank로 고정했다. Mean-loss delta는 bank2 selection minus primary selection이다.')
p.append(table(['Method','Contact agreement','Stick agreement','Contact loss delta','Stick loss delta'],[[m]+[pct(x) for x in s['ranking']['matched_nominal'][m]['bank2_agreement']]+[f(x) for x in s['ranking']['matched_nominal'][m]['bank2_loss_delta']] for m in ['adaptive','adaptive_global','fixed128','global128','native','nested512']]))
p.append('### Nested2048 reference: 각 cell의 첫 64 worlds\n\nReference risk regret는 nested2048가 추정한 risk 아래의 차이이며 실제 Bayes regret가 아니다. Agreement만 높이고 actual loss가 나빠질 수 있으므로 양쪽을 구분한다.')
p.append(table(['Cell / method','Contact agreement','Stick agreement','Contact reference regret','Stick reference regret'],[[c+'/'+m]+[pct(x) for x in s['ranking'][c][m]['reference_top1_agreement']]+[f(x) for x in s['ranking'][c][m]['reference_risk_regret']] for c,_,_ in CELLS for m in ['adaptive','adaptive_global','fixed32','fixed128','global128','native','nested512']]))
p.append('### Components와 stopping reason\n\nK counts는 region당 K=[8,16,32,64,128]의 순서다. Reason counts는 [gap-separated, cap-unresolved, risk-tie, fixed-size] 순서다. Processed components는 모든 expansion stages와 두 replicas에서 처리한 component 수의 합이다. Final payload와 다른 지표다.')
p.append(table(['Cell / method / task','K counts','Mean K / stages','Reason counts','Mean processed components'],[[c+'/'+m+'/'+t,list(x['K_counts'].values()),f"{x['mean_K']:.2f} / {x['mean_stages']:.2f}",x['reason_counts'],f"{x['mean_total_replica_components_processed']:.1f}"] for c,_,_ in CELLS for m in ['adaptive','adaptive_global','fixed32','fixed128','global128'] for t,x in s['expansion'][c][m].items()]))
p.append(table(['Cell / method / task','Cap worlds where sampling term ≥ gap','Gap-separated reference subset n','Reference disagreements in that subset'],[[c+'/'+m+'/'+t,x['cap_sampling_alone_exceeds_gap'],x['settled_reference_n'],x['settled_reference_disagreements']] for c,_,_ in CELLS for m in ['adaptive','adaptive_global'] for t,x in s['expansion'][c][m].items()]))
d=read(R/'diagnostics.json')
p.append('### Post-analysis label-support audit와 stopping groups')
p.append('Risk tie는 risk range≤10⁻¹²라는 코드 조건이다. 반드시 floating-point roundoff만을 뜻하지 않는다. 아래는 full factual posterior에는 두 label regions가 있지만 두 global replicas가 각각 single-label support가 된 cases다. Loss나 policy를 변경하지 않은 explanatory audit다.')
p.append(table(['Task','Global risk ties','Ties with both factual regions','Both replicas single-label','Minority mass [min,median,max]'],[[t,v['global_risk_ties'],v['ties_with_both_factual_regions'],v['ties_with_both_replicas_single_label'],v['minority_mass_in_collapsed_ties_quantiles']] for t,v in d.items()]))
p.append(table(['Task / group','n','Seed2 agreement','Mean Brier loss','Loss minus fixed128'],[[t+'/'+g,x['n'],pct(x['seed2_agreement']) if x['n'] else 'NA',f(x['loss']) if x['n'] else 'NA',f(x['loss_minus_fixed128']) if x['n'] else 'NA'] for t,v in d.items() for g,x in v['groups'].items()]))
p.append('Groups는 primary stopping result로 사후 구분했으므로 서로 다른 groups 사이의 mean-loss 차이를 causal effect로 해석하지 않는다. Support collapse가 발견됐다고 해서 global method의 aggregate loss가 유의하게 나빴다고 주장하지도 않는다.')
p.append('## 6. Compute 비용\n\nRTX PRO6000 Blackwell Server Edition, CPU2 threads, float64, batch1. 아래 primary compute 표는 equivalent optimized implementation의 동일 timing pass다. 고정된 첫 20 matched worlds × 2 tasks × 3 repeats로 method당120 measurements를 수행했다. Factual fitting, public targets, outer-bank fitting, compression, stopping checks, 선택한 actual observation update를 포함한다. Bank construction과 static forward cache setup은 별도로 기록했다. GPU와 CPU는 순차 실행했다. V16의 5-task aggregate latency와 직접 비교하지 않는다.')
names=list(s['timing_gpu'])
p.append(table(['Method','GPU median ms','GPU p90 ms','CPU median ms','GPU contact ms','GPU stick ms'],[[m,f"{1000*s['timing_gpu'][m]['median_s']:.3f}",f"{1000*s['timing_gpu'][m]['p90_s']:.3f}",f"{1000*s['timing_cpu'][m]['median_s']:.3f}"]+[f"{1000*s['timing_gpu'][m]['task_medians_s'][t]:.3f}" for t in ETASKS] for m in names]))
p.append('### 구현 최적화 전후의 observed latency')
p.append(table(['Method','Original GPU ms','Optimized GPU ms','Original CPU ms','Optimized CPU ms'],[[m]+[f"{1000*s[key][m]['median_s']:.3f}" for key in ['timing_original_gpu','timing_gpu','timing_original_cpu','timing_cpu']] for m in names]))
p.append('Policy가 고정된 후 CDF caching과 logging-related synchronization을 줄였다. 15,360 cases의 equivalence를 검증했다. Original logs/results도 보존했다. Baselines는 같은 새 pass에서 다시 측정했으며 pass 사이의 latency variation도 표에 남겼다. 특히 CPU nested timing 변동이 있어 absolute speedup을 다른 hardware의 보장으로 해석하지 않는다.')
p.append(table(['Method','Bank/cache bytes','Two-replica compact payload max bytes','Observed CUDA temporary peak bytes','Warm setup ms'],[[m,s['timing_gpu'][m]['bank_cache_bytes'],s['timing_gpu'][m]['compact_two_replica_bytes'],s['timing_gpu'][m]['temporary_peak_bytes'],f"{1000*s['timing_gpu'][m]['setup_s']:.3f}"] for m in names]))
p.append('Temporary peak는 40 unique timing cases에서 관찰한 추가 CUDA allocated memory다. 전체 가능한 inputs에 대한 bound가 아니다. Bank/cache는 두 banks 및 noise plans를 포함하며 Python objects, CUDA context/allocator reserve는 포함하지 않는다. Compact payload는 full factual weights나 banks를 없앤 memory saving을 뜻하지 않는다. Physical sensing/reset costs는 이 compute 시간에 포함되지 않는다.')
p.append(table(['Method','CPU/GPU action mismatches / 40','GPU prediction max error','CPU prediction max error'],[[m,len(s['timing_cpu'][m]['action_mismatches']),f"{s['timing_gpu'][m]['max_prediction_error']:.3e}",f"{s['timing_cpu'][m]['max_prediction_error']:.3e}"] for m in names]))
p.append('![Decision loss, compute, and Top-1 stability](results/tradeoff.png)')
p.append('Figure는 point estimates다. Uncertainty는 위 paired CI 표에서 확인한다. Ranking column의 deterministic은 같은 bank에서 acquisition randomness가 없는 policy, not audited는 이번 seed-repeat audit에 포함하지 않은 method를 뜻한다.')
p.append(Path('REPORT_DISCUSSION_KO.md').read_text())
p.append('''## 8. 실행·재현·무결성

서버 경로: `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/adaptive_acquisition_v17`. Allocation2680491, node56의 idle GPU를 확인한 뒤 guard 아래 실행했다. 기존 사용자 jobs나 allocation은 변경하지 않았다. Original GPU worker와 optimized-equivalence worker를 순차 실행했다. 각 GPU worker 종료 후 해당 CPU timing을 측정하고 statistics를 완료한 뒤 결과를 seal했다.

핵심 파일은 `PROTOCOL.md`, `adaptive.py`, `adaptive_fast.py`, `run_policy.py`, `runtime_fast.py`다. Policies와 expansion histories는 `results/policies`, independent-seed/bank/reference audits는 `results/audits`, actual predictions는 `results/evaluations`에 있다. `results/summary.json`, `results/world_losses.npz`, `results/decision_actions.npz`에서 결과를 재분석할 수 있다.

`validate.py`는 aliasing/higher-order counterexamples, label mass, numerical ties와 optimized LMMSE equivalence를 점검했다. `seal.py`는 policy sources와 frozen files, actual losses/probabilities, stopping reason, negative controls, GPU/CPU output equivalence, V16 sealed files 보존 여부를 검증한다. `INTEGRITY.json`과 `MANIFEST.json`에 검증 내역과 SHA256 hashes를 저장했다.

재실행은 sealed directory를 덮어쓰지 않고 새 directory에서 수행한다. Allocated idle GPU에서 `guard.py run.py`, GPU 완료 후 `timing.py --cpu`, 이어서 idle GPU에서 `guard.py run_fast.py`, GPU 완료 후 `timing_fast.py --cpu`, `analyze.py`, `diagnostics.py`, `plot_results.py`, `write_report.py`, `seal.py` 순서다. Seal 전에 main/fast worker exit records를 각각 gpu_worker_exit.json과 fast_worker_exit.json에 보존한다. CPU scripts는 `BELIEF_DEVICE=cpu`와 빈 `CUDA_VISIBLE_DEVICES`를 지정한다. Report 생성에는 `REPORT_CONCLUSION_KO.md`, `REPORT_METHODS_KO.md`, `REPORT_DISCUSSION_KO.md`가 필요하다.
''')
Path('REPORT_KO.md').write_text('\n\n'.join(p));print('Wrote REPORT_KO.md',flush=True)
