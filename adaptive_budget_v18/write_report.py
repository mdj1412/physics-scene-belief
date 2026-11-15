from common import *
from run_policy import CELLS,ETASKS,ADAPT,MIX,NEST
from analyze import METHODS
s=read(R/'summary.json')
def f(x):return f'{x:.6f}'
def signed(x):return f'{x:+.3e}' if 0<abs(x)<1e-7 else f'{x:+.6f}'
def ci(x):return signed(x['mean'])+' ['+signed(x['low'])+', '+signed(x['high'])+']'
def pct(x):return f'{100*x:.2f}%'
def table(h,rows):return '\n'.join(['| '+' | '.join(h)+' |','|'+'|'.join(['---']*len(h))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
p=['# Adaptive allocation of compression and outer integration budgets — v18\n\n작성일: 2026-11-13 KST. Server implementation, fresh-world experiments, independent audit와 GPU/CPU online timing을 수행했다.']
p.append(Path('REPORT_CONCLUSION_KO.md').read_text());p.append(Path('REPORT_METHODS_KO.md').read_text())
p.append('## 4. Decision loss\n\nBrier loss이며 낮을수록 좋다. 각 cell은 같은 512 worlds의 paired conditions이고 world당16 repeats를 먼저 평균했다.')
for c,_,_ in CELLS:p.append('### '+c+'\n\n'+table(['Method']+ETASKS,[[m]+[f(s['means'][c][t][m]) for t in ETASKS] for m in METHODS]))
p.append('### Primary contrasts\n\nAdaptive minus baseline이다. Negative difference는 adaptive가 유리하다. CI는12 comparisons에 대한 adjusted99.58333%다. Numerical precision에 가까운 차이를 실질적인 효과로 해석하지 않는다.')
p.append(table(['Task / comparison','Difference [adjusted CI]'],[[k,ci(v)] for k,v in s['primary'].items()]))
p.append('### 사전 screens')
p.append(table(['Task / adaptive-to-baseline','Mean-loss ratio [95% CI]'],[[t+'/'+m,f"{v['mean']:.4f} [{v['low']:.4f}, {v['high']:.4f}]"] for t,vals in s['ratios'].items() for m,v in vals.items()]))
p.append('Screen 결과: `'+str(s['screens'])+'`. Superiority 미검출을 equivalence로 해석하지 않는다.')
p.append('### Ablations와 저비용 baselines: secondary95% CI')
p.append(table(['Task / adaptive minus baseline','Difference [95% CI]'],[[t+'/'+m,ci(s['secondary']['matched_nominal/'+t+'/adaptive-'+m])] for t in ETASKS for m in ['component_only','outer_only','fixed32_512','fixed32_4096','native','qmc64_native','nested512','nested4096']]))
p.append('## 5. Ranking 안정성과 budget allocation')
p.append('### Corrected independent acquisition audit: 512 matched worlds\n\nPrimary master91531 대 disjoint master101531의 Top-1 agreement다. Seed2 loss delta는 같은 primary actual decoder에서 seed2 selection minus primary selection이다.')
p.append(table(['Method','Contact agreement','Stick agreement','Contact loss delta [95% CI]','Stick loss delta [95% CI]'],[[m]+[pct(x) for x in s['ranking']['matched_nominal'][m]['seed2_agreement']]+[ci(x) for x in s['ranking']['matched_nominal'][m]['seed2_loss_delta']] for m in MIX+list(NEST)]))
p.append('Component-only 대비 agreement 증가량: '+str({t:{k:round(100*v,3) for k,v in x.items()} for t,x in s['repeatability_improvement'].items()})+' percentage points. CI는 secondary95%다.')
p.append('### Independent nested8192 reference: 첫64 worlds/cell\n\nReference agreement와 reference risk regret는 finite-bank numerical diagnostics이며 exact Bayes regret가 아니다.')
p.append(table(['Cell / method','Contact agreement','Stick agreement','Contact reference regret','Stick reference regret'],[[c+'/'+m]+[pct(x) for x in s['ranking'][c][m]['reference_agreement']]+[f(x) for x in s['ranking'][c][m]['reference_risk_regret']] for c,_,_ in CELLS for m in ['adaptive','balanced','component_only','outer_only','fixed128_4096','global128_4096','native','nested2048','nested4096']]))
p.append('### Final budgets와 unresolved cases\n\nK counts는 [16,32,64,128], M counts는 [512,1024,2048,4096] 순서다. Reasons는 [gap-separated,cap-unresolved,risk-tie,fixed]다.')
p.append(table(['Cell / method / task','K counts','M counts','Reasons','Mean stages','Mean K/M increases','Mean component–observation evaluations'],[[c+'/'+m+'/'+t,list(x['K_counts'].values()),list(x['M_counts'].values()),x['reason_counts'],f"{x['mean_stages']:.2f}",f"{x['mean_K_steps']:.2f}/{x['mean_M_steps']:.2f}",f"{x['mean_work']:.0f}"] for c,_,_ in CELLS for m in ADAPT for t,x in s['budgets'][c][m].items()]))
p.append(table(['Cell / task','Adaptive unresolved: compression-dominant','Adaptive unresolved: outer-dominant'],[[c+'/'+t,x['cap_compression_dominant'],x['cap_outer_dominant']] for c,_,_ in CELLS for t,x in s['budgets'][c]['adaptive'].items()]))
p.append('![Final K and M budgets](results/budgets.png)')
p.append('### Error proxies와 observed discrepancy')
rows=[]
for c,_,_ in CELLS:
    for t,d in s['calibration'][c].items():
        for key in ['compression','outer_perturbation']:
            x=d[key]
            if x:rows.append([c+'/'+t+'/'+key,x['n'],f(x['median_proxy']),f(x['median_observed_discrepancy']),pct(x['observed_exceeds_proxy_fraction']),d['ambiguous_pair_count']])
p.append(table(['Audit','n','Median proxy','Median observed discrepancy','Observed exceeds proxy','Ambiguous pair exclusions'],rows))
p.append('Compression discrepancy는 같은 outer samples에서 compact와 full-inner risk gap의 차이다. Outer discrepancy는 같은 compact representatives에서 independent outer scrambles로 계산한 두 gap estimates의 차이다. 이 둘은 true error의 동일한 정의가 아니며, 특히 outer discrepancy의 exceedance rate를 confidence-interval miscoverage로 부를 수 없다. 4-scramble uncertainty, shared compression bias, adaptive selection과 finite-bank error가 남는다.')
p.append('## 6. Online compute와 memory\n\nRTX PRO6000 Blackwell Server Edition, CPU2 threads, float64, batch1. 고정된20 matched worlds×2tasks×3repeats=120 measurements/method다. 같은 pass에서 strong baselines를 측정했다. 전체 bank/static plans의 construction은 setup으로 별도 표시하고, factual fits·public targets·representative/outer sample realization·cache allocation·all refinements·actual update는 online 시간에 포함했다. GPU와 CPU timing은 순차 실행했다.')
names=list(s['timing_gpu'])
p.append(table(['Method','GPU median ms','GPU p90 ms','CPU median ms','CPU contact ms','CPU stick ms','GPU contact ms','GPU stick ms'],[[m,f"{1000*s['timing_gpu'][m]['median_s']:.3f}",f"{1000*s['timing_gpu'][m]['p90_s']:.3f}",f"{1000*s['timing_cpu'][m]['median_s']:.3f}"]+[f"{1000*s['timing_cpu'][m]['task_medians_s'][t]:.3f}" for t in ETASKS]+[f"{1000*s['timing_gpu'][m]['task_medians_s'][t]:.3f}" for t in ETASKS] for m in names]))
p.append(table(['Method','Original GPU ms','Optimized GPU ms','Static bank/plan bytes','Acquisition cache max bytes','Observed CUDA temporary peak bytes','Warm setup ms'],[[m,f"{1000*s['timing_original_gpu'][m]['median_s']:.3f}",f"{1000*s['timing_gpu'][m]['median_s']:.3f}",s['timing_gpu'][m]['bank_plan_bytes'],s['timing_gpu'][m]['acquisition_cache_max_bytes'],s['timing_gpu'][m]['temporary_peak_bytes'],f"{1000*s['timing_gpu'][m]['setup_s']:.3f}"] for m in names]))
p.append('Acquisition cache에는 representatives와 hypothetical samples 및 prefix log sums를 포함한다. Static full banks는 여전히 필요하다. Temporary peak는40 unique timing cases에서 관찰한 추가 CUDA allocated bytes이며 global memory bound가 아니다. CUDA context/allocator reserve, Python objects와 실제 sensing/reset 비용은 별도다. Hardware와 workload가 다른 곳의 latency를 보장하지 않는다.')
p.append(table(['Method','CPU/GPU action mismatches /40','GPU prediction max error','CPU prediction max error'],[[m,len(s['timing_cpu'][m]['action_mismatches']),f"{s['timing_gpu'][m]['max_prediction_error']:.3e}",f"{s['timing_cpu'][m]['max_prediction_error']:.3e}"] for m in names]))
p.append('![Loss, compute and Top-1 agreement](results/tradeoff.png)\n\nFigure는 point estimates이며 uncertainty는 paired CI 표를 따른다. Deterministic methods의 acquisition-seed repeatability를 stochastic audit의 근거처럼 사용하지 않는다.')
p.append(Path('REPORT_DISCUSSION_KO.md').read_text())
p.append('''## 8. 실행과 재현 자료

Server directory: `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/adaptive_budget_v18`. Allocation2680491, node56의 idle GPU를 확인하고 guard 아래 실행했다. 기존 사용자 jobs/allocation은 변경하지 않았다.

`budget.py`, `budget_fast.py`, `runtime_fast.py`가 핵심 구현이고 `PROTOCOL.md`, `IMPLEMENTATION_AUDIT.md`, `AUDIT_CORRECTION.md`에 사전 설계와 implementation/audit 수정의 이유를 기록했다. `results/policies`에는 primary choices, budgets, histories, same-sample full-inner risks가 있다. `results/independent_audit.npz`가 corrected independent audit이며 원래 audit는 `results/audits`에 보존했다. `results/evaluations`, `results/world_losses.npz`, `results/summary.json`에서 actual outcomes와 통계를 재분석할 수 있다.

Source/policy freeze, cached-vs-dense validation, optimized-kernel equivalence, online prediction checks와 old V17 file hashes를 검증했다. `INTEGRITY.json`과 `MANIFEST.json`에 검증 결과 및 SHA256을 남겼다. 모든 simulation/inference/statistics는 allocated compute node에서 실행했고 local mirror는 검토와 전달에 사용했다.

재실행은 새 directory에서 allocated idle GPU로 `guard.py run.py`를 실행한다. Main worker exit를 gpu_worker_exit.json에 보존한 뒤 `guard.py run_fast.py`로 equivalence, corrected independent audit, optimized timing을 수행하고 fast_worker_exit.json을 보존한다. 이후 CPU에서 `timing_fast.py --cpu`, `analyze.py`, `plot_results.py`, `write_report.py`, `seal.py` 순서로 실행한다. CPU scripts는 BELIEF_DEVICE=cpu 및 빈 CUDA_VISIBLE_DEVICES를 사용한다. Sealed results를 덮어쓰지 않는다.
''')
Path('REPORT_KO.md').write_text('\n\n'.join(p));print('Wrote REPORT_KO.md')
