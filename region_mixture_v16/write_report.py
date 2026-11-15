from pathlib import Path
from common import *
s=read(R/'summary.json');v=read(R/'validation.json');g=read(R/'counterexample_global.json')
tasks=['smooth_velocity','contact_velocity','contact_event','slip_velocity','slip_event'];cells=['matched_nominal','shifted_nominal','shifted_correct']
def f(x):return f'{x:.6f}'
def ci(x):return f"{x['mean']:+.6f} [{x['low']:+.6f}, {x['high']:+.6f}]"
def table(h,rows):return '\n'.join(['| '+' | '.join(h)+' |','|'+'|'.join(['---']*len(h))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
p=['# Decision-region conditional predictive mixtures — v16\n\n작성일: 2026-11-13 KST. 서버에서 implementation validation, fresh-world inference/evaluation, GPU·CPU timing과 statistical analysis를 수행했다.']
p.append(Path('REPORT_CONCLUSION_KO.md').read_text())
p.append(r'''## 2. 구현: Pair distance를 mixture Bayes risk로 교체

V15 pair score는 query predictions가 다른 개별 hypotheses 사이의 Gaussian separation을 평균했다. V16은 각 region 안의 observation likelihood를 먼저 합친 뒤 posterior prediction과 risk를 계산한다. Binary event region r의 posterior mass를 πᵣ, conditional representatives를 θᵣⱼ라고 하면 다음과 같다.

\[
\hat f_r(y\mid a)=\frac1{K_r}\sum_{j=1}^{K_r}\mathcal N(y;h_a(\theta_{rj}),R_a),\quad
\hat p(r\mid y,a)=\frac{\pi_r\hat f_r(y\mid a)}{\sum_s\pi_s\hat f_s(y\mid a)}.
\]

Binary Brier loss의 acquisition은 다음 risk를 최소화한다.

\[
\hat R(a)=\mathbb E_{y\sim\sum_r\pi_r\hat f_r}
[\hat p(1\mid y,a)(1-\hat p(1\mid y,a))].
\]

두 region의 conditional mixtures가 동일하면 posterior probability가 바뀌지 않아 utility가 0이다. 개별 components가 멀리 떨어져 있다는 이유만으로 information이 있다고 판단하지 않는다. 이는 새로운 decision-theoretic objective가 아니라 기존 expected Bayes risk를 compact mixture에서 근사하는 구현이다.

`region32`는 QMC16384 factual posterior에서 각 nonempty region별 K=32를 deterministic systematic resampling한다. 각 representative의 weight는 πᵣ/K이며 region mass를 유지한다. Sensor covariance는 원래 Gaussian noise 그대로다. Region을 단일 Gaussian으로 moment matching하지 않는다. Posterior mass≤1e−10인 region만 제외하고 renormalize한다.

Binary tasks는 contact/stick label0과1의 두 regions다. Continuous velocity tasks는 첫 velocity coordinate의 posterior weighted quartiles로 4개 strata를 정의하고 full 2D target을 유지한다. 이는 prediction-space stratification이며 exact continuous decision-equivalence classes라고 부르지 않는다. K=32는 전체32 components가 아니다. Binary에서는 최대 64, velocity에서는 최대 128이다.

Hypothetical observations는 component마다4 Gaussian noise samples를 scrambled Sobol plan으로 생성한다. 같은 randomness를 actions 사이에 공유한다. 이렇게 얻은 compact posterior variance를 평균하는 coherent estimator다. 같은 compact model에서 generation과 inference를 하므로 compression이 만든 optimism은 남을 수 있다. 별도의 cross-fitted variant로 이를 점검한다.

**이 방법은 bank-free compact belief가 아니다.** QMC factual posterior를 계산한 다음 acquisition representation을 줄인다. Primary actual-observation decoder도 동일 QMC16을 쓴다. 별도로 decoder까지 mixture로 바꾼 `region32_compact`를 평가했다. Timing에는 factual fit과 compression을 포함한다.

| Method | 비교 목적 |
|---|---|
| `region32` | Primary conditional-mixture acquisition, QMC16 actual decoder |
| `global32` | 같은 total components의 global posterior resampling, region별 mass 보존을 하지 않음 |
| `region64` | Region당64 components, compression resolution 확인 |
| `region_cf32` | Independent QMC outer bank에서도 conditional mixture를 만들고 inner decoder의 outer-target loss를 계산 |
| `region32_compact` | region32와 같은 action, actual decoder도 compact mixture |
| `native` / `qmc64_native` | Optimized LMMSE16/64 acquisition, same posterior-mean prediction |
| `nested128` / `nested512` | Optimized independent-inner/outer QMC16 expected-loss baseline |
| `pair` | V15 pair-disagreement score |
| `query`, `factual`, `probe`, `none` | Fixed-action와 no-observation diagnostics |

## 3. Protocol과 physical setting

Primary는 **새로운 512 iid worlds**다. Seed5100101+137×index, actual observation noise16 repeats, nominal/shifted paired conditions이며 V15의 evaluated worlds를 primary로 재사용하지 않았다. 같은 law families와 model priors의 new-world replication이므로 new-law-family generalization이라고 부르지 않는다. Pilot8 seed5000101은 correctness용이다.

V15의 expanded prior를 그대로 유지한다. Geometry two families × frictionless/Coulomb four components는 각각1/8, flexible-law two components는 각각1/4다. Boundary x(y)=d+s y+a f_g(y), d∼U[−4,4]mm, s∼U[−.4,.4], a∼U[−.1,.1], e∼U[.35,.9], b₀∼U[−4,2], b₁∼U[−2,2]다. L=25mm, f₀=L(y/L)²/2, f₁=L[(y/L)²/2−.35(y/L)⁴].

Nominal flexible tangential impulse는 vₜ sigmoid[b₀+b₁ log((|vₜ|+.02)/.5)]다. Shifted flexible law는 vₜ sigmoid(b₀)[.2+.8/(1+(|vₜ|/s₀)²)], s₀=.1+.7 sigmoid(b₁)다. Base laws는 변화시키지 않은 negative controls다. Cells는 matched/nominal, shifted/nominal, shifted/correct-law다. Correct-law는 privileged diagnostic이다.

Factual contact y=0, incoming(1,.35); factual profiles9개(−10…10mm), profile noise1.2mm, outgoing velocity noise.07m/s다. Query y=25mm이며 contact gate threshold는 public factual profiles의 straight-line extrapolation이다. No-contact는 incoming velocity(1,.9)를 유지한다. Slip query는 public slope estimate로 median restitution의 Coulomb transition을 겨냥하고, residual tangential velocity absolute≤1e−8을 exact stick label로 정의한다.

Actions는 query_view(19…31mm profiles7개, noise1mm), factual_view(−8…8mm profiles7개, noise.35mm), probe(y=0, incoming(1,−.7), velocity noise.05m/s)다. Selectors는 3개 actions 중 1개를 선택하며 sensing cost는 같다고 가정한다. No-observation은 별도의 diagnostic이며 early-stop policy를 학습하거나 평가하지 않았다. Event loss는 Brier, velocity loss는 half squared Euclidean error다. Event target에 √2 scaling을 적용해 common loss 계산과 일치시켰다.

이 simulator는 finite-reach gate와 instantaneous local impulse model이다. Full trajectory collision detection, real RGB geometry, real robot intervention을 검증한 것은 아니다. Query context는 public evidence에서만 정하고 hidden truths로 favorable cases를 고르지 않았다.

Primary 12 contrasts는 matched contact/stick events에서 region32 대 native/nested512/global32/pair/query/probe다. Independent world 단위 30,000 paired bootstrap, Bonferroni99.58333% CI다. 16 repeats는 world별로 평균하고 독립 표본으로 세지 않는다. 나머지는 secondary95% CI다.

별도의 approximation screen은 두 event tasks 모두에서 region32/nested512 mean-loss ratio의95% upper≤1.05, GPU online median speedup≥2를 요구했다. 5%는 사전 engineering margin이며 application-derived tolerance가 아니다. CI가0을 포함하는 것과 equivalence를 혼동하지 않는다. 독립 method contribution에는 native와 equal-budget global 대비 개선, fixed-rule 대비 유효한 trade-off가 추가로 필요하다.
''')
p.append('## 4. Counterexamples: 무엇을 고쳤는가')
p.append(table(['Example','No-observation risk','Region32 informative risk','Quadrature informative reference','Region32 uninformative risk','Global informative / uninformative'],[[k,f(x['region_risks'][0]),f(x['region_risks'][1]),f(x['exact_risk_good']),f(x['region_risks'][2]),f(g[k]['global_risks'][1])+' / '+f(g[k]['global_risks'][2])] for k,x in v.items() if isinstance(x,dict)]))
p.append('V15 aliasing example은 labels [0,0,1,1], equal weights, noiseSD1에서 uninformative means[−4,+4,−4,+4], informative means[−.5,−.5,+.5,+.5]를 사용했다. Old pair는 uninformative action에0.249916의 positive score를 주었다. Region mixture는 그 action에서 risk를0.25로 유지했다. Higher-order example은 class0의±5, class1의−10/0/+10 mixtures가 모두 mean0, variance25지만 모양이 다르다. Conditional Gaussian moment reduction과 LMMSE covariance는 이 information을 놓치지만 mixture는 보존했다. Reference는 Gauss–Hermite100 quadrature로서 모든 경우의 closed-form exact value는 아니다. **Equal-budget global mixture도 두 예제를 통과했다. 따라서 이 mechanism 개선을 region stratification만의 novelty로 주장하지 않는다.**')
p.append('## 5. Decision loss')
methods=['region32','global32','region64','region_cf32','region32_compact','native','nested128','nested512','pair','query','factual','probe','none','qmc64_native']
for cell in cells:p.append('### '+cell+'\n\n'+table(['Method']+tasks,[[m]+[f(s['means'][cell][task][m]) for task in tasks] for m in methods if m in s['means'][cell][tasks[0]]]))
p.append('### Primary contrasts\n\nNegative difference는 region32가 유리하다는 뜻이다.')
p.append(table(['Task / comparison','Difference [adjusted CI]','Relative mean change'],[[k,ci(x['delta']),f"{100*x['relative_mean_change']:+.2f}%"] for k,x in s['primary'].items()]))
audit=read(R/'statistical_audit.json')
p.append('### Statistical numerical audit와 비슷한 compute 비용의 baselines')
p.append('Primary stick-event 대 LMMSE의 adjusted CI upper가 0에 가까워, policy나 primary criterion을 바꾸지 않고 bootstrap Monte Carlo error를 추가 점검했다. Independent seeds 5개에서 각각 100,000 bootstrap resamples를 사용한 upper bounds는 모두 음수였다. 이는 같은 worlds에서의 numerical audit이며 새 worlds 또는 QMC banks에 대한 replication이 아니다. Primary 30,000-resample CI는 그대로 유지한다.')
p.append(table(['Bootstrap seed','Adjusted lower','Adjusted upper'],[[x['seed'],f"{x['low']:+.8f}",f"{x['high']:+.8f}"] for x in audit['primary_stick_native_bootstrap_MC_audit']]))
p.append('다음 비교는 post-analysis secondary 95% CI다. 두 baseline 모두 region32와 비슷한 GPU compute 범위에 있다.')
p.append(table(['Task / region32 minus baseline','Difference [95% CI]'],[[k,ci(x)] for k,x in audit['secondary_near_budget_comparisons'].items()]))
p.append('### Predeclared approximation screen')
p.append(table(['Task','Mean-loss ratio [95% CI]'],[[k,f"{x['mean']:.4f} [{x['low']:.4f}, {x['high']:.4f}]"] for k,x in s['noninferiority'].items()]))
p.append('Screen result: `'+str(s['approximation_screen'])+'`. Ratio criterion과 speed criterion을 분리해서 해석한다.')
p.append('### Observation choices')
p.append(table(['Cell / method','Contact [none,query,factual,probe]','Stick [none,query,factual,probe]'],[[c+'/'+m,s['actions'][c]['contact_event'][m],s['actions'][c]['slip_event'][m]] for c in cells for m in ['region32','global32','native','nested512']]))
p.append('## 6. Compression, cross-fitting과 ranking sensitivity')
p.append(table(['Cell / task','Region64 − region32 [95% CI]','Cross-fitted region32 − primary [95% CI]','Compact decoder − full decoder [95% CI]'],[[c+'/'+t]+[ci(s['secondary'][f'{c}/{t}/{a}-region32']) for a in ['region64','region_cf32','region32_compact']] for c in cells for t in ['contact_event','slip_event']]))
for c in cells:p.append(table([c,'Contact agreement / loss change','Stick agreement / loss change'],[[name]+[f"{100*x['agreement'][k]:.2f}% / {x['loss_change'][k]:+.6f}" for k in [2,4]] for name,x in s['audit'][c].items() if name!='reference_regret']))
p.append('각 cell의 첫 32 worlds의 Q8과 alternate Sobol seed는 main region32와, nested2048은 nested512와 비교했다. Main policy는 audit로 변경하지 않았다. Independent-seed nested2048의 내부 risk regret도 summary에 기록했으나 actual Bayes regret로 부르지 않는다. Fresh worlds의 bootstrap CI는 fixed QMC bank의 모든 numerical uncertainty를 포함하지 않는다.')
p.append(table(['Cell','Factual ESS median','Selected ESS medians (five tasks)','Max components','Max omitted mass'],[[c,f(s['diagnostics'][c]['factual_ess_median']),[round(x,2) for x in s['diagnostics'][c]['selected_ess_medians']],s['diagnostics'][c]['components_max'],s['diagnostics'][c]['omitted_mass_max']] for c in cells]))
p.append('## 7. Online compute와 memory\n\nRTX PRO6000 Blackwell Server Edition, CPU 2 threads, float64, batch1. 고정된 20 fresh matched worlds × 5 tasks × 3 repeats로 method당 300 measurements를 측정했다. Factual fit, target generation, compression/selection, selected actual observation update를 포함했다. GPU online choices/predictions가 stored evaluation과 일치하는지 확인했다. CPU는 아래 numerical audit에 별도로 기록했다. GPU/CPU timing은 순차 실행했다. CPU와 GPU 숫자를 섞어 speedup을 만들지 않았다.')
names=['region32','global32','region64','region_cf32','region32_compact','native','nested128','nested512','qmc64_native','pair','query','probe']
p.append(table(['Method','GPU median ms','CPU median ms','GPU contact-event ms','GPU stick-event ms'],[[m,f"{s['timing_gpu'][m]['median_s']*1000:.3f}",f"{s['timing_cpu'][m]['median_s']*1000:.3f}"]+[f"{s['timing_gpu'][m]['task_medians_s'][t]*1000:.3f}" for t in ['contact_event','slip_event']] for m in names]))
p.append(table(['Method','Bank/cache bytes','Compact payload max bytes','Observed GPU temporary peak bytes','Warm setup ms'],[[m,s['timing_gpu'][m]['bank_cache_bytes'],s['timing_gpu'][m]['compact_payload_max_bytes'],s['timing_gpu'][m]['temporary_peak_bytes'],f"{1000*s['timing_gpu'][m]['setup_s']:.3f}"] for m in names]))
p.append('CPU timing의 원래 strict cross-device check는 region64의 한 action 차이에서 중단됐다. 원본 source와 failure log를 보존하고, 별도의 timing_cpu_audit.py에서 CPU가 실제 선택한 action의 full decoder prediction을 검증하며 timing을 다시 측정했다. 아래는 100 unique cases 중 GPU와 달랐던 action 수다. CPU 결과로 GPU decision-loss 결과를 교체하지 않았다.')
p.append(table(['Method','CPU/GPU action mismatch count / 100','CPU prediction max error'],[[m,len(s['timing_cpu'][m]['cross_device_action_mismatches']),f"{s['timing_cpu'][m]['max_prediction_error']:.3e}"] for m in names]))
p.append('Compact payload는 representative weights, targets, whitened action means다. Full factual weight array와 bank storage를 없앤 값이 아니다. Cache에는 bank/features와 precomputed noise plan을 포함한다. Temporary peak는100 fixed cases에서 관찰한 추가 CUDA allocated bytes로 모든 possible input의 global bound가 아니다. Python objects, CUDA context, allocator reserve, physical sensing/reset cost는 별도다. Setup은 warm process construction이다.')
p.append('![Matched loss and online compute](results/tradeoff.png)')
p.append(Path('REPORT_DISCUSSION_KO.md').read_text())
p.append('''## 9. 재현 자료와 실행 상태

Server directory: `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/region_mixture_v16`. Allocation2680491, node56의 idle GPU를 확인하고 guard 아래 실행했다. 기존 jobs/allocation을 변경하지 않았다. 모든 simulation/inference/statistics는 allocated compute node에서 수행했다.

`PROTOCOL.md`, `mixture.py`, `baselines.py`, `runtime.py`가 핵심 설계와 구현이다. `validate.py`, `counterexample_global.py`는 mechanism 및 baseline-specificity checks다. `run_policy.py`는 public evidence만 읽고 policies를 freeze한 뒤 `evaluate.py`가 private outcomes를 평가한다. `timing.py`, `analyze.py`가 online costs와 paired statistics를 계산한다.

Raw predictions/actions, numerical audit, timing은 `results/evaluations`, `results/policies`, `results/audits`, `results/world_losses.npz`, `results/summary.json`, `results/timing_gpu.json`, `results/timing_cpu.json`에 있다. Source/policy hashes, worker exit records와 phase markers를 남겼다. `INTEGRITY.json`, `MANIFEST.json`은 final verification과 SHA256 manifest다. V15 sealed artifacts의 hash를 대조했다.

재실행은 sealed directory를 덮어쓰지 않고 새 directory에 code를 복사한다. Allocated idle GPU에서 `guard.py run.py`, 이어서 CPU timing `timing_cpu_audit.py`, `analyze.py`, `counterexample_global.py`, `statistical_audit.py`, `write_report.py`, `seal.py` 순서다. CPU-only scripts는 `BELIEF_DEVICE=cpu`로 실행한다.
''')
Path('REPORT_KO.md').write_text('\n\n'.join(p));print('Wrote REPORT_KO.md')
