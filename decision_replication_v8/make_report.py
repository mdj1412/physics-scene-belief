import json,hashlib
from pathlib import Path
import numpy as np
P=Path('.').resolve();A=P/'results/analysis';s=json.loads((A/'summary.json').read_text())
methods=['bridge','standard4096','standard_fixed8192','crossfit_fixed8192','standard_fixed16384','crossfit_fixed16384','standard_budget','crossfit_budget']
labels={'bridge':'Tempered bridge','standard4096':'Standard SNIS4096','standard_fixed8192':'Standard SNIS8192','crossfit_fixed8192':'Cross-fitted SNIS8192','standard_fixed16384':'Standard SNIS16384','crossfit_fixed16384':'Cross-fitted SNIS16384','standard_budget':'Standard SNIS budget','crossfit_budget':'Cross-fitted SNIS budget'}
primary=['standard4096','crossfit_fixed16384','standard_budget','crossfit_budget']
def fmt(x):return f'{x:.8f}'
def ci(x):return '['+', '.join(fmt(v) for v in x)+']'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(str,r))+' |' for r in rows]])
def decision(m):
 r=s['contrasts'][m];lo,hi=r['ci_bonferroni']
 return 'Bridge 우세 확인' if hi<0 else 'Bridge 열세 확인' if lo>0 else '차이 미확정'
winners=[labels[m] for m in primary if s['contrasts'][m]['ci_bonferroni'][1]<0]
budget_win=all(s['contrasts'][m]['ci_bonferroni'][1]<0 for m in ['standard_budget','crossfit_budget'])
small_win=s['contrasts']['standard4096']['ci_bonferroni'][1]<0
if budget_win:headline='Bridge의 이득은 independent corner replication과 compute-budget controls에서도 유지됐다. 다만 이 결과는 synthetic corner condition에 한정되며 새로운 SMC algorithm이나 real-world generalization의 증거는 아니다.'
elif small_win:headline='작은 factual bank 대비 이득은 independent worlds에서 재현됐지만, compute-budget controls까지 통과하는 일반적인 Bridge 우세는 확인되지 않았다. 핵심 기여 후보는 universal algorithmic superiority보다 finite-bank failure를 구분하는 diagnostic에 가깝다.'
else:headline='v7의 exploratory corner signal이 이번 independent replication과 강한 controls를 모두 통과했다고 말할 수 없다. Numerical posterior 개선만으로 actual decision-loss 개선을 주장하는 방향은 현재 증거로 부족하다.'
budget_rows=[]
for k,label in [('bridge_budget','Bridge time (s)'),('snis_budget_seconds','SNIS time (s)'),('snis_budget_n','SNIS particles'),('budget_utilization','Budget utilization')]:
 vals=[s['budget'][k][a] for a in ['min','median','max']]
 cells=[f'{v*100:.2f}%' if k=='budget_utilization' else f'{int(v):,}' if k=='snis_budget_n' else f'{v:.3f}' for v in vals]
 budget_rows.append([label,*cells])
text=f'''# Corner + precise view independent replication v8

작성일: 2026-11-11 KST. 설명은 한글, technical terms는 English로 작성했다.

## 1. 결과와 판단

**{headline}**

새로운 **64 corner worlds × 32 observation-noise repeats × 6 queries**에서 비교했다. 모든 selector는 같은 actual observations와 같은 full-SMC decoder를 사용했다. 표의 negative difference는 Bridge가 더 낮은 actual decision loss를 얻었다는 뜻이다.

{table(['Primary comparison: Bridge − baseline','Mean difference','World-cluster 95% CI','Familywise 98.75% CI','판정'],[[labels[m],fmt(s['contrasts'][m]['mean']),ci(s['contrasts'][m]['ci95']),ci(s['contrasts'][m]['ci_bonferroni']),decision(m)] for m in primary])}

Confidence interval이 0을 포함하면 equivalence나 모든 조건에서 효과가 없다는 뜻이 아니다. 이번 64-world sample에서 차이를 확정하지 못했다는 뜻이다. Four primary contrasts의 nominal familywise coverage는 percentile bootstrap intervals에 Bonferroni 방식을 적용해 조정했다. N64는 결과를 보기 전에 고정했으며 별도의 minimum useful effect나 equivalence margin을 사전 설정한 실험은 아니다. 나머지 tables와 mechanism diagnostics는 descriptive다.

![Actual decision loss와 paired differences](results/analysis/figures/decision_loss.png)

### 1.1 이번 결과에서 가장 중요한 해석

Standard SNIS4096 대비 Bridge의 MSE 감소는 약 **78.3%**로 재현됐다. 그러나 같은 16384 particles를 사용하는 SNIS에서 squared-shift를 cross-fitted utility로 바꾸면 MSE가 0.00082841에서 0.00045356으로 낮아졌다. Bridge MSE는 0.00045325다. 따라서 작은 Standard SNIS에 대한 큰 차이를 tempering만의 기여로 해석할 수 없다.

Cross-fitted SNIS16384의 평균 acquisition compute는 **{s['fixed_compute']['fixed16384']['mean_seconds']:.2f}s**, Bridge는 **{s['budget']['bridge_budget']['mean']:.2f}s**였다. Shared costs를 제외한 이 비교에서는 약 4분의 1 시간의 baseline에 대한 추가 Bridge 이득이 확인되지 않았다. Cross-fitted SNIS8192도 descriptive MSE가 0.00045335로 가까웠다. 이는 formal equivalence test의 통과라는 뜻은 아니다.

Standard SNIS budget 대비 Bridge의 mean MSE 감소는 약 **8.0%**이며 primary adjusted interval도 음수다. 다만 upper endpoint가 0에 가깝다. Bootstrap Monte Carlo precision을 확인한 supplementary 500000 draws의 adjusted interval은 {ci(s['bootstrap_sensitivity']['ci9875'])}로 같은 방향이었다. **Cross-fitted budget baseline에 대해서는 개선을 확인하지 못했다.** 주장의 핵심 경계는 bank 크기뿐 아니라 utility estimator다.

**내 판단은 현재 Bridge를 독립적인 핵심 방법론 기여로 추진하는 것을 보류하는 것이다.** Full-SMC reference에 대한 posterior mean accuracy는 개선됐지만, 훨씬 저렴한 cross-fitted baseline이 actual decision loss를 거의 같은 수준으로 낮춘다. 아래 nuisance control은 이 benchmark의 불필요한 dimension도 상당한 numerical difficulty를 만들었음을 보여준다.

## 2. v7에서 무엇을 재검증했는가

v7의 corner + precise view 16 worlds에서 Bridge MSE는 0.00037445, Standard SNIS4096은 0.00282173, SNIS16384는 0.00099234였다. 당시 family subgroup은 exploratory였고 전체 48 worlds에서는 추가 이득을 확인하지 못했다. 이번에는 그 16 worlds를 재사용하지 않고 같은 corner prior에서 64 worlds를 새로 생성했다. 유리한 contact geometry나 factual observation을 골라 넣지 않았다.

Physics는 2D rounded-rectangle contact이며 latent variables는 geometry offset, restitution, friction, rectangle half-height, corner rounding이다. Known geometry family, bounded uniform prior, Coulomb impulse와 instantaneous velocity prediction을 유지했다. View는 boundary/curvature extraction을 calibrate한 **Gaussian continuous proxy**이고 precise condition은 native covariance를 0.0625배로 줄인다. Raw RGB 또는 real camera에서 이 noise level을 달성했다는 뜻이 아니다.

v7과 완전히 같은 random draw의 재실행은 아니다. 이번에는 independent outer generator bank를 8192에서 16384로 늘렸고, 모든 world에서 H512를 고정했으며, Bridge의 initial factual bank를 별도의 SMC4096으로 만들었다. v7은 larger factual bank의 subset을 사용했다. 따라서 independent condition replication과 estimator/compute controls이지, 하나의 code bit만 바꾼 causal ablation은 아니다.

Latent ranges는 geometry offset ±8 mm, restitution 0.15–0.85, friction 0–0.6, half-height 22–34 mm, rounding 0–6 mm다. Actions는 view와 source-y intervention −22, +22, +9 mm이며, queries는 −30, −17, −7, +4, +15, +28 mm intervention 후 t=0.8s의 velocity다. Dynamic observations는 finite-frame displacement 기반 velocity이며 Gaussian standard deviation은 0.025 m/s다.

Factual evidence는 world당 한 번 생성한다. 반복 32회는 **추가 observation noise**에 대한 반복이며, factual evidence까지 32회 재생성한 실험이 아니다. Independent test worlds가 새로운 family 또는 out-of-distribution physical class라는 뜻도 아니다.

## 3. 비교 방법과 공정성

| Method | Factual bank와 utility | 목적 |
| --- | --- | --- |
| Tempered bridge | Initial SMC4096; H512 × 4 actions; conditional N1024; 12 MH moves/stage; 2 runs cross-product | Factual posterior에서 hypothetical observation으로 이동 |
| Standard SNIS4096 | 첫 independent A stream의 4096 particles; squared-shift utility | v7의 작은 bank signal replication |
| Standard/Cross-fitted SNIS8192 | Independent A/B 각4096; 같은 총 particles를 공유 | Utility estimator control |
| Standard/Cross-fitted SNIS16384 | Independent A/B 각8192; 각 stream은 독립 SMC populations의 equal-particle mixture | 강한 fixed-size baseline 및 estimator control |
| Standard/Cross-fitted SNIS budget | Bridge 시간 한도까지 bank 확장; 같은 최종 particles에 두 utilities 적용 | Compute budget control |

All methods는 동일한 independent posterior-predictive hypothetical observations와 동일한 current predictive mean을 사용한다. SNIS 자신의 bank에서 hypothetical observations를 뽑는 self-inclusion 효과를 피했다. Bridge의 두 conditional runs는 stochastic seeds가 다르지만 같은 finite factual initial bank를 공유한다. Cross-fitted SNIS는 독립 factual SMC streams로 구성했으므로 서로 다른 공유 approximation error가 남을 수 있다.

Actual evaluation decoder는 처음부터 joint factual + selected-observation likelihood를 사용하는 full SMC N2048, moves18이다. 모든 actions에 대해 동일한 32 noisy observations를 decode한 다음 frozen choices를 적용했다. 이를 통해 selector가 자신의 decoder를 써서 발생하는 우위를 분리했다. 모든 64 policies가 고정된 다음 actual decoding을 시작했으며 checksum을 검증했다.

### 3.1 Cross-fitting의 의미와 한계

Current predictive mean을 $m_0$, observation 이후 mean estimates를 $\\hat m_A,\\hat m_B$라 하면 다음 utilities를 쓴다.

$$U_{{standard}}=\\tfrac12\\|\\hat m_{{pooled}}-m_0\\|^2,$$

$$U_{{crossfit}}=\\tfrac12(\\hat m_A-m_0)^\\top(\\hat m_B-m_0).$$

Independent unbiased mean estimates라는 이상적인 조건에서는 cross-product가 squared estimation noise의 positive variance term을 제거한다. 실제 SNIS에는 self-normalization bias가 있고 factual SMC approximation도 finite하다. 또한 common finite $m_0$의 error는 남는다. Independent streams 내부 particles도 genealogy 때문에 i.i.d.가 아니다. 따라서 unbiased EVSI를 얻었다고 주장하지 않는다. Pooled SNIS mean의 normalizer와 두 stream normalizers는 서로 다르므로 standard utility에서 단순히 sample variance를 빼는 것과도 정확히 같지 않다. Negative cross-fitted utilities는 clip하지 않았으며, 모든 methods는 네 actions 중 반드시 하나를 선택한다.

### 3.2 Compute budget accounting

같은 GPU에서 GPU synchronization을 포함한 wall time을 측정했다. Method별 factual bank 생성, query forward simulation, action-feature 계산, importance accumulation, conditional summaries와 utility 계산을 포함한다. Shared outer generator, shared actual decoder, process startup, disk I/O는 비교 budget에서 제외한다. GPU FLOPs나 최적화된 모든 가능한 SNIS implementation에 대한 최적성 비교는 아니다.

Bridge에는 첫 A bank의 feature-cache 생성 비용도 부과했다. Bridge에는 모두 필요하지 않은 cache이므로 이 규칙은 SNIS의 허용 시간을 약간 넓히는 보수적인 선택이다. Standard/Cross-fitted SNIS budget은 같은 samples와 계산을 공유하며 둘의 utility 계산을 모두 포함한다.

{table(['항목','최소','중앙값','최대'],budget_rows)}

Budget utilization은 SNIS에 실제 포함된 checkpoint time / Bridge time이며 평균은 93.87%다. 95% 미만 worlds는 **{len(s['budget']['below95_worlds'])}/64**였다. 따라서 결과는 정확히 동일한 elapsed seconds가 아니라 **동일한 time ceiling 안의 비교**다. Completed checkpoint granularity와 runtime variation을 숨기지 않는다. 마지막 chunk가 budget을 넘으면 그 particles는 primary policy에 사용하지 않는다. 실제 profiling에서는 overshoot 계산도 수행했으며 전체 SNIS compute time 합은 {s['budget']['all_snis_compute_seconds']['sum']:.2f}s, primary checkpoint 합은 {s['budget']['snis_budget_seconds']['sum']:.2f}s다. Deadline에서 직전 완료 checkpoint를 사용하는 정책으로 해석해야 한다.

![Compute-budget audit](results/analysis/figures/compute_budget.png)

## 4. Actual decision-loss 결과

Loss는 counterfactual velocity의 두 components에 대한 squared error 평균, 즉 $\\|\\hat v-v\\|^2/2$다. Loss 단위는 m²/s²다. World 내부의 32 repeats와 6 queries를 평균하고 worlds 사이에서 평균했다. Selection action은 query마다 달라질 수 있다. Six query-specific choices는 각 query를 별도 decision episode로 보는 것이며 한 scene에서 여섯 observations를 동시에 취득하는 protocol은 아니다.

{table(['Method','Mean actual MSE'],[[labels[m],fmt(s['means'][m])] for m in methods])}

No-observation factual-mean baseline MSE는 {fmt(s['no_observation_mse'])}다. 이는 descriptive comparator이며 no-observation을 선택 가능한 fifth action으로 넣지는 않았다.

### 4.1 Utility estimator만 바꾸면 어떻게 되는가

다음은 같은 bank에서 Cross-fitted − Standard SNIS의 loss difference다. Negative면 cross-fitting이 유리하다. 이 비교들은 primary multiplicity family에 포함하지 않은 descriptive controls다.

{table(['같은 particle budget','Difference','95% CI','선택이 달라진 queries /384'],[[k,fmt(v['mean']),ci(v['ci95']),v['disagreement']] for k,v in s['estimator_effects'].items()])}

Bridge 대비 작은 Standard SNIS의 차이가 cross-fitting 또는 bank expansion만으로 줄어든다면, 그것은 Bridge만의 장점이라는 주장보다 estimator noise 또는 finite support 문제로 설명될 여지가 크다. 반대로 두 controls를 통과해도 tempering 자체만의 causal contribution은 별도의 matched-estimator / proposal ablation이 필요하다.

### 4.2 World heterogeneity와 noise 반복

{table(['Baseline','Bridge wins /64','Bridge losses /64','Leave-one-world-out mean difference 범위'],[[labels[m],s['contrasts'][m]['world_wins'],s['contrasts'][m]['world_losses'],ci(s['contrasts'][m]['leave_one_out_range'])] for m in primary])}

Strict wins/losses에는 numerical magnitude가 아주 작은 차이도 포함되므로 win count 자체를 significance로 해석하지 않는다. Observation repeats가 많아도 independent worlds 수는 64다.

{table(['Baseline, R32','Mean difference','World SEM','Conditional-repeat SEM'],[[labels[m],fmt(s['noise']['32'][m]['mean']),fmt(s['noise']['32'][m]['world_sem']),fmt(s['noise']['32'][m]['conditional_repeat_sem'])] for m in primary])}

Conditional-repeat SEM에는 observation noise뿐 아니라 해당 observations의 finite-SMC decoder variation도 섞여 있다. World SEM에는 world별 geometry/material/factual-evidence 차이와 남은 numerical variation이 포함된다. 두 항을 물리적 noise의 순수한 causal decomposition으로 해석하지 않는다.

![World heterogeneity](results/analysis/figures/world_heterogeneity.png)

오른쪽 panel은 작은 differences를 보기 위해 y-axis를 확대한 것으로, 두 panels의 y-axis scale은 다르다.

## 5. 남은 ranking ambiguity

H512 hypothetical observations를 bootstrap해 top-1 gap과 winner regret를 계산했다. Active query 기준은 method 자신의 maximum gross estimated gain >5e-5다. Practically stable은 선택된 action의 bootstrap 95% upper regret가 max(5e-5, maximum gross gain의 5%) 이하다. 이 기준은 실제 ground truth를 읽지 않는다. Bank와 conditional SMC realizations를 고정한 outer Monte Carlo diagnostic이므로 모든 posterior approximation uncertainty를 포함하지 않는다.

{table(['Method','Active queries','Decisive top-1','Practically stable','Half-sample agreement'],[[labels[m],s['ranking'][m]['active'],s['ranking'][m]['decisive'],s['ranking'][m]['practical'],f"{s['ranking'][m]['half_agreement']*100:.1f}%"] for m in methods])}

Bridge와 Cross-fitted SNIS16384의 전체 action disagreements는 {s['contrasts']['crossfit_fixed16384']['choice_disagreement']}/384지만, Bridge-active queries에서는 **{s['contrasts']['crossfit_fixed16384']['active_choice_disagreement']}/235**뿐이다. Inactive queries의 실제 loss contribution은 0이다. 따라서 raw top-1 disagreement를 그대로 decision failure로 세면 크게 과장된다. Decisive Bridge queries 134개에서는 두 cross-fitted baselines와 actual loss difference contribution이 정확히 0이었다.

Inactive 또는 near-tied query에서는 floating-point 차이로 action identity가 달라질 수 있다. 각 method의 active subset이 달라질 수 있으므로 위 비율을 identical evaluation subset의 accuracy로 읽으면 안 된다. Decisive와 practically stable은 별개의 기준이다. 아래 contribution partition에서는 decisive를 먼저 배정해 exclusive groups를 만든다.

{table(['Bridge ranking group','Queries','Contribution: vs Standard4k','Contribution: vs Cross-fitted budget'],[[k,v['queries'],fmt(v['contrasts']['standard4096']['mean']),fmt(v['contrasts']['crossfit_budget']['mean'])] for k,v in s['ambiguity_contributions'].items()])}

Contribution은 해당 group 외 queries를 0으로 두고 전체 64×6으로 평균한다. 따라서 행의 합은 전체 paired difference와 정확히 같다. 이는 group 내부 평균 효과와 다른 estimand이며, 소수 group을 별도 유의한 발견으로 제시하지 않는다. Group별 95% intervals는 `summary.json`에 저장했다.

![Ranking과 observation-noise convergence](results/analysis/figures/ranking_noise.png)

### 5.1 ESS collapse

{table(['Bank','Precise-view median ESS','ESS≤1.01 fraction'],[[k,f"{v['view']['median']:.3f}",f"{v['view']['collapse_fraction']*100:.1f}%"] for k,v in s['ess'].items()])}

ESS는 importance weights의 concentration을 측정한다. 작은 ESS가 곧 잘못된 action 또는 높은 actual loss라는 뜻은 아니다. 반대로 full SMC의 stage ESS가 높아도 posterior mode coverage나 mixing이 완전하다는 보장은 없다. 이번 판단의 primary endpoint는 실제 paired decision loss다.

## 6. Numerical reference와 decoder audit

미리 정한 worlds 0,1,2,3,8,9,10,11에서 full SMC N2048, moves18, two runs를 사용해 acquisition reference를 만들었다. 모든 methods는 동일한 첫 H128 hypothetical observations만 사용해 비교했다. Reference도 finite approximation이며 ground-truth acquisition oracle이 아니다. Conditional mean RMSE는 pooled posterior mean의 reference error이며 Standard/Cross-fitted SNIS는 같은 bank일 때 동일한 mean diagnostic을 공유한다.

{table(['Method','Reference active queries','Action agreement','Mean reference regret','Conditional mean RMSE'],[[labels[m],s['reference'][m]['active'],f"{s['reference'][m]['agreement']*100:.1f}%",fmt(s['reference'][m]['regret']),fmt(s['reference'][m]['conditional_mean_rmse'])] for m in methods])}

Bridge의 conditional mean RMSE는 Cross-fitted SNIS16384보다 약 8.9배 작았지만, full-reference action agreement의 차이는 100% 대 96.6%였고 actual decision loss의 추가 개선은 확인되지 않았다. Posterior mean accuracy 자체와 observation selection의 가치가 분리되는 사례다.

Worlds 0–3의 같은 R32 actual observations에 decoder N4096을 추가했다.

{table(['Bridge − baseline','N2048 paired difference','N4096 paired difference','변화'],[[labels[m],fmt(v['n2048_difference']),fmt(v['n4096_difference']),fmt(v['change'])] for m,v in s['decoder_audit'].items()])}

Four-world audit는 decoder sensitivity를 점검하는 범위이며 전체 64 worlds의 numerical convergence 증명이 아니다. Full-reference action agreement가 높더라도 특정 fixed world의 realized loss를 최소화하는 action과 반드시 같지는 않다. Bayesian acquisition은 posterior-averaged risk에 대한 선택이다.

### 6.1 Model-aware nuisance-dimension control

Actual evaluation 0/64인 시점에 source inspection으로 추가한 supplementary diagnostic이다. 기존 primary contrasts에는 넣지 않았다. Known corner geometry에서 half-height $d$는 lower boundary와 lower-left arc에서 소거된다. 모든 actions/queries의 최대 source-y는 28 mm이고 upper core의 최소 y는 44 mm이므로 upper geometry가 dynamics에 영향을 주지 않는다. Independent uniform prior의 $d$를 **Gaussian view likelihood에서 analytically integrate**하면 필요 없는 dimension의 importance-weight concentration을 피할 수 있다.

4096 prior samples와 $d$의 양 끝/중간 값으로 모든 dynamics를 검증한 maximum forward error는 {s['nuisance_validation']['d_invariance_max_error']:.3g}였다. Marginal likelihood와 numerical 1D quadrature의 maximum log-likelihood error는 {s['nuisance_validation']['quadrature_loglik_max_error']:.3g}였다. Covariance correlations와 bounded prior를 유지한 Gaussian CDF difference를 사용했다. Second channel을 단순 삭제한 것이 아니다.

Fixed SNIS16384와 동일한 four SMC populations/seeds를 재생성하고 **view weights만** 바꿨다. Dynamic conditional means의 replay 일치를 검증했다. Auxiliary choices는 private evidence를 읽지 않는 code에서 별도로 freeze했으며, main decoding 이후 계산됐지만 diagnostic 설계와 code는 main actual evaluation 전에 고정했다. 모든 actual losses의 분석은 이 freeze 이후 수행했다.

{table(['Rao–Blackwellized SNIS16k','Actual MSE','Bridge − RB','Descriptive 95% CI','Mean compute seconds'],[[k,fmt(v['mse']),fmt(v['bridge_minus']),ci(v['ci95']),f"{v['mean_seconds']:.3f}"] for k,v in s['nuisance'].items()])}

{table(['Estimator','RB − original SNIS16k','95% CI','View median ESS','Reference action agreement'],[[k,fmt(v['minus_original16k']),ci(v['minus_original16k_ci95']),f"{v['view_median_ess']:.3f}",f"{v['reference_agreement']*100:.1f}%"] for k,v in s['nuisance'].items()])}

Prior boundary 및 tail의 별도 quadrature check에서도 maximum log-likelihood error는 {s['nuisance_bounds_validation']['max_loglik_error']:.3g}였다.

같은 16k particles에서 d marginalization만으로 view median ESS는 **1.013 → 25.840**, collapse fraction은 **48.03% → 0.085%**로 바뀌었다. Standard utility의 actual MSE는 크게 개선됐지만, 이미 cross-fitted인 SNIS의 actual MSE는 추가로 개선되지 않았다. 즉 ESS를 높이는 것만으로 decision-loss 이득이 보장되지는 않는다. RB Standard 대비 Bridge의 작은 차이는 descriptive 95% interval에서만 음수이고, RB Cross-fitted의 interval upper endpoint는 0이다. 이 supplementary result를 새로운 primary success로 전환하지 않는다.

이 baseline은 simulator의 analytic structure를 활용한다. Unknown geometry나 general real images에 바로 적용되는 방법은 아니지만, **현재 benchmark의 강한 model-aware baseline**이다. Bridge가 작은 full-dimensional bank만 이기는 결과를 geometry–restitution reasoning의 고유한 장점으로 읽지 않기 위해 필요하다. 이 control은 exact equal-time baseline이 아니라 더 적은 compute로 dimension을 처리하는 구조적 대안이다.

![Nuisance control](results/analysis/figures/nuisance_control.png)

## 7. 논문 기여 가능성 평가

{headline}

특히 cheap Rao–Blackwellized baseline과의 차이가 미확정이거나 반전되면, primary budget comparison에서 이득이 있더라도 analytic nuisance handling보다 Bridge가 필요하다는 결론은 성립하지 않는다. 현재 toy benchmark의 likelihood structure를 실질적인 physics reasoning 난이도와 구분해야 한다.

새로운 주장으로 삼기 어려운 부분은 명확하다. Tempered SMC / resample–move 자체, downstream utility를 기준으로 observation을 고르는 발상, 그리고 selection과 decoder를 분리해 평가하는 설계만으로 novelty를 주장하기 어렵다. [Inside-Out SMC², ICML 2024](https://proceedings.mlr.press/v235/iqbal24a.html)는 dynamical systems의 experimental design에 nested SMC를 사용한다. 이번 정적 five-parameter collision diagnostic과 문제는 다르지만 nested SMC를 experimental design에 도입했다는 것 자체가 새롭지는 않다.

[Amortized Bayesian Experimental Design for Decision-Making](https://arxiv.org/html/2411.02064)은 downstream decision utility를 직접 고려한다. [Action-BED v2](https://arxiv.org/html/2606.23662v2)는 loss-driven design/action optimization을 다루므로 task-driven objective라는 큰 방향도 기존 연구와 겹친다. 이 보고서는 이 논문들의 model이나 training protocol을 직접 재현한 성능 비교를 수행하지 않았다.

현 단계에서 검토할 수 있는 contribution은 **finite-bank approximation, utility-estimator noise, action-gap ambiguity가 actual decision loss에 어떻게 연결되는지 분리한 diagnostic과 resource-aware refinement criterion**이다. 다만 criterion 자체는 이번에 제안·검증한 새 algorithm이 아니다. 이번에 구현한 것은 independent replication과 controls다.

이번 결과에 따른 구체적인 선택은 **현재 toy에서 Bridge를 더 정교하게 만드는 것을 우선순위에서 내리는 것**이다. 먼저 cross-fitted SNIS와 task-relevant nuisance handling을 기본 baseline으로 삼아야 한다. 이후 unknown geometry, image-likelihood mismatch 또는 더 복잡한 contact topology에서 이 baseline들이 실제로 실패하는지를 검증해야 한다. 그 실패가 확인되기 전에는 selective refinement algorithm 개발이 이 benchmark의 numerical artifact를 해결하는 데 그칠 위험이 있다.

현재 결과 자체는 재현 가능한 diagnostic/negative-result study로 가치가 있지만, major conference의 독립적인 method contribution으로는 약하다. 이를 발전시키려면 task-relevant error indicator, resource allocation rule 또는 model mismatch에 대한 새로운 결과가 필요하다. 이번 실험은 그런 새 algorithm을 구현했다는 증거가 아니다.

후속 의사결정은 다음처럼 제한하는 것이 타당하다.

1. Fixed small bank만 이기고 compute-budget baseline을 이기지 못하면, 일반적인 Bridge superiority 주장을 중단한다. 빠른 cached SNIS가 충분한 조건과 finite-bank failure 조건을 정리하는 diagnostic contribution으로 범위를 축소한다.
2. 두 budget controls보다 이득이 남으면, independent additional seeds/worlds와 더 넓은 geometry family에서 복제하고 same-estimator ablation을 먼저 수행한다. 이후 action gap과 estimated numerical error를 함께 사용해 일부 queries에만 refinement를 배정하는 정책을 비교한다.
3. 현재 2D known-family proxy에서의 차이를 real visual physics나 unknown object representation의 성과로 확대하지 않는다. 실제 연구 확장은 image likelihood mismatch, occlusion, unknown geometry와 collision boundary의 counterfactual reliability가 있는 환경에서 강한 baseline과 함께 검증해야 한다.
4. 결과가 유리해지도록 contact 조건을 계속 좁히거나 posterior를 인위적으로 더 날카롭게 하는 후속 실험은 독립적인 scientific motivation 없이 primary evidence로 사용하지 않는다.

## 8. 실행·재현 evidence

- Server project: `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/decision_replication_v8`.
- 짧은 allocations 2676324, 2676325는 worlds preparation과 high-index cases의 bounded acquisition을 수행했다. 나머지 acquisition 및 모든 actual evaluation은 node56 allocation 2680491의 idle GPU에서 수행했다. Resource scheduling 변경은 `RESOURCE_ADDENDUM.md`에 기록했고 inference protocol은 유지했다. Existing jobs/allocations는 변경하지 않았다.
- Main GPU 환경은 NVIDIA RTX PRO 6000 Blackwell Server Edition, driver 580.126.09, Python 3.9.18, PyTorch 2.7.1+cu128, NumPy 1.26.4다. Local statistical analysis와 plotting은 별도 NumPy/Matplotlib 환경에서 수행했다.
- Main conditional populations: **{s['integrity']['conditional_populations']:,}**. Unique conditional seeds: **{s['integrity']['unique_conditional_seeds']}**. 모든 final beta=1; minimum stage ESS/N={s['integrity']['min_stage_ess_ratio']:.12f}.
- Conditional SMC + summaries compute time: {s['integrity']['conditional_compute_seconds']:.2f}s. 이 값은 bank 생성, outer generator, whole pipeline I/O를 포함하지 않는 부분 합이다.
- All 64 independent truth hashes가 서로 다르고, frozen policy checksum 및 choices를 다시 계산해 일치함을 검증했다.
- Simulator/likelihood/calibration은 v7과 동일한 source bytes다. Gaussian quadrature와 direct NumPy SNIS, streaming accumulation, pooled/cross-product utilities, ESS, seed uniqueness 검증을 통과했다.
- World/repeat/query/action indexing, exclusive contribution sum, repeated-noise를 world 수로 오인하지 않는 analysis checks를 통과했다.

주요 artifacts:

- [사전 protocol](PROTOCOL.md)
- [실행 및 data boundary](README.md)
- [Resource scheduling addendum](RESOURCE_ADDENDUM.md)
- [Nuisance control 사전 기록](NUISANCE_ADDENDUM.md)
- [Cross-fitting과 decision error 해석](THEORY_NOTE_KO.md)
- [실험 코드](pipeline.py)
- [분석 코드](analyze.py)
- [전체 수치](results/analysis/summary.json)
- [World별 loss table](results/analysis/world_losses.csv)
- [Paired loss arrays](results/analysis/paired_losses.npz)
- [Ranking diagnostics](results/analysis/ranking.json)
- [고정 inference source hashes](source_hashes.json)

별도 `server_run_manifest.json`에 완료 artifacts의 SHA256를 저장한다. 위 결과는 synthetic simulation이며 실제 물체 영상이나 hardware pilot 결과는 포함하지 않는다.
'''
assert not any(ord(c)<32 and c!='\n' for c in text)
(P/'REPORT_KO.md').write_text(text)
# Delivered local report uses absolute file links; server version keeps relative links.
import re
local=re.sub(r'\]\((?!https?://)([^)]+)\)',lambda m:']('+str((P/m.group(1)).resolve())+')',text)
out=P.parents[2]/'outputs/physics_scene_belief_corner_replication_v8_report.md';out.write_text(local)
print(str(out))
