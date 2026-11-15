# Adaptive decision-region compression — v17

작성일: 2026-11-13 KST. Allocated server에서 구현, fresh-world experiments, ranking audits 및 GPU/CPU timing을 완료했다.

## 1. 결론

**Acquisition error와 action gap에 따라 components를 늘리는 adaptive compression을 구현했지만, 이번 실험에서는 ranking 안정성과 compute 효율을 함께 개선하지 못했다.** Decision loss는 nested512와 가까웠고 사전 5% noninferiority 조건도 충족했지만, GPU에서는 nested512보다 느렸으며 더 저렴한 fixed-size mixtures와 strong rules 대비 이득도 입증하지 못했다.

Fresh 512 worlds × 16 actual observation repeats의 matched condition은 다음과 같다. Latency는 동일한 optimized timing pass에서 얻은 두 event tasks의 aggregated median이며, 낮을수록 좋다.

| Method | Contact Brier loss | Stick Brier loss | GPU online ms |
|---|---:|---:|---:|
| **Adaptive region** | **0.052373** | **0.078740** | **14.581** |
| Adaptive global | 0.052302 | 0.079426 | 8.723 |
| Fixed region K32 | 0.052373 | 0.077076 | 5.475 |
| Fixed region K128 | 0.052373 | 0.078740 | 5.949 |
| Fixed global K128 budget | 0.052372 | 0.079331 | 6.245 |
| Optimized LMMSE16 | 0.052296 | 0.083311 | 3.784 |
| Optimized LMMSE64 | 0.049676 | 0.080689 | 4.406 |
| Nested128 | 0.052298 | 0.078737 | 4.727 |
| Nested512 | 0.052372 | 0.079387 | 10.322 |
| Always-query | 0.052295 | 0.098184 | 1.647 |
| Always-probe | 0.120361 | 0.077625 | 1.625 |

Adaptive region의 stick-event loss는 LMMSE16보다 평균 약 5.49% 낮지만 adjusted CI가 0을 포함한다. Difference는 **−0.004571 [−0.013684,+0.003870]**이다. V16에서 관찰한 LMMSE 대비 개선이 이번 새 worlds에서도 통계적으로 명확하게 재현됐다고 말할 수 없다. Adaptive global 및 fixed global 대비 우위도 확인되지 않았다.

Nested512 대비 mean-loss ratio의 95% upper는 contact **1.000045**, stick **1.003879**로 사전 1.05 기준 이하다. 하지만 GPU latency는 **14.58 ms 대 10.32 ms**로 오히려 약 41% 높다. 따라서 accuracy와 2× speedup을 함께 요구한 approximation screen은 실패했다. Fixed128과 mean loss가 사실상 같은데 비용은 약 2.45배다.

Ranking 안정성도 개선되지 않았다. Acquisition seed를 바꾸었을 때 stick-event의 Top-1 agreement는 adaptive region **77.54%**, fixed32 **79.49%**, fixed128 **77.73%**, adaptive global **88.48%**, nested512 **76.56%**였다. Adaptive minus fixed32는 **−1.95 percentage points**, secondary 95% CI **[−6.05,+2.34]**다. Nested512도 변동성이 있으므로 이를 exact ranking oracle로 취급하지 않는다.

실패 조건은 구체적이다. Contact에서는 region당 K16에서 381/512 worlds가 끝났지만, stick에서는 385/512 worlds가 K128까지 갔고 **357개가 cap-unresolved**였다. 그중 **278개는 paired sampling-error 항만으로도 action gap 이상**이었다. Fixed outer512 samples에서 components만 늘리는 방식은 이런 불확실성을 해소하지 못한다.

추가 observation 자체는 유용했다. No-observation loss는 contact 0.181039, stick 0.098995였다. 그러나 adaptive selection은 contact에서 always-query를, stick에서 always-probe를 넘는 추가 가치를 보여 주지 못했다. Stable ranking이나 더 많은 components를 쓰는 것 자체가 좋은 decision을 보장하지 않는다.

**현재의 component-only adaptive compression을 독립적인 method paper의 핵심으로 추진하는 것은 권하지 않는다.** 이번 단계의 성과는 error source를 분리해, representation resolution을 늘려야 할 경우와 outer integration budget을 바꿔야 할 경우가 다름을 실험으로 확인한 것이다. 후속 방법은 이 차이를 실제 resource allocation으로 연결해야 하며, 단순히 cap을 더 높이는 접근은 우선순위가 낮다.


## 2. 구현한 adaptive compression

Factual evidence를 본 후 QMC16384 inner posterior를 얻고, event label 0/1의 두 decision regions로 나눈다. 각 region의 posterior mass를 유지하면서 region당 K개의 systematic representatives를 추출한다. Representative마다 원래 Gaussian sensor likelihood를 유지한다. K는 전체 components 수가 아니다. 두 regions가 모두 존재하면 replica당 2K개이며, 이번 방법은 두 replicas를 사용한다.

별도의 seed로 생성한 QMC16384 outer posterior에서 512개의 hypothetical world–observation pairs를 만든다. 이 outer sample은 simulator ground truth나 실제 미래 observation이 아니다. Public factual evidence에서 추론한 posterior predictive samples다. 모든 candidate actions와 expansion stages가 같은 outer samples를 공유한다.

각 compact replica가 hypothetical observation 이후 예측한 event probability와 outer sample의 event label 사이의 Brier loss를 계산한다. 두 replicas의 risks를 평균한다. 두 prediction을 평균한 ensemble decoder의 risk와는 다르다. 실제 observation을 받은 뒤에는 공통 QMC16384 full decoder를 사용하므로 primary 비교는 acquisition 차이를 측정한다.

현재 best action을 b, 다른 action을 a, 두 action의 estimated risk 차이를 gₐ라고 두면 stopping score는 다음과 같다.

\[
g_a(K)=\hat R_a(K)-\hat R_b(K),
\]
\[
e_a(K)=2\widehat{SE}_{\rm paired}(\ell_a-\ell_b)
+|g_a(K)-g_a(K_{\rm prev})|
+\tfrac12|g_a^{(1)}(K)-g_a^{(2)}(K)|.
\]

K를 **8→16→32→64→128**로 늘린다. K≥16에서 모든 competitor에 대해 gₐ>eₐ이면 멈춘다. 현재 best가 이전 단계와 달라도 이전 risk vector에서 현재 b와의 차이를 다시 계산한다. 모든 action risks의 range가 10⁻¹² 이하면 numerical tie로 기록하고 query view를 선택한다. K128에서도 gap이 분리되지 않으면 `cap-unresolved`로 기록하고 현재 estimated risk가 가장 작은 action을 선택한다.

첫 error 항은 outer sampling variation, 두 번째는 successive-resolution drift, 세 번째는 compression replicas 사이의 variation을 겨냥한다. **이 합은 true acquisition error의 보장된 upper bound가 아니다.** QMC rows는 iid가 아니며, 두 replicas만으로 shared bias를 탐지하지 못할 수 있고, successive stopping에 대한 coverage 보정도 없다. 따라서 `gap separated`는 내부 heuristic을 만족했다는 뜻이며 certified optimal action이라는 뜻이 아니다.

이번 구현은 outer sample count를 512로 고정한다. Sampling error가 action gap보다 크면 components만 늘려서는 해결되지 않을 수 있다. 이 상황의 빈도를 별도로 기록했다. 각 K에서는 같은 systematic phases를 쓰되 quantile locations가 달라져 representatives를 다시 구성한다. 기존 components를 그대로 유지하며 일부만 추가하는 nested coreset implementation은 아니다. Timing에는 이 재계산과 stopping checks를 모두 포함했다.

Posterior mass≤10⁻¹⁰인 region만 pruning하고 renormalize한다. Omitted mass를 기록한다. Inner/outer banks, full decoder와 cached forward predictions를 계속 사용하므로 bank-free compact belief라고 부르지 않는다.

| Method | 역할 |
|---|---|
| adaptive | Region별 compression과 위 stopping rule |
| adaptive_global | 같은 최대 total components, outer samples, replicas, stopping rule의 global resampling |
| fixed32 / fixed128 | 같은 cross-fitted risk objective에서 region당 K32/K128을 고정; stopping의 필요성 확인 |
| global128 | Fixed128과 같은 total components의 global resampling |
| region32 / global32 | V16 coherent mixture risk, representative당 hypothetical noise Q4 |
| native / qmc64_native | Optimized LMMSE16 / LMMSE64 |
| nested128 / nested512 | Independent inner/outer QMC16을 사용하는 optimized nested expected-loss baseline |
| query / probe / factual / none | Fixed rules 및 no-observation diagnostic |

Adaptive methods의 stopping rule을 켠 비교와 끈 비교는 같은 cross-fitted objective를 사용한다. V16 coherent methods와의 차이는 adaptivity뿐 아니라 risk estimator까지 포함하므로 pure adaptivity effect라고 해석하지 않는다.

## 3. 실험 설계와 해석 범위

새로운 **512 iid worlds**, world당 **16 actual observation noise repeats**를 사용했다. Seed는 6100101+137×world index이며 correctness pilot 8개는 별도 seed6000101이다. V16의 worlds를 primary test로 재사용하지 않았다. Contact-event와 stick-event 두 tasks에 집중했고, continuous velocity task의 성능을 이번 결과로 주장하지 않는다.

Model은 V16의 unknown boundary geometry, restitution과 contact-law uncertainty를 유지한다. Boundary는 x(y)=d+s y+a f_g(y), 두 geometry families와 frictionless/Coulomb/flexible laws를 사용한다. Prior는 geometry×두 base laws의 네 components에 각각 1/8, geometry×flexible law의 두 components에 각각 1/4이다. d∼U[−4,4] mm, s∼U[−0.4,0.4], a∼U[−0.1,0.1], e∼U[0.35,0.9], b₀∼U[−4,2], b₁∼U[−2,2]다.

L=25 mm, z=y/L일 때 f₀=Lz²/2, f₁=L(z²/2−0.35z⁴)다. Nominal flexible tangential impulse는 vₜ sigmoid[b₀+b₁ log((|vₜ|+0.02)/0.5)], shifted law는 vₜ sigmoid(b₀)[0.2+0.8/(1+(|vₜ|/s₀)²)]이고 s₀=0.1+0.7 sigmoid(b₁)다. Base laws는 바꾸지 않는다.

Factual evidence는 y=0에서 incoming velocity (1,0.35)의 outgoing velocity와 −10…10 mm의 boundary profiles 9개다. Noise SD는 각각 0.07 m/s와 1.2 mm다. Query y=25 mm에서 public factual profiles를 직선으로 extrapolate한 값이 contact/no-contact threshold다. Stick query의 incoming direction 역시 public slope estimate로 정하고 residual tangential velocity absolute≤10⁻⁸을 stick label로 쓴다. Hidden truth로 favorable worlds를 선택하지 않는다.

Candidate observations는 query view(19…31 mm의 profiles 7개, SD1 mm), factual view(−8…8 mm의 profiles 7개, SD0.35 mm), probe(y=0, incoming velocity(1,−0.7), outgoing velocity SD0.05 m/s)다. Selectors는 셋 중 하나를 선택한다. Sensing cost는 같다고 가정하며 no-observation은 별도 diagnostic이다.

Matched nominal, shifted data/nominal inference, shifted data/correct-law inference의 세 cells를 평가했다. Correct-law는 privileged diagnostic이다. Physics는 finite-reach contact gate와 instantaneous local impulse model이며 full trajectory solver, raw RGB reconstruction, real robot validation은 아니다.

Primary inference banks는 inner seed2026111201, outer seed2026111202, LMMSE64 seed2026111203이다. Primary acquisition seed81531을 고정하고 actual outcomes를 보기 전에 policies를 freeze했다. 512 matched worlds 모두에서 seed81532로 adaptive/fixed cross-fitted mixtures와 nested512를 반복했다. 여기서는 hypothetical samples와 compression phases가 함께 달라진다. 두 원인을 따로 분리한 factorial audit은 아니다.

각 cell의 첫 64 worlds에는 seed81533의 nested2048 reference를 사용했다. 첫 64 matched worlds에서는 inner/outer bank seeds2026111301/2026111302로 policy를 다시 계산했다. Bank audit의 actual loss는 공통 primary decoder로 평가하여 acquisition 차이를 분리했다. 모든 posterior seeds에 대한 일반화를 증명하는 충분한 multi-seed study는 아니다.

Ranking 안정성은 주로 **Top-1 action agreement**로 측정한다. Full ordering의 모든 permutation stability를 측정한 것은 아니다. LMMSE와 fixed rules는 같은 bank에서 deterministic하므로 acquisition seed를 바꾸어도 action이 같다. 이것이 정확한 ranking임을 뜻하지 않으므로 independent-bank sensitivity와 reference agreement를 함께 본다. Nested2048도 exact Bayes oracle은 아니다.

Primary 12 contrasts는 두 tasks에서 adaptive 대 native/nested512/adaptive_global/fixed128/global128/task-specific fixed rule(contact의 query, stick의 probe)다. World별로 16 repeats를 평균한 뒤 30,000 paired bootstrap과 Bonferroni 99.58333% CI를 사용했다. 나머지는 secondary 95% CI다. Fixed banks와 acquisition plans에 조건부인 CI이므로 모든 numerical uncertainty를 포함하지 않는다.

사전 approximation screen은 두 tasks 모두에서 adaptive/nested512 mean-loss ratio의 95% upper≤1.05 및 GPU aggregated median speedup≥2다. Adaptivity screen은 fixed32 대비 Top-1 repeatability 10 percentage points 이상 증가, fixed128 대비 GPU median 20% 이상 절약, 각 task의 fixed128 대비 loss-ratio upper≤1.05다. 분석 코드는 repeatability 조건을 두 tasks 모두에 적용했다. Contact에서 baseline agreement가 이미 높은 경우 이 조건에는 ceiling limitation이 있으므로 screen 실패만으로 모든 가치를 부정하지 않고 task별 수치와 cost를 함께 해석한다. Threshold를 outcomes에 맞춰 바꾸지 않는다.


## 4. Decision loss

각 cell의 512 worlds × 16 observation repeats 결과다. 낮을수록 좋다. V16과 worlds가 다르므로 V16 보고서의 평균과 직접 비교해 개선율을 만들지 않는다.

### matched_nominal

| Method | contact_event | slip_event |
|---|---|---|
| adaptive | 0.052373 | 0.078740 |
| adaptive_global | 0.052302 | 0.079426 |
| fixed32 | 0.052373 | 0.077076 |
| fixed128 | 0.052373 | 0.078740 |
| global128 | 0.052372 | 0.079331 |
| region32 | 0.052295 | 0.078231 |
| global32 | 0.052298 | 0.076965 |
| native | 0.052296 | 0.083311 |
| qmc64_native | 0.049676 | 0.080689 |
| nested128 | 0.052298 | 0.078737 |
| nested512 | 0.052372 | 0.079387 |
| query | 0.052295 | 0.098184 |
| probe | 0.120361 | 0.077625 |
| factual | 0.100581 | 0.102486 |
| none | 0.181039 | 0.098995 |

### shifted_nominal

| Method | contact_event | slip_event |
|---|---|---|
| adaptive | 0.051947 | 0.077189 |
| adaptive_global | 0.051879 | 0.077427 |
| fixed32 | 0.051947 | 0.076312 |
| fixed128 | 0.051947 | 0.077189 |
| global128 | 0.051947 | 0.077948 |
| region32 | 0.051872 | 0.074090 |
| global32 | 0.051874 | 0.074585 |
| native | 0.051872 | 0.077388 |
| qmc64_native | 0.049770 | 0.076540 |
| nested128 | 0.051884 | 0.076349 |
| nested512 | 0.051947 | 0.077818 |
| query | 0.051872 | 0.088699 |
| probe | 0.118477 | 0.075249 |
| factual | 0.101222 | 0.092653 |
| none | 0.181077 | 0.092809 |

### shifted_correct

| Method | contact_event | slip_event |
|---|---|---|
| adaptive | 0.050721 | 0.068911 |
| adaptive_global | 0.050661 | 0.068906 |
| fixed32 | 0.050721 | 0.068471 |
| fixed128 | 0.050721 | 0.068911 |
| global128 | 0.050720 | 0.069073 |
| region32 | 0.050661 | 0.070780 |
| global32 | 0.050669 | 0.067479 |
| native | 0.050661 | 0.068037 |
| qmc64_native | 0.049753 | 0.066451 |
| nested128 | 0.050609 | 0.070387 |
| nested512 | 0.050720 | 0.069161 |
| query | 0.050661 | 0.081771 |
| probe | 0.113066 | 0.067282 |
| factual | 0.099679 | 0.084391 |
| none | 0.179656 | 0.082595 |

### Primary paired contrasts

Adaptive minus baseline이다. 음수는 adaptive가 유리하다. CI는 12 comparisons를 보정한 99.58333% CI다.

| Task / comparison | Difference [adjusted CI] |
|---|---|
| contact_event/adaptive-native | +0.000077 [-0.000001, +0.000353] |
| contact_event/adaptive-nested512 | +0.000001 [-0.000000, +0.000004] |
| contact_event/adaptive-adaptive_global | +0.000071 [+1.001e-13, +0.000347] |
| contact_event/adaptive-fixed128 | +0.000000 [+0.000000, +0.000000] |
| contact_event/adaptive-global128 | +0.000001 [-0.000000, +0.000004] |
| contact_event/adaptive-query | +0.000078 [+1.056e-08, +0.000354] |
| slip_event/adaptive-native | -0.004571 [-0.013684, +0.003870] |
| slip_event/adaptive-nested512 | -0.000647 [-0.002429, +0.000695] |
| slip_event/adaptive-adaptive_global | -0.000686 [-0.002321, +0.000574] |
| slip_event/adaptive-fixed128 | +1.436e-15 [+0.000000, +7.103e-15] |
| slip_event/adaptive-global128 | -0.000591 [-0.002197, +0.000682] |
| slip_event/adaptive-probe | +0.001114 [-0.003213, +0.005704] |

10⁻¹³ 수준의 차이나 CI endpoint는 numerical precision에 가까우므로 실질적 superiority로 해석하지 않는다. 특히 adaptive와 fixed128의 losses는 이번 sample에서 사실상 같았다.

### Approximation 및 adaptivity screens

| Task / adaptive-to-baseline | Mean-loss ratio [95% CI] |
|---|---|
| contact_event/nested512 | 1.0000 [1.0000, 1.0000] |
| contact_event/fixed128 | 1.0000 [1.0000, 1.0000] |
| slip_event/nested512 | 0.9919 [0.9776, 1.0039] |
| slip_event/fixed128 | 1.0000 [1.0000, 1.0000] |

사전 screen 결과: `{'nested_loss': True, 'nested_speedup': 0.7079011056766171, 'approximation_pass': False, 'repeatability_plus_10pp': False, 'fixed128_time_saving': -1.4512009315501455, 'fixed128_loss': True, 'adaptivity_pass': False}`. Superiority가 검출되지 않은 것과 equivalence는 다르다.

### Observation choices

| Cell / method | Contact [none,query,factual,probe] | Stick [none,query,factual,probe] |
|---|---|---|
| matched_nominal/adaptive | [0, 503, 5, 4] | [0, 187, 16, 309] |
| matched_nominal/adaptive_global | [0, 511, 0, 1] | [0, 272, 5, 235] |
| matched_nominal/fixed128 | [0, 503, 5, 4] | [0, 184, 16, 312] |
| matched_nominal/global128 | [0, 506, 3, 3] | [0, 223, 13, 276] |
| matched_nominal/native | [0, 502, 6, 4] | [0, 224, 15, 273] |
| matched_nominal/nested512 | [0, 504, 5, 3] | [0, 144, 27, 341] |
| shifted_nominal/adaptive | [0, 505, 3, 4] | [0, 203, 19, 290] |
| shifted_nominal/adaptive_global | [0, 511, 0, 1] | [0, 292, 2, 218] |
| shifted_nominal/fixed128 | [0, 505, 3, 4] | [0, 202, 19, 291] |
| shifted_nominal/global128 | [0, 508, 1, 3] | [0, 229, 10, 273] |
| shifted_nominal/native | [0, 503, 7, 2] | [0, 215, 10, 287] |
| shifted_nominal/nested512 | [0, 504, 3, 5] | [0, 157, 29, 326] |
| shifted_correct/adaptive | [0, 502, 4, 6] | [0, 195, 25, 292] |
| shifted_correct/adaptive_global | [0, 512, 0, 0] | [0, 295, 3, 214] |
| shifted_correct/fixed128 | [0, 502, 4, 6] | [0, 194, 25, 293] |
| shifted_correct/global128 | [0, 507, 4, 1] | [0, 233, 16, 263] |
| shifted_correct/native | [0, 504, 6, 2] | [0, 121, 0, 391] |
| shifted_correct/nested512 | [0, 503, 6, 3] | [0, 151, 28, 333] |

## 5. Ranking 안정성과 error diagnostics

### Independent acquisition seed: 512 matched worlds

Hypothetical sampling과 compression phases를 함께 바꾼 Top-1 action agreement다. Loss 변화는 seed2 minus primary이며 secondary 95% CI다.

| Method | Contact agreement | Stick agreement | Contact loss change | Stick loss change |
|---|---|---|---|---|
| adaptive | 98.63% | 77.54% | -0.000008 [-0.000022, +7.972e-08] | -0.000384 [-0.002312, +0.001510] |
| adaptive_global | 99.41% | 88.48% | +0.000063 [-0.000018, +0.000209] | -0.000013 [-0.001895, +0.002046] |
| fixed32 | 97.66% | 79.49% | -0.000077 [-0.000222, +1.230e-08] | +0.000488 [-0.000816, +0.001963] |
| fixed128 | 98.63% | 77.73% | -0.000008 [-0.000022, +7.972e-08] | -0.000384 [-0.002312, +0.001510] |
| global128 | 98.63% | 85.74% | -0.000007 [-0.000021, +1.503e-08] | -0.000312 [-0.001903, +0.001289] |
| nested512 | 98.83% | 76.56% | -0.000007 [-0.000021, +1.446e-08] | -0.000638 [-0.002277, +0.000954] |

Fixed32 대비 adaptive의 agreement 증가량: {'contact_event': {'mean': 0.977, 'low': 0.0, 'high': 1.953}, 'slip_event': {'mean': -1.953, 'low': -6.055, 'high': 2.344}} percentage points. 여기의 CI는 secondary95%다.

### Independent posterior banks: 첫 64 matched worlds

Acquisition banks만 교체하고 실제 prediction decoder는 primary bank로 고정했다. Mean-loss delta는 bank2 selection minus primary selection이다.

| Method | Contact agreement | Stick agreement | Contact loss delta | Stick loss delta |
|---|---|---|---|---|
| adaptive | 100.00% | 85.94% | 0.000000 | -0.000387 |
| adaptive_global | 100.00% | 84.38% | 0.000000 | 0.004245 |
| fixed128 | 100.00% | 85.94% | 0.000000 | -0.000387 |
| global128 | 98.44% | 87.50% | -0.000000 | -0.000155 |
| native | 98.44% | 90.62% | -0.000000 | 0.004583 |
| nested512 | 100.00% | 81.25% | 0.000000 | 0.002088 |

### Nested2048 reference: 각 cell의 첫 64 worlds

Reference risk regret는 nested2048가 추정한 risk 아래의 차이이며 실제 Bayes regret가 아니다. Agreement만 높이고 actual loss가 나빠질 수 있으므로 양쪽을 구분한다.

| Cell / method | Contact agreement | Stick agreement | Contact reference regret | Stick reference regret |
|---|---|---|---|---|
| matched_nominal/adaptive | 100.00% | 68.75% | 0.000000 | 0.000240 |
| matched_nominal/adaptive_global | 100.00% | 65.62% | 0.000000 | 0.000192 |
| matched_nominal/fixed32 | 98.44% | 68.75% | 0.000145 | 0.000314 |
| matched_nominal/fixed128 | 100.00% | 68.75% | 0.000000 | 0.000240 |
| matched_nominal/global128 | 98.44% | 67.19% | 0.000001 | 0.000167 |
| matched_nominal/native | 98.44% | 48.44% | 0.000001 | 0.004345 |
| matched_nominal/nested512 | 100.00% | 82.81% | 0.000000 | 0.000098 |
| shifted_nominal/adaptive | 100.00% | 62.50% | 0.000000 | 0.000253 |
| shifted_nominal/adaptive_global | 100.00% | 59.38% | 0.000000 | 0.000888 |
| shifted_nominal/fixed32 | 98.44% | 67.19% | 0.000091 | 0.000274 |
| shifted_nominal/fixed128 | 100.00% | 62.50% | 0.000000 | 0.000253 |
| shifted_nominal/global128 | 100.00% | 60.94% | 0.000000 | 0.000193 |
| shifted_nominal/native | 98.44% | 43.75% | 0.000002 | 0.004630 |
| shifted_nominal/nested512 | 100.00% | 79.69% | 0.000000 | 0.000139 |
| shifted_correct/adaptive | 98.44% | 65.62% | 0.000005 | 0.000499 |
| shifted_correct/adaptive_global | 100.00% | 59.38% | 0.000000 | 0.000811 |
| shifted_correct/fixed32 | 100.00% | 57.81% | 0.000000 | 0.001021 |
| shifted_correct/fixed128 | 98.44% | 65.62% | 0.000005 | 0.000499 |
| shifted_correct/global128 | 100.00% | 64.06% | 0.000000 | 0.000199 |
| shifted_correct/native | 98.44% | 64.06% | 0.000005 | 0.001041 |
| shifted_correct/nested512 | 98.44% | 70.31% | 0.000005 | 0.000195 |

### Components와 stopping reason

K counts는 region당 K=[8,16,32,64,128]의 순서다. Reason counts는 [gap-separated, cap-unresolved, risk-tie, fixed-size] 순서다. Processed components는 모든 expansion stages와 두 replicas에서 처리한 component 수의 합이다. Final payload와 다른 지표다.

| Cell / method / task | K counts | Mean K / stages | Reason counts | Mean processed components |
|---|---|---|---|---|
| matched_nominal/adaptive/contact_event | [2, 381, 85, 10, 34] | 27.00 / 2.40 | [484, 26, 2, 0] | 183.9 |
| matched_nominal/adaptive/slip_event | [1, 49, 35, 42, 385] | 105.23 / 4.49 | [143, 357, 12, 0] | 809.8 |
| matched_nominal/adaptive_global/contact_event | [37, 376, 83, 7, 9] | 20.64 / 2.17 | [465, 6, 41, 0] | 133.1 |
| matched_nominal/adaptive_global/slip_event | [195, 62, 43, 30, 182] | 56.92 / 2.89 | [148, 154, 210, 0] | 423.3 |
| matched_nominal/fixed32/contact_event | [0, 0, 512, 0, 0] | 32.00 / 1.00 | [0, 0, 0, 512] | 127.8 |
| matched_nominal/fixed32/slip_event | [0, 0, 512, 0, 0] | 32.00 / 1.00 | [0, 0, 0, 512] | 127.9 |
| matched_nominal/fixed128/contact_event | [0, 0, 0, 0, 512] | 128.00 / 1.00 | [0, 0, 0, 512] | 511.0 |
| matched_nominal/fixed128/slip_event | [0, 0, 0, 0, 512] | 128.00 / 1.00 | [0, 0, 0, 512] | 511.5 |
| matched_nominal/global128/contact_event | [0, 0, 0, 0, 512] | 128.00 / 1.00 | [0, 0, 0, 512] | 511.0 |
| matched_nominal/global128/slip_event | [0, 0, 0, 0, 512] | 128.00 / 1.00 | [0, 0, 0, 512] | 511.5 |
| shifted_nominal/adaptive/contact_event | [2, 386, 78, 12, 34] | 26.97 / 2.39 | [483, 27, 2, 0] | 183.7 |
| shifted_nominal/adaptive/slip_event | [2, 53, 29, 37, 391] | 105.88 / 4.49 | [141, 357, 14, 0] | 814.9 |
| shifted_nominal/adaptive_global/contact_event | [35, 382, 79, 10, 6] | 20.17 / 2.16 | [468, 4, 40, 0] | 129.3 |
| shifted_nominal/adaptive_global/slip_event | [224, 57, 45, 31, 155] | 50.72 / 2.68 | [137, 135, 240, 0] | 373.7 |
| shifted_nominal/fixed32/contact_event | [0, 0, 512, 0, 0] | 32.00 / 1.00 | [0, 0, 0, 512] | 127.8 |
| shifted_nominal/fixed32/slip_event | [0, 0, 512, 0, 0] | 32.00 / 1.00 | [0, 0, 0, 512] | 127.8 |
| shifted_nominal/fixed128/contact_event | [0, 0, 0, 0, 512] | 128.00 / 1.00 | [0, 0, 0, 512] | 511.0 |
| shifted_nominal/fixed128/slip_event | [0, 0, 0, 0, 512] | 128.00 / 1.00 | [0, 0, 0, 512] | 511.0 |
| shifted_nominal/global128/contact_event | [0, 0, 0, 0, 512] | 128.00 / 1.00 | [0, 0, 0, 512] | 511.0 |
| shifted_nominal/global128/slip_event | [0, 0, 0, 0, 512] | 128.00 / 1.00 | [0, 0, 0, 512] | 511.0 |
| shifted_correct/adaptive/contact_event | [2, 379, 82, 14, 35] | 27.50 / 2.42 | [479, 31, 2, 0] | 187.9 |
| shifted_correct/adaptive/slip_event | [2, 39, 33, 29, 409] | 109.19 / 4.57 | [120, 375, 17, 0] | 841.4 |
| shifted_correct/adaptive_global/contact_event | [32, 375, 87, 6, 12] | 21.41 / 2.20 | [466, 6, 40, 0] | 139.2 |
| shifted_correct/adaptive_global/slip_event | [221, 54, 36, 27, 174] | 54.27 / 2.76 | [120, 156, 236, 0] | 402.1 |
| shifted_correct/fixed32/contact_event | [0, 0, 512, 0, 0] | 32.00 / 1.00 | [0, 0, 0, 512] | 127.8 |
| shifted_correct/fixed32/slip_event | [0, 0, 512, 0, 0] | 32.00 / 1.00 | [0, 0, 0, 512] | 127.8 |
| shifted_correct/fixed128/contact_event | [0, 0, 0, 0, 512] | 128.00 / 1.00 | [0, 0, 0, 512] | 511.0 |
| shifted_correct/fixed128/slip_event | [0, 0, 0, 0, 512] | 128.00 / 1.00 | [0, 0, 0, 512] | 511.0 |
| shifted_correct/global128/contact_event | [0, 0, 0, 0, 512] | 128.00 / 1.00 | [0, 0, 0, 512] | 511.0 |
| shifted_correct/global128/slip_event | [0, 0, 0, 0, 512] | 128.00 / 1.00 | [0, 0, 0, 512] | 511.0 |

| Cell / method / task | Cap worlds where sampling term ≥ gap | Gap-separated reference subset n | Reference disagreements in that subset |
|---|---|---|---|
| matched_nominal/adaptive/contact_event | 23 | 61 | 0 |
| matched_nominal/adaptive/slip_event | 278 | 15 | 0 |
| matched_nominal/adaptive_global/contact_event | 4 | 59 | 0 |
| matched_nominal/adaptive_global/slip_event | 113 | 16 | 0 |
| shifted_nominal/adaptive/contact_event | 22 | 60 | 0 |
| shifted_nominal/adaptive/slip_event | 284 | 17 | 1 |
| shifted_nominal/adaptive_global/contact_event | 3 | 58 | 0 |
| shifted_nominal/adaptive_global/slip_event | 104 | 16 | 0 |
| shifted_correct/adaptive/contact_event | 24 | 60 | 0 |
| shifted_correct/adaptive/slip_event | 306 | 11 | 0 |
| shifted_correct/adaptive_global/contact_event | 6 | 58 | 0 |
| shifted_correct/adaptive_global/slip_event | 117 | 11 | 0 |

### Post-analysis label-support audit와 stopping groups

Risk tie는 risk range≤10⁻¹²라는 코드 조건이다. 반드시 floating-point roundoff만을 뜻하지 않는다. 아래는 full factual posterior에는 두 label regions가 있지만 두 global replicas가 각각 single-label support가 된 cases다. Loss나 policy를 변경하지 않은 explanatory audit다.

| Task | Global risk ties | Ties with both factual regions | Both replicas single-label | Minority mass [min,median,max] |
|---|---|---|---|---|
| contact_event | 41 | 39 | 39 | [1.27657854753771e-06, 0.01456792954025789, 0.13156462703258318] |
| slip_event | 210 | 209 | 209 | [2.499422671785271e-10, 0.0006560392011636584, 0.06044337799552192] |

| Task / group | n | Seed2 agreement | Mean Brier loss | Loss minus fixed128 |
|---|---|---|---|---|
| contact_event/adaptive/gap_separated | 484 | 100.00% | 0.055205 | 0.000000 |
| contact_event/adaptive/cap_unresolved | 26 | 73.08% | 0.003691 | 0.000000 |
| contact_event/adaptive/risk_tie | 2 | 100.00% | 0.000000 | 0.000000 |
| contact_event/adaptive_global/gap_separated | 465 | 100.00% | 0.057433 | 0.000000 |
| contact_event/adaptive_global/cap_unresolved | 6 | 83.33% | 0.008319 | 0.000000 |
| contact_event/adaptive_global/risk_tie | 41 | 95.12% | 0.000550 | -0.000887 |
| slip_event/adaptive/gap_separated | 143 | 99.30% | 0.129774 | 0.000000 |
| slip_event/adaptive/cap_unresolved | 357 | 68.91% | 0.060944 | 0.000000 |
| slip_event/adaptive/risk_tie | 12 | 75.00% | 0.000000 | 0.000000 |
| slip_event/adaptive_global/gap_separated | 148 | 99.32% | 0.132946 | 0.000000 |
| slip_event/adaptive_global/cap_unresolved | 154 | 70.78% | 0.125696 | 0.002009 |
| slip_event/adaptive_global/risk_tie | 210 | 93.81% | 0.007775 | 0.000200 |

Groups는 primary stopping result로 사후 구분했으므로 서로 다른 groups 사이의 mean-loss 차이를 causal effect로 해석하지 않는다. Support collapse가 발견됐다고 해서 global method의 aggregate loss가 유의하게 나빴다고 주장하지도 않는다.

## 6. Compute 비용

RTX PRO6000 Blackwell Server Edition, CPU2 threads, float64, batch1. 아래 primary compute 표는 equivalent optimized implementation의 동일 timing pass다. 고정된 첫 20 matched worlds × 2 tasks × 3 repeats로 method당120 measurements를 수행했다. Factual fitting, public targets, outer-bank fitting, compression, stopping checks, 선택한 actual observation update를 포함한다. Bank construction과 static forward cache setup은 별도로 기록했다. GPU와 CPU는 순차 실행했다. V16의 5-task aggregate latency와 직접 비교하지 않는다.

| Method | GPU median ms | GPU p90 ms | CPU median ms | GPU contact ms | GPU stick ms |
|---|---|---|---|---|---|
| adaptive | 14.581 | 19.604 | 6.516 | 9.045 | 18.898 |
| adaptive_global | 8.723 | 17.601 | 3.481 | 8.275 | 16.589 |
| fixed32 | 5.475 | 6.019 | 2.996 | 4.990 | 5.963 |
| fixed128 | 5.949 | 6.976 | 3.494 | 5.797 | 6.909 |
| global128 | 6.245 | 6.765 | 3.416 | 5.463 | 6.597 |
| region32 | 3.628 | 3.956 | 1.683 | 3.226 | 3.723 |
| global32 | 3.582 | 4.178 | 1.563 | 3.412 | 3.986 |
| native | 3.784 | 4.070 | 1.950 | 3.312 | 3.955 |
| qmc64_native | 4.406 | 4.851 | 5.202 | 4.252 | 4.650 |
| nested128 | 4.727 | 5.373 | 28.302 | 4.070 | 5.083 |
| nested512 | 10.322 | 11.621 | 89.627 | 9.655 | 10.694 |
| query | 1.647 | 2.083 | 0.929 | 1.339 | 1.973 |
| probe | 1.625 | 1.987 | 0.927 | 1.271 | 1.894 |

### 구현 최적화 전후의 observed latency

| Method | Original GPU ms | Optimized GPU ms | Original CPU ms | Optimized CPU ms |
|---|---|---|---|---|
| adaptive | 15.807 | 14.581 | 7.414 | 6.516 |
| adaptive_global | 8.771 | 8.723 | 4.178 | 3.481 |
| fixed32 | 6.016 | 5.475 | 3.300 | 2.996 |
| fixed128 | 6.376 | 5.949 | 3.718 | 3.494 |
| global128 | 5.651 | 6.245 | 3.715 | 3.416 |
| region32 | 3.551 | 3.628 | 1.653 | 1.683 |
| global32 | 3.185 | 3.582 | 1.676 | 1.563 |
| native | 3.261 | 3.784 | 1.871 | 1.950 |
| qmc64_native | 4.251 | 4.406 | 5.063 | 5.202 |
| nested128 | 4.575 | 4.727 | 18.295 | 28.302 |
| nested512 | 9.700 | 10.322 | 70.887 | 89.627 |
| query | 1.465 | 1.647 | 0.981 | 0.929 |
| probe | 1.328 | 1.625 | 0.937 | 0.927 |

Policy가 고정된 후 CDF caching과 logging-related synchronization을 줄였다. 15,360 cases의 equivalence를 검증했다. Original logs/results도 보존했다. Baselines는 같은 새 pass에서 다시 측정했으며 pass 사이의 latency variation도 표에 남겼다. 특히 CPU nested timing 변동이 있어 absolute speedup을 다른 hardware의 보장으로 해석하지 않는다.

| Method | Bank/cache bytes | Two-replica compact payload max bytes | Observed CUDA temporary peak bytes | Warm setup ms |
|---|---|---|---|---|
| adaptive | 17334304 | 73728.0 | 4283904 | 11.332 |
| adaptive_global | 17334304 | 73728.0 | 4283904 | 12.373 |
| fixed32 | 17334304 | 18432.0 | 1835008 | 10.703 |
| fixed128 | 17334304 | 73728.0 | 4229120 | 8.098 |
| global128 | 17334304 | 73728.0 | 4229120 | 10.021 |
| region32 | 8765440 | None | 1572864 | 5.947 |
| global32 | 8765440 | None | 1572864 | 4.318 |
| native | 24117248 | None | 1572864 | 5.329 |
| qmc64_native | 96468992 | None | 6291456 | 8.347 |
| nested128 | 17309696 | None | 50998784 | 8.319 |
| nested512 | 17334272 | None | 51008000 | 8.465 |
| query | 8650752 | None | 1572864 | 5.031 |
| probe | 8650752 | None | 1572864 | 4.725 |

Temporary peak는 40 unique timing cases에서 관찰한 추가 CUDA allocated memory다. 전체 가능한 inputs에 대한 bound가 아니다. Bank/cache는 두 banks 및 noise plans를 포함하며 Python objects, CUDA context/allocator reserve는 포함하지 않는다. Compact payload는 full factual weights나 banks를 없앤 memory saving을 뜻하지 않는다. Physical sensing/reset costs는 이 compute 시간에 포함되지 않는다.

| Method | CPU/GPU action mismatches / 40 | GPU prediction max error | CPU prediction max error |
|---|---|---|---|
| adaptive | 0 | 9.770e-15 | 8.660e-15 |
| adaptive_global | 0 | 9.770e-15 | 8.660e-15 |
| fixed32 | 0 | 9.770e-15 | 9.465e-15 |
| fixed128 | 0 | 9.770e-15 | 8.660e-15 |
| global128 | 0 | 9.770e-15 | 8.660e-15 |
| region32 | 0 | 1.843e-14 | 3.308e-14 |
| global32 | 0 | 9.770e-15 | 1.421e-14 |
| native | 0 | 1.893e-14 | 2.004e-14 |
| qmc64_native | 0 | 2.420e-14 | 1.488e-14 |
| nested128 | 0 | 9.770e-15 | 8.660e-15 |
| nested512 | 0 | 9.770e-15 | 8.660e-15 |
| query | 0 | 1.893e-14 | 2.004e-14 |
| probe | 0 | 6.717e-15 | 9.992e-15 |

![Decision loss, compute, and Top-1 stability](results/tradeoff.png)

Figure는 point estimates다. Uncertainty는 위 paired CI 표에서 확인한다. Ranking column의 deterministic은 같은 bank에서 acquisition randomness가 없는 policy, not audited는 이번 seed-repeat audit에 포함하지 않은 method를 뜻한다.

## 7. 해석과 논문 기여 가능성

### 7.1 Early stopping은 작동했지만 전체 compute는 줄지 않았다

Contact-event의 평균 최종 K는 27.0으로 cap128보다 훨씬 작았다. 두 replicas와 모든 stages에서 처리한 components의 평균 합도 약 184개였다. 그러나 stages 사이에 compression, likelihood evaluation, risk comparison을 반복하므로 최종 K가 작다는 사실만으로 latency 절약이 보장되지는 않는다.

Stick-event에서는 평균 최종 K가 105.23, 평균 stages가 4.49였다. 총 처리한 components는 평균 약 810개로 fixed128의 최대 512개보다 많다. Sampling-error term이 action gap보다 큰 상태에서 representation만 확장하면 비용을 쓰고도 stopping 조건을 만족시키지 못한다.

Original adaptive GPU median은 15.81 ms였다. Region indices/CDF를 재사용하고 logging용 GPU scalar의 CPU 변환을 줄인 equivalent implementation에서는 14.58 ms였다. 세 cells × 512 worlds × 2 tasks × 5 mixture methods, **15,360 cases 전체에서 actions, stopping metadata와 histories가 일치했고 maximum risk difference는 0**이었다. 최적화로 policy를 바꾸거나 좋은 결과를 다시 선택하지 않았다. Baselines도 같은 새 timing pass에서 다시 측정했다.

이러한 비용은 최소 가능한 algorithmic lower bound가 아니다. Batched stages, asynchronous stopping 또는 GPU-specific fusion을 더 개발할 여지는 있다. 다만 현재의 measured implementation에서 adaptive speed advantage는 없으며, “components 수가 줄었으니 compute도 절약했다”는 주장은 성립하지 않는다. Exact kernel-level profiling으로 원인을 완전히 분해한 것은 아니다.

### 7.2 Acquisition error의 한 종류만 줄인다는 한계

이 실험의 numerical error는 적어도 세 종류다. 첫째는 compact inner mixture가 full posterior predictive response를 얼마나 잘 표현하는지에 따른 compression error다. 둘째는 hypothetical outcomes를 512개로 적분하는 outer integration error다. 셋째는 finite QMC factual posterior와 contact model 자체의 error다.

K를 늘리는 것은 주로 첫 번째를 겨냥한다. 두 번째는 고정된 상태이며 세 번째 역시 변화하지 않는다. 따라서 components를 늘릴수록 모든 error가 줄어든다는 전제가 맞지 않는다. Matched stick의 357 cap-unresolved cases 중 278개에서 sampling term만으로 gap 이상이었던 결과는 이 한계를 직접 보여 준다. 단, heuristic SE의 크기만으로 true error의 인과적 비중을 정확히 추정했다고 주장하지는 않는다.

Coarse stages에서 gap이 충분히 큰 cases는 일찍 종료하고, ambiguous cases는 cap까지 간다는 동작 자체는 구현됐다. 첫 64 matched worlds 중 gap-separated인 contact 61개, stick 15개는 모두 nested2048와 action이 같았다. 그러나 이는 작은 diagnostic subset이며 valid confidence coverage나 false-certification probability의 증명이 아니다.

### 7.3 안정성과 정확성은 다른 평가다

Adaptive region의 first64 matched stick reference agreement는 68.75%, nested512는 82.81%였다. 반면 independent acquisition seed 사이에서는 adaptive가 77.54%, nested512가 76.56%로 가까웠다. Reference agreement, self-repeatability, actual decision loss는 서로 다른 질문에 답한다.

Adaptive global이 더 stable하다는 수치도 주의해서 읽어야 한다. Global compression은 rare label region의 mass를 정확히 유지하지 않는다. 이 때문에 compact representation에서 prediction이 거의 constant가 되면 actions가 동률이 되고 query view로 결정될 수 있다. Section 5의 post-analysis label-support audit에서 matched stick의 global risk ties 210개 중 209개는 full factual posterior에 두 regions가 있는데 두 compact replicas가 single-label이 된 경우였다. 이 subset의 minority mass median은 약 0.000656, maximum은 약 0.06044였다. Contact에서는 41개 ties 중 39개가 해당했고 minority mass maximum은 약 0.13156이었다. 높은 agreement를 정확한 uncertainty representation으로 곧바로 해석하지 않는다.

동시에 global methods의 actual losses가 adaptive region보다 명확히 나쁘다는 근거도 없었다. Region mass를 보존한다는 구조적 장점이 실제 decision loss를 줄이는 충분한 이유는 아니다. 현재 결과는 global mixture에 대한 독자적 superiority를 지지하지 않는다.

Independent-bank audit에서도 stick Top-1 agreement는 adaptive 85.94%, global128 87.50%, LMMSE16 90.63%, nested512 81.25%였다. 이 audit는 64 worlds에 한정되며 full multi-seed generalization을 보장하지 않는다. Bank가 바뀌어도 같은 action을 택하는 LMMSE가 항상 더 정확하다는 뜻도 아니다.

### 7.4 Model mismatch와 실제 검증 범위

Shifted data/nominal inference의 stick loss는 adaptive 0.077189, native 0.077388, nested512 0.077818이었다. Correct-law diagnostic에서는 0.068911, 0.068037, 0.069161이었다. Model을 올바르게 제공해도 adaptive region이 LMMSE를 일관되게 이기는 것은 아니다.

현재 연구는 noisy boundary profiles와 local impulse surrogate를 사용한다. Raw image의 ambiguity, geometry reconstruction error, full collision trajectories와 실제 영상의 domain gap을 이 실험에서 검증하지 않았다. Factual bank도 계속 필요하다. 따라서 이 결과를 real-world visual active perception이나 bank-free scene belief의 성과로 확대하지 않는다.

### 7.5 관련 연구와 novelty 판단

Dinkel 등의 2026 preprint **Efficient Bayesian Optimal Experimental Design for Expensive Computational Models over Finite Design Sets**는 design 간 EIG 차이에 common random numbers와 Rao–Blackwellization을 적용하고, paired bootstrap comparisons에 따라 designs를 제거하거나 sample budget을 늘린다. Sections 2.3–2.4와 Algorithm 1을 확인했다. V17은 EIG 대신 event Brier risk를 사용하고 inner conditional-mixture size를 조절한다는 차이가 있다. 그러나 action 간 차이가 불확실할 때 계산을 더 쓴다는 원리 자체는 새롭지 않다. 해당 ADE algorithm을 이 physics setting에서 재현해 비교한 것은 아니다. [원문](https://arxiv.org/html/2607.16933v1)

Zhu와 Hainy의 2026 preprint **Sequential Bayesian Experimental Design for Prediction in Physical Experiments Informed by Computer Models**는 predictive MI 계산에 Gaussian mixture compression을 사용한다. Whitening, k-means moment matching과 KNN-guided Runnalls reduction을 제시한다. V17의 label-conditioned systematic representatives와 gap-based expansion은 구성상 다르지만, predictive mixture를 줄여 acquisition 비용을 낮춘다는 큰 아이디어 역시 선행연구와 겹친다. Generic GMM compression에 대한 직접적인 superiority는 아직 검증하지 않았다. [원문, Section 4.1](https://arxiv.org/html/2603.16756v1#S4.SS1)

Garivier와 Kaufmann의 **Optimal Best Arm Identification with Fixed Confidence**는 Track-and-Stop의 sampling/stopping strategy와 asymptotic optimality를 제시한다. V17의 sequential heuristic은 해당 theoretical guarantees를 따르지 않는다. Biased nested approximation, dependent QMC samples와 compression error를 가진 여기의 설정에 그 보장을 그대로 적용할 수 없다. [원문 출판 페이지](https://proceedings.mlr.press/v49/garivier16a.html)

따라서 “adaptive compression을 넣었다”만으로는 method novelty가 약하다. 기존 adaptive design 및 mixture reduction과 비교했을 때, representation error와 integration error를 구별하여 필요한 계산만 늘리고, 실제 downstream risk를 보존한다는 추가 근거가 있어야 한다.

### 7.6 다음 단계의 우선순위

현재 데이터를 보고 cap만 높이거나 threshold를 낮추는 방식으로 성공 여부를 바꾸지 않는다. 다음 가설을 평가한다면 새로운 protocol과 worlds를 사용해야 한다.

1. **Error-source-aware budget allocation:** Compression drift나 replica discrepancy가 큰 경우 K를, randomized outer integration variation이 큰 경우 outer sample count를 늘린다. Posterior-bank discrepancy가 지배적이면 acquisition integration을 더 해도 해결되지 않으므로 별도로 다룬다. 이 분기는 아직 구현·검증한 방법이 아니다.
2. **Exact ranking보다 bounded decision regret:** Actions가 사실상 동등하면 ranking을 끝까지 분리하는 데 비용을 쓰지 않는 stopping을 고려한다. Allowed risk tolerance는 test 결과가 아니라 application cost에서 정해야 한다. 이 역시 새로운 원리라고 주장하기 전에 best-arm/good-arm identification 및 adaptive OED와의 차이를 확인해야 한다.
3. **강한 fixed baseline을 먼저 넘는 검증:** Fixed32/128, global compression, nested128, task-specific query/probe rules를 동시에 비교한다. Loss와 actual latency의 Pareto frontier를 개선하지 못하면 adaptive method claim을 중단한다. 실제 optimized CPU median은 adaptive 6.52 ms, nested512 89.63 ms로 약 13.8× 빠르다. 그러나 fixed128은 3.49 ms, global128은 3.42 ms로 더 저렴하다. CPU에서 nested512 대비 빨라도 더 싼 fixed mixture가 같은 loss를 내면 adaptivity의 필요성은 입증되지 않는다.
4. **실제 geometry/observation으로의 연결:** 여러 hidden contact modes와 rare but decision-relevant regions가 실제 시각적 관측에서도 나타나는지 확인한다. Synthetic corner 사례만으로 region stratification의 일반적 가치를 주장하지 않는다.

이번 실험의 가장 명확한 finding은 **“acquisition uncertainty가 남았다는 이유만으로 components를 늘리는 것은 충분하지 않으며, 고정된 outer integration error가 병목이면 더 큰 compact mixture가 오히려 비용만 늘릴 수 있다”**이다. 유용한 failure analysis이지만, 현재 adaptive rule 자체의 우수성을 입증한 결과는 아니다.


## 8. 실행·재현·무결성

서버 경로: `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/adaptive_acquisition_v17`. Allocation2680491, node56의 idle GPU를 확인한 뒤 guard 아래 실행했다. 기존 사용자 jobs나 allocation은 변경하지 않았다. Original GPU worker와 optimized-equivalence worker를 순차 실행했다. 각 GPU worker 종료 후 해당 CPU timing을 측정하고 statistics를 완료한 뒤 결과를 seal했다.

핵심 파일은 `PROTOCOL.md`, `adaptive.py`, `adaptive_fast.py`, `run_policy.py`, `runtime_fast.py`다. Policies와 expansion histories는 `results/policies`, independent-seed/bank/reference audits는 `results/audits`, actual predictions는 `results/evaluations`에 있다. `results/summary.json`, `results/world_losses.npz`, `results/decision_actions.npz`에서 결과를 재분석할 수 있다.

`validate.py`는 aliasing/higher-order counterexamples, label mass, numerical ties와 optimized LMMSE equivalence를 점검했다. `seal.py`는 policy sources와 frozen files, actual losses/probabilities, stopping reason, negative controls, GPU/CPU output equivalence, V16 sealed files 보존 여부를 검증한다. `INTEGRITY.json`과 `MANIFEST.json`에 검증 내역과 SHA256 hashes를 저장했다.

재실행은 sealed directory를 덮어쓰지 않고 새 directory에서 수행한다. Allocated idle GPU에서 `guard.py run.py`, GPU 완료 후 `timing.py --cpu`, 이어서 idle GPU에서 `guard.py run_fast.py`, GPU 완료 후 `timing_fast.py --cpu`, `analyze.py`, `diagnostics.py`, `plot_results.py`, `write_report.py`, `seal.py` 순서다. Seal 전에 main/fast worker exit records를 각각 gpu_worker_exit.json과 fast_worker_exit.json에 보존한다. CPU scripts는 `BELIEF_DEVICE=cpu`와 빈 `CUDA_VISIBLE_DEVICES`를 지정한다. Report 생성에는 `REPORT_CONCLUSION_KO.md`, `REPORT_METHODS_KO.md`, `REPORT_DISCUSSION_KO.md`가 필요하다.
