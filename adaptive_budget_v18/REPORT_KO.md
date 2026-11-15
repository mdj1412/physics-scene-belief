# Adaptive allocation of compression and outer integration budgets — v18

작성일: 2026-11-13 KST. Server implementation, fresh-world experiments, independent audit와 GPU/CPU online timing을 수행했다.

## 1. 핵심 결론

**Compression error와 outer integration error를 따로 추정해 K와 M을 늘리는 adaptive budget allocation을 구현하고, 서버에서 512개의 새로운 worlds × 16 observation noise repeats로 검증했다. Component-only의 ranking 불안정성은 줄었지만, strong fixed mixture를 넘어서는 decision-loss–compute 우위는 입증하지 못했다.**

Matched control의 stick-event에서 independent acquisition-seed Top-1 agreement는 component_only 76.37%에서 adaptive 82.62%로 6.25 percentage points 증가했다. Paired secondary95% CI는 [+2.34,+10.16] percentage points다. Outer samples를 고정하고 components만 늘리는 방법에 남던 문제가 실제로 줄어든 것이다. 다만 이 CI가 +5 percentage points를 모두 넘는 것은 아니며, 사전 screen의 +5 기준은 point estimate 기준이다.

Adaptive의 stick Brier loss는 0.071078, fixed128/M4096은 0.071307, global128/M4096은 0.070957, nested2048은 0.071242였다. Primary adjusted CI에서는 이 strong baselines 대비 loss 감소가 확인되지 않았다. Global mixture는 Top-1 agreement도 90.82%로 더 높았다. 따라서 region preservation과 두 error proxies의 사용 자체가 우수한 acquisition을 보장한다는 주장은 지지되지 않는다.

GPU online median을 두 tasks에 걸쳐 모으면 adaptive 7.87ms, fixed128/M4096 6.03ms, global128/M4096 5.87ms, nested2048 27.86ms다. Nested 대비 3.54× speedup과 사전 loss-ratio tolerance를 만족하여 approximation screen은 통과했다. 그러나 adaptive는 fixed128/M4096보다 30.5% 느려 adaptivity screen은 실패했다. **어려운 stick task만 보면 adaptive 18.52ms 대 nested2048 28.28ms로 speedup은 1.53×이며, fixed128/M4096 6.14ms보다 약 3.02× 느리다.** Aggregate median만으로 어려운 cases의 효율을 설명하면 과장된다.

CPU에서는 early stopping이 도움이 됐다. Contact median은 adaptive 4.08ms 대 fixed128/M4096 50.09ms, stick은 43.64ms 대 50.81ms였다. 그러나 작은 fixed128/M512는 stick 6.04ms이며, 그 baseline 대비 adaptive의 loss 감소도 primary CI에서 확인되지 않았다. CPU 결과도 strong fixed-budget frontier를 넘었다는 충분한 증거는 아니다.

이번 결과의 가장 분명한 가치는 **compression refinement만으로 해결되지 않는 outer-integration 불확실성을 드러내고, 두 축의 refinement가 어디까지 도움을 주는지와 실제 latency가 왜 따라오지 않는지 검증한 것**이다. 현재 버전을 독립적인 새 method의 완성으로 보기는 어렵다. 아래에 positive evidence, counterevidence, 구현·검증 자료와 선행연구 대비 주장 가능한 범위를 모두 남겼다.


## 2. Adaptive budget allocation의 구현

K는 label region당 representative 수, M=4m은 전체 hypothetical sample 수다. K는 16→32→64→128, m은 128→256→512→1024로 늘어난다. K/2 comparison을 위해 K8 prefix도 유지한다. 두 label regions가 모두 존재하면 replica당 components는 2K개이며, compression replicas는 2개다.

각 region의 factual posterior mass를 유지하면서 nested scrambled-Sobol quantiles로 representatives를 얻는다. K를 늘릴 때 기존 representatives가 prefix에 남는다. Original sensor Gaussian kernels와 event labels를 그대로 사용한다. Posterior mass≤10⁻¹⁰인 region만 pruning하고 renormalize하며 omitted mass를 기록한다.

Independent outer QMC16384 posterior에서 4개의 독립적인 randomized QMC scrambles로 hypothetical world–observation pairs를 만든다. 하나의 scramble 안에서는 m개의 samples가 서로 독립인 것은 아니다. Error estimate에는 각 scramble의 평균을 사용한다. Hypothetical labels는 candidate posterior에서 나온 값이며 실제 hidden world의 target이 아니다.

현재 risk가 가장 작은 action을 b라 하고 competitor a와의 gap을 gₐ라고 하면 다음 error proxies를 계산한다.

\[
g_a=\hat R_a-\hat R_b,
\qquad e_{C,a}=|g_a(K,M)-g_a(K/2,M)|+\tfrac12|g_a^{(1)}-g_a^{(2)}|,
\]
\[
e_{O,a}=2\,\mathrm{SD}\{g_a^{[1]},\ldots,g_a^{[4]}\}/\sqrt4.
\]

Superscript (1)/(2)는 compression replica, [1]…[4]는 outer scramble이다. Risk는 두 compact decoders가 independent outer targets에 대해 내는 Brier loss의 평균이다. 두 predictions를 평균한 ensemble decoder의 risk와는 다르다. K/2와 K의 비교는 같은 outer samples를 사용한다.

모든 competitor에서 gₐ>eC,ₐ+eO,ₐ이면 `gap-separated`로 멈춘다. 그렇지 않으면 gₐ−eC,ₐ−eO,ₐ가 가장 작은 competitor를 기준으로 더 큰 error proxy의 축을 두 배로 늘린다. 그 축이 cap이면 다른 축을 늘리고, 두 caps에 도달하면 `cap-unresolved`로 기록한다. 모든 action risks의 range가 10⁻¹² 이하면 `risk-tie`로 멈추고 query view를 선택한다.

**이 error proxies는 formal confidence bounds가 아니다.** Compression drift와 두 replicas가 shared bias를 놓칠 수 있고, 4개 outer randomizations는 variance estimate에 충분히 큰 표본이라고 보장할 수 없다. Factor2가 adaptive stopping의 95% coverage를 뜻하지도 않는다. Model mismatch와 finite factual-bank error는 별도로 남는다.

Risk-tie도 물리적으로 모든 observations가 무용하다는 증명은 아니다. 특히 global resampling은 minority label support를 놓쳐 constant prediction을 만들 수 있다. Primary region mixtures는 label mass를 보존하지만, 그 구조적 성질만으로 실제 decision loss 우위가 보장되지는 않는다.

### Cached implementation과 strong fixed baseline

Likelihoods의 class-wise log sums를 cache한다. K가 늘면 새 components와 기존 hypothetical samples의 조합만, M이 늘면 새 hypothetical samples와 현재 components의 조합만 평가한다. 따라서 한 실행에서 평가한 component–observation combinations는 최종 K×M rectangle과 일치한다. 중복 계산을 매번 새로 시작하지 않는다.

Adaptive engine은 최대 K의 representatives와 최대 M의 hypothetical coordinates를 미리 준비한다. Adaptive K/M은 실제 likelihood evaluation prefix를 정한다. 따라서 sample preparation과 representation memory가 최종 선택된 작은 budget에 비례해서 모두 줄었다고 주장하지 않는다. 이 preparation, cache allocation과 stopping checks를 online timing에 포함한다.

Original kernel은 logcumsumexp를, equivalent optimized kernel은 block logsumexp와 logaddexp를 사용한다. Fixed mixtures는 불필요한 adaptive prefix/error checks 없이 direct risk kernel을 사용한다. 전체 primary policies에 대해 action/risk equivalence와 adaptive stopping histories를 검증한 뒤 같은 optimized timing pass에서 비교한다. Original GPU timing도 보존한다.

## 3. Comparators와 protocol

| Method | 변경하는 budget / 역할 |
|---|---|
| adaptive | Error proxy에 따라 K 또는 M 선택 |
| balanced | 불확실하면 가능한 두 축을 함께 증가 |
| component_only | M512 고정, K만 증가 |
| outer_only | K128 고정, M만 증가 |
| fixed32_512 / fixed128_512 | Small outer budget의 strong fixed mixtures |
| fixed32_4096 / fixed128_4096 | Large outer budget의 strong fixed mixtures |
| global32_512 / global128_4096 | 같은 total components의 global resampling |
| native / qmc64_native | Optimized LMMSE16 / LMMSE64 |
| nested512 / nested2048 / nested4096 | 같은 4 outer scrambles를 사용하는 full-inner QMC16384 risk |
| query / probe / factual / none | Fixed rules와 no-observation diagnostic |

새로운 **512 iid worlds**(seed7100101+137×index), world당 **16 actual observation noise repeats**를 사용한다. Correctness pilot8은 seed7000101이다. Primary tasks는 contact-event와 stick-event다. 저장된 task 이름 `slip_event`는 slip/stick transition의 stick label을 평가한다. V17과 worlds, representative sequence, outer estimator가 달라졌으므로 V17의 평균과 직접 비교해 개선율을 만들지 않는다. Ablations는 V18 안에서 같은 estimator를 사용한다.

Physical model은 V17과 같은 local impulse surrogate다. Boundary x(y)=d+s y+a f_g(y), L=25mm, f₀=L(y/L)²/2, f₁=L[(y/L)²/2−0.35(y/L)⁴]를 쓴다. Geometry2 × base contact laws2의 네 components에 prior mass 각각1/8, geometry2 × flexible law의 두 components에 각각1/4를 둔다. Parameters는 d∈[−4,4]mm, s∈[−0.4,0.4], a∈[−0.1,0.1], restitution e∈[0.35,0.9], b₀∈[−4,2], b₁∈[−2,2]의 uniform priors다.

Factual evidence는 y=0, incoming velocity(1,0.35)의 outgoing velocity와 −10…10mm boundary profiles9개다. Noise SD는 각각0.07m/s와1.2mm다. Public factual profile extrapolation으로 y=25mm의 contact threshold를 만들며, public slope estimate로 slip/stick query direction을 정한다. Stick label은 residual tangent absolute≤10⁻⁸이다.

Candidate observations는 query view(19…31mm, profiles7, SD1mm), factual view(−8…8mm, profiles7, SD0.35mm), probe(y=0, incoming(1,−0.7), outgoing velocity SD0.05m/s)다. Selector는 하나를 선택하며 sensing cost는 같다고 가정한다. 실제 observation 이후의 decoder는 공통 full QMC16384이다. LMMSE64만 QMC65536 decoder를 사용한다.

Matched nominal, shifted data/nominal inference, shifted data/correct-law inference를 평가한다. Shift는 flexible law의 speed dependence를 바꾸며 base laws는 그대로다. Correct-law는 privileged diagnostic이다. Nominal tangential impulse는 vₜ sigmoid[b₀+b₁ log((|vₜ|+0.02)/0.5)], shifted impulse는 vₜ sigmoid(b₀)[0.2+0.8/(1+(|vₜ|/s₀)²)], s₀=0.1+0.7 sigmoid(b₁)다.

Primary banks는 inner2026111201, outer2026111202, QMC64 2026111203이다. Primary master seed91531을 고정했다. Main policies를 freeze한 뒤 private outcomes를 평가했다. Actual target, hidden parameter, unselected actual observation은 policy 입력으로 사용하지 않는다.

Primary12 contrasts는 adaptive 대 fixed128_512/fixed128_4096/global128_4096/nested2048/balanced/task-specific fixed rule을 두 tasks에서 비교한다. World별 repeats를 먼저 평균한 뒤 30,000 paired bootstrap과 Bonferroni99.58333% CI를 사용한다. 나머지는 secondary95% CI다.

Approximation screen은 두 task 모두에서 nested2048 대비 mean-loss ratio의 95% upper≤1.05와 GPU median speedup≥2다. Adaptivity screen은 stick repeatability가 component_only보다 5 percentage points 이상 높고, 두 tasks의 fixed128_4096 대비 ratio upper≤1.05이며 GPU median이20% 이상 낮은지를 본다. Contact에는 ceiling 때문에 별도 agreement-gain threshold를 요구하지 않는다. 5%/20%는 engineering thresholds이며 application-derived costs가 아니다.

### Independent audit의 수정과 error-source 검증

Code review에서 original alternative master92531의 outer stream이 primary representative seed92531과 겹치는 것을 발견했다. 원래 audit는 보존했고, 결과를 살펴보기 전에 disjoint master101531로 secondary audit를 교체하기로 기록했다. Primary policy, data와 thresholds는 바꾸지 않았다. 자세한 기록은 `AUDIT_CORRECTION.md`다.

최종 repeatability는 corrected master101531의 512 matched worlds와 primary를 비교한다. Supplemental script는 public evidence와 primary stopping budgets만 읽고, private outcomes를 읽지 않으며 결과를 별도로 freeze한다. 두 runs에서는 representatives와 outer samples가 함께 달라진다.

각 cell의 첫64 worlds에는 별도 master93531의 nested8192 reference를 사용한다. 이는 exact Bayes oracle이 아니다. 같은 hypothetical samples에서의 full-inner risk와 compact risk 차이를 통해 compression proxy를 점검한다. 첫64 matched worlds에서는 primary representatives를 고정하고 corrected outer scrambles만 바꾸어 outer proxy를 점검한다.

저장된 worst-gap 값으로 competitor를 유일하게 식별할 수 없는 near-tie cases는 proxy audit에서 제외하고 그 수를 기록한다. Outer audit는 두 independent estimates의 차이를 보는 것으로, true integration error의 직접 측정이나 nominal coverage test가 아니다. Ranking은 주로 Top-1 agreement이며 full ordering stability 전체를 의미하지 않는다.

이 실험은 raw RGB, full collision trajectories 또는 real robot을 검증하지 않는다. Factual banks와 cached candidate forward predictions도 계속 필요하다. 따라서 bank-free belief나 end-to-end visual physics inference의 성능으로 확대 해석하지 않는다.


## 4. Decision loss

Brier loss이며 낮을수록 좋다. 각 cell은 같은 512 worlds의 paired conditions이고 world당16 repeats를 먼저 평균했다.

### matched_nominal

| Method | contact_event | slip_event |
|---|---|---|
| adaptive | 0.048176 | 0.071078 |
| balanced | 0.048176 | 0.071417 |
| component_only | 0.048113 | 0.072088 |
| outer_only | 0.048176 | 0.072230 |
| fixed32_512 | 0.048112 | 0.074751 |
| fixed128_512 | 0.048113 | 0.072080 |
| fixed32_4096 | 0.048176 | 0.071840 |
| fixed128_4096 | 0.048176 | 0.071307 |
| global32_512 | 0.048116 | 0.071798 |
| global128_4096 | 0.048176 | 0.070957 |
| native | 0.048176 | 0.077228 |
| qmc64_native | 0.046066 | 0.077706 |
| nested512 | 0.048179 | 0.072331 |
| nested2048 | 0.048176 | 0.071242 |
| nested4096 | 0.048176 | 0.071105 |
| query | 0.048176 | 0.087303 |
| probe | 0.130483 | 0.072715 |
| factual | 0.096659 | 0.090007 |
| none | 0.183904 | 0.097282 |

### shifted_nominal

| Method | contact_event | slip_event |
|---|---|---|
| adaptive | 0.047877 | 0.068577 |
| balanced | 0.047877 | 0.068576 |
| component_only | 0.047814 | 0.068365 |
| outer_only | 0.047877 | 0.068467 |
| fixed32_512 | 0.047813 | 0.068754 |
| fixed128_512 | 0.047814 | 0.068358 |
| fixed32_4096 | 0.047877 | 0.069463 |
| fixed128_4096 | 0.047877 | 0.068466 |
| global32_512 | 0.047817 | 0.067305 |
| global128_4096 | 0.047877 | 0.067193 |
| native | 0.047876 | 0.073187 |
| qmc64_native | 0.046368 | 0.073719 |
| nested512 | 0.047880 | 0.067338 |
| nested2048 | 0.047877 | 0.067621 |
| nested4096 | 0.047877 | 0.067227 |
| query | 0.047877 | 0.082787 |
| probe | 0.119723 | 0.067616 |
| factual | 0.095780 | 0.086498 |
| none | 0.178791 | 0.093339 |

### shifted_correct

| Method | contact_event | slip_event |
|---|---|---|
| adaptive | 0.047518 | 0.063622 |
| balanced | 0.047518 | 0.063622 |
| component_only | 0.047534 | 0.064253 |
| outer_only | 0.047518 | 0.063618 |
| fixed32_512 | 0.047529 | 0.065613 |
| fixed128_512 | 0.047534 | 0.064249 |
| fixed32_4096 | 0.047518 | 0.064797 |
| fixed128_4096 | 0.047518 | 0.063618 |
| global32_512 | 0.047529 | 0.065412 |
| global128_4096 | 0.047518 | 0.065053 |
| native | 0.047542 | 0.065243 |
| qmc64_native | 0.046226 | 0.065537 |
| nested512 | 0.047518 | 0.064139 |
| nested2048 | 0.047518 | 0.063799 |
| nested4096 | 0.047518 | 0.063609 |
| query | 0.047518 | 0.081643 |
| probe | 0.119268 | 0.066387 |
| factual | 0.095240 | 0.086095 |
| none | 0.178142 | 0.092482 |

### Primary contrasts

Adaptive minus baseline이다. Negative difference는 adaptive가 유리하다. CI는12 comparisons에 대한 adjusted99.58333%다. Numerical precision에 가까운 차이를 실질적인 효과로 해석하지 않는다.

| Task / comparison | Difference [adjusted CI] |
|---|---|
| contact_event/adaptive-fixed128_512 | +0.000063 [-2.829e-12, +0.000317] |
| contact_event/adaptive-fixed128_4096 | -1.773e-22 [-8.867e-22, +0.000000] |
| contact_event/adaptive-global128_4096 | +1.339e-09 [+0.000000, +4.659e-09] |
| contact_event/adaptive-nested2048 | +1.326e-10 [-7.093e-22, +6.630e-10] |
| contact_event/adaptive-balanced | +0.000000 [+0.000000, +0.000000] |
| contact_event/adaptive-query | +1.472e-09 [+1.469e-13, +4.889e-09] |
| slip_event/adaptive-fixed128_512 | -0.001002 [-0.005019, +0.001988] |
| slip_event/adaptive-fixed128_4096 | -0.000229 [-0.001592, +0.000424] |
| slip_event/adaptive-global128_4096 | +0.000121 [-0.004112, +0.004158] |
| slip_event/adaptive-nested2048 | -0.000164 [-0.004438, +0.003847] |
| slip_event/adaptive-balanced | -0.000339 [-0.001694, +1.591e-16] |
| slip_event/adaptive-probe | -0.001637 [-0.006948, +0.003351] |

### 사전 screens

| Task / adaptive-to-baseline | Mean-loss ratio [95% CI] |
|---|---|
| contact_event/nested2048 | 1.0000 [1.0000, 1.0000] |
| contact_event/fixed128_4096 | 1.0000 [1.0000, 1.0000] |
| slip_event/nested2048 | 0.9977 [0.9589, 1.0360] |
| slip_event/fixed128_4096 | 0.9968 [0.9850, 1.0041] |

Screen 결과: `{'nested_loss': True, 'nested_speedup': 3.539331839771331, 'approximation_pass': True, 'fixed_loss': True, 'fixed_time_saving': -0.30480849097645124, 'stick_repeatability_plus5pp': True, 'adaptivity_pass': False}`. Superiority 미검출을 equivalence로 해석하지 않는다.

### Ablations와 저비용 baselines: secondary95% CI

| Task / adaptive minus baseline | Difference [95% CI] |
|---|---|
| contact_event/component_only | +0.000063 [-1.415e-12, +0.000190] |
| contact_event/outer_only | +0.000000 [+0.000000, +0.000000] |
| contact_event/fixed32_512 | +0.000064 [+0.000000, +0.000193] |
| contact_event/fixed32_4096 | +6.547e-10 [-5.320e-22, +1.964e-09] |
| contact_event/native | +1.472e-09 [+1.418e-11, +3.504e-09] |
| contact_event/qmc64_native | +0.002110 [-0.000434, +0.004700] |
| contact_event/nested512 | -0.000003 [-0.000009, +2.011e-09] |
| contact_event/nested4096 | +8.003e-19 [-5.320e-22, +2.401e-18] |
| slip_event/component_only | -0.001010 [-0.003659, +0.001094] |
| slip_event/outer_only | -0.001152 [-0.003448, +0.000212] |
| slip_event/fixed32_512 | -0.003673 [-0.007291, -0.000571] |
| slip_event/fixed32_4096 | -0.000762 [-0.002619, +0.000999] |
| slip_event/native | -0.006150 [-0.012459, -0.000339] |
| slip_event/qmc64_native | -0.006628 [-0.013090, -0.000415] |
| slip_event/nested512 | -0.001253 [-0.004484, +0.001830] |
| slip_event/nested4096 | -0.000028 [-0.002768, +0.002684] |

## 5. Ranking 안정성과 budget allocation

### Corrected independent acquisition audit: 512 matched worlds

Primary master91531 대 disjoint master101531의 Top-1 agreement다. Seed2 loss delta는 같은 primary actual decoder에서 seed2 selection minus primary selection이다.

| Method | Contact agreement | Stick agreement | Contact loss delta [95% CI] | Stick loss delta [95% CI] |
|---|---|---|---|---|
| adaptive | 99.41% | 82.62% | -7.873e-10 [-2.229e-09, +0.000000] | -0.000201 [-0.002991, +0.002668] |
| balanced | 99.41% | 83.40% | -7.873e-10 [-2.229e-09, +0.000000] | -0.001194 [-0.003418, +0.000401] |
| component_only | 99.22% | 76.37% | +0.000063 [-1.964e-09, +0.000190] | -0.000360 [-0.003019, +0.001939] |
| outer_only | 99.41% | 82.81% | -7.873e-10 [-2.229e-09, +0.000000] | -0.000983 [-0.003226, +0.000611] |
| fixed32_512 | 98.44% | 77.93% | +0.000065 [-1.631e-11, +0.000195] | -0.002152 [-0.004888, +0.000160] |
| fixed128_512 | 99.22% | 75.78% | +0.000063 [-1.964e-09, +0.000190] | -0.000247 [-0.002889, +0.002051] |
| fixed32_4096 | 99.02% | 80.66% | +6.465e-10 [-2.425e-11, +1.964e-09] | +0.001400 [-0.001144, +0.004597] |
| fixed128_4096 | 99.41% | 83.20% | -7.873e-10 [-2.229e-09, +0.000000] | -0.000983 [-0.003226, +0.000611] |
| global32_512 | 99.02% | 83.59% | +0.000060 [-0.000009, +0.000190] | +0.001489 [-0.000643, +0.003716] |
| global128_4096 | 99.61% | 90.82% | +1.325e-09 [+0.000000, +3.321e-09] | -0.000496 [-0.002856, +0.001287] |
| nested512 | 98.83% | 76.37% | -0.000003 [-0.000009, +1.341e-09] | -0.000387 [-0.002256, +0.001455] |
| nested2048 | 99.02% | 84.57% | -6.540e-10 [-1.964e-09, +2.122e-12] | -0.001390 [-0.003499, -0.000197] |
| nested4096 | 99.41% | 88.87% | -7.873e-10 [-2.229e-09, +0.000000] | -0.001015 [-0.002984, +0.000020] |

Component-only 대비 agreement 증가량: {'contact_event': {'mean': 0.195, 'low': -0.586, 'high': 0.977}, 'slip_event': {'mean': 6.25, 'low': 2.344, 'high': 10.156}} percentage points. CI는 secondary95%다.

### Independent nested8192 reference: 첫64 worlds/cell

Reference agreement와 reference risk regret는 finite-bank numerical diagnostics이며 exact Bayes regret가 아니다.

| Cell / method | Contact agreement | Stick agreement | Contact reference regret | Stick reference regret |
|---|---|---|---|---|
| matched_nominal/adaptive | 96.88% | 65.62% | 0.000001 | 0.000052 |
| matched_nominal/balanced | 96.88% | 64.06% | 0.000001 | 0.000076 |
| matched_nominal/component_only | 96.88% | 67.19% | 0.000001 | 0.000177 |
| matched_nominal/outer_only | 96.88% | 65.62% | 0.000001 | 0.000052 |
| matched_nominal/fixed128_4096 | 96.88% | 65.62% | 0.000001 | 0.000076 |
| matched_nominal/global128_4096 | 98.44% | 68.75% | 0.000000 | 0.000034 |
| matched_nominal/native | 98.44% | 53.12% | 0.000000 | 0.004161 |
| matched_nominal/nested2048 | 98.44% | 76.56% | 0.000001 | 0.000024 |
| matched_nominal/nested4096 | 98.44% | 76.56% | 0.000001 | 0.000013 |
| shifted_nominal/adaptive | 96.88% | 59.38% | 0.000001 | 0.000109 |
| shifted_nominal/balanced | 96.88% | 57.81% | 0.000001 | 0.000134 |
| shifted_nominal/component_only | 96.88% | 62.50% | 0.000001 | 0.000128 |
| shifted_nominal/outer_only | 96.88% | 59.38% | 0.000001 | 0.000109 |
| shifted_nominal/fixed128_4096 | 96.88% | 59.38% | 0.000001 | 0.000134 |
| shifted_nominal/global128_4096 | 98.44% | 57.81% | 0.000000 | 0.000066 |
| shifted_nominal/native | 98.44% | 60.94% | 0.000000 | 0.003910 |
| shifted_nominal/nested2048 | 98.44% | 76.56% | 0.000001 | 0.000061 |
| shifted_nominal/nested4096 | 98.44% | 79.69% | 0.000001 | 0.000017 |
| shifted_correct/adaptive | 96.88% | 65.62% | 0.000000 | 0.000065 |
| shifted_correct/balanced | 96.88% | 67.19% | 0.000000 | 0.000065 |
| shifted_correct/component_only | 96.88% | 57.81% | 0.000000 | 0.000345 |
| shifted_correct/outer_only | 96.88% | 65.62% | 0.000000 | 0.000065 |
| shifted_correct/fixed128_4096 | 96.88% | 67.19% | 0.000000 | 0.000065 |
| shifted_correct/global128_4096 | 98.44% | 62.50% | 0.000000 | 0.000077 |
| shifted_correct/native | 95.31% | 64.06% | 0.001188 | 0.000414 |
| shifted_correct/nested2048 | 98.44% | 78.12% | 0.000000 | 0.000038 |
| shifted_correct/nested4096 | 98.44% | 84.38% | 0.000000 | 0.000025 |

### Final budgets와 unresolved cases

K counts는 [16,32,64,128], M counts는 [512,1024,2048,4096] 순서다. Reasons는 [gap-separated,cap-unresolved,risk-tie,fixed]다.

| Cell / method / task | K counts | M counts | Reasons | Mean stages | Mean K/M increases | Mean component–observation evaluations |
|---|---|---|---|---|---|---|
| matched_nominal/adaptive/contact_event | [426, 50, 20, 16] | [477, 13, 11, 11] | [497, 10, 5, 0] | 1.40 | 0.27/0.13 | 284064 |
| matched_nominal/adaptive/slip_event | [71, 64, 48, 329] | [151, 51, 27, 283] | [238, 260, 14, 0] | 5.10 | 2.24/1.86 | 3727104 |
| matched_nominal/balanced/contact_event | [421, 51, 23, 17] | [421, 51, 23, 17] | [497, 10, 5, 0] | 1.29 | 0.29/0.29 | 399264 |
| matched_nominal/balanced/slip_event | [64, 57, 58, 333] | [64, 57, 58, 333] | [223, 279, 10, 0] | 3.29 | 2.29/2.29 | 4325760 |
| matched_nominal/component_only/contact_event | [421, 47, 12, 32] | [512, 0, 0, 0] | [485, 22, 5, 0] | 1.33 | 0.33/0.00 | 156960 |
| matched_nominal/component_only/slip_event | [64, 50, 35, 363] | [512, 0, 0, 0] | [164, 332, 16, 0] | 3.36 | 2.36/0.00 | 615552 |
| matched_nominal/outer_only/contact_event | [0, 0, 0, 512] | [485, 12, 5, 10] | [497, 10, 5, 0] | 1.10 | 0.00/0.10 | 933120 |
| matched_nominal/outer_only/slip_event | [0, 0, 0, 512] | [164, 39, 19, 290] | [224, 274, 14, 0] | 2.85 | 0.00/1.85 | 4048896 |
| shifted_nominal/adaptive/contact_event | [413, 62, 20, 17] | [481, 12, 5, 14] | [497, 10, 5, 0] | 1.42 | 0.30/0.12 | 302496 |
| shifted_nominal/adaptive/slip_event | [63, 54, 49, 346] | [143, 42, 28, 299] | [218, 281, 13, 0] | 5.27 | 2.32/1.94 | 3929952 |
| shifted_nominal/balanced/contact_event | [409, 65, 21, 17] | [409, 65, 21, 17] | [497, 10, 5, 0] | 1.31 | 0.31/0.31 | 401568 |
| shifted_nominal/balanced/slip_event | [59, 47, 64, 342] | [59, 47, 64, 342] | [208, 292, 12, 0] | 3.35 | 2.35/2.35 | 4446048 |
| shifted_nominal/component_only/contact_event | [409, 60, 13, 30] | [512, 0, 0, 0] | [483, 24, 5, 0] | 1.34 | 0.34/0.00 | 157344 |
| shifted_nominal/component_only/slip_event | [59, 43, 34, 376] | [512, 0, 0, 0] | [151, 345, 16, 0] | 3.42 | 2.42/0.00 | 631008 |
| shifted_nominal/outer_only/contact_event | [0, 0, 0, 512] | [482, 14, 3, 13] | [497, 10, 5, 0] | 1.12 | 0.00/0.12 | 959232 |
| shifted_nominal/outer_only/slip_event | [0, 0, 0, 512] | [158, 34, 22, 298] | [211, 285, 16, 0] | 2.90 | 0.00/1.90 | 4140288 |
| shifted_correct/adaptive/contact_event | [417, 59, 19, 17] | [471, 19, 5, 17] | [493, 14, 5, 0] | 1.45 | 0.29/0.16 | 332064 |
| shifted_correct/adaptive/slip_event | [69, 47, 44, 352] | [150, 33, 24, 305] | [211, 286, 15, 0] | 5.27 | 2.33/1.95 | 3964704 |
| shifted_correct/balanced/contact_event | [411, 67, 16, 18] | [411, 67, 16, 18] | [494, 13, 5, 0] | 1.30 | 0.30/0.30 | 400416 |
| shifted_correct/balanced/slip_event | [64, 56, 39, 353] | [64, 56, 39, 353] | [204, 299, 9, 0] | 3.33 | 2.33/2.33 | 4512288 |
| shifted_correct/component_only/contact_event | [411, 56, 17, 28] | [512, 0, 0, 0] | [481, 26, 5, 0] | 1.34 | 0.34/0.00 | 156192 |
| shifted_correct/component_only/slip_event | [64, 41, 32, 375] | [512, 0, 0, 0] | [156, 339, 17, 0] | 3.40 | 2.40/0.00 | 628128 |
| shifted_correct/outer_only/contact_event | [0, 0, 0, 512] | [481, 8, 5, 18] | [493, 14, 5, 0] | 1.14 | 0.00/0.14 | 1012992 |
| shifted_correct/outer_only/slip_event | [0, 0, 0, 512] | [164, 28, 12, 308] | [203, 292, 17, 0] | 2.91 | 0.00/1.91 | 4192512 |

| Cell / task | Adaptive unresolved: compression-dominant | Adaptive unresolved: outer-dominant |
|---|---|---|
| matched_nominal/contact_event | 7 | 3 |
| matched_nominal/slip_event | 216 | 44 |
| shifted_nominal/contact_event | 5 | 5 |
| shifted_nominal/slip_event | 229 | 52 |
| shifted_correct/contact_event | 8 | 6 |
| shifted_correct/slip_event | 236 | 50 |

![Final K and M budgets](results/budgets.png)

### Error proxies와 observed discrepancy

| Audit | n | Median proxy | Median observed discrepancy | Observed exceeds proxy | Ambiguous pair exclusions |
|---|---|---|---|---|---|
| matched_nominal/contact_event/compression | 507 | 0.020074 | 0.014038 | 30.97% | 5 |
| matched_nominal/contact_event/outer_perturbation | 63 | 0.014149 | 0.007054 | 22.22% | 5 |
| matched_nominal/slip_event/compression | 498 | 0.003073 | 0.000989 | 17.87% | 14 |
| matched_nominal/slip_event/outer_perturbation | 61 | 0.000892 | 0.000440 | 21.31% | 14 |
| shifted_nominal/contact_event/compression | 507 | 0.019891 | 0.013147 | 27.42% | 5 |
| shifted_nominal/slip_event/compression | 499 | 0.002773 | 0.000835 | 14.83% | 13 |
| shifted_correct/contact_event/compression | 507 | 0.021002 | 0.012501 | 21.70% | 5 |
| shifted_correct/slip_event/compression | 497 | 0.002945 | 0.000957 | 18.31% | 15 |

Compression discrepancy는 같은 outer samples에서 compact와 full-inner risk gap의 차이다. Outer discrepancy는 같은 compact representatives에서 independent outer scrambles로 계산한 두 gap estimates의 차이다. 이 둘은 true error의 동일한 정의가 아니며, 특히 outer discrepancy의 exceedance rate를 confidence-interval miscoverage로 부를 수 없다. 4-scramble uncertainty, shared compression bias, adaptive selection과 finite-bank error가 남는다.

## 6. Online compute와 memory

RTX PRO6000 Blackwell Server Edition, CPU2 threads, float64, batch1. 고정된20 matched worlds×2tasks×3repeats=120 measurements/method다. 같은 pass에서 strong baselines를 측정했다. 전체 bank/static plans의 construction은 setup으로 별도 표시하고, factual fits·public targets·representative/outer sample realization·cache allocation·all refinements·actual update는 online 시간에 포함했다. GPU와 CPU timing은 순차 실행했다.

| Method | GPU median ms | GPU p90 ms | CPU median ms | CPU contact ms | CPU stick ms | GPU contact ms | GPU stick ms |
|---|---|---|---|---|---|---|---|
| adaptive | 7.872 | 19.120 | 5.517 | 4.077 | 43.641 | 5.713 | 18.523 |
| balanced | 8.266 | 15.722 | 6.308 | 4.092 | 44.935 | 5.761 | 15.145 |
| component_only | 7.752 | 10.878 | 4.152 | 2.847 | 6.886 | 5.657 | 10.474 |
| outer_only | 7.485 | 16.186 | 8.103 | 7.148 | 41.875 | 6.587 | 15.638 |
| fixed32_512 | 5.433 | 6.788 | 3.274 | 2.773 | 3.675 | 4.651 | 5.573 |
| fixed128_512 | 5.331 | 5.797 | 5.934 | 5.454 | 6.045 | 4.540 | 5.509 |
| fixed32_4096 | 5.615 | 6.475 | 13.051 | 12.252 | 13.523 | 4.776 | 5.876 |
| fixed128_4096 | 6.033 | 6.448 | 50.356 | 50.089 | 50.813 | 5.179 | 6.141 |
| global32_512 | 5.206 | 5.693 | 3.288 | 2.583 | 3.447 | 4.426 | 5.427 |
| global128_4096 | 5.869 | 6.510 | 54.606 | 49.375 | 57.433 | 5.061 | 5.997 |
| native | 3.126 | 3.981 | 1.770 | 1.389 | 1.855 | 2.749 | 3.250 |
| qmc64_native | 4.166 | 4.765 | 5.326 | 4.489 | 5.845 | 3.994 | 4.420 |
| nested512 | 9.787 | 10.645 | 83.033 | 83.033 | 83.020 | 9.143 | 10.124 |
| nested2048 | 27.861 | 29.757 | 317.585 | 318.173 | 316.434 | 27.494 | 28.279 |
| nested4096 | 51.800 | 56.462 | 642.110 | 641.311 | 643.487 | 51.279 | 52.648 |
| query | 1.433 | 1.870 | 1.007 | 0.637 | 1.117 | 1.104 | 1.631 |
| probe | 1.413 | 1.676 | 0.970 | 0.624 | 1.107 | 1.045 | 1.574 |

| Method | Original GPU ms | Optimized GPU ms | Static bank/plan bytes | Acquisition cache max bytes | Observed CUDA temporary peak bytes | Warm setup ms |
|---|---|---|---|---|---|---|
| adaptive | 7.157 | 7.872 | 17571840 | 2789888 | 79408640 | 8.696 |
| balanced | 7.206 | 8.266 | 17571840 | 2789888 | 66922496 | 8.337 |
| component_only | 7.164 | 7.752 | 17342464 | 438784 | 11134976 | 8.752 |
| outer_only | 6.931 | 7.485 | 17571840 | 2789888 | 66922496 | 9.003 |
| fixed32_512 | 5.966 | 5.433 | 17336320 | 115840 | 6088192 | 8.349 |
| fixed128_512 | 6.563 | 5.331 | 17342464 | 193024 | 19665408 | 9.242 |
| fixed32_4096 | 6.842 | 5.615 | 17565696 | 746624 | 39609856 | 11.767 |
| fixed128_4096 | 9.980 | 6.033 | 17571840 | 823808 | 152932864 | 8.437 |
| global32_512 | 5.784 | 5.206 | 17336320 | 115840 | 6088192 | 8.690 |
| global128_4096 | 8.919 | 5.869 | 17571840 | 823808 | 152932864 | 8.600 |
| native | 3.044 | 3.126 | 24117248 | 0 | 1572864 | 4.247 |
| qmc64_native | 4.044 | 4.166 | 96468992 | 0 | 6291456 | 6.577 |
| nested512 | 9.588 | 9.787 | 17334272 | 0 | 50999296 | 8.908 |
| nested2048 | 27.586 | 27.861 | 17432576 | 0 | 51008512 | 11.751 |
| nested4096 | 52.299 | 51.800 | 17563648 | 0 | 51020800 | 8.634 |
| query | 1.337 | 1.433 | 8650752 | 0 | 1572864 | 5.199 |
| probe | 1.341 | 1.413 | 8650752 | 0 | 1572864 | 4.727 |

Acquisition cache에는 representatives와 hypothetical samples 및 prefix log sums를 포함한다. Static full banks는 여전히 필요하다. Temporary peak는40 unique timing cases에서 관찰한 추가 CUDA allocated bytes이며 global memory bound가 아니다. CUDA context/allocator reserve, Python objects와 실제 sensing/reset 비용은 별도다. Hardware와 workload가 다른 곳의 latency를 보장하지 않는다.

| Method | CPU/GPU action mismatches /40 | GPU prediction max error | CPU prediction max error |
|---|---|---|---|
| adaptive | 0 | 3.997e-15 | 1.310e-14 |
| balanced | 0 | 3.997e-15 | 1.310e-14 |
| component_only | 0 | 3.997e-15 | 1.310e-14 |
| outer_only | 0 | 3.997e-15 | 1.310e-14 |
| fixed32_512 | 0 | 3.997e-15 | 1.310e-14 |
| fixed128_512 | 0 | 3.997e-15 | 1.310e-14 |
| fixed32_4096 | 0 | 3.997e-15 | 1.310e-14 |
| fixed128_4096 | 0 | 3.997e-15 | 1.310e-14 |
| global32_512 | 0 | 3.997e-15 | 1.310e-14 |
| global128_4096 | 0 | 3.997e-15 | 1.310e-14 |
| native | 0 | 3.997e-15 | 1.310e-14 |
| qmc64_native | 0 | 9.992e-15 | 2.376e-14 |
| nested512 | 0 | 3.997e-15 | 1.310e-14 |
| nested2048 | 0 | 3.997e-15 | 1.310e-14 |
| nested4096 | 0 | 3.997e-15 | 1.310e-14 |
| query | 0 | 5.329e-15 | 1.310e-14 |
| probe | 0 | 9.548e-15 | 1.044e-14 |

![Loss, compute and Top-1 agreement](results/tradeoff.png)

Figure는 point estimates이며 uncertainty는 paired CI 표를 따른다. Deterministic methods의 acquisition-seed repeatability를 stochastic audit의 근거처럼 사용하지 않는다.

## 7. 결과 해석과 논문 기여 가능성

### 7.1 두 error sources를 구분하면 무엇이 달라졌는가

Matched contact에서는 426/512 worlds가 K16에서, 477/512가 M512에서 멈췄다. Adaptive의 평균 component–observation evaluations는 284,064, balanced는 399,264였다. Stick에서는 각각 3,727,104 대 4,325,760으로 adaptive가 더 적었다. 두 축을 무조건 함께 키우는 것보다 불필요한 likelihood evaluations를 줄이는 효과는 관찰됐다. 이 count는 준비·memory·error checks·kernel launches를 포함한 FLOPs나 latency가 아니다.

Component-only의 stick cap-unresolved는 332/512였고 adaptive는 260/512였다. Independent-seed agreement도 76.37%에서 82.62%로 높아졌다. 하지만 outer_only 82.81%, balanced 83.40%, fixed128/M4096 83.20%보다 높은 것은 아니다. 따라서 개선의 근거는 “outer refinement를 허용하면 도움이 된다”까지이며 “현재 두-error routing이 최선이다”까지는 아니다.

Adaptive stick의 278/512 worlds는 최대 K128/M4096 rectangle까지 계산했고 260/512는 stopping criterion을 만족하지 못했다. 이 unresolved 중 216은 compression proxy, 44는 outer proxy가 더 컸다. Outer sampling을 늘린 뒤에도 대부분의 unresolved cases에서 compression proxy 또는 작은 action gap이 병목으로 남는다. 이 숫자를 실제 compression error가 지배한다는 증명으로 읽어서는 안 된다.

Same-outer full-inner audit에서 contact의 observed gap discrepancy가 compression proxy보다 큰 비율은 30.97%, stick은 17.87%였다. Independent outer perturbation audit에서는 각각 22.22%, 21.31%였다. 특히 후자는 두 noisy estimates의 차이이므로 nominal confidence coverage의 실패율이 아니다. 이 결과와 shared-bias 가능성을 함께 보면, 현재 eC+eO를 reliable error certificate로 제시할 수 없다. Adaptive allocation은 heuristic으로 평가해야 한다.

### 7.2 Ranking 안정성과 decision loss가 일치하지 않는 이유

Ranking은 작은 risk 차이에도 action이 바뀌면 disagreement로 센다. 반면 actual decision loss는 실제 world와 observation noise에 대한 평균이다. 첫64 matched worlds의 independent nested8192 reference와 adaptive의 stick agreement는 65.63%이지만 reference risk regret는 약 5.19×10⁻⁵였다. Top-1 disagreement 모두가 큰 손실을 뜻하지 않는다는 수치다. 이 reference도 finite bank와 finite outer samples를 사용하므로 정확한 Bayes regret는 아니다.

Global128/M4096은 stick agreement 90.82%, mean loss 0.070957, GPU median 5.87ms로 강한 comparator다. Minority label support를 놓칠 수 있다는 구조적 약점만으로 이를 약한 baseline으로 제외할 수 없다. Region-preserving mixture가 필요한 조건을 별도 검증해야 하며, 이번 matched average에서는 adaptive의 우위를 보이지 않았다.

No observation의 stick loss 0.097282에 비해 adaptive는 0.071078로 낮았다. 그러나 always-probe도 0.072715이고 adaptive-minus-probe의 adjusted CI는 [−0.006948,+0.003351]이다. 추가 observation의 가치와 sophisticated observation selection의 추가 가치는 구분해야 한다. Contact에서는 query-only가 사실상 같은 loss를 훨씬 낮은 compute로 달성해, adaptive selection을 입증하기 어려운 ceiling 조건이다.

Shifted-law/nominal-inference stick loss는 adaptive 0.068577, global128/M4096 0.067193, nested2048 0.067621, always-probe 0.067616이었다. Privileged correct-law inference를 제공하면 adaptive는 0.063622였다. Budget refinement는 model mismatch를 자동으로 해결하지 않는다. Correct-law와 nominal의 차이는 acquisition뿐 아니라 posterior/decoder 차이도 포함하므로 순수한 selector 효과로 볼 수 없다.

### 7.3 Compute 결과를 어떻게 읽어야 하는가

GPU에서 cached likelihood evaluations를 줄여도 sequential stopping checks, synchronization과 작은 kernel launches가 든다. Adaptive stick은 평균 5.10 stages, balanced는 3.29 stages였다. Work count가 적은 adaptive의 stick GPU median이 18.52ms로 balanced 15.14ms보다 오히려 컸다. 이 관찰은 GPU에서 regular batched fixed computation이 유리할 수 있다는 해석을 지지한다. 상세한 profiler로 원인별 시간을 분해한 것은 아니므로 causal attribution은 추가 검증이 필요하다.

Original과 block-logsumexp kernel은 30,720 cases에서 선택 action, risks와 adaptive stopping이 일치했고 최대 risk 차이는 5.25×10⁻¹⁵였다. Fixed methods는 adaptive checks를 수행하지 않는 direct kernel로 측정했다. Original adaptive aggregate GPU median 7.16ms보다 optimized pass의 7.87ms가 오히려 높았다. 두 passes의 작은 시간 차이는 kernel 및 runtime variation을 함께 포함하므로 “optimization으로 adaptive를 가속했다”는 주장을 하지 않는다. 최종 비교에는 동일한 optimized pass의 모든 methods를 사용한다.

CPU aggregate median은 adaptive 5.52ms 대 fixed128/M4096 50.36ms지만, 두 tasks가 섞인 이 값은 hard-case 비용을 가린다. Stick의 감소는 50.81ms에서 43.64ms로 약14.1%이며, cheap fixed mixtures와의 tradeoff도 남는다. 따라서 GPU 결과를 CPU에 그대로 일반화하거나 aggregate median으로 일관된 9× CPU speedup을 주장하지 않는다.

이번 timing은 같은 hardware에서20 worlds×2tasks×3repeats다. 512-world loss 평가와는 표본 크기가 다르고 timing uncertainty CI를 제공하지 않는다. Batch1, float64, analytic cached forward model이라는 조건도 제한적이다. Expensive simulator나 real image encoder를 포함했을 때 budget allocation이 유리해질 가능성은 있지만 이번 결과가 그 가능성을 검증한 것은 아니다.

### 7.4 가장 가까운 선행연구와 novelty

Bartuska 등의 **Multilevel randomized quasi-Monte Carlo estimator for nested integration**는 nested integration에서 inner sample count와 inner integrand의 discretization accuracy를 levels로 다루고, bias와 statistical error를 분석하며 inner/outer sample allocation을 연구한다. V18은 discontinuous contact/stick event의 Brier-risk ranking에 conditional-mixture approximation을 쓴다는 차이가 있다. 그러나 inner approximation error와 outer integration error를 구별한다는 발상 자체는 새로운 기여가 아니다. 해당 multilevel estimator나 그 regularity assumptions를 이번 simulator에 적용해 이론적으로 검증한 것은 아니다. [원문](https://arxiv.org/html/2412.07723v1)

Sorokin 등의 **Fast Bayesian Multilevel Quasi-Monte Carlo**는 Bayesian cubature/GP uncertainty를 이용해, sample count를 두 배로 늘렸을 때 예상되는 uncertainty 감소와 비용을 비교하여 계산을 배분한다. V18은 단순히 현재 eC와 eO 중 큰 값을 기준으로 축을 고른다. Expected error reduction per unit cost를 추정하는 방법이 아니므로 optimal resource allocation이라고 부를 수 없다. 또한 V18의 4-scramble error proxy를 이 논문의 Bayesian error guarantee와 동일시할 수 없다. [원문](https://arxiv.org/html/2510.24604v1)

Dinkel 등의 2026 preprint **Efficient Bayesian Optimal Experimental Design for Expensive Computational Models over Finite Design Sets**는 EIG 차이를 common random numbers, Rao–Blackwellization과 paired bootstrap으로 비교하고, 불확실한 designs에 계산을 추가하는 adaptive elimination을 제시한다. V18은 event Brier loss와 compact conditional mixtures에 초점을 맞추지만, action gap의 불확실성에 따라 계산을 늘린다는 원리에는 선행연구가 있다. 해당 ADE method를 직접 구현해 이 데이터에서 비교한 것은 아니다. [원문](https://arxiv.org/html/2607.16933v1)

Zhu와 Hainy의 2026 preprint **Sequential Bayesian Experimental Design for Prediction in Physical Experiments Informed by Computer Models**는 predictive design에서 Gaussian mixture compression을 사용한다. V18은 label-conditioned nested representatives를 사용하지만, mixture를 압축해 nested acquisition 계산을 줄인다는 큰 아이디어만으로 novelty를 주장할 수는 없다. Generic GMM reduction과의 직접 비교도 남아 있다. [원문, Section 4.1](https://arxiv.org/html/2603.16756v1#S4.SS1)

### 7.5 주장할 수 있는 범위와 후속 연구의 조건

현재 loss는 event probability의 Brier loss다. 실제 robot control success, intervention cost, safety constraint나 task reward를 평가한 것은 아니다. 작은 analytic local-impulse model과 cached candidate forward predictions를 사용하며, raw images와 실제 영상에서의 geometry/contact-model error가 포함되지 않았다.

Method contribution을 주장하려면 fixed-budget mixtures와 balanced refinement를 넘는 accuracy–compute frontier, error proxy의 reliability, application-specific decision tolerance가 함께 필요하다. Marginal mean loss가 좋아 보인다는 이유만으로 acquisition error decomposition이 정확했다고 결론 내릴 수 없다. Conversely, ranking agreement가 높아져도 실제 loss나 compute가 나아지지 않으면 실용적인 contribution이 약하다.

추가 설계가 필요하다면 현재 error 크기만 비교하기보다 각 refinement의 **예상 decision-risk 감소 / incremental compute**를 추정하는 방향이 후보가 된다. 그러나 이 역시 cost-aware numerical integration과 adaptive experimental design의 기존 연구와 직접 비교해야 한다. Exact ranking 대신 application에서 정한 허용 regret를 사용하는 stopping도 검토할 수 있다. 이 보고서에서 그런 후속 방법을 구현하거나 성능을 입증했다고 주장하지 않는다.


## 8. 실행과 재현 자료

Server directory: `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/adaptive_budget_v18`. Allocation2680491, node56의 idle GPU를 확인하고 guard 아래 실행했다. 기존 사용자 jobs/allocation은 변경하지 않았다.

`budget.py`, `budget_fast.py`, `runtime_fast.py`가 핵심 구현이고 `PROTOCOL.md`, `IMPLEMENTATION_AUDIT.md`, `AUDIT_CORRECTION.md`에 사전 설계와 implementation/audit 수정의 이유를 기록했다. `results/policies`에는 primary choices, budgets, histories, same-sample full-inner risks가 있다. `results/independent_audit.npz`가 corrected independent audit이며 원래 audit는 `results/audits`에 보존했다. `results/evaluations`, `results/world_losses.npz`, `results/summary.json`에서 actual outcomes와 통계를 재분석할 수 있다.

Source/policy freeze, cached-vs-dense validation, optimized-kernel equivalence, online prediction checks와 old V17 file hashes를 검증했다. `INTEGRITY.json`과 `MANIFEST.json`에 검증 결과 및 SHA256을 남겼다. 모든 simulation/inference/statistics는 allocated compute node에서 실행했고 local mirror는 검토와 전달에 사용했다.

재실행은 새 directory에서 allocated idle GPU로 `guard.py run.py`를 실행한다. Main worker exit를 gpu_worker_exit.json에 보존한 뒤 `guard.py run_fast.py`로 equivalence, corrected independent audit, optimized timing을 수행하고 fast_worker_exit.json을 보존한다. 이후 CPU에서 `timing_fast.py --cpu`, `analyze.py`, `plot_results.py`, `write_report.py`, `seal.py` 순서로 실행한다. CPU scripts는 BELIEF_DEVICE=cpu 및 빈 CUDA_VISIBLE_DEVICES를 사용한다. Sealed results를 덮어쓰지 않는다.
