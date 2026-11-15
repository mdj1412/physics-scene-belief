# Decision-region conditional predictive mixtures — v16

작성일: 2026-11-13 KST. 서버에서 implementation validation, fresh-world inference/evaluation, GPU·CPU timing과 statistical analysis를 수행했다.

## 1. 결론

**Conditional predictive mixture를 유지하면 v15의 false-information counterexample을 해결한다. Fresh-world 실험에서도 pair selector의 실패를 상당히 줄였다. 다만 region별 compression의 독자적 이득과 nested baseline을 대체할 만큼 안정적인 accuracy–compute trade-off는 아직 입증하지 못했다.**

새로운 512 iid worlds와 world당 16 observation noise repeats를 사용했다. Primary matched condition의 결과는 다음과 같다. Event loss는 Brier loss이며 낮을수록 좋다. GPU latency는 factual fit부터 selected observation의 posterior update까지 포함한 5개 tasks의 aggregated median이다.

| Method | Contact-event loss | Stick-event loss | GPU online ms |
|---|---:|---:|---:|
| **Region32** | **0.044491** | **0.066922** | **4.265** |
| Equal-size global mixture | 0.044537 | 0.067871 | 3.771 |
| Optimized LMMSE16 | 0.044758 | 0.073383 | 3.392 |
| Optimized LMMSE64 | 0.042943 | 0.070734 | 4.438 |
| Nested128 | 0.044291 | 0.066739 | 4.767 |
| Nested512 | 0.044319 | 0.065929 | 10.068 |
| V15 pair selector | 0.055675 | 0.072749 | 3.204 |
| Always-query | 0.044473 | 0.077896 | 1.584 |
| Always-probe | 0.130390 | 0.068684 | 1.542 |

Region32는 pair 대비 contact-event loss를 **20.1%**, stick-event loss를 **8.0%** 줄였고 두 차이의 primary adjusted CI가 모두 0 아래였다. LMMSE16 대비 stick-event 평균 개선은 **8.8%**이며 difference는 −0.006462, adjusted 99.58333% CI [−0.013470, −0.000015]였다. CI upper가 0에 가까우므로 강한 보편적 우위로 해석하지 않는다. 별도 bootstrap Monte Carlo audit에서도 방향은 유지됐지만 fixed QMC banks의 uncertainty를 해소한 검증은 아니다.

Nested512 대비 GPU online speedup은 **2.36×**다. 그러나 사전에 정한 approximation screen은 **실패**했다. Contact-event mean-loss ratio의 95% upper는 1.01596으로 1.05 이하였지만 stick-event는 **1.05773**이었다. 따라서 “nested 수준의 loss를 5% 이내로 보존하면서 2배 이상 빠르다”는 결론을 내릴 수 없다. 평균 차이가 작거나 superiority CI가 0을 포함한다는 이유로 equivalence를 주장하지 않는다.

Global mixture 대비 개선은 두 events 모두 adjusted CI에서 확인되지 않았다. Global mixture도 두 counterexamples를 해결했다. 따라서 이번 진전의 근거는 **pairwise separation 대신 conditional mixtures를 합한 뒤 decision risk를 계산한 것**에 있으며, **region stratification만의 필요성**까지 입증한 것은 아니다.

추가 observation 자체는 유용하다. Region32의 contact/stick losses는 no-observation의 0.192616/0.088885보다 낮았다. 하지만 adaptive selection의 추가 가치는 별도 문제다. Contact에서는 always-query와 사실상 같은 loss이며 always-query가 훨씬 저렴하다. Stick에서는 always-probe 대비 평균 2.57% 낮지만 adjusted CI가 0을 포함한다. Strong fixed rule을 넘는 근거는 충분하지 않다.

현재 판단은 **유용한 acquisition correction과 후속 연구용 baseline은 확보했지만, 독립적인 method paper의 핵심으로 확정할 단계는 아니다**이다. 특히 compact decoder까지 사용하면 matched stick loss가 0.076230으로 악화된다. 이번 결과는 full factual posterior를 대체한 compact belief의 성공이 아니라, QMC factual posterior 위에서 acquisition을 압축한 결과다.


## 2. 구현: Pair distance를 mixture Bayes risk로 교체

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


## 4. Counterexamples: 무엇을 고쳤는가

| Example | No-observation risk | Region32 informative risk | Quadrature informative reference | Region32 uninformative risk | Global informative / uninformative |
|---|---|---|---|---|---|
| v15_aliasing | 0.250000 | 0.199175 | 0.198986 | 0.250000 | 0.199175 / 0.250000 |
| higher_order | 0.250000 | 0.007321 | 0.008089 | 0.250000 | 0.007321 / 0.250000 |

V15 aliasing example은 labels [0,0,1,1], equal weights, noiseSD1에서 uninformative means[−4,+4,−4,+4], informative means[−.5,−.5,+.5,+.5]를 사용했다. Old pair는 uninformative action에0.249916의 positive score를 주었다. Region mixture는 그 action에서 risk를0.25로 유지했다. Higher-order example은 class0의±5, class1의−10/0/+10 mixtures가 모두 mean0, variance25지만 모양이 다르다. Conditional Gaussian moment reduction과 LMMSE covariance는 이 information을 놓치지만 mixture는 보존했다. Reference는 Gauss–Hermite100 quadrature로서 모든 경우의 closed-form exact value는 아니다. **Equal-budget global mixture도 두 예제를 통과했다. 따라서 이 mechanism 개선을 region stratification만의 novelty로 주장하지 않는다.**

## 5. Decision loss

### matched_nominal

| Method | smooth_velocity | contact_velocity | contact_event | slip_velocity | slip_event |
|---|---|---|---|---|---|
| region32 | 0.014741 | 0.072609 | 0.044491 | 0.005855 | 0.066922 |
| global32 | 0.014904 | 0.072617 | 0.044537 | 0.006002 | 0.067871 |
| region64 | 0.014893 | 0.071604 | 0.044491 | 0.005832 | 0.069161 |
| region_cf32 | 0.014437 | 0.071073 | 0.044567 | 0.005791 | 0.067020 |
| region32_compact | 0.015444 | 0.076584 | 0.045794 | 0.006244 | 0.076230 |
| native | 0.014828 | 0.071743 | 0.044758 | 0.005928 | 0.073383 |
| nested128 | 0.014480 | 0.070961 | 0.044291 | 0.005775 | 0.066739 |
| nested512 | 0.014421 | 0.071001 | 0.044319 | 0.005780 | 0.065929 |
| pair | 0.015188 | 0.093836 | 0.055675 | 0.006140 | 0.072749 |
| query | 0.020103 | 0.071589 | 0.044473 | 0.008797 | 0.077896 |
| factual | 0.021666 | 0.149052 | 0.100008 | 0.009876 | 0.085404 |
| probe | 0.014679 | 0.192923 | 0.130390 | 0.005940 | 0.068684 |
| none | 0.022499 | 0.274492 | 0.192616 | 0.010368 | 0.088885 |
| qmc64_native | 0.014292 | 0.068354 | 0.042943 | 0.005621 | 0.070734 |

### shifted_nominal

| Method | smooth_velocity | contact_velocity | contact_event | slip_velocity | slip_event |
|---|---|---|---|---|---|
| region32 | 0.011664 | 0.068672 | 0.045091 | 0.006325 | 0.066489 |
| global32 | 0.011806 | 0.068683 | 0.045073 | 0.006294 | 0.068543 |
| region64 | 0.011774 | 0.067516 | 0.045091 | 0.006262 | 0.067210 |
| region_cf32 | 0.011502 | 0.067583 | 0.045132 | 0.006299 | 0.068007 |
| region32_compact | 0.012049 | 0.072918 | 0.047140 | 0.006588 | 0.075625 |
| native | 0.011144 | 0.067523 | 0.045078 | 0.006246 | 0.069341 |
| nested128 | 0.011277 | 0.067511 | 0.045078 | 0.006261 | 0.065484 |
| nested512 | 0.011450 | 0.067523 | 0.045073 | 0.006341 | 0.066478 |
| pair | 0.011886 | 0.091104 | 0.059404 | 0.006537 | 0.071581 |
| query | 0.014166 | 0.067621 | 0.045073 | 0.008770 | 0.072453 |
| factual | 0.015913 | 0.142142 | 0.099270 | 0.009962 | 0.080240 |
| probe | 0.012237 | 0.181710 | 0.126026 | 0.006395 | 0.067522 |
| none | 0.016651 | 0.267591 | 0.192778 | 0.010314 | 0.082983 |
| qmc64_native | 0.011067 | 0.064762 | 0.043492 | 0.006158 | 0.065628 |

### shifted_correct

| Method | smooth_velocity | contact_velocity | contact_event | slip_velocity | slip_event |
|---|---|---|---|---|---|
| region32 | 0.008071 | 0.066025 | 0.045053 | 0.005986 | 0.060866 |
| global32 | 0.008183 | 0.066014 | 0.045366 | 0.006004 | 0.061245 |
| region64 | 0.008099 | 0.065883 | 0.045048 | 0.005962 | 0.059839 |
| region_cf32 | 0.008045 | 0.066259 | 0.045068 | 0.005993 | 0.059078 |
| region32_compact | 0.008278 | 0.071896 | 0.045978 | 0.006173 | 0.072082 |
| native | 0.008056 | 0.065791 | 0.045048 | 0.006010 | 0.058345 |
| nested128 | 0.008208 | 0.065383 | 0.044658 | 0.005990 | 0.057453 |
| nested512 | 0.007955 | 0.065384 | 0.044658 | 0.005932 | 0.058237 |
| pair | 0.008894 | 0.080720 | 0.056308 | 0.006561 | 0.066013 |
| query | 0.010001 | 0.065791 | 0.045048 | 0.008247 | 0.069053 |
| factual | 0.012148 | 0.140470 | 0.098786 | 0.009569 | 0.076033 |
| probe | 0.008743 | 0.170921 | 0.119956 | 0.006076 | 0.058358 |
| none | 0.013115 | 0.265770 | 0.190960 | 0.009862 | 0.075695 |

### Primary contrasts

Negative difference는 region32가 유리하다는 뜻이다.

| Task / comparison | Difference [adjusted CI] | Relative mean change |
|---|---|---|
| contact_event/region32-native | -0.000267 [-0.001386, +0.000059] | -0.60% |
| contact_event/region32-nested512 | +0.000171 [-0.000211, +0.001086] | +0.39% |
| contact_event/region32-global32 | -0.000046 [-0.000241, +0.000011] | -0.10% |
| contact_event/region32-pair | -0.011185 [-0.020948, -0.003322] | -20.09% |
| contact_event/region32-query | +0.000017 [-0.000000, +0.000077] | +0.04% |
| contact_event/region32-probe | -0.085899 [-0.106097, -0.066156] | -65.88% |
| slip_event/region32-native | -0.006462 [-0.013470, -0.000015] | -8.81% |
| slip_event/region32-nested512 | +0.000993 [-0.003237, +0.005020] | +1.51% |
| slip_event/region32-global32 | -0.000949 [-0.004859, +0.002596] | -1.40% |
| slip_event/region32-pair | -0.005827 [-0.012341, -0.000105] | -8.01% |
| slip_event/region32-query | -0.010974 [-0.021679, -0.000948] | -14.09% |
| slip_event/region32-probe | -0.001762 [-0.006621, +0.002845] | -2.57% |

### Statistical numerical audit와 비슷한 compute 비용의 baselines

Primary stick-event 대 LMMSE의 adjusted CI upper가 0에 가까워, policy나 primary criterion을 바꾸지 않고 bootstrap Monte Carlo error를 추가 점검했다. Independent seeds 5개에서 각각 100,000 bootstrap resamples를 사용한 upper bounds는 모두 음수였다. 이는 같은 worlds에서의 numerical audit이며 새 worlds 또는 QMC banks에 대한 replication이 아니다. Primary 30,000-resample CI는 그대로 유지한다.

| Bootstrap seed | Adjusted lower | Adjusted upper |
|---|---|---|
| 140101 | -0.01362562 | -0.00011043 |
| 140102 | -0.01356148 | -0.00013728 |
| 140103 | -0.01348950 | -0.00015416 |
| 140104 | -0.01349267 | -0.00021927 |
| 140105 | -0.01356564 | -0.00020869 |

다음 비교는 post-analysis secondary 95% CI다. 두 baseline 모두 region32와 비슷한 GPU compute 범위에 있다.

| Task / region32 minus baseline | Difference [95% CI] |
|---|---|
| contact_event/nested128 | +0.000200 [-0.000091, +0.000698] |
| contact_event/qmc64_native | +0.001548 [-0.000673, +0.003902] |
| slip_event/nested128 | +0.000183 [-0.002555, +0.002789] |
| slip_event/qmc64_native | -0.003812 [-0.009310, +0.001423] |

### Predeclared approximation screen

| Task | Mean-loss ratio [95% CI] |
|---|---|
| contact_event | 1.0039 [0.9969, 1.0160] |
| slip_event | 1.0151 [0.9720, 1.0577] |

Screen result: `{'loss_criteria': False, 'gpu_speedup': 2.360741321003479, 'passes': False}`. Ratio criterion과 speed criterion을 분리해서 해석한다.

### Observation choices

| Cell / method | Contact [none,query,factual,probe] | Stick [none,query,factual,probe] |
|---|---|---|
| matched_nominal/region32 | [0, 499, 11, 2] | [0, 70, 95, 347] |
| matched_nominal/global32 | [0, 502, 6, 4] | [0, 225, 72, 215] |
| matched_nominal/native | [0, 496, 12, 4] | [0, 196, 21, 295] |
| matched_nominal/nested512 | [0, 498, 11, 3] | [0, 148, 43, 321] |
| shifted_nominal/region32 | [0, 500, 10, 2] | [0, 59, 83, 370] |
| shifted_nominal/global32 | [0, 500, 7, 5] | [0, 251, 48, 213] |
| shifted_nominal/native | [0, 498, 11, 3] | [0, 186, 9, 317] |
| shifted_nominal/nested512 | [0, 502, 8, 2] | [0, 154, 37, 321] |
| shifted_correct/region32 | [0, 497, 14, 1] | [0, 78, 61, 373] |
| shifted_correct/global32 | [0, 503, 5, 4] | [0, 259, 51, 202] |
| shifted_correct/native | [0, 496, 14, 2] | [0, 105, 2, 405] |
| shifted_correct/nested512 | [0, 499, 6, 7] | [0, 180, 39, 293] |

## 6. Compression, cross-fitting과 ranking sensitivity

| Cell / task | Region64 − region32 [95% CI] | Cross-fitted region32 − primary [95% CI] | Compact decoder − full decoder [95% CI] |
|---|---|---|---|
| matched_nominal/contact_event | -0.000000 [-0.000000, +0.000000] | +0.000076 [-0.000028, +0.000232] | +0.001303 [-0.000255, +0.002997] |
| matched_nominal/slip_event | +0.002239 [-0.000006, +0.004736] | +0.000099 [-0.004038, +0.004716] | +0.009308 [+0.003312, +0.016184] |
| shifted_nominal/contact_event | -0.000000 [-0.000000, +0.000000] | +0.000042 [-0.000038, +0.000168] | +0.002049 [+0.000531, +0.003773] |
| shifted_nominal/slip_event | +0.000722 [-0.001033, +0.002630] | +0.001518 [-0.001881, +0.005562] | +0.009136 [+0.003702, +0.015578] |
| shifted_correct/contact_event | -0.000006 [-0.000017, +0.000000] | +0.000015 [-0.000008, +0.000040] | +0.000925 [-0.000574, +0.002513] |
| shifted_correct/slip_event | -0.001027 [-0.003987, +0.001873] | -0.001788 [-0.005301, +0.001767] | +0.011216 [+0.005519, +0.018064] |

| matched_nominal | Contact agreement / loss change | Stick agreement / loss change |
|---|---|---|
| region32_Q8 | 100.00% / +0.000000 | 78.12% / -0.000558 |
| region32_seed2 | 100.00% / +0.000000 | 78.12% / -0.002691 |
| nested2048 | 100.00% / +0.000000 | 78.12% / +0.005130 |

| shifted_nominal | Contact agreement / loss change | Stick agreement / loss change |
|---|---|---|
| region32_Q8 | 100.00% / +0.000000 | 65.62% / -0.000051 |
| region32_seed2 | 100.00% / +0.000000 | 75.00% / -0.002177 |
| nested2048 | 100.00% / +0.000000 | 78.12% / +0.004324 |

| shifted_correct | Contact agreement / loss change | Stick agreement / loss change |
|---|---|---|
| region32_Q8 | 100.00% / +0.000000 | 78.12% / +0.000004 |
| region32_seed2 | 100.00% / +0.000000 | 81.25% / -0.000042 |
| nested2048 | 100.00% / +0.000000 | 75.00% / +0.000036 |

각 cell의 첫 32 worlds의 Q8과 alternate Sobol seed는 main region32와, nested2048은 nested512와 비교했다. Main policy는 audit로 변경하지 않았다. Independent-seed nested2048의 내부 risk regret도 summary에 기록했으나 actual Bayes regret로 부르지 않는다. Fresh worlds의 bootstrap CI는 fixed QMC bank의 모든 numerical uncertainty를 포함하지 않는다.

| Cell | Factual ESS median | Selected ESS medians (five tasks) | Max components | Max omitted mass |
|---|---|---|---|---|
| matched_nominal | 108.895455 | [22.17, 27.26, 27.42, 22.37, 21.53] | [128.0, 128.0, 64.0, 128.0, 64.0] | 1.9414220896044375e-11 |
| shifted_nominal | 110.685058 | [23.23, 27.6, 27.93, 23.18, 22.25] | [128.0, 128.0, 64.0, 128.0, 64.0] | 1.9414220896044375e-11 |
| shifted_correct | 116.636260 | [26.27, 28.76, 28.68, 25.24, 24.32] | [128.0, 128.0, 64.0, 128.0, 64.0] | 1.681462651552598e-11 |

## 7. Online compute와 memory

RTX PRO6000 Blackwell Server Edition, CPU 2 threads, float64, batch1. 고정된 20 fresh matched worlds × 5 tasks × 3 repeats로 method당 300 measurements를 측정했다. Factual fit, target generation, compression/selection, selected actual observation update를 포함했다. GPU online choices/predictions가 stored evaluation과 일치하는지 확인했다. CPU는 아래 numerical audit에 별도로 기록했다. GPU/CPU timing은 순차 실행했다. CPU와 GPU 숫자를 섞어 speedup을 만들지 않았다.

| Method | GPU median ms | CPU median ms | GPU contact-event ms | GPU stick-event ms |
|---|---|---|---|---|
| region32 | 4.265 | 2.972 | 3.097 | 3.564 |
| global32 | 3.771 | 2.871 | 2.745 | 3.315 |
| region64 | 4.325 | 3.642 | 3.163 | 3.564 |
| region_cf32 | 6.656 | 5.133 | 4.202 | 5.274 |
| region32_compact | 4.126 | 2.979 | 2.806 | 3.298 |
| native | 3.392 | 2.072 | 2.659 | 3.747 |
| nested128 | 4.767 | 22.904 | 3.872 | 4.798 |
| nested512 | 10.068 | 71.686 | 9.029 | 9.998 |
| qmc64_native | 4.438 | 5.743 | 3.972 | 4.365 |
| pair | 3.204 | 2.274 | 2.623 | 3.636 |
| query | 1.584 | 1.076 | 1.112 | 1.633 |
| probe | 1.542 | 1.063 | 1.056 | 1.610 |

| Method | Bank/cache bytes | Compact payload max bytes | Observed GPU temporary peak bytes | Warm setup ms |
|---|---|---|---|---|
| region32 | 8765440 | 19456 | 2543104 | 5.233 |
| global32 | 8765440 | 19456 | 2543104 | 4.230 |
| region64 | 8765440 | 38912 | 9147904 | 5.785 |
| region_cf32 | 17416192 | 19456 | 2955776 | 7.589 |
| region32_compact | 8765440 | 19456 | 2543104 | 5.922 |
| native | 24117248 | None | 1589248 | 7.891 |
| nested128 | 17309696 | None | 51262976 | 7.553 |
| nested512 | 17334272 | None | 51275264 | 9.314 |
| qmc64_native | 96468992 | None | 6356992 | 5.432 |
| pair | 8650752 | None | 3877376 | 3.593 |
| query | 8650752 | None | 1589248 | 3.702 |
| probe | 8650752 | None | 1589248 | 3.833 |

CPU timing의 원래 strict cross-device check는 region64의 한 action 차이에서 중단됐다. 원본 source와 failure log를 보존하고, 별도의 timing_cpu_audit.py에서 CPU가 실제 선택한 action의 full decoder prediction을 검증하며 timing을 다시 측정했다. 아래는 100 unique cases 중 GPU와 달랐던 action 수다. CPU 결과로 GPU decision-loss 결과를 교체하지 않았다.

| Method | CPU/GPU action mismatch count / 100 | CPU prediction max error |
|---|---|---|
| region32 | 0 | 9.631e-15 |
| global32 | 0 | 9.631e-15 |
| region64 | 1 | 9.631e-15 |
| region_cf32 | 0 | 9.631e-15 |
| region32_compact | 0 | 1.799e-14 |
| native | 0 | 6.106e-15 |
| nested128 | 0 | 9.631e-15 |
| nested512 | 0 | 9.631e-15 |
| qmc64_native | 0 | 9.881e-15 |
| pair | 0 | 9.631e-15 |
| query | 0 | 6.106e-15 |
| probe | 0 | 9.881e-15 |

Compact payload는 representative weights, targets, whitened action means다. Full factual weight array와 bank storage를 없앤 값이 아니다. Cache에는 bank/features와 precomputed noise plan을 포함한다. Temporary peak는100 fixed cases에서 관찰한 추가 CUDA allocated bytes로 모든 possible input의 global bound가 아니다. Python objects, CUDA context, allocator reserve, physical sensing/reset cost는 별도다. Setup은 warm process construction이다.

![Matched loss and online compute](results/tradeoff.png)

## 8. 결과 해석과 논문 기여 가능성

### 8.1 V15에서 무엇이 달라졌는가

V15 counterexample에서는 label 0과 label 1의 observation distributions가 같아도 일부 hypotheses 사이의 거리가 컸다. Pair score는 이 거리를 information으로 잘못 계산했다. V16은 region 안의 likelihoods를 먼저 marginalize하므로 같은 distributions에 대해 정확히 같은 posterior label probability를 얻는다. Uninformative action의 Brier risk는 0.25이며 informative action의 compact estimate는 0.199175였다. Reference quadrature는 0.198986이다.

이 수정은 distribution의 multimodality를 유지할 이유도 보여준다. Mean과 variance가 같고 higher-order structure만 다른 두 label-conditioned mixtures에서도 mixture acquisition은 information을 감지했다. 하지만 이는 의도적으로 만든 mechanism check다. 실제 physical worlds에서 이런 구조가 충분히 자주 나타나고 실질적인 선택 이득으로 연결된다는 사실까지 이 예제만으로 증명할 수는 없다.

Equal-size global mixture가 두 counterexamples에서 region mixture와 같은 risk를 얻었다는 사실이 중요하다. **Counterexample을 고친 데 필요한 것은 mixture-aware risk 계산이며, 현재 실험은 region별 allocation이 필수라는 근거를 주지 않았다.** 같은 region의 hypotheses를 하나의 Gaussian으로 합쳐도 항상 충분하다는 주장 역시 higher-order example 때문에 성립하지 않는다.

### 8.2 Observation selection의 이득은 어디에 남는가

Matched contact-event에서 region32는 512 worlds 중 499개에 query view를 선택했다. Always-query의 loss가 0.044473으로 region32의 0.044491보다 오히려 조금 낮다. Action counts가 다르다는 사실만으로 adaptive selection의 가치가 생기지는 않는다. 현재 contact task에서는 query-type rule이 매우 강하다.

Stick-event에서는 region32가 query/factual/probe를 각각 70/95/347회 선택했고, global mixture는 225/72/215회, nested512는 148/43/321회였다. Ranking은 상당히 다르지만 mean loss는 가까웠다. 이는 여러 actions의 실제 utility 차이가 작거나 서로 다른 worlds에서의 개선과 악화가 평균에서 상쇄될 수 있음을 시사한다. 이 둘을 현재 aggregate result만으로 완전히 분리했다고 주장하지 않는다.

Region32는 LMMSE16보다 stick-event에서 유리했다. 그러나 비슷한 GPU compute를 쓰는 LMMSE64와 nested128도 강하다. LMMSE64는 4.438 ms, nested128은 4.767 ms이며 region32는 4.265 ms다. 두 baseline에 대한 secondary paired CI는 0을 포함한다. 특히 nested128의 stick loss 0.066739는 region32의 0.066922보다 낮다. 느린 nested512 하나와의 speedup만으로 compute frontier를 개선했다고 주장할 수 없다.

### 8.3 Ranking ambiguity와 compression의 한계

첫 32 matched worlds의 stick task에서는 acquisition noise를 Q4에서 Q8로 바꾸거나 Sobol seed를 바꿀 때 region32 action agreement가 각각 78.125%였다. Nested512와 independent-seed nested2048의 agreement도 78.125%다. 따라서 높은 sample count의 nested reference도 exact oracle로 취급할 수 없다. Q8/alternate-seed 결과는 primary policy를 수정하거나 best seed를 고르는 데 사용하지 않았다.

K를 32에서 64로 늘려도 matched stick loss가 단조 개선되지 않았다. Region64의 loss는 0.069161로 region32보다 높다. Cross-fitted region32는 0.067020이며 비용은 6.656 ms로 늘었다. 이것은 compression resolution, hypothetical-noise integration, finite posterior bank가 서로 얽혀 있다는 경고다. 이번 결과만으로 원인을 하나로 단정할 수 없다.

Actual decoder까지 compact mixture로 바꾼 경우 matched stick loss는 0.076230이었다. Full decoder 대비 difference는 **+0.009308**, secondary 95% CI **[+0.003312,+0.016184]**다. Shifted nominal/correct 조건에서도 악화의 CI가 0 위였다. Acquisition에 쓸 만큼의 approximation과 실제 posterior prediction에 필요한 approximation은 구분해야 한다. Factual ESS median은 약 109, selected-observation ESS medians는 대략 22–27이어서 finite QMC representation의 한계도 남는다.

CPU/GPU action 차이는 region64의 world 11, contact-velocity에서만 나타났다. 해당 compact risk는 actions 사이에 약 6×10⁻¹⁷ 수준으로 모두 사실상 0이었고, floating-point cancellation으로 tie-breaking이 달라졌다. 전체 100 unique timing cases에서 다른 methods의 action 차이는 없었다. CPU prediction을 실제 선택한 action의 GPU full-decoder reference와 대조한 maximum error는 전체 methods에서 1.8×10⁻¹⁴ 미만이었다. 이 현상을 substantive ranking uncertainty와 혼동하지 않으며 failure log를 보존했다.

### 8.4 Model mismatch와의 관계

Shifted data에 nominal inference를 쓴 stick-event loss는 region32 0.066489, native 0.069341, nested512 0.066478이었다. Correct-law diagnostic에서는 각각 0.060866, 0.058345, 0.058237이었다. 올바른 model을 제공했을 때에도 region32가 LMMSE보다 항상 좋아지는 것은 아니다. Correct-law condition은 law를 이미 안다는 privileged diagnostic이며 실용적인 mismatch adaptation 방법이 아니다.

Flexible-law worlds는 247개, unchanged base-law controls는 265개다. Controls의 factual data, observations, targets와 주요 methods의 losses가 paired nominal/shifted 조건에서 완전히 같은지 확인했다. 이는 model-mismatch manipulation의 구현 검증이다. 이 결과를 acquisition approximation만으로 model mismatch를 해결했다는 주장으로 확대하지 않는다.

### 8.5 가장 가까운 선행연구와 남는 차별점

Predictive quantity를 직접 겨냥한 experimental design은 이미 연구됐다. Zhong 등의 **Goal-Oriented Bayesian Optimal Experimental Design for Nonlinear Models using Markov Chain Monte Carlo**는 nonlinear QoI에 대한 EIG를 nested Monte Carlo, MCMC와 KDE로 추정한다. V16은 contact/stick prediction의 Brier loss를 직접 사용하는 compact estimator이지만, prediction-oriented acquisition이라는 목적 자체가 새롭지는 않다. [원문, v2](https://arxiv.org/html/2403.18072v2)

Zhu와 Hainy의 2026 preprint **Sequential Bayesian Experimental Design for Prediction in Physical Experiments Informed by Computer Models**는 KOH model의 predictive MI를 다루며 Gaussian mixture compression을 사용한다. Section 4.1에는 whitening, k-means moment matching, KNN-guided Runnalls merging이 제시된다. V16은 event-conditioned mass를 유지한 systematic representatives와 원래 sensor kernels를 사용한다는 구현 차이가 있다. 하지만 predictive mixture를 압축해 nested integration을 줄인다는 큰 아이디어는 선행연구와 겹친다. 해당 compression algorithm을 직접 구현해 비교한 것은 아니므로 성능 우위를 주장하지 않는다. [원문, Section 4.1](https://arxiv.org/html/2603.16756v1#S4.SS1)

Long의 **Multimodal Information Gain in Bayesian Design of Experiments**도 여러 modes를 mixture로 유지하는 Laplace approximation과 nested importance sampling을 다룬다. 따라서 “multimodality를 보존하면 single-Gaussian approximation보다 나을 수 있다”만으로는 충분한 novelty가 되지 않는다. V16의 conditional sensor-kernel mixture와는 구성 및 utility가 다르지만, 이 차이가 실용적 이득을 만드는지 별도로 입증해야 한다. [원문](https://arxiv.org/html/2108.07224v1)

위 논문들을 근거로 내리는 해석은 다음과 같다. **남을 수 있는 기여는 새로운 Bayes-risk objective가 아니라, downstream decision을 보존하는 compression을 언제·얼마나 수행해야 하는지를 검증 가능하게 정하는 방법**이다. 현재의 fixed K와 systematic resampling은 이를 위한 baseline이다.

### 8.6 다음 연구의 진행 기준

현재 구현은 reproducible negative/positive evidence를 함께 갖춘 실험으로 보존한다. 다음 방법을 개발한다면 fixed K를 사후 조정하는 방향보다 **acquisition error와 action gap에 따라 필요한 regions만 확장하는 adaptive compression**을 우선한다. 단, 다음 내용은 아직 구현하거나 성능을 입증한 결과가 아니라 후속 가설이다.

1. **Decision-loss-aware compression error:** 각 action의 compact risk와 independent held-out posterior reference 사이의 error를 추정한다. Top-two action gap이 error보다 작으면 components 또는 hypothetical samples를 추가한다. Confidence bound나 certificate를 주장하려면 sequential stopping과 reference error까지 반영한 별도 정당화가 필요하다.
2. **Region stratification의 필요성 검증:** Rare decision regions, 여러 contact modes, overlapping observation mixtures를 포함하는 새로운 predeclared distribution에서 equal-budget global resampling, generic GMM compression, optimized nested128 및 strong fixed rules와 비교한다. Counterexample를 알고 고른 몇 worlds만으로 성능을 주장하지 않는다.
3. **선택과 decoder를 분리한 compute frontier:** Acquisition만 compact하게 만들 때의 이득을 먼저 검증하고, compact decoder는 별도 가설로 다룬다. Full factual bank와 compression 비용을 계속 포함한다. GPU와 CPU 각각의 runtime–loss frontier를 보고, 두 event tasks의 사전 noninferiority 조건을 모두 통과해야 approximation claim을 한다.
4. **Simulation 밖의 검증:** 현재는 local impulse surrogate와 직접적인 noisy profiles다. Raw RGB, occlusion, geometry extraction error와 실제 contact trajectory를 포함한 검증이 없으므로 vision/robotics paper의 실증 수준에는 미치지 못한다. 별도 환경에서 동일한 conditional-mixture failure와 해결 효과가 나타나야 한다.

이번 단계에서 논문 기여를 정직하게 요약하면, **“pairwise hypothesis separation은 decision information을 과대평가할 수 있고, compact predictive mixtures로 이를 교정할 수 있다. 하지만 이 교정만으로 region-aware compression의 독자적 성능이나 nested-equivalent accuracy는 보장되지 않는다”**이다. 이 finding은 의미가 있으나 현재 수치만으로 넓은 범위의 새 method를 확정하기에는 부족하다.


## 9. 재현 자료와 실행 상태

Server directory: `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/region_mixture_v16`. Allocation2680491, node56의 idle GPU를 확인하고 guard 아래 실행했다. 기존 jobs/allocation을 변경하지 않았다. 모든 simulation/inference/statistics는 allocated compute node에서 수행했다.

`PROTOCOL.md`, `mixture.py`, `baselines.py`, `runtime.py`가 핵심 설계와 구현이다. `validate.py`, `counterexample_global.py`는 mechanism 및 baseline-specificity checks다. `run_policy.py`는 public evidence만 읽고 policies를 freeze한 뒤 `evaluate.py`가 private outcomes를 평가한다. `timing.py`, `analyze.py`가 online costs와 paired statistics를 계산한다.

Raw predictions/actions, numerical audit, timing은 `results/evaluations`, `results/policies`, `results/audits`, `results/world_losses.npz`, `results/summary.json`, `results/timing_gpu.json`, `results/timing_cpu.json`에 있다. Source/policy hashes, worker exit records와 phase markers를 남겼다. `INTEGRITY.json`, `MANIFEST.json`은 final verification과 SHA256 manifest다. V15 sealed artifacts의 hash를 대조했다.

재실행은 sealed directory를 덮어쓰지 않고 새 directory에 code를 복사한다. Allocated idle GPU에서 `guard.py run.py`, 이어서 CPU timing `timing_cpu_audit.py`, `analyze.py`, `counterexample_global.py`, `statistical_audit.py`, `write_report.py`, `seal.py` 순서다. CPU-only scripts는 `BELIEF_DEVICE=cpu`로 실행한다.
