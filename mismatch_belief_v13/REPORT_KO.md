# Compact belief의 model mismatch detection과 model expansion — v13 서버 실험 보고서

작성일: 2026-11-12 KST. Server: `datacenter3` → `node56`, 기존 allocation `2680491`의 idle GPU 1개. GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition. 모든 simulation, inference, timing과 statistical analysis는 allocated compute node에서 실행했다.

## 1. 결론

**Predictive residual detection → observation selection 변경 → 필요한 경우 model expansion이라는 pipeline을 구현하고 실행했다.** 최초 방법의 실패를 분석한 뒤, action별로 detector를 calibration하는 수정안을 별도의 independent worlds에서 다시 검증했다.

수정안은 새로운 replication에서 원래 gate보다 saturating law의 decision loss를 약 **3.3%**, speed-weakening law를 약 **5.3%** 줄였다. 두 비교 모두 six-contrast Bonferroni-adjusted CI에서 감소가 확인되었다. 그러나 **동일한 flexible model을 사용하는 expanded QMC보다 좋다는 증거는 없고, compute 비용도 훨씬 높다.** 현재 method를 논문 수준의 우위가 확인된 결과로 판단하지 않는다.

이번 결과가 주는 연구 방향은 더 구체적이다. **“Factual residual이 크면 mismatch가 있다”는 가정은 이 문제에서 약하다. Factual evidence를 설명할 수 있는 여러 contact laws가 query에서 다른 결과를 낼 수 있으므로, 다음 observation이 model 간 query disagreement를 얼마나 해소하는지를 봐야 한다.** 이 해석은 현재 실패 원인과 일관된 후속 가설이며, 새로운 algorithm의 성공을 입증한 결과는 아니다.

## 2. 구현한 방법

### 2.1 기존 compact belief를 유지한 detector

Base는 v12의 GH5, 4-step Gaussian-sum assumed-density filter다. Boundary family 2개 × contact model 2개의 4 components를 사용하고, component마다 latent mean 3개, full covariance 9개, log weight 1개를 저장한다. Numerical belief payload는 **416 bytes/world**다. Full posterior나 QMC의 weights/moments/teacher labels를 입력받지 않는다.

Pre-observation score는 다음과 같이 계산한다.

1. Factual boundary profile 9개만으로 geometry belief를 갱신한다.
2. 이 belief에서 아직 fit하지 않은 factual outgoing velocity의 mixture predictive density를 계산한다.
3. Whitened coordinates의 negative log predictive density를 score로 사용한다.

따라서 factual velocity를 먼저 fit한 뒤 같은 velocity의 reconstruction residual을 평가하는 detector가 아니다. Predictive density는 각 Gaussian component의 measurement moments로 만든 approximate Gaussian mixture이며, 정확한 likelihood integral은 아니다.

Pre-score가 threshold보다 크면 `probe`, 그렇지 않으면 기존 LMMSE gain으로 observation을 고른다. Post-score는 factual posterior로부터 **실제로 선택하여 받은 observation**의 predictive negative log density를 평가한다. Pre 또는 post flag가 있으면 expanded model로 factual evidence와 selected observation을 replay한다. Flag가 없으면 base model만 갱신한다.

Matched calibration 512 worlds에서 pre-threshold를 정하고, 별도의 matched calibration 512 worlds에서 policy를 실행해 post-threshold를 정했다. 각 threshold는 finite-sample rank로 계산한 97.5% upper quantile이다. Shifted-law data나 test targets로 threshold를 fit하지 않았다. 이 실험의 empirical false-alarm calibration이며, 모든 covariates에서 5% error를 보장하는 theorem으로 해석하지 않는다.

### 2.2 Flexible model expansion

기존 frictionless/Coulomb models를 유지하면서 geometry family별로 다음 flexible tangential-response model을 하나씩 추가했다.

\[
j_t=v_t\,\sigma\left(b_0+b_1\log\frac{|v_t|+0.02}{0.5}\right),
\quad b_0\sim U[-4,2],\quad b_1\sim U[-2,2].
\]

Normal restitution은 기존 model을 유지한다. Tangential impulse는 incoming tangential velocity와 같은 부호이고 크기가 그보다 작으므로 이 response는 tangential energy를 증가시키지 않는다. 기존 4 models에 prior mass 0.5, 추가 2 models에 0.5를 배분했다. Expanded compact는 6 components × 5D latent state와 GH3, 4-step update를 사용한다. Existing models에서 추가 두 latent dimensions는 nuisance이며 response에 영향을 주지 않는다.

Expanded belief payload는 **1,488 bytes/world**다. Runtime은 replay를 위해 factual evidence 11 float64, 즉 88 bytes와 selected observation buffer도 필요하다. Base/expanded state가 잠시 함께 존재하는 temporary memory, Python/Tensor overhead, shared quadrature cache는 별도다. 실제 process memory가 416 또는 1,488 bytes라는 뜻이 아니다.

이는 특정 held-out law의 정답 공식을 menu에 넣는 방식은 아니다. 다만 viscous law는 `b₁=0`인 special case로 표현할 수 있고, 다른 laws도 유한한 관측 범위에서 근사할 수 있다. “모든 test laws가 expanded candidate support 밖에 있다”는 주장은 하지 않는다. Model expansion은 고정된 flexible family를 추가하는 방법이며 arbitrary symbolic law discovery는 아니다.

## 3. Worlds와 comparison design

Base geometry, visibility, target query, observation budget은 v12와 같다. Synthetic 2D point-mass instantaneous contact를 사용한다. Boundary는 `x(y)=s y+a f_g(y)`, `L=25 mm`, `z=y/L`이며 candidate families는 `f₀=Lz²/2`, `f₁=L(z²/2−0.35z⁴)`다. `s∈[−0.4,0.4]`, `a∈[−0.1,0.1]`, restitution `e∈[0.35,0.9]`의 uniform priors를 공유한다.

Factual contact는 `y=0`, incoming velocity `(1,0.35)`이고 query는 `y=25 mm`, incoming velocity `(1,0.9)`다. Factual evidence는 `−10…10 mm`의 boundary samples 9개와 outgoing velocity다. Noise SD는 각각 `1.2 mm`, `0.07 m/s`다. 추가 observation은 한 번만 받는다.

| Action | Measurement | Noise SD |
|---|---|---|
| `query_view` | `19…31 mm` boundary samples 7개 | `1 mm` |
| `factual_view` | `−8…8 mm` boundary samples 7개 | `0.35 mm` |
| `probe` | `y=0`, incoming `(1,−0.7)`의 outgoing velocity | `0.05 m/s` |

모든 방법의 추가 observation budget은 동일하다. Physical sensing/reset latency는 따로 측정하지 않았으며, action cost는 같게 취급했다. 이번 실험은 raw RGB/video encoder나 실제 물체를 검증한 것이 아니라 boundary-preserving samples와 collision velocity를 직접 사용하는 synthetic diagnostic이다.

각 independent base world마다 여섯 paired conditions를 구성했다.

| Condition | Frictional class의 tangential impulse |
|---|---|
| Matched | `sign(vt) min(|vt|, 0.22(1+e)vn)` |
| Viscous replication | `λ vt`, `λ∼U[0.25,0.65]` |
| Saturating full | `sign(vt) α s₀ tanh(|vt|/s₀)`, `α∼U[0.35,0.85]`, `s₀∼U[0.25,0.65]` |
| Weakening full | `vt[0.12+β/(1+(|vt|/v₀)²)]`, `β∼U[0.3,0.6]`, `v₀∼U[0.2,0.6]` |
| Saturating/weakening half | 해당 shifted impulse와 matched impulse를 50:50 혼합 |

Boundary, restitution와 sensor noise는 paired conditions 사이에 공유한다. Frictionless class는 모든 조건에서 변화 없는 negative control이다. 최초 cohort는 256 worlds 중 실제 law가 바뀌는 130 worlds와 negative controls 126 worlds다. Follow-up cohort는 각각 126/130 worlds다. Observation noise는 world-condition마다 16회 반복했다. Independent sample unit은 noise repeat가 아니라 **base world**다.

Viscous는 v12에서 이미 본 family이므로 replication으로 표시했다. Saturating/weakening은 이번 최초 test에서 calibration에 쓰지 않은 새 generating functional forms다. Follow-up은 이미 결과를 본 이 law families의 **새 worlds**에서 재검증한 것이므로, 최초 confirmatory test와 합쳐 family-level generalization을 과장하지 않는다.

Primary gate 외에 base compact LMMSE/always-probe, selection-only gate, expansion-only gate, always-probe + gate, always-expanded compact LMMSE/always-probe를 비교했다. Strong baselines는 base QMC16384와 expanded QMC16384/65536의 LMMSE 및 always-probe다. Expanded QMC는 compact와 같은 model menu/prior를 사용한다. 65536에는 second scramble도 실행했다. QMC는 exact candidate likelihood reweighting을 사용하며, 새로운 observation 이후에도 weights를 update한다.

LMMSE acquisition은 `0.5 tr(C_TY (C_YY+R)⁻¹ C_YT)`로 모든 native selectors에 공통이다. Exact nested expected decision loss를 계산하는 selector는 아니다. Decision loss는 `||predicted query velocity−true query velocity||²/2`, 단위 `(m/s)²`다. Physical robot action success가 아니라 continuous prediction decision을 평가했다.

## 4. 최초 cohort: 단순 residual gate는 충분하지 않았다


| Method | Matched | Viscous | Saturating full | Weakening full |
|---|---|---|---|---|
| base_native | 0.005406 | 0.020760 | 0.012071 | 0.015269 |
| base_probe | 0.006826 | 0.020133 | 0.013883 | 0.018565 |
| select_gate | 0.005426 | 0.020741 | 0.012075 | 0.015273 |
| expand_only | 0.005484 | 0.020097 | 0.012054 | 0.015309 |
| primary_gate | 0.005513 | 0.020075 | 0.012054 | 0.015316 |
| probe_gate | 0.007039 | 0.014569 | 0.013772 | 0.016664 |
| expanded_native | 0.008660 | 0.009659 | 0.011384 | 0.012355 |
| expanded_probe | 0.008934 | 0.009193 | 0.011813 | 0.012650 |
| qmc16_native | 0.005332 | 0.020651 | 0.012021 | 0.014909 |
| qmc16_probe | 0.006706 | 0.020780 | 0.013765 | 0.018568 |
| expanded_qmc16_native | 0.009056 | 0.009580 | 0.011080 | 0.010473 |
| expanded_qmc16_probe | 0.009663 | 0.009215 | 0.011669 | 0.011555 |
| expanded_qmc64_native | 0.008972 | 0.009565 | 0.011142 | 0.010503 |
| expanded_qmc64_probe | 0.009624 | 0.009075 | 0.011603 | 0.011568 |


Primary gate는 base QMC + always-probe보다 saturating/weakening에서 낮은 loss를 보인다. 그러나 base compact native와 거의 같은 결과다. 따라서 이 차이를 새 detector나 expansion의 기여로 돌릴 수 없다. 특히 flexible family를 동일하게 제공한 expanded QMC와 비교하면 우위가 유지되지 않는다.

최초 six primary contrasts에는 30,000 paired world-bootstrap resamples와 Bonferroni-adjusted 99.1667% CI를 적용했다. Negative difference는 primary gate가 유리하다는 뜻이다.


| Condition / baseline | Mean loss difference | Adjusted CI |
|---|---|---|
| viscous/qmc16_probe | -0.000704 | [-0.003202, +0.001804] |
| viscous/expanded_qmc16_probe | +0.010861 | [+0.007244, +0.014878] |
| saturating_full/qmc16_probe | -0.001712 | [-0.003128, -0.000245] |
| saturating_full/expanded_qmc16_probe | +0.000385 | [-0.002317, +0.003123] |
| weakening_full/qmc16_probe | -0.003252 | [-0.004737, -0.001930] |
| weakening_full/expanded_qmc16_probe | +0.003760 | [+0.001016, +0.006673] |


Matched loss ratio(primary/base compact native)는 **1.0199**, 95% CI **[1.0046,1.0420]**이다. 사전에 정한 upper bound 1.10 기준은 통과했지만, 평균적으로 약 2%의 손해가 있다는 결과도 함께 봐야 한다. Harm이 전혀 없다는 뜻은 아니다. 두 새 full laws에서 두 strong baselines를 모두 이겨야 한다는 사전 success criterion은 실패했다.

### Detector가 무엇을 놓쳤는가


| Condition | Pre-score AUROC¹ | Primary expansion: 전체 | Primary expansion: 실제 변경 | Probe+gate expansion: 실제 변경 |
|---|---|---|---|---|
| matched | 0.455 | 2.95% | 3.80% | 4.66% |
| viscous | 0.481 | 3.56% | 5.00% | 42.36% |
| saturating_full | 0.466 | 2.61% | 3.12% | 35.82% |
| weakening_full | 0.491 | 2.59% | 3.08% | 43.46% |


¹ AUROC는 changed-class score와 matched-world score의 비교다. Matched 행의 값은 shifted-law detection 성능이 아니며 참고값이다. 실제 full shifts의 AUROC는 0.47–0.49로 separation이 거의 없었다. 이는 **현재 score가 실패했다는 증거**이며, factual evidence에서 가능한 모든 detector가 실패한다는 정보론적 증명은 아니다.

Primary gate의 matched expansion rate는 2.95%, world-bootstrap 95% CI [1.73%,4.49%]다. 반면 실제 law가 변경된 worlds에서도 expansion은 viscous 5.00%, saturating 3.13%, weakening 3.08%에 그쳤다. Always-probe 후 gate를 적용하면 changed worlds의 expansion rate가 약 36–43%로 올라갔다. 어떤 observation을 받는지가 mismatch를 볼 수 있는지에 영향을 준다.

또한 pooled post-threshold는 15.37이었다. Primary calibration에서 query_view가 338회, probe가 174회 선택되었고, 7D view와 2D probe의 NLL scale이 달랐다. Test에서 probe의 median score는 약 4–5, view는 약 11이었다. Whitened coordinates를 사용하더라도 measurement dimension이 다른 scores를 하나의 threshold로 처리한 점이 probe detection을 불리하게 만들었다.

Common-action diagnostic에서도 gate의 한계가 보인다. Primary와 같은 observations를 받은 뒤 항상 expanded decoder를 쓰면 최초 cohort의 viscous/saturating full/weakening full loss는 각각 **0.011018 / 0.009976 / 0.011812**다. Primary gate는 **0.020075 / 0.012054 / 0.015316**으로 더 높았다. 그러나 matched에서는 always-expanded decoder가 **0.008203**, primary가 **0.005513**이므로 무조건 expansion하는 것도 답이 아니다.

Half-shift도 같은 trade-off를 보인다. Saturating half에서 base native / primary / expanded QMC64 native는 **0.007533 / 0.007799 / 0.010899**, weakening half에서는 **0.009217 / 0.009298 / 0.012015**다. Flexible prior를 넓히면 severe mismatch에는 도움이 되지만 matched·mild regimes에서 prediction variance와 model ambiguity가 늘어날 수 있다. 이 설명은 관측된 trade-off의 해석이며, 원인을 완전히 분리한 ablation은 아니다.

![최초 cohort의 loss와 GPU latency](results/loss_latency.png)

그림은 means만 표시한다. 작은 mean 차이는 표의 CI 없이 superiority로 해석하지 않는다.

## 5. Action-conditioned calibration: independent replication

최초 결과를 보존한 뒤, post-threshold를 matched calibration의 **실제 선택 action별 subgroup**에서 다시 정했다. Primary query_view threshold는 **15.5911**, probe threshold는 **6.9520**이다. Pre-threshold, acquisition, model/prior와 inference 설정은 바꾸지 않았다. Factual-view subgroup은 없어서 기존 pooled threshold를 fallback으로 두었고, 이번 test에서 해당 action은 선택되지 않았다.

새 seed의 independent 256 worlds로 original gate와 actionwise gate를 같은 조건에서 비교했다. 아래가 후속 결과다.


| Method | Matched | Viscous | Saturating full | Weakening full |
|---|---|---|---|---|
| original_gate | 0.006600 | 0.021150 | 0.011777 | 0.014238 |
| actionwise_gate | 0.006595 | 0.019842 | 0.011388 | 0.013481 |
| base_native | 0.006417 | 0.022158 | 0.011662 | 0.014083 |
| base_probe | 0.006786 | 0.020174 | 0.013930 | 0.017537 |
| qmc16_native | 0.005823 | 0.021397 | 0.011675 | 0.013677 |
| qmc16_probe | 0.006765 | 0.020911 | 0.013908 | 0.017640 |
| expanded_probe | 0.009394 | 0.008768 | 0.010502 | 0.011193 |
| expanded_qmc16_native | 0.009776 | 0.009071 | 0.009940 | 0.010445 |
| expanded_qmc16_probe | 0.011186 | 0.008706 | 0.011141 | 0.011294 |
| expanded_qmc64_native | 0.009681 | 0.008967 | 0.009966 | 0.010453 |
| expanded_qmc64_probe | 0.011162 | 0.008663 | 0.011010 | 0.011168 |


후속 six primary contrasts는 두 full laws에서 actionwise gate와 original gate / base QMC16-probe / expanded QMC16-probe의 비교다. 이 six comparisons 안에서 다시 Bonferroni-adjusted 99.1667% CI를 사용했다. 첫 cohort와 합쳐 p-value나 CI를 계산하지 않았다.


| Condition / baseline | Actionwise − baseline | Adjusted CI |
|---|---|---|
| saturating_full/original_gate | -0.000389 | [-0.000860, -0.000018] |
| saturating_full/qmc16_probe | -0.002521 | [-0.004538, -0.000361] |
| saturating_full/expanded_qmc16_probe | +0.000247 | [-0.002470, +0.003056] |
| weakening_full/original_gate | -0.000757 | [-0.001541, -0.000144] |
| weakening_full/qmc16_probe | -0.004159 | [-0.006350, -0.001906] |
| weakening_full/expanded_qmc16_probe | +0.002188 | [-0.000177, +0.004639] |


Action-conditioned calibration은 original gate 대비 saturating full에서 약 **3.3%**, weakening full에서 약 **5.3%** 개선했다. 두 adjusted CI가 모두 0 아래다. 그러나 expanded QMC16-probe 대비 adjusted CIs는 0을 포함하며, superiority는 확인되지 않았다. Saturating/weakening의 mean 자체도 expanded QMC보다 높다. CI가 0을 포함한다는 사실을 equivalence로 해석하지 않는다.

추가로 native QMC와 비교한 descriptive 95% CI는 다음과 같다. 이는 후속 primary contrasts에 포함하지 않은 진단이다.


| Condition / baseline | Actionwise − baseline | Descriptive 95% CI |
|---|---|---|
| saturating_full/base_native | -0.000274 | [-0.000640, +0.000073] |
| saturating_full/qmc16_native | -0.000287 | [-0.001159, +0.000692] |
| saturating_full/expanded_qmc64_native | +0.001421 | [-0.000416, +0.003279] |
| weakening_full/base_native | -0.000602 | [-0.001199, -0.000084] |
| weakening_full/qmc16_native | -0.000195 | [-0.001221, +0.000910] |
| weakening_full/expanded_qmc64_native | +0.003028 | [+0.001406, +0.004717] |


Always-probe만 비교했을 때 보이던 차이가 strong native selector까지 유지되는지 반드시 확인해야 한다. 현재 evidence로는 actionwise detector의 modest improvement와 end-to-end superiority를 구분해야 한다.

Actionwise gate의 matched/base-native loss ratio는 **1.0279**, 95% CI **[1.0045,1.0623]**다. 10% harm margin은 통과했지만 약 2.8%의 matched penalty가 남았다. Matched expansion rate는 original 4.17%, actionwise 4.35%다. 실제 shifted-class expansion rate는 saturating full에서 **2.43% → 10.81%**, weakening full에서 **2.58% → 16.22%**로 증가했다. Calibration 수정이 detection과 loss에 영향을 준 것은 재현되었지만, 여전히 대부분의 changed worlds에서는 expansion하지 못한다.

## 6. Compute와 memory 비용

Timing은 float64, batch1, 같은 allocated hardware에서 측정했다. 최초 cohort의 fixed48 cases ×3 repetitions에 warmup과 CUDA synchronization을 적용했다. CPU는 사전 protocol에 포함한 동일 code의 2-thread 실행이다. 한 번의 factual inference, detection/selection, 실제 selected observation의 update/expansion/prediction까지 포함한다. Offline all-action evaluation cache로 runtime을 대신하지 않았다.


| Method | GPU median ms | GPU p90 ms | CPU median ms |
|---|---|---|---|
| base_native | 24.963 | 28.764 | 4.743 |
| base_probe | 20.582 | 23.891 | 3.954 |
| primary_gate | 38.142 | 41.019 | 7.240 |
| probe_gate | 33.352 | 46.092 | 6.396 |
| expanded_native | 29.156 | 32.502 | 6.402 |
| expanded_probe | 22.425 | 26.126 | 5.572 |
| qmc16_probe | 0.712 | 0.796 | 0.204 |
| expanded_qmc16_probe | 0.705 | 0.829 | 0.204 |
| expanded_qmc64_native | 3.228 | 3.824 | 3.399 |
| expanded_qmc64_probe | 1.721 | 1.832 | 0.609 |


최초 fixed timing의 primary gate cases에는 우연히 expansion event가 없었다. 이 사실을 숨기지 않고 condition별 flag/nonflag strata에서 4 events씩 추가로 측정했다. Primary gate의 GPU branch median은 nonflag **36.858 ms**, expansion **48.439 ms**다. 실제 flag fractions로 conditional means를 가중한 estimated mean은 **37.402 ms**다. Primary median과 이 stratified estimate는 서로 다른 statistics다.

수정안과 QMC는 follow-up의 동일48 cases에서도 측정했다. 아래 fixed medians는 같은 timing boundary를 사용한다.


| Follow-up method | GPU median ms | CPU median ms |
|---|---|---|
| original_gate | 38.326 | 7.060 |
| actionwise_gate | 37.774 | 7.052 |
| qmc16_probe | 0.671 | 0.246 |
| expanded_qmc16_probe | 0.675 | 0.194 |
| expanded_qmc64_probe | 1.818 | 0.555 |
| expanded_qmc64_native | 3.482 | 3.304 |


수정안의 GPU flag/nonflag stratum medians는 **49.206 / 37.112 ms**, population-weighted estimated mean은 **38.157 ms**다. Expansion하지 않아도 profile-only inference와 predictive density calculation 비용을 지불한다. Conditional expansion이 항상-expanded inference보다 빠르다는 가정은 현재 구현에서 성립하지 않았다.

Numerical state는 base **416 bytes**, expanded **1,488 bytes**, QMC16 **131,072 bytes**, QMC64 **524,288 bytes/world**다. Primary gate의 shared sigma-point cache는 **15,664 bytes**, QMC16의 shared forward/moment cache는 **28,442,624 bytes**, QMC64는 **113,770,496 bytes**다. Cache는 고정된 query/sensors에서 여러 worlds가 공유한다. Setup/cold-start, sensor reset, Python/Tensor object overhead를 포함한 total application cost는 이 숫자와 다르다.

QMC는 고정된 forward tables와 moment features를 재사용하는 반면 compact는 많은 작은 matrix operations를 반복한다. 이 regime에서는 state가 작다는 장점이 latency 우위로 이어지지 않았다. Hardware-specific kernel fusion이나 changing-query cache invalidation까지 검증한 것은 아니지만, 현재 결과로 strong QMC보다 compute-efficient하다고 주장할 수 없다.

## 7. Numerical checks와 한계

- Expanded update의 linear-Gaussian closed-form validation 최대 error는 **9.21e-15**였다. Pilot covariance PSD, probability normalization와 finite predictions를 확인했다.
- Online runtime prediction과 frozen all-action evaluation prediction을 실제 timing cases에서 비교했다. Action/flag 일치와 numerical prediction agreement를 assertions로 검증했다. 실제 gate에서는 selected observation 하나만 전달한다.
- 최초 cohort의 expanded QMC16 selected-observation ESS median은 condition별 약 **133–148**, QMC64는 **528–593**이었다. QMC64 second scramble의 full-law mean loss 차이는 대체로 수만분의1 수준이며, 최초 six conditions의 seed-difference 95% CIs는 모두 0을 포함했다. 두 seeds만 확인했으므로 exact integration convergence를 보증하지 않는다.
- Frictionless negative controls의 paired loss differences는 모든 분석 methods에서 0이었다. Source/policy/data hashes, compact-before-QMC ordering, original v12 artifact integrity도 별도로 검사한다.
- 최초 실행은 calibration NPZ의 duplicate key 때문에 test 생성 전에 실패했다. 저장 key를 수정하고 새 results에서 실행했으며, 실패 기록은 `pretest_attempt1/`에 보존했다. Model나 threshold rule을 이 오류를 이유로 바꾸지 않았다.
- Initial cohort와 follow-up cohort는 서로 다른 worlds지만 matched calibration을 공유한다. Follow-up은 첫 결과에서 발견한 calibration 문제에 대한 post-primary adaptation이다. 한 번의 성공적인 replication을 대규모 generalization evidence로 확대하지 않는다.
- Current model menu/prior, fixed query/visibility와 세 observation actions에 국한된다. Extra flexible models의 prior mass0.5는 고정했고, prior-mass sensitivity나 hierarchical shrinkage는 비교하지 않았다.
- Expanded compact는 GH3, base는 GH5를 사용한다. Expanded compact–QMC 차이에는 finite quadrature/assumed-density error가 포함된다. Model family의 표현력과 inference approximation을 동일시하지 않는다.
- Measurement dimension별 calibration을 고쳤어도 threshold 최적화가 decision regret를 직접 줄이는 것은 아니다. Matched-data quantile는 anomaly rarity를 맞추는 도구이며, 어느 observation이 query prediction을 개선하는지의 objective와 다르다.
- 모든 결과는 synthetic single contact다. Rotation, deformable contact, multi-contact sequence, real camera noise, physical robot cost를 검증하지 않았다.

## 8. 논문 기여 가능성과 다음 방향

### 기존 연구와 구분해야 할 부분

Predictive model checking을 discrepancy로 수행하는 개념은 오래된 연구다. Gelman, Meng, Stern은 posterior predictive assessment와 model fitness의 discrepancy를 다룬다. 이번 prequential approximate-NLL detector가 그 논문의 검정 절차와 동일한 것은 아니지만, “predictive residual로 model을 검사한다”는 개념 자체를 novelty로 제시할 수 없다. [Statistica Sinica 원문 페이지](https://www3.stat.sinica.edu.tw/statistica/j6n4/j6n41/j6n41.htm)

Model discrepancy가 Bayesian experimental design을 왜곡하고, 선택된 observations로 discrepancy model을 보완하는 방향도 선행연구가 있다. Yang, Chen, Wu의 *Active Learning of Model Discrepancy with Bayesian Experimental Design*는 discrepancy learning과 sequential experimental design을 결합한다. 이번 실험은 작은 contact-law mixture와 one-step prediction이라는 다른 setting이지만, 단순한 “detect 후 model correction”만으로 해당 연구와 명확히 구별되지 않는다. [2025 arXiv v2 원문](https://arxiv.org/html/2502.05372v2)

Downstream loss를 중심으로 observation을 설계하고 explicit posterior 계산을 피하는 방향도 *Action-BED*와 겹친다. 이번 LMMSE gate는 그 방법의 재현이나 직접 비교가 아니므로 우위를 주장하지 않는다. [2026 arXiv v2 원문](https://arxiv.org/html/2606.23662v2)

### 현재 evidence로 가능한 claim

| Claim | 판단 |
|---|---|
| Full posterior 없이 mismatch detection과 conditional expansion을 구현했다 | 완료 |
| Action-conditioned calibration이 pooled gate를 개선했다 | Independent worlds의 두 full laws에서 지지 |
| Factual residual만으로 unseen contact law를 잘 감지한다 | 지지되지 않음 |
| Strong expanded QMC보다 decision loss가 낮다 | 입증되지 않음 |
| Conditional compact expansion이 QMC보다 빠르다 | 현재 구현에서 실패 |
| Method paper의 중심 기여가 충분하다 | 현재 evidence로 부족 |

현재 연구를 폐기할 필요는 없지만, “compact mismatch detector가 strong baselines를 이긴다”는 주장으로 밀어붙이는 것은 적절하지 않다. Positive result는 calibration 수정의 modest gain이며, 더 큰 연구 문제는 **factual agreement와 query disagreement를 구분한 observation design**이다.

다음 가설은 다음과 같이 구체화하는 편이 낫다. 아직 실행하지 않은 계획이다.

1. **Disagreement-aware probe selection:** Factual evidence를 비슷하게 설명하는 base/flexible hypotheses의 query predictions가 갈릴 때, 그 차이를 줄이는 observation을 고른다. Gate를 residual 크기 하나에만 의존시키지 않는다. 동일 one-observation budget에서 always-probe와 expanded QMC native를 계속 baseline으로 유지한다.
2. **Decision-regret-aware expansion:** Anomaly probability보다 expanded model이 query decision에 줄 수 있는 이득을 추정한다. Matched/mild conditions에서 불필요한 variance를 늘리는 expansion을 억제해야 한다.
3. **Prior와 representation을 분리:** Flexible prior mass와 shrinkage를 matched/proxy dev에서 고정하고, compact와 QMC에 같은 prior를 준다. Test law를 알아야만 좋은 prior를 고를 수 있는 방법이면 일반적 mismatch 대응이라고 보기 어렵다.
4. **실제 기여의 통과 조건:** 새로운 law families와 worlds에서 expanded QMC native/always-probe 대비 loss 감소가 남아야 한다. Fixed memory 또는 same-device wall-clock budget 아래에서도 advantage가 있어야 하며, matched penalty를 사전에 정한 수준으로 제한해야 한다. 현재 experiment가 이 조건을 만족했다고 말하지 않는다.

이번 v13은 실패 원인을 분해하고 한 가지 calibration 문제를 independent replication으로 고친 연구 단계다. Major-conference method contribution으로 발전하려면 detector refinement를 넘어, model ambiguity가 observation의 가치를 어떻게 바꾸는지에 대한 새로운 objective 또는 검증 가능한 compute–risk trade-off가 필요하다.

## 9. Artifact와 실행 기록

Server directory:

```text
/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/mismatch_belief_v13
```

`PROTOCOL.md`가 최초 사전 design, `ACTIONWISE_ADDENDUM.md`가 새 worlds에서의 후속 design이다. `results/`는 최초 cohort, `results_actionwise/`는 independent replication, `pretest_attempt1/`는 test 이전 저장 오류 기록이다. 기존 v12 files는 변경하지 않았다.

| 파일 | 역할 |
|---|---|
| `compact.py`, `flexible.py`, `detection.py` | Base/expanded compact inference와 predictive scores |
| `runtime.py` | Selected observation만 받는 실제 online gate |
| `worlds.py`, `bank.py` | Private data generation, strong QMC baselines |
| `run_compact.py`, `run_banks.py`, `assemble.py` | Calibration, policy freeze, evaluation |
| `benchmark.py`, `benchmark_branches.py` | 동일 hardware timing, rare expansion branch 보완 |
| `actionwise_replication.py`, `actionwise_timing.py` | 새 worlds와 action-conditioned gate |
| `actionwise_baseline_timing.py` | Replication의 동일 cases에서 QMC timing |
| `analyze.py`, `analyze_actionwise.py`, `diagnostics.py` | Primary/secondary statistics와 failure decomposition |
| `results*/summary.json`, `world_losses.npz` | Aggregate statistics와 independent-world losses |
| `results*/decision_predictions.npz`, `flags.npz` 또는 `expansion_flags.npz` | Frozen decisions와 expansion 기록 |
| `INTEGRITY.json`, `MANIFEST.json` | 무결성 검증과 SHA256 manifest |

Python 3.9.18, PyTorch 2.7.1+cu128, NumPy 1.26.4 환경을 사용했다. GPU worker들은 idle-process guard 아래 실행했고, 다른 jobs/allocation은 취소하거나 변경하지 않았다. GPU allocation 자체는 사용자의 것이므로 유지한다.

재실행할 때는 sealed results를 덮어쓰지 말고 code를 새 directory로 복사한다. Slurm allocation 안에서 idle GPU를 확인한 다음 `run_compact.py` → `second_phase.py` → `benchmark.py --cpu` → `analyze.py` 순서로 실행한다. Follow-up은 `actionwise_replication.py` → `actionwise_timing.py` 및 `--cpu` → `actionwise_baseline_timing.py` 및 `--cpu` → `analyze_actionwise.py` 순서다. Scripts의 GPU/CPU timing은 겹쳐 실행하지 않는다. 실제 실행은 `../continuous_v2/.venv/bin/python`과 `srun`을 사용했다. Logs, worker exit records, source snapshots와 policy hashes가 각 phase의 실행 evidence다.
