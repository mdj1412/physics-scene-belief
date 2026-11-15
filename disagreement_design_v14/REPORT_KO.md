# Factual-equivalent hypotheses의 query disagreement를 이용한 observation selection — v14

작성일: 2026-11-12 KST. `datacenter3`를 통해 `node56`의 기존 allocation `2680491`, idle NVIDIA RTX PRO 6000 Blackwell Server Edition 1개에서 실행했다. Python 3.9.18, PyTorch 2.7.1+cu128, NumPy 1.26.4를 사용했다. 모든 simulation, inference, timing과 statistical analysis는 allocated compute node에서 수행했다.

## 1. 결론

**Factual fit은 비슷하지만 query prediction이 다른 hypotheses를 구별하도록 observation을 선택하는 방법을 구현했다.** Expanded QMC와 독립 compact belief 양쪽에 같은 selector를 적용하고, 새로운 **256 base worlds × 6 paired conditions × actual observation noise 16회**에서 평가했다.

이번 결과는 “mechanism은 작동하지만 strong baseline 대비 decision-quality 이득은 입증되지 않았다”다. Physical mechanism checks에서는 ambiguity의 원인에 따라 `probe`와 `query_view`를 다르게 선택했다. 그러나 saturating/weakening full conditions의 six primary comparisons에서 loss difference의 adjusted CI는 모두 0을 포함했다. 기존 expanded QMC의 LMMSE selector가 더 빠르면서 mean loss도 조금 낮았다.

계산상 장점은 있다. QMC16 disagreement selector의 GPU median latency는 **6.358 ms**, cross-fitted nested expected-loss selector는 **15.759 ms**로 약 **2.48배** 차이가 났다. CPU에서는 **2.214 ms 대 34.506 ms**, 약 **15.58배**다. 하지만 QMC16 LMMSE는 GPU **2.170 ms**, always-probe는 **0.709 ms**였다. 따라서 nested calculation의 저렴한 surrogate로서의 가능성과, 전체 strong-baseline frontier에서의 우위는 구분해야 한다.

**현재 형태의 독립적인 method-paper 기여는 부족하다.** Query-relevant uncertainty를 줄인다는 개념 자체도 기존 decision-focused experimental design과 겹친다. 이번 결과만으로 새로운 acquisition 원리나 실제 unseen-law robustness를 확보했다고 주장하지 않는다.

## 2. 무엇을 구현했는가

### 2.1 Primary comparison은 같은 belief에서 selector만 교체

Primary method는 `qmc16_pair`다. `qmc16_native`, `qmc16_probe`, `qmc16_nested128`과 같은 expanded QMC16384 factual posterior 및 actual-observation decoder를 사용한다. 이렇게 해야 결과 차이를 inference approximation이 아니라 observation selection에 귀속할 수 있다.

별도로 `compact_pair`를 구현했다. 이는 기존 expanded compact의 6 Gaussian components, 5D latent mean/covariance와 GH3/4-step update를 사용한다. Compact phase는 QMC module을 import하지 않고 policy freeze와 evaluation을 마친 뒤 QMC phase를 실행했다. 따라서 compact variant는 full posterior의 moments를 받아 압축한 방법이 아니다. 반면 primary QMC variant는 QMC posterior를 사용하는 방법이므로, full posterior를 완전히 대체했다는 claim을 하지 않는다.

V13의 conditional mismatch gate는 이번 comparison에 넣지 않았다. Model menu와 prior를 고정한 상태에서 disagreement selection 자체를 검증했다. Existing base models 4개에 total prior mass0.5, flexible contact models 2개에0.5를 둔다. Flexible tangential impulse는 다음과 같다.

\[
j_t=v_t\,\sigma\left(b_0+b_1\log\frac{|v_t|+0.02}{0.5}\right),
\qquad b_0\sim U[-4,2],\quad b_1\sim U[-2,2].
\]

Normal restitution과 geometry priors는 v13과 같다. Expanded models의 prior mass나 filter 설정을 test 결과에 따라 조정하지 않았다.

### 2.2 Query-weighted pair discrimination score

Factual posterior에서 deterministic systematic inverse-CDF resampling으로 **K=256 hypotheses**의 committee를 만든다. Compact는 자신의 sigma-point distribution, QMC는 자신의 factual weights를 사용한다. Posterior weights는 resampling frequency를 통해 반영되며, resampled hypotheses에 weights를 다시 곱하지 않는다.

Hypothesis i의 factual log likelihood를 `ℓᵢ`, query prediction을 `Tᵢ`, action a의 predicted observation mean을 `μₐᵢ`라고 하자. 다음 조건을 만족하는 pair만 사용한다.

\[
E_{ij}=\mathbf1\{\ell_i,\ell_j\ge\max_k\ell_k-8\}
\mathbf1\{|\ell_i-\ell_j|\le2\}.
\]

이 실험에서 “factual fit이 비슷하다”는 위 log-likelihood band로 정의했다. `|ℓᵢ−ℓⱼ|≤2`는 likelihood ratio 최대 약7.39배를 허용하므로, 엄밀한 observational equivalence나 완전히 같은 fit은 아니다. Committee 내 relative fit을 제한할 뿐 absolute model adequacy를 보장하지 않는다. 두 cutoffs는 test 전에 고정한 heuristics다.

Query disagreement와 observation discriminability는 다음과 같다.

\[
D_{ij}=\frac12\|T_i-T_j\|^2,
\qquad B_a(i,j)=1-\exp\left[-\frac18
(\mu_{ai}-\mu_{aj})^T R_a^{-1}(\mu_{ai}-\mu_{aj})\right].
\]

`Bₐ`는 동일 covariance의 두 Gaussian observation distributions 사이 Bhattacharyya overlap의 complement다. 두 hypotheses가 같은 query prediction을 내면 `Dᵢⱼ=0`, observation으로 구별되지 않으면 `Bₐ=0`이다. 최종 score는 다음의 eligible-pair average다.

\[
S(a)=\frac{\sum_{ij}E_{ij}D_{ij}B_a(i,j)}{\max(1,\sum_{ij}E_{ij})}.
\]

Denominator는 action에 공통이므로 action ranking을 바꾸지 않는다. 이 score가 **exact expected Bayes-risk reduction이라는 주장은 하지 않는다.** Soft posterior update의 실제 효과를 pairwise Gaussian separation으로 근사한 surrogate다. 모든 scores가 수치적으로0이면 같은 belief의 LMMSE로 fallback하도록 구현했으나, 이번 test에서 fallback은 발생하지 않았다.

두 ablations를 함께 고정했다. `unfiltered`는 factual-fit band를 제거하고 query weight를 유지한다. `agnostic`은 factual-fit band를 유지하지만 `Dᵢⱼ=1`로 두어 query prediction과 무관한 discrimination을 수행한다.

### 2.3 Cross-fitted nested expected-loss baseline

Inner QMC16384는 seed20261112, hypothetical observations를 만드는 outer QMC16384는 independent scramble seed20261113을 사용한다. 두 banks 모두 factual evidence로 reweight한다. Outer posterior에서 hypothesis를 고르고 Gaussian observation noise를 더해 128 hypothetical observations를 생성한다. Scrambled Sobol randomness는 actions 사이에 공유한다.

Inner bank는 각 hypothetical observation을 반영한 posterior predictive mean/second moment를 계산한다. Acquisition은 current query variance와 expected updated query variance의 차이다. Actual selected observation 이후 prediction은 inner bank만 사용한다. 이 baseline의 timing에는 두 factual fits와 outer cache가 모두 포함된다.

이는 같은 유한 bank에서 outer/inner calculation을 모두 하는 경우의 self-reuse를 줄이기 위한 cross-fitting이다. Finite inner bank와128 outer samples의 approximation은 남으므로 exact oracle로 부르지 않는다. 또한 true shifted law를 알고 있는 oracle이 아니라, 동일 candidate model 내부의 expected-loss baseline이다.

## 3. Mechanism checks와 experimental setup

### 3.1 같은 query·visibility에서도 observation을 다르게 고르는가

Implementation validation에서 다음 두 사례를 만들었다. Candidate equations만 사용한 mechanism checks이며, test dataset에서 성능이 좋은 사례를 골라낸 것이 아니다.

| Ambiguity | Factual prediction 차이 | 두 query predictions `(vx,vy)` | 선택 |
|---|---|---|---|
| Contact response | 정확히0 | `(-0.6000,0.7097)` / `(-0.6000,0.3386)` | `probe` |
| Geometry family | Boundary maximum0.0224 mm, factual velocity 동일 | `(-0.4733,0.7270)` / `(-0.6404,0.4697)` | `query_view` |

Contact case에서는 geometry observations가 두 hypotheses를 구별하지 못해 view scores가0이고 probe score는0.031132였다. Geometry case에서는 probe가 같은 response를 내서 score가0이고 query_view score는0.016479였다. **Factual evidence가 같거나 거의 같아도 query outcome을 가르는 ambiguity의 종류에 따라 필요한 observation이 달라지는 동작은 구현되었다.** 이 두 사례가 평균 decision loss 우위를 입증하는 것은 아니다.

### 3.2 새로운 worlds와 동일 observation budget

Test seed는 `3100101 + 137×world_index`, pilot16 seed는 `3000101 + 137×index`다. Pilot은 correctness checks에만 사용했고 hyperparameter 선택에 쓰지 않았다. Test의 256 worlds 중 실제 law가 변경되는 frictional class는 **135**, unchanged frictionless negative control은 **121**이다. Independent sample unit은 base world이며 actual noise16 repeats를 독립 worlds로 세지 않는다.

Synthetic 2D point-mass instantaneous collision을 사용한다. Boundary는 `x(y)=s y+a f_g(y)`, `L=25 mm`, `z=y/L`이며 `f₀=Lz²/2`, `f₁=L(z²/2−0.35z⁴)`다. Priors는 `s∼U[−0.4,0.4]`, `a∼U[−0.1,0.1]`, `e∼U[0.35,0.9]`다.

Factual contact는 `y=0`, incoming velocity `(1,0.35) m/s`, query는 `y=25 mm`, incoming velocity `(1,0.9) m/s`로 고정했다. Factual evidence는 `−10…10 mm` boundary samples9개와 outgoing velocity다. Noise SD는 각각1.2 mm와0.07 m/s다. 추가 action은 아래 셋이고 최대 한 번만 받는다. `none`은 no-additional-observation diagnostic이다.

| Action | Observation | Noise SD |
|---|---|---|
| `query_view` | `19…31 mm` boundary samples7개 | 1 mm |
| `factual_view` | `−8…8 mm` boundary samples7개 | 0.35 mm |
| `probe` | `y=0`, incoming `(1,−0.7)`의 outgoing velocity | 0.05 m/s |

World마다 six paired conditions를 만들고 parameters와 factual/additional noise를 공유했다.

| Condition | Frictional class의 tangential impulse |
|---|---|
| `matched` | 기존 Coulomb law, `sign(vt) min(|vt|,0.22(1+e)vn)` |
| `viscous` | `λvt`, `λ∼U[0.25,0.65]` |
| `saturating_full` | `sign(vt) αs₀ tanh(|vt|/s₀)`, `α∼U[0.35,0.85]`, `s₀∼U[0.25,0.65]` |
| `weakening_full` | `vt[0.12+β/(1+(|vt|/v₀)²)]`, `β∼U[0.3,0.6]`, `v₀∼U[0.2,0.6]` |
| 각 `half` condition | 해당 shifted impulse와 Coulomb impulse를50:50 혼합 |

이 law families는 v13에서 이미 다룬 것이다. **새로운 worlds에서의 replication**이며, 처음 보는 law family까지 일반화했다고 표현하지 않는다. 이번에는 boundary family를 별도로 OOD로 바꾸지 않았다. 또한 `matched`는 기존 base-law control이라는 뜻이다. Expanded inference prior는 flexible family에0.5 mass를 주지만 이 control의 true generator는 base laws만 사용하므로, population prior까지 완전히 일치하는 Bayesian model-consistent control은 아니다.

Loss는 `||predicted query velocity−true query velocity||²/2`, 단위 `(m/s)²`다. Continuous prediction decision을 평가하며 physical robot control success는 아니다. Actual sensing/reset time은 측정하지 않았고 action costs를 같게 취급했다.

## 4. Decision loss: strong baselines 대비 우위가 확인되지 않음


| Method | Base-law control | Viscous | Saturating full | Weakening full |
|---|---|---|---|---|
| qmc16_pair | 0.009421 | 0.009437 | 0.010759 | 0.010507 |
| qmc16_native | 0.009697 | 0.010413 | 0.010467 | 0.010201 |
| qmc16_probe | 0.009837 | 0.009398 | 0.010851 | 0.010587 |
| qmc16_nested128 | 0.009656 | 0.009365 | 0.010899 | 0.010474 |
| qmc64_pair | 0.009500 | 0.009607 | 0.010648 | 0.010502 |
| qmc64_native | 0.009684 | 0.010564 | 0.010284 | 0.010208 |
| qmc64_probe | 0.009832 | 0.009352 | 0.010828 | 0.010534 |
| compact_pair | 0.008699 | 0.009438 | 0.011014 | 0.011704 |
| compact_native | 0.008614 | 0.009434 | 0.010495 | 0.011114 |
| compact_probe | 0.009094 | 0.009278 | 0.011061 | 0.011361 |


`native`는 LMMSE acquisition이다. Primary comparisons는 두 full laws에서 QMC16 pair와 QMC16 native / always-probe / nested128의 six contrasts다. 30,000 paired world-bootstrap resamples와 Bonferroni-adjusted **99.1667% CI**를 사용했다. Negative difference는 pair selector가 유리하다는 뜻이다.


| Condition / baseline | Pair − baseline | Adjusted CI |
|---|---|---|
| saturating_full/qmc16_native | +0.000292 | [-0.000497, +0.001047] |
| saturating_full/qmc16_probe | -0.000092 | [-0.000907, +0.000860] |
| saturating_full/qmc16_nested128 | -0.000140 | [-0.000663, +0.000264] |
| weakening_full/qmc16_native | +0.000306 | [-0.000523, +0.001243] |
| weakening_full/qmc16_probe | -0.000080 | [-0.000962, +0.000883] |
| weakening_full/qmc16_nested128 | +0.000033 | [-0.000244, +0.000404] |


Six adjusted CIs 모두0을 포함한다. 이는 equivalence의 증명이 아니라, 이번 sample에서 superiority를 확보하지 못했다는 뜻이다. Pair의 mean loss는 native LMMSE보다 saturating에서 약2.79%, weakening에서 약3.00% 높았다. Always-probe 대비 mean 감소는 각각 약0.84%,0.75%로 작고 불확실했다.

Matched control의 pair/native mean loss ratio는 **0.9715**, 95% CI **[0.9112,1.0434]**다. 사전 engineering margin인 upper bound1.10은 통과했지만, 두 full laws에서 세 baseline 모두를 이겨야 한다는 primary success criterion은 실패했다.

Nested 대비 ratio의 adjusted CIs는 saturating **[0.9406,1.0255]**, weakening **[0.9766,1.0390]**다. 이 숫자는 이번 setting에서 nested의 저렴한 approximation 가능성을 시사한다. 그러나 사전에 정하지 않은 equivalence claim을 새로 부여하거나, 더 단순한 native LMMSE와의 비교를 생략할 근거는 아니다.

### 실제 law가 변경된 worlds만 보면

아래는 frictional class135 worlds의 descriptive means다. Negative controls로 인해 결과가 희석되거나 다른 효과가 합쳐지는 것을 확인하기 위해 분리했다.


| Method | Viscous, changed | Saturating full, changed | Weakening full, changed |
|---|---|---|---|
| qmc16_pair | 0.011181 | 0.013687 | 0.013209 |
| qmc16_native | 0.012634 | 0.012736 | 0.012233 |
| qmc16_probe | 0.010571 | 0.013325 | 0.012824 |
| qmc16_nested128 | 0.010983 | 0.013893 | 0.013087 |
| compact_pair | 0.011239 | 0.014226 | 0.015535 |


실제 changed subset에서는 pair가 always-probe보다 두 full laws에서 평균 loss가 오히려 높다. 전체256 worlds에서 보이는 아주 작은 mean gain을 “unseen contact law 개선”으로 설명하면 안 된다. Changed-subset difference의 별도 descriptive 95% CIs는 `results/changed_subset_contrasts.json`에 저장했다. 이 subgroup은 primary six contrasts와 구분한다.

Half shifts에서도 pair는 native보다 평균 loss가 높았다. Saturating half는 **0.010870 대0.010664**, weakening half는 **0.011996 대0.011340**이다. 이번 score의 일관된 decision-quality advantage는 확인되지 않았다.

![Decision loss와 GPU latency](results/loss_latency.png)

각 panel은 means를 표시하고 y-axis 범위가 다르다. 작은 순위 차이의 판단에는 위 paired CIs를 함께 사용해야 한다.

## 5. Selection이 실제로 어떻게 달라졌는가


| Condition | LMMSE query/probe | Pair query/probe | Nested query/probe | Pair–nested agreement |
|---|---|---|---|---|
| saturating_full | 93/163 | 64/192 | 58/198 | 92.19% |
| weakening_full | 104/152 | 72/184 | 72/184 | 92.19% |


Pair는 native보다 probe를 더 자주 선택한다. Saturating full에서는43 worlds, weakening full에서는38 worlds에서 native와 action이 달랐다. 이 changed-action subsets에서 pair−native mean loss는 각각 **+0.001739**, **+0.002059**였다. 행동은 달라졌지만 그 변경이 평균적으로 이득을 만들지는 못했다. Query/visibility가 고정되어도 action이 달라지는 것은 성공 조건의 일부일 뿐, loss 감소를 대체하지 않는다.

![Observation 선택 빈도](results/action_choices.png)

### Ablations와 common decoder


| Method / ablation | Base-law control | Viscous | Saturating full | Weakening full |
|---|---|---|---|---|
| qmc16_pair | 0.009421 | 0.009437 | 0.010759 | 0.010507 |
| qmc16_unfiltered | 0.009259 | 0.009525 | 0.010675 | 0.010722 |
| qmc16_agnostic | 0.009887 | 0.009699 | 0.010579 | 0.010380 |
| compact_pair_qmc16_decoder | 0.009267 | 0.009421 | 0.010783 | 0.010975 |
| qmc16_pair_compact_decoder | 0.008834 | 0.009447 | 0.011008 | 0.011116 |


Factual-fit band를 제거하거나 query weight를 제거해도 두 full laws에서 primary pair보다 유의하게 나쁘다는 evidence가 없다. 해당 secondary comparisons의95% CIs는0을 포함한다. 따라서 **factual-equivalence band와 query weighting이 꼭 필요한 contribution이라는 ablation evidence도 아직 없다.** Band의 eligible-pair fraction은 QMC에서 약84%, compact에서 약70%였고 fallback은 모두0이었다.

Weakening full에서 compact pair의 loss는0.011704였다. 같은 compact choices를 QMC decoder로 바꾸면0.010975, QMC pair choices를 compact decoder로 바꾸면0.011116이다. Compact의 결과에는 acquisition과 inference approximation이 모두 관여한다. 이 swap만으로 두 효과가 독립적이거나 단순 합산 가능하다고 주장하지 않는다.

## 6. Compute와 memory 비용

Same GPU 및 CPU2threads, float64, batch1, fixed48 cases×3 timing repetitions을 사용했다. Warmup 후 factual fitting, selection, actual selected observation update/prediction을 포함해 측정했다. GPU stages에는 explicit synchronization을 사용했다. Online benchmark는 selected observation 하나만 받고, offline all-action cache를 이용해 prediction을 대신하지 않는다. Stored evaluation과 action/prediction이 일치하는지도 검사했다.


| Method | GPU median ms | GPU p90 ms | CPU median ms |
|---|---|---|---|
| qmc16_pair | 6.358 | 6.840 | 2.214 |
| qmc16_native | 2.170 | 2.257 | 1.055 |
| qmc16_probe | 0.709 | 0.830 | 0.216 |
| qmc16_nested128 | 15.759 | 16.103 | 34.506 |
| qmc64_pair | 6.738 | 8.046 | 2.629 |
| qmc64_native | 3.467 | 3.921 | 3.437 |
| compact_pair | 28.549 | 31.730 | 7.618 |
| compact_native | 29.681 | 33.544 | 6.295 |
| compact_probe | 22.784 | 25.966 | 5.289 |


Pair는 nested128보다 GPU 약2.48배, CPU 약15.58배 빠르다. 사전에 정한 nested 대비2배 speed screening과 matched harm margin은 충족한다. 그러나 GPU에서 QMC16 native보다 약2.93배, always-probe보다 약8.96배 느리다. **전체 strong-baseline frontier에서의 compute superiority는 아니다.** Compact pair도28.549 ms로 QMC16 native보다 훨씬 느렸다.

State와 shared cache는 다음 numerical payload다. Python/Tensor objects, allocator reserve, CUDA context와 실제 sensor 비용은 포함하지 않는다.


| Method | Belief state bytes/world | Shared cache bytes | 4-case observed temporary peak bytes |
|---|---|---|---|
| compact_pair | 1488 | 11664 | 3271168 |
| qmc16_pair | 131072 | 28442624 | 3399680 |
| qmc16_native | 131072 | 28442624 | 524288 |
| qmc16_nested128 | 262144 | 56893440 | 67653632 |
| qmc64_pair | 524288 | 113770496 | 3792896 |


Committee의 parameter payload는 `256×7×8=14,336 bytes`다. Float64 pair matrix 하나는524,288 bytes이며 여러 distance/weight arrays가 필요하다. Nested128은 inner weights tensor 하나가 `128×16384×8=16,777,216 bytes`다. 실제 four-case CUDA allocated-memory audit에서는 QMC16 pair의 temporary peak 약3.40 MB, nested128 약67.65 MB였다. 이는 고정된4 cases에서 관찰한 값이며, 모든 inputs의 global peak 보증은 아니다. Process startup/library workspace가 섞일 수 있고 shared cache는 측정 직전 allocation으로 별도 제외했다.

Setup은 warm process에서 method construction을 측정했다. QMC16 pair는 GPU약5.689 ms, nested128은14.916 ms였다. Process startup이나 CUDA context initialization 전체를 포함한 cold-start가 아니다. QMC의 forward/moment cache는 fixed query/sensors에서 반복 재사용한다. Query가 계속 바뀌는 regime의 cache invalidation 비용은 측정하지 않았다. CPU와 GPU의 speed ratios를 섞어서 하나의 speedup으로 만들지 않았다.

## 7. Ranking ambiguity와 재현성

사전에 고정한 first16 independent worlds×6 conditions, 총96 cases에서 numerical audit를 수행했다.

| Audit | Main selector와 action agreement | 해석 |
|---|---|---|
| Nested128 → nested512, 같은 Sobol seed | 100% | 이 subset에서는 main ranking 변화 없음 |
| Nested128 → nested512, 다른 Sobol seed | 98.96% | Viscous case1개에서 변화 |
| Pair K256 → K512 | 93.75% | 동일 negative-control world의6 paired cases에서 변화 |

Pair의6 changes는6개의 independent failures가 아니다. Base world10, frictionless class 하나가 모든 conditions에서 같은 방식으로 바뀐 것이다. 각 condition의 audit mean loss change는 **+0.000133624**였다. Audit를 보고 main K나 policy를 변경하지 않았다.

Different-seed nested audit의 viscous mean loss change는 **−0.000001370**, 두 full laws에서는0이었다. 따라서 현재 subset에서는 nested ranking noise가 primary 실패의 주원인이라는 설명이 지지되지 않는다. 다만 나머지240 worlds까지 uniform stability를 보장하지 않는다. Pair 역시 finite committee size와 deterministic resampling order의 영향을 받는다.

Compact pilot의 minimum covariance eigenvalue는 약0.01429였고 model probability normalization, finite predictions를 검사했다. Online GPU/CPU timing의 predictions는 stored evaluation과 수치 오차 수준에서 일치했다. Frictionless negative controls121 worlds의 paired outcome/loss는 conditions 사이에 동일했다. 모든 primary policies는 private outcomes 평가 전에 freeze했고, source/data/policy hashes와 compact-before-QMC ordering을 기록했다.

이번 expanded QMC16/64의 primary bank scramble은 고정되어 있다. Nested outer bank는 independent scramble이지만, 이것이 모든 inference policies의 multi-seed robustness test를 대신하지는 않는다. Confidence intervals는 independent worlds의 variation을 반영하며 모든 model/prior/implementation uncertainty를 포함하지 않는다.

## 8. 논문 기여 가능성에 대한 평가

### 선행연구와 겹치는 개념

EC²는 hypotheses 사이의 weighted edges를 두고 observation으로 decision-relevant ambiguity를 제거하는 Bayesian active learning을 다룬다. 따라서 “hypothesis pairs를 구별하는 observation을 선택한다”는 발상 자체는 새롭지 않다. 이번 방법은 continuous query difference와 Gaussian overlap을 사용하는 surrogate이며, EC²의 exact objective를 구현한 것이 아니다. 그 논문의 adaptive-submodularity나 approximation guarantees를 이번 score에 가져올 수 없다. [Golovin, Krause, Ray, NeurIPS 2010 원문](https://proceedings.neurips.cc/paper_files/paper/2010/file/1e6e0a04d20f50967c64dac2d639a577-Paper.pdf)

Decision에 충분한 hypotheses를 같은 region으로 묶어 정보를 수집하는 방향도 기존 연구다. HEC는 overlapping decision regions에 대한 information gathering을 다루며 robot manipulation 관련 localization에도 적용했다. 현재 velocity-MSE weighted pair score와 문제 설정은 다르지만, “모든 latent uncertainty가 아니라 decision에 중요한 uncertainty를 줄인다”는 주장을 novelty로 내세울 수 없다. [Javdani et al., AISTATS 2014 원문](https://proceedings.mlr.press/v33/javdani14.pdf)

MOCU 기반 experimental design은 uncertainty의 operational cost 또는 expected regret를 줄이는 실험을 선택한다. 이번 nested expected-loss baseline과 decision-weighted selector는 이 넓은 decision-theoretic 계열 안에서 비교해야 한다. [Boluki, Qian, Dougherty, 2018 원문](https://arxiv.org/pdf/1805.01143)

최근 Action-BED 역시 downstream expected future loss를 중심으로 experiment/action policies를 최적화한다. 이번에는 그 learned policy를 재현하거나 직접 비교하지 않았으므로 해당 방법보다 우수하다는 주장은 하지 않는다. [Action-BED, 2026 arXiv v2](https://arxiv.org/html/2606.23662v2)

### 확보한 결과와 확보하지 못한 결과

| 후보 claim | 판단 |
|---|---|
| Factual agreement와 query disagreement를 구분하는 selector 구현 | 완료 |
| 같은 query/visibility에서도 ambiguity에 맞춰 observation 변경 | Mechanism checks와 test action variation에서 확인 |
| Strong expanded QMC/always-probe보다 낮은 decision loss | 이번 primary comparisons에서 입증되지 않음 |
| Nested expected-loss calculation보다 낮은 latency/temporary memory | 현재 fixed-query implementation에서 확인 |
| Cheap LMMSE보다 우수한 accuracy–compute trade-off | 확인되지 않음 |
| Factual-fit band/query weighting의 필수성 | Ablation evidence 부족 |
| 독립 method paper의 중심 contribution | 현재 evidence로 부족 |

Nested와 비슷한 loss를 더 적은 compute로 얻는 것은 engineering result로 남길 수 있다. 하지만 더 단순한 LMMSE가 더 싸고 평균 loss도 낮은 현재 setting에서는 왜 새 surrogate가 필요한지 설명하기 어렵다. 또한 well-specified model에서 exact Bayes-optimal selector보다 loss가 낮아야만 approximation method가 가치 있는 것은 아니다. 이번 사전 superiority screen의 실패와, 제한된 approximation-cost result를 구분해야 한다. 그렇다고 test 이후 criterion을 바꾸어 최초 실험의 성공으로 재분류하지는 않는다.

실제 changed-class에서 pair와 nested가 native보다 불리할 수 있는 이유로는 candidate prior/model mismatch가 있다. Internally expected loss를 정확하게 줄이는 observation이 실제 shifted population의 loss도 반드시 줄이는 것은 아니다. 현재 data가 이 설명과 일관되지만, matched full-mixture generator와 OOD generator를 분리해 원인을 확인한 실험은 아직 하지 않았으므로 확정 원인으로 표현하지 않는다.

### 내가 권하는 다음 판단 기준

현재 toy setting에서 committee size나 cutoffs를 반복 조정해 positive result를 찾는 방향은 추천하지 않는다. 다음 실험을 한다면 아래 구분이 먼저 필요하다. 아직 실행하지 않은 계획이다.

1. **Model-consistent control과 mismatch control 분리:** Expanded prior와 정확히 같은 generator에서 nested/approximate selector의 risk를 비교하고, 그다음 withheld-law population을 적용한다. Selection approximation error와 model mismatch error를 분리해야 한다.
2. **LMMSE가 실제로 실패하는 decision regime 확인:** 현재는 contact 발생을 전제로 한 single impulse와 velocity-MSE다. Contact/no-contact, slip/stick transition, downstream binary success 등에서 multimodal prediction이 decision을 바꾸는지 먼저 확인한다. 단순히 scene을 복잡하게 만드는 것만으로 contribution이 생기지는 않는다.
3. **필요한 approximation을 입증:** Native LMMSE가 미리 정한 accuracy target을 못 맞추는 regime에서 nested reference를 근사하면서 compute를 줄여야 한다. Pair surrogate의 regret error bound 또는 ranking reliability 진단이 있으면 기존 edge-cutting/decision-focused design과의 차이가 더 명확해질 수 있다. 현재는 그러한 bound가 없다.
4. **외부 검증과 cost realism:** Fixed sensor profiles를 넘어 실제 영상 geometry uncertainty와 sensor action costs를 포함하고, new worlds뿐 아니라 new law families에서도 frozen policy를 검증해야 한다.

이번 v14의 가장 확실한 성과는 “원하는 ambiguity mechanism을 구현하고, strong controls에서 그 한계를 확인했다”는 것이다. Research direction의 문제 정의는 여전히 유효하지만, 현재 selector를 major-conference method contribution으로 포장할 근거는 부족하다.

## 9. 실행 파일과 재현 자료

Server project directory:

```text
/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/disagreement_design_v14
```

| 파일 | 역할 |
|---|---|
| `PROTOCOL.md` | Test 전 objective, cutoffs, primary contrasts, audit/timing design |
| `acquisition.py` | Pair score, committee generation, cross-fitted nested selector |
| `flexible.py`, `compact.py`, `bank.py`, `physics.py` | 고정한 inference/forward models |
| `worlds.py` | Private generator; policy는 hidden parameters를 받지 않음 |
| `run_compact.py`, `run_banks.py` | Independent compact phase와 QMC baseline phase |
| `runtime.py`, `benchmark.py` | Selected observation만 받는 online inference와 GPU/CPU timing |
| `assemble.py`, `analyze.py`, `diagnostics.py` | Frozen decisions, paired statistics, memory/ranking diagnostics |
| `results/policies/`, `results/evaluations/` | Policies와 per-action actual-noise predictions |
| `results/summary.json`, `world_losses.npz` | Aggregate results와 independent-world losses |
| `results/ranking_audit.npz`, `diagnostics.json` | Numerical audit와 temporary-memory evidence |
| `INTEGRITY.json`, `MANIFEST.json` | Source/policy/results 검증 및 SHA256 manifest |

실행에는 `../continuous_v2/.venv/bin/python`과 Slurm `srun`을 사용했다. GPU workers는 foreign GPU process가 나타나면 자신의 process만 중단하는 guard 아래 실행했다. 기존 jobs/allocation은 변경하지 않았으며 사용자의 allocation은 유지한다. V13 sealed artifacts를 변경하지 않았는지도 검사한다.

재실행은 sealed results를 덮어쓰지 말고 새 directory에 code를 복사해 수행한다. Allocated compute node에서 idle GPU를 확인한 뒤 `run_compact.py` → `second_phase.py` → `benchmark.py --cpu` → `analyze.py` 순서로 실행한다. Memory audit는 `diagnostics.py`, 보고서 생성은 `write_report.py`, 최종 검증은 `seal.py`다. GPU/CPU timing과 다른 compute workloads를 겹쳐 실행하지 않는다. Logs, worker exit records와 phase timestamps가 실제 실행 evidence다.
