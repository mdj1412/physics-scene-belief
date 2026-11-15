# Factual grazing contact의 correlated posterior: 서버 실험과 연구 가치 재검토

작성일: 2026-11-04. 이전 `continuous_v2`를 확장한 `correlated_v3` 결과다. 설명은 한글로, technical terms는 English로 표기한다. Primary experiment와 numerical audit를 모두 `datacenter3`의 기존 Slurm allocation에서 실행했다.

**Factual grazing contact가 geometry와 restitution을 보상 관계로 묶는 curved posterior를 만들었다. 작은 uniform grid와 prior importance particles는 narrow posterior에서 prediction과 observation selection 모두 실패했다. 그러나 strong baseline인 tempered SMC와 conditional quadrature는 reference에 가까운 결과를 회복했다. 현재 결과는 posterior dependence의 중요성을 보여주지만, 새로운 inference 또는 acquisition algorithm이 필요하다는 증거는 아니다.**

또한 실제 추가 observation을 선택하는 adaptive policy의 이점은 예상보다 작았다. Validation으로 정한 query-type rule과의 MSE 차이에 대한 95% CI가 0을 포함한다. 이번 결과에서 가장 강한 메시지는 “새 acquisition rule이 더 좋다”가 아니라 “posterior의 support와 dependence를 잃으면, 더 정확한 factual observation을 받아도 downstream 결과가 악화될 수 있다”이다.

## 1. 이전 실험에서 무엇을 바꿨는가

이전 실험에서는 geometry와 restitution의 prior 및 observation likelihood가 factorized되어 있었다. 이번에는 **실제 collision 이후의 normalized outgoing velocity**를 factual measurement로 사용했다. 동일한 velocity를 설명하는 여러 geometry–restitution 조합이 생기므로 posterior가 factorize되지 않는다.

이번에는 이 단순한 frictionless collision의 exact event solution을 primary forward model로 사용했다. 기존 PyBullet lookup을 작은 noise까지 그대로 사용하면 interpolation, timestep, collision-margin 차이가 posterior approximation의 실패처럼 보일 수 있기 때문이다. 별도 PyBullet audit에서 이 차이를 확인하고 원인을 분리했다. 따라서 아래 main table은 **exact physical model과 일치하는 synthetic sensor setting의 결과**이며, PyBullet image observation이나 real robot 결과가 아니다.

| 항목 | 설정 |
|---|---|
| 물체 | radius 0.02 m sphere, 고정된 box, half-width 0.03 m |
| Dynamics | Friction·gravity·damping이 없는 single rigid impact |
| Unknowns | Lateral geometry offset `g`, restitution `e` |
| Prior | `|g| ~ Uniform[0.032, 0.0498] m`, `e ~ Uniform[0.1, 0.9]`, 처음에는 independent |
| One-branch | `g > 0`를 알고 있으며, branch 내부의 curved dependence를 추정 |
| Two-branch | `sign(g)`가 ±일 prior probability가 각각 0.5; factual velocity만으로 sign을 구분할 수 없음 |
| Factual sensor | `y_f = v_x_out / v_x_in + Gaussian noise` |
| Factual noise SD | `0.03, 0.01, 0.003, 0.001` |
| Additional view | `y_view = g + N(0, σ_g²)`, `σ_g`는 family마다 1–4 mm |
| Additional probe | 별도 reset의 head-on restitution measurement, `y_probe = e + N(0, σ_e²)`, `σ_e`는 0.04–0.15 |
| Query | 모두 `translate_wall_y(δ)`이며 `δ`만 달라짐 |
| Query 배치 | −80, −50, −25, −8, +4, +12, +25, +50 mm 근처에 ±2 mm jitter; hidden truth와 독립 |
| Target | Counterfactual normalized outgoing `v_x`, contact event probability |
| Acquisition budget | Factual observation 이후 추가 observation 최대 1회; `view`, `probe`, `stop` |
| Utility | Expected target MSE reduction − `0.0005 × observation count` |

Additional sensors는 calibrated scalar sensor로 이상화했다. RGB detection이나 learned perception을 구현한 결과로 해석하지 않는다. Factual measurement는 outgoing velocity의 한 component만 관측하므로 ambiguity가 남는다. Full 2D velocity와 contact normal까지 정밀하게 관측하면 식별 가능성이 크게 달라질 수 있다.

## 2. Correlated posterior가 생기는 이유

Grazing 구간에서 `u=(|g|−0.03)/0.02`, `A(g)=1−u²`라고 두면,

\[
f(g,e)=v_x^{out}/v_x^{in}=1-(1+e)A(g),
\qquad
v_y^{out}/v_x^{in}=-(1+e)\sqrt{A(g)}\,u\,\mathrm{sign}(g).
\]

Factual likelihood는 다음과 같다.

\[
p(g,e\mid y_f)\propto p(g)p(e)
\exp\left[-\frac{(y_f-f(g,e))^2}{2\sigma_f^2}\right].
\]

`|g|`가 커지면 collision이 더 grazing해지고 `A(g)`가 작아진다. 같은 `y_f`를 설명하려면 `e`가 커져야 한다. Small-noise limit에서 posterior는 대략

\[
e(g)=\frac{1-y_f}{A(g)}-1
\]

를 따르는 curved ridge에 집중된다. Prior에 임의 covariance를 넣은 것이 아니라 **factual collision likelihood가 dependence를 만든 것**이다.

Two-branch에서는 `f(g,e)=f(−g,e)`이므로 서로 대칭인 두 ridge가 남는다. 이 경우 `Corr(g,e)`가 0이어도 independent가 아니다. 각 branch의 `|g|`와 `e`는 강하게 결합되어 있고, `translate_wall_y(δ)` 이후 두 branch의 contact 결과는 달라질 수 있다.

Held-out family에서 `σ_f=0.001`일 때 reference의 평균 `Corr(|g|, e)`는 **0.9793**였다. Two-branch의 평균 `Corr(g,e)`는 numerical zero이지만 동일한 absolute-geometry dependence가 유지된다. Histogram MI는 binning에 의존하는 보조 diagnostic으로만 사용했다.

![Correlated posterior의 support 비교](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/correlated_v3/results/analysis/posterior_support.png)

그림은 사전에 저장 대상으로 정한 첫 held-out family 32의 결과다. Red cross는 truth이며 inference에 제공되지 않는다. Upper row는 one-branch, lower row는 two-branch다. 각 row에서 x-axis 범위를 통일했다. 작은 grid와 prior particles는 ridge의 일부만 남기며, SMC와 conditional quadrature는 구조를 유지한다. `factorized_marginals`는 geometry와 restitution의 marginal을 대략 유지하면서 dependence를 제거한 ablation이다. Scatter의 개별 point 크기는 weight를 반영하며, 연속 density plot과 동일하지 않다.

## 3. 비교 방법과 공정성

| Method | Posterior 구성 | 실제 support 크기 | 해석 |
|---|---|---:|---|
| Reference | Posterior-adapted `256×16` conditional quadrature | 4,096 / 8,192 | One-/two-branch의 approximate Bayes reference |
| Grid17×17 | Fixed Cartesian trapezoid grid + factual likelihood | 289 / 578 | 매우 작은 uniform grid |
| Grid33×17 | 동일, 더 촘촘한 geometry grid | 561 / 1,122 | 이전 small-grid 수준과 가까운 조건 |
| Grid65×33 | 동일, 두 축을 더 촘촘히 배치 | 2,145 / 4,290 | Grid scaling baseline |
| Prior512 / Prior2048 | Prior에서 sampling 후 importance weighting | 512 / 2,048 | Proposal adaptation과 rejuvenation 없음 |
| Posterior512 | 4,097-point marginal CDF에서 `g` sampling, conditional truncated Normal에서 `e` sampling | 512 | Model structure를 이용하는 strong sampling baseline |
| SMC512 | Adaptive likelihood tempering, systematic resampling, covariance-scaled Metropolis rejuvenation | 512 | Generic continuous inference에 가까운 strong baseline |
| Conditional_quad512 | Factual-compatible geometry interval과 conditional restitution CDF에서 quadrature | 항상 512 | One-branch `32×16`, two-branch `2×16×16` |
| Factorized_marginals | Reference의 각 marginal을 64-point quantile로 근사한 뒤 product 구성 | 4,096 | Dependence 제거 diagnostic; reference를 사용하는 privileged computation |
| Query-type rule | Validation에서 고른 fixed `view`/`probe`/`stop` + reference posterior | Reference와 동일 | Noise level과 branch regime을 알고 선택하는 강한 fixed rule |

Conditional quadrature는 `e | g,y_f`가 truncated Normal임을 이용한다. 이때 `mean=(1−y_f)/A(g)−1`, `sd=σ_f/A(g)`다. Geometry integration interval은 likelihood의 ±8σ 범위에 맞춰 줄이고 Gauss–Legendre nodes를 배치한다. 일반 black-box simulator에 곧바로 적용할 수 있는 새로운 adaptive-grid algorithm이 아니다.

SMC의 next temperature는 conditional ESS가 약 `0.8N`이 되도록 고른다. 매 단계 resampling 후 10회 Metropolis move를 수행한다. Two-branch의 sign은 factual likelihood에 대해 정확히 대칭이므로 마지막에 exact conditional sign refresh를 사용한다. 따라서 일반적인 unknown multimodal landscape의 mode exploration을 해결했다는 결과는 아니다. 추가 view/probe 이후에는 모든 particle baseline을 동일하게 likelihood reweighting하며, 별도 rejuvenation은 하지 않았다.

모든 방법은 동일한 prior, factual measurement, query values, candidate sensors, acquisition cost를 사용한다. Truth는 inference grid와 독립적인 continuous draw다. Acquisition은 hidden `g,e`, target truth, 아직 선택하지 않은 sensor realization을 입력으로 받지 않는다. Evaluation에서 양쪽 sensor를 모두 계산하는 것은 matched-information 비교를 위한 것이며 policy에는 전달되지 않는다.

**동일 atom 수와 동일 compute budget은 다르다.** SMC는 support가 512여도 여러 temperature와 Metropolis proposal을 평가한다. Conditional sampling도 4,097개의 marginal evaluations를 사용한다. 따라서 아래 결과를 동일 simulator-call budget의 우위로 해석하지 않는다. 이번에는 accuracy, support size, 실제 소요 시간과 evaluation count를 함께 기록했다.

## 4. Data split과 numerical validation

- 96 independent family: 16 dev, 16 validation, 64 held-out test.
- 각 family의 continuous truth, sensor quality, query jitter와 noise는 seed로 재현 가능하다.
- 2 branch regimes × 4 factual noise levels × 8 numeric queries. Held-out은 4,096 query contexts다.
- Stochastic inference는 3개의 independent inference seed로 반복했다. 동일 family의 worlds/noise/queries는 method 간 공유한다.
- 총 18 method/seed configurations로 110,592개의 raw metric row를 저장했다. 이 수가 independent sample size는 아니다.
- CI는 query와 seed를 family 내부에서 평균한 뒤 64 family를 paired bootstrap 10,000회 resampling했다. Multiple comparison correction은 하지 않은 exploratory CI다.
- Reference refinement는 dev 128 conditions에서 `512×32` quadrature와 513-point observation integration으로 비교했다. 모든 action이 같았다.
- 별도 2,100,225-point uniform grid로 dev 16 conditions의 posterior moments를 교차 검증했다.

Primary reference는 exact posterior의 analytic closed form 전체가 아니라 **convergence를 검사한 numerical approximation**이다. Contact indicator는 discontinuous하므로 velocity-based decision accuracy만으로 모든 contact probability의 완전한 수렴을 보장하지 않는다.

Reference refinement의 maximum acquisition-gain 차이는 `2.19e-05`, maximum predictive-mean 차이는 `1.06e-05`였다. 독립 uniform-grid audit의 maximum counterfactual mean 차이는 `5.58e-06`였다.

## 5. Narrow posterior에서의 결과

다음 표는 가장 작은 `σ_f=0.001`에서 one-/two-branch를 동일 비중으로 평균한 held-out 결과다. MSE는 normalized velocity의 squared error다. Incoming speed 0.6 m/s로 해석하면 physical velocity MSE는 표의 MSE에 0.36을 곱한다.

Design regret는 reference posterior에서 선택한 action의 expected MSE reduction minus cost가 최선보다 얼마나 낮은지다. 실제 미래를 아는 clairvoyant oracle과 비교한 값이 아니다. `Coverage90`은 central predictive interval의 empirical coverage이며, no-contact의 point mass 때문에 정확히 90%가 되어야 하는 지표는 아니다.

| Method | MSE ↓ | CRPS ↓ | Brier ↓ | Design regret ↓ | Action agreement ↑ | Coverage90 |
|---|---:|---:|---:|---:|---:|---:|
| reference | 0.008004 | 0.034058 | 0.005188 | 0.0000000 | 100.00% | 96.00% |
| grid17x17 | 0.042994 | 0.124754 | 0.015296 | 0.0078409 | 77.54% | 44.14% |
| grid33x17 | 0.027233 | 0.090412 | 0.010802 | 0.0035999 | 86.91% | 50.00% |
| grid65x33 | 0.013667 | 0.052552 | 0.009447 | 0.0006518 | 97.27% | 78.32% |
| prior512 | 0.271341 | 0.259044 | 0.129196 | 0.1095358 | 58.37% | 39.62% |
| prior2048 | 0.064355 | 0.095607 | 0.032132 | 0.0191664 | 93.13% | 62.86% |
| posterior512 | 0.007989 | 0.034193 | 0.005459 | 0.0000001 | 99.87% | 96.03% |
| smc512 | 0.008063 | 0.034168 | 0.005070 | 0.0000003 | 99.97% | 95.93% |
| conditional_quad512 | 0.008049 | 0.034288 | 0.005381 | 0.0000000 | 100.00% | 95.21% |
| factorized_marginals | 0.013020 | 0.045957 | 0.006410 | 0.0008332 | 92.58% | 95.70% |
| query_type_rule | 0.008143 | 0.034489 | 0.005467 | — | 79.79% | 95.51% |

| Reference 대비 paired difference | Mean difference [95% CI] |
|---|---:|
| Grid33×17의 MSE 증가 | 0.019230 [0.012644, 0.026448] |
| Prior512의 MSE 증가 | 0.263337 [0.206048, 0.321831] |
| SMC512의 MSE 차이 | 0.000059 [-0.000109, 0.000231] |
| Conditional_quad512의 MSE 차이 | 0.000045 [-0.000023, 0.000116] |
| Factorized marginals의 MSE 증가 | 0.005016 [0.002694, 0.007534] |
| Query-type rule의 MSE 차이 | 0.000139 [-0.000007, 0.000410] |

Small grid와 prior particles의 열화는 크고 CI도 0을 넘는다. 반면 SMC와 conditional quadrature의 차이에 대한 CI는 0을 포함한다. 이는 “완전히 동등함”을 입증한 equivalence test는 아니지만, 이 실험에서는 reference와 비교한 큰 열화가 관측되지 않았다는 뜻이다.

## 6. Noise가 작아질수록 어디서 실패하는가

| Factual noise SD | Reference MSE | Grid33×17 MSE | Prior512 MSE | SMC512 MSE | Conditional_quad512 MSE |
|---:|---:|---:|---:|---:|---:|
| 0.03 | 0.008462 | 0.008448 | 0.009818 | 0.008478 | 0.008468 |
| 0.01 | 0.008087 | 0.010001 | 0.010461 | 0.008241 | 0.008076 |
| 0.003 | 0.008019 | 0.017806 | 0.078670 | 0.008049 | 0.008042 |
| 0.001 | 0.008004 | 0.027233 | 0.271341 | 0.008063 | 0.008049 |

![Noise sweep](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/correlated_v3/results/analysis/noise_sweep.png)

### Grid: ridge를 가로지르는 resolution이 부족하다

Factual noise가 넓으면 작은 grid도 posterior를 충분히 덮는다. Noise가 작아지면 대부분의 grid point가 ridge에서 벗어나고, 우연히 likelihood가 큰 1–2개 point가 posterior mass를 차지한다. 이때 geometry와 restitution의 서로 다른 plausible 조합을 유지하지 못한다.

예를 들어 `g=0.042 m, e=0.5`에서 `A=0.64`, `∂f/∂g=90 m⁻¹`이다. `33-point geometry grid`의 spacing은 약 0.556 mm이며, 이 방향의 factual velocity 변화는 약 0.050이다. Restitution spacing 0.05도 velocity를 약 0.032만큼 바꾼다. 둘 다 `σ_f=0.001`보다 크다. 다만 이것은 local scale 설명이며, diagonal ridge와 grid의 정렬까지 포함하는 엄밀한 error bound는 아니다.

### Prior particles: likelihood weighting 이후 support가 사실상 사라진다

Prior에서 뽑힌 fixed particles는 좁은 posterior tube를 거의 맞추지 못한다. Weight만 바꾸면 새로운 support를 만들 수 없다. Two-branch에서는 한 branch에 남은 우연한 particle이 대부분의 weight를 가져가고, 다른 branch가 없다고 잘못 믿게 된다. 그러면 sign을 알려주는 view의 value도 작게 평가할 수 있다.

- One-branch Prior512: median ESS `1.07` / 512.
- Two-branch Prior512: median ESS `1.23` / 512, branch mass가 5% 미만 또는 95% 초과로 쏠린 비율 `68.23%`.


여기서 ESS는 importance weights의 concentration을 나타낸다. Resampling된 SMC의 weight ESS=512가 512개의 독립적인 posterior sample을 뜻하지는 않는다. SMC의 실효 정확도는 reference 대비 prediction/action 및 seed variation으로 확인했다.

### Strong baseline: representation을 늘리기보다 배치를 바꾸면 회복된다

Conditional quadrature는 ridge의 conditional structure에 맞춰 nodes를 놓는다. Posterior sampling은 factual evidence를 반영한 proposal을 쓴다. SMC는 likelihood를 점진적으로 강화하면서 particles를 이동시킨다. 모두 narrow posterior에 mass를 두는 방식이다. 따라서 이번 결과가 지지하는 것은 **prior proposal을 고정한 importance sampling과 uniform grid의 한계**이며, particles 또는 small-support approximation 전체의 한계가 아니다.

## 7. Dependence와 observation selection을 분리한 해석

`factorized_marginals`는 reference의 marginal quantile들을 product로 조합한 numerical ablation이다. 완벽히 동일한 continuous marginals는 아니며 64-point approximation을 사용한다. 이를 true independent posterior라고 부르지 않는다.

`σ_f=0.001`에서 dependence를 제거하면 full-pipeline MSE가 `0.008004`에서 `0.013020`로 증가했다. 모든 방법에 **같은 view**를 강제로 제공해도 factorized ablation의 MSE 증가는 `0.005932 [0.002238, 0.009931]`였다. 따라서 이 조건에서의 문제는 observation 선택만으로 설명되지 않으며, geometry measurement가 restitution posterior를 함께 갱신해야 한다는 점이 중요하다.

같은 query type 안에서 최적 action이 달라지는지도 다시 검사했다. 다음은 `σ_f=0.001`, 64 held-out family × 8 queries의 reference 선택이다.

| Regime | View | Probe | Stop | 한 family 안에 action이 2개 이상 |
|---|---:|---:|---:|---:|
| One-branch | 10 | 305 | 197 | 64/64 |
| Two-branch | 512 | 0 | 0 | 0/64 |

**Two-branch에서는 모든 query가 view를 선택했다.** 이 setting은 sign ambiguity를 측정하는 데는 유용하지만, 어려운 acquisition ranking benchmark는 아니다. One-branch에서는 action이 달라지지만 대부분 probe와 stop 사이의 차이다. Useful view와 useful probe가 자주 뒤바뀌는 과제라고 과장해서는 안 된다.

Validation-only query-type rule은 모든 noise level에서 one-branch에는 probe, two-branch에는 view를 선택했다. 이 rule은 regime과 noise level을 알고 있는 강한 fixed baseline이며 numerical query value와 factual measurement 값에 따라 action을 바꾸지 않는다.

전체 noise 평균 MSE는 adaptive reference `0.008143`, query-type rule `0.008289`였다. Paired difference는 `0.000146 [-0.000011, 0.000439]`다. Observation count는 `0.808` 대 `1.000`으로 adaptive policy가 더 적지만, **MSE 우위는 통계적으로 명확하지 않다.**

## 8. PyBullet audit: timestep 외에 collision geometry도 달랐다

96 continuous parameter cases에 대해 4 timestep의 PyBullet rollouts를 실행했다. 기존 `physics.py`의 설정을 그대로 사용한 첫 audit에서는 timestep을 줄여도 analytic sharp-box solution과 차이가 남았다. 조사 결과 PyBullet box의 default collision margin이 0.001 m였다.

Rounded box는 core half-width를 `0.03−margin`, sphere와 합쳐지는 corner radius를 `0.02+margin`으로 해석할 수 있다. 이 rounded model의 exact solution과 비교하거나, PyBullet box margin을 0으로 설정하면 차이가 크게 줄었다. 따라서 처음의 차이를 전부 timestep error라고 해석하면 틀린다.

| `dt=1/30720` 비교 | Normalized velocity component RMSE | Maximum error |
|---|---:|---:|
| Default margin PyBullet vs sharp exact | 0.022596 | 0.074622 |
| Default margin PyBullet vs rounded exact | 0.000269 | 0.001044 |
| Zero-margin PyBullet vs sharp exact | 0.000328 | 0.001301 |

총 PyBullet rollouts는 480회다. 첫 384회의 contact/no-contact 판정은 exact model과 모두 일치했다. 그러나 velocity의 numerical discrepancy는 완전히 0이 아니다. Primary table은 이런 오차를 observation noise에 숨기지 않고 exact model 안에서 posterior approximation만 비교했다.

이 audit는 실제 연구 방향에도 중요하다. Visual mesh와 collision proxy가 조금만 달라도 material parameter가 geometry/model error를 대신 설명할 수 있다. 그렇다고 이번 audit만으로 real-world model discrepancy correction을 검증한 것은 아니다. 다음에는 같은 factual fit을 보이는 서로 다른 collision proxies가 held-out physical intervention에서 얼마나 다르게 실패하는지 확인해야 한다.

## 9. 실행 evidence, 비용, 재현 방법

Remote directory:

`/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/correlated_v3`

기존 `continuous_v2/.venv`를 재사용했고 이전 실험 source/data/results는 수정하지 않았다. PyTorch 2.7.1+cu128, CUDA 12.8, float64 numerical inference를 사용했다. Neural network training이나 foundation-model inference는 수행하지 않았다. GPU는 posterior likelihood integration과 acquisition calculation에 사용했다.

| Slurm job | Node | Main shard elapsed time | Peak CUDA allocation |
|---|---|---:|---:|
| 2642474 | node55 | 100.13 s | 430.9 MiB |
| 2642475 | node56 | 38.67 s | 430.9 MiB |
| 2642476 | node56 | 40.06 s | 430.9 MiB |
| 2642473 | node55 | 92.85 s | 430.9 MiB |

4개 GPU 모두 사용했지만 이 2D problem은 작아 GPU memory나 compute를 포화시키지 않는다. Node별 elapsed time 차이도 있으므로 이를 대규모 scalability benchmark로 해석하지 않는다. 기존 allocation을 취소하거나 사용자의 다른 process를 종료하지 않았다.

Held-out 평균으로 SMC512는 약 57,459 factual likelihood evaluations를 사용했다. Prior512는 512개, Prior2048은 2,048개다. SMC의 평균 end-to-end method 계산 시간은 약 22.7 ms, Prior512는 약 8.7 ms였다. 이 측정은 8개 query와 모든 sensor choice의 evaluation을 포함한 이번 구현의 시간이다. Simulator가 비싸면 이 비율은 달라진다.

`runtime.json` 집계의 reference method timer는 앞서 계산한 reference를 재사용하는 시간만 측정하므로 다른 method와 speed comparison에 사용할 수 없다. Factorized ablation 역시 reference 계산을 선행하므로 standalone efficiency baseline이 아니다. 모든 method의 실제 atom 수와 likelihood evaluation 수는 `timings_*.csv`에 저장했다.

Main prediction 결과 생성 후 histogram MI 계산의 tiny-probability multiplication에서 underflow를 발견해 log-marginal 형태로 수정했다. Original diagnostics를 `original_diagnostics/`에 보존하고 동일 seed로 diagnostic만 재계산했다. 다른 diagnostic의 maximum change가 `1e-10` 미만인지 검사하며, prediction/action raw metrics는 변경하지 않았다. Main run 당시 `core.py`는 `provenance/core_at_main_run.py`에 별도로 보존했다.

새로 실행할 때는 current allocation이 idle인지 확인하고 `launch.sh`의 job IDs를 현재 값으로 바꿔야 한다. 기존 결과를 보존하려면 `run.py --out`에 새 directory를 지정한다. Analysis script의 `ROOT`도 해당 directory로 맞춘다. Code와 data를 login node에서 실행하지 않는다.

```bash
# 할당받은 compute node의 Slurm step 안에서 실행
../continuous_v2/.venv/bin/python run.py --shard 0 --out results/new_run
# shard 0,1,2,3을 각각 실행한 다음
../continuous_v2/.venv/bin/python analyze.py
../continuous_v2/.venv/bin/python audit_physics.py
../continuous_v2/.venv/bin/python audit_margin.py
../continuous_v2/.venv/bin/python audit_reference.py
```

## 10. 이번 결과로 주장할 수 있는 것과 없는 것

| 주장 | 판단 |
|---|---|
| Factual grazing contact가 geometry–restitution dependence를 만든다 | Exact model과 numerical posterior에서 확인 |
| 더 작은 factual noise가 fixed approximate inference의 결과를 악화시킬 수 있다 | Noise sweep에서 확인 |
| 작은 uniform grid와 prior importance particles가 ridge/branch를 잃는다 | ESS, branch mass, predictive score, design regret로 확인 |
| Posterior dependence가 추가 observation 이후의 prediction에 중요하다 | Matched-view factorization ablation이 지지 |
| 모든 particle method가 실패한다 | 틀림. SMC와 posterior sampling이 잘 작동 |
| 더 큰 learned joint scene model이 필요하다 | 현재 실험으로 입증하지 못함 |
| Query-conditioned acquisition이 fixed rule보다 큰 이득을 준다 | 현재 setting에서는 뒷받침되지 않음 |
| Real visual scene이나 실제 robot에서 효과가 검증되었다 | 아직 아님 |
| 최신 논문보다 우수하다 | 논문 implementation을 재현하지 않았으므로 주장하지 않음 |

이후의 선행연구 검토와 연구 방향 제안은 이 한계를 전제로 한다.

---

# Correlated physical belief 연구의 선행연구와 가치 재검토

검토일: 2026-11-04. 아래 평가는 이번 `correlated_v3` 실험 결과를 전제로 한다. 2026년 논문의 상당수는 preprint이므로, 확인하지 않은 학회 acceptance를 가정하지 않는다. 검색으로 발견한 논문 중 가까운 연구의 원문 method와 관련 experiment 부분을 읽었으며, 전체 분야를 빠짐없이 검토했다는 의미는 아니다.

**문제의 중요성은 충분하지만, 현재 구현은 새로운 method의 필요성을 입증한 상태가 아니다.** Factual grazing contact에서 geometry–restitution coupling, 작은 grid의 under-resolution, prior importance sampling의 weight collapse를 재현했다. 그러나 standard SMC와 problem-specific conditional quadrature가 대부분의 gap을 없앴다. 또한 validation으로 고른 query-type rule과 비교한 adaptive selection의 MSE 차이는 작다. 따라서 지금 결과를 “새로운 joint scene belief algorithm의 성공”으로 포장하면 안 된다.

## 1. 이번에 특히 중요하게 확인한 최근 연구

### A Bayesian Approach for Task-Specific Next-Best-View Selection with Uncertain Geometry — Zhu, Sellán, Terenin, 2026

원문 §3.1–3.2.3, §4.3과 author project page를 확인했다. Author page는 SIGGRAPH 2026으로 표시한다. Stochastic Poisson Surface Reconstruction의 posterior samples로 가능한 scan을 생성하고, downstream utility의 expected improvement를 기준으로 viewpoint를 고른다. Classification·segmentation뿐 아니라 heat diffusion의 coldest-point 발견도 포함한다. 즉, **uncertain geometry를 유지하면서 physics task에 필요한 observation을 고르는 접근은 이미 직접적인 선행연구가 있다.** [원문](https://arxiv.org/html/2605.05095) · [author page](https://papers.avt.im/next-view-planning/)

우리의 남은 후보는 static surface scan을 넘어, factual motion이 geometry와 material을 결합해서 제약하고, discrete contact mode와 continuous parameter가 함께 counterfactual을 결정하는 설정이다. 다만 heat diffusion을 contact dynamics로 바꾸는 것만으로 충분한 method contribution이 된다고 볼 수는 없다. 이 논문을 단순 uncertainty-reduction baseline으로 묘사해서도 안 된다.

### Informationally Decoupled Trajectory Design for Sim-to-Real System Identification — Shin et al., 2026

2026-10-27의 v2 원문 §III–V를 확인했다. IDTD는 서로 다른 parameter의 효과가 보상되어 trajectory fit이 좋아도 parameter estimate가 틀리는 문제를 명시적으로 다룬다. Nominal parameter에서 계산한 Fisher Information Matrix의 Schur complement, normalization, segment-wise coverage로 exploration을 설계한다. Linear system, 여러 robot simulator, real K1 humanoid를 평가한다. [원문](https://arxiv.org/html/2610.10905)

우리의 차별점 후보는 parameter identification 전체가 아니라 **특정 counterfactual에 필요한 uncertainty를 남기거나 줄이는 것**, 그리고 local FIM으로 표현하기 어려운 여러 contact branch의 posterior다. 그러나 “parameter들이 서로 보상하므로 active experiment가 필요하다” 자체는 이 논문과 직접 겹친다. IDTD의 full trajectory-design algorithm을 이번 scalar sensor 실험에서 재현한 것은 아니다.

### Active Tactile Exploration for Rigid Body Pose and Shape Estimation — Gordon et al., 2025/2026

2026-02-24의 v2 원문 §III–VII를 중심으로 확인했다. 움직일 수 있는 rigid object의 pose와 continuous convex geometry를 tactile observation으로 함께 추정한다. Violation-implicit loss로 contact constraint를 다루고, local information approximation에 기반한 EIG로 exploration을 고른다. Simulation과 real robot 결과가 있다. 원문은 dynamic belief가 multimodal해져 단일 Gaussian propagation이 부적절할 수 있다는 문제도 논의한다. [원문](https://arxiv.org/html/2510.13595)

따라서 contact-aware geometry learning이나 active sensing 자체도 새 주장이 아니다. 우리에게 남는 검증은 local approximation이 놓치는 branch가 query-specific predictive distribution과 action ranking을 얼마나 바꾸는지다. 이번 결과는 그 가능성을 보여주는 작은 diagnostic이며, 해당 논문에 대한 성능 우위를 보인 것은 아니다.

### System Identification under Constraints and Disturbance: A Bayesian Estimation Approach — Martinez, Tonneau, Mastalli, 2026

원문 introduction, constrained formulation, hybrid dynamics와 results의 구성 및 설명을 확인했다. Joint state–parameter estimation, contact/loop constraints, friction, disturbance projection과 energy observation을 함께 다룬다. Physical model이나 constraint를 잘못 표현하면 parameter가 그 오류를 흡수한다는 지적이 중요하다. [원문](https://arxiv.org/html/2602.16358)

이번 box collision margin audit와 연결되는 교훈은, posterior를 정교하게 계산해도 잘못된 forward model을 믿고 있으면 inference가 틀릴 수 있다는 것이다. 이 논문의 constrained optimization을 arbitrary multimodal posterior sampling과 동일시하지 않는다. 우리에게는 **approximation error와 model discrepancy를 분리한 평가**가 필요하다.

## 2. Acquisition 원리와 numerical inference의 직접 선행연구

| 연구 | 원문에서 확인한 내용 | 이번 연구에 주는 제약 |
|---|---|---|
| [Action-BED, 2026](https://arxiv.org/html/2606.23662), §3–4·6.2–6.3 | Expected future loss를 기준으로 design policy와 downstream action policy를 공동 학습한다. Explicit posterior 없이도 학습할 수 있고 pendulum 및 visual patch acquisition을 평가한다. | Downstream loss를 acquisition objective로 쓰는 원리를 새롭게 주장할 수 없다. 이번 EVSI 계산은 Action-BED implementation이 아니다. |
| [GO-CBED, 2025 preprint](https://arxiv.org/html/2507.07359), §3–5·Appendix C.4 | Causal QoI에 대한 non-myopic EIG, variational posterior와 policy를 학습한다. Structural uncertainty와 기존 observational data도 다룬다. | Query-specific intervention, multimodal causal uncertainty, observational conditioning 자체는 이미 포함된다. 이번 one-step 실험과 horizon이 다르다. |
| [Amortized Bayesian Experimental Design for Decision-Making, 2024/2025](https://arxiv.org/html/2411.02064), §2.3–4 | Decision Utility Gain과 Transformer Neural Decision Process를 제안한다. Posterior approximation이 decision에 중요한 영역을 놓칠 수 있다는 점도 논의한다. | “좋은 posterior fit보다 좋은 decision이 중요하다”는 문제의식도 새 원리는 아니다. |
| [Nesting Particle Filters for Experimental Design in Dynamical Systems, ICML 2024](https://proceedings.mlr.press/v235/iqbal24a.html), [원문 §3–4](https://arxiv.org/html/2402.07868) | Inside-Out SMC²와 particle MCMC로 dynamical system의 sequential design을 다룬다. Naive prior weighting의 high variance를 설명한다. | Prior particles의 실패를 particle methods 전체의 한계로 일반화하면 안 된다. 이번 tempered SMC는 이 논문의 재현이 아니다. |
| [BayesSim, RSS 2019](https://arxiv.org/html/1906.01728), §IV–V | Simulator parameter의 likelihood-free posterior를 mixture density로 추정하고 domain randomization에 활용한다. | Physical parameter를 point estimate 대신 distribution으로 표현하는 것 자체는 오래된 접근이다. Diagonal mixture component도 여러 component의 결합으로 dependence를 표현할 수 있다. |
| [SPI-Active, 2025](https://arxiv.org/html/2505.14266), §3–5 | Parallel sampling 기반 physical parameter fitting과 Fisher Information 기반 active exploration을 결합한다. | GPU sampling과 active system identification의 결합만으로는 차별화되지 않는다. 우리의 작은 inference benchmark가 실제 locomotion transfer 성과를 대체하지 못한다. |

여기서 논문별 숫자를 가져와 우리 결과와 직접 비교하지 않았다. Observation interface, dynamics, task loss, horizon, training budget이 다르기 때문이다. 이번에 실제 실행한 strong baseline은 **tempered SMC, conditional posterior sampling, posterior-adapted conditional quadrature**다.

## 3. Video와 physical digital twin 방향에서 가까운 연구

| 연구 | 관련 원문과 핵심 | 우리가 주장할 수 있는 범위 |
|---|---|---|
| [VideoPhysEdit, 2026](https://arxiv.org/html/2609.35134v1), §3·Appendix C.2–C.3 | Video에서 executable rigid scene을 맞추고 physical edit 이후 trajectory를 생성한다. 초기 관측에 비슷하게 맞는 최대 3개 candidate를 이후 contact까지 유지하는 절차가 있다. | 기존 방법이 ambiguity를 전혀 다루지 않는다는 주장은 틀리다. Candidate selection과 calibrated posterior propagation의 차이를 실제 downstream 평가로 보여야 한다. |
| [Asking the World, 2026](https://arxiv.org/html/2609.39135v2), §3·Appendix A.2–A.3 | CEM으로 world를 fitting하고 question에 맞춰 simulation과 intervention을 선택한다. Representation과 world modeling도 수정할 수 있다. | Fitted simulator에서 virtual probe를 수행하는 것과 실제 environment의 새 measurement를 받는 acquisition을 구별해야 한다. 이번 실험은 후자의 이상화된 setting이다. |
| [PhysTwin, 2025](https://arxiv.org/html/2503.17973), §4–5 | Three RGBD videos에서 geometry, topology, physical parameters, appearance를 맞추고 unseen interaction을 평가한다. | 실제 visual twin의 어려움과 규모가 우리 2D toy보다 크다. 여기서 실패한 geometry grid를 PhysTwin의 실패 증거로 제시할 수 없다. |
| [Active System Identification for Physics Twins, 2026 ongoing project](https://jacobgit2.github.io/projects/phystwin-active-sysid/) | 같은 factual motion에 맞는 twin들이 unseen action에서 달라지고, mean-field uncertainty가 이를 숨길 수 있음을 분석한다. Ensemble disagreement가 실제 refit payoff를 잘 예측하지 못한 결과와 model error 문제도 공개한다. | Peer-reviewed paper로 취급하지 않는다. 다만 “factual fit의 비식별성→ensemble→active interaction”이라는 생각은 공개 선행 작업과도 겹친다. |

이번 survey에서 느끼는 흐름은 단순히 world model을 만드는 데서 **그 model이 어떤 intervention까지 믿을 만한지, 어떤 추가 observation이 그 신뢰를 바꾸는지**를 묻는 방향으로 넓어지고 있다는 것이다. 이는 위 primary sources를 종합한 내 해석이며, 학회 전체의 정량적인 trend analysis는 아니다.

## 4. 지금 연구가 좋은 연구인가

**좋은 출발점이지만, 현재 상태로는 독립적인 main-conference method paper의 근거가 약하다.** 다음 네 가지를 분리해서 평가해야 한다.

1. **Problem relevance는 높다.** Factual fit이 좋아도 unseen intervention의 prediction이 틀리는 문제는 reconstruction, robotics, simulation 모두에 실용적이다. 이번 실험은 그 원인을 측정 가능한 parameter coupling으로 만들었다.
2. **Diagnostic evidence는 유용하다.** Noise가 줄어들수록 부정확한 posterior approximation의 downstream 결과가 오히려 악화되고, sign branch를 잃으면 observation selection도 크게 틀어진다. 어떤 오류인지 검증할 수 있는 reference가 있다는 장점이 있다.
3. **Method novelty는 아직 없다.** Tempering, resampling, Metropolis rejuvenation, conditional integration은 기존 도구다. 우리의 conditional quadrature가 잘된다는 사실은 새로운 대형 model이 필요하다는 주장에 반대되는 증거다.
4. **Acquisition의 실용적 추가 이득은 아직 약하다.** Two-branch setting에서는 reference가 모든 query에서 view를 선택한다. One-branch에서는 query마다 action이 달라지지만 대부분 probe와 stop의 차이다. Query-type rule과의 MSE gap이 작다는 결과를 숨겨서는 안 된다.

여기서 toy를 더 어렵게 만들어 특정 baseline을 계속 실패시키는 방향은 권하지 않는다. 실패 조건을 만들기보다, 실제 scene reconstruction에서 자연스럽게 생기는 ambiguity를 수집하고 **SMC를 포함한 강한 기준선으로도 남는 비용·신뢰성 문제**를 찾아야 한다.

## 5. 내가 선택할 다음 연구 방향

우선 제목 수준의 방향을 **“Calibrated Counterfactual Prediction under Contact-Model Ambiguity”**로 좁히겠다. Active acquisition은 이 문제를 해결하는 수단으로 두고, 아직 입증되지 않은 새로운 acquisition 원리를 중심 주장으로 삼지 않겠다.

핵심 질문은 다음과 같다. “Factual observation에 모두 부합하는 geometry, material, contact model이 있을 때, 어떤 counterfactual을 믿을 수 있으며, 어떤 추가 measurement가 모델들 사이의 차이를 실제로 줄이는가?” 이번 audit의 1 mm collision margin도 이 질문을 구체화한다. Collision proxy의 작은 차이가 restitution 변화로 흡수될 수 있기 때문이다.

### Contribution 후보 A: 검증 가능한 benchmark와 failure taxonomy

Continuous parameter뿐 아니라 contact surface assignment, occluded support, sharp/rounded collision proxy, friction regime을 포함한다. Factual fit은 비슷하지만 counterfactual은 다른 scene을 만들고, 실제 추가 view 또는 probe를 얻을 수 있게 한다. 평가에는 proper predictive scores, calibration, event probability, acquisition cost, simulator-call budget을 함께 둔다.

이번 결과는 benchmark의 sanity check로 쓸 수 있다. 하지만 benchmark paper를 주장하려면 여러 scene family와 현실적인 visual observation, 가능하면 physical data가 필요하다. PCVE-RigidBench 등 기존 benchmark와의 구체적인 추가 기능도 비교해야 한다.

### Contribution 후보 B: Contact branch를 보존하는 효율적인 inference

Posterior density 전체를 균일하게 잘 맞추기보다, query에 영향을 주는 contact event와 observation response가 다른 hypotheses를 보존하는 compression 또는 allocation을 연구할 수 있다. 그러나 loss-calibrated inference와 decision-aware BED의 일반 원리를 새로 제안한다고 하면 안 된다.

논문 기여가 되려면 최소한 다음을 보여야 한다. 동일한 simulator-call budget에서 standard SMC, MCMC, conditional/adaptive quadrature, posterior surrogate보다 낮은 design regret 또는 높은 predictive coverage를 얻어야 한다. 또한 여러 query와 unseen contact configuration에 transfer되어야 한다. 지금 2D setting에서는 conditional quadrature가 너무 잘 풀기 때문에, 이 후보의 필요성은 아직 입증되지 않았다.

### Contribution 후보 C: Model discrepancy를 판별하는 closed loop

추가 observation이 parameter uncertainty를 줄이는 경우와, 기존 model class 자체가 observation을 설명하지 못하는 경우를 구분한다. Held-out probe의 posterior predictive check가 반복적으로 실패하면 noise를 임의로 크게 만드는 대신 contact proxy나 model class를 다시 검토한다. 필요한 경우 prediction을 보류하거나 범위를 넓힌다.

이 또한 generic Bayesian model checking 자체가 새 기여인 것은 아니다. Visual reconstruction에서 만들어진 collision geometry의 오류를 구체적으로 진단하고 수정하며, counterfactual reliability를 개선하는 실험이 있어야 한다. 이번 margin audit는 이 후보를 검토할 실질적인 이유를 제공하지만, 아직 closed-loop correction을 구현한 결과는 아니다.

## 6. 진행 여부를 결정할 다음 검증

| 검증 | 계속할 근거 | 방향을 바꿔야 할 신호 |
|---|---|---|
| Shape·pose·friction·restitution·contact mode를 함께 바꾼 3개 이상의 scene family | 동일 compute budget에서 strong posterior baseline과 실용적인 차이가 반복됨 | SMC만으로 모든 문제가 충분히 빠르게 해결됨 |
| Noisy RGB/RGBD observation | GT state 없이도 calibrated uncertainty와 acquisition 효과가 유지됨 | Sensor calibration 오류가 모든 이득을 압도함 |
| Query-type rule, parameter EIG, task-aware policy의 비교 | 여러 유용한 action 사이에서 의미 있는 선택 이득이 있음 | 차이가 거의 stop 여부에만 있고 fixed rule과 MSE가 비슷함 |
| Correct vs misspecified contact model | Parameter uncertainty와 model discrepancy를 구분해 대응할 수 있음 | Narrow posterior가 단순히 잘못된 simulator를 더 확신함 |
| Small real-world pilot | Held-out physical intervention에서 coverage와 prediction이 개선됨 | Factual fitting loss만 좋아지고 unseen action은 악화됨 |

내 판단은 **연구를 중단하기보다 중심 주장을 수정하는 편이 좋다**는 것이다. 현재 실험은 새로운 algorithm을 정당화하기보다, 어떤 일반 원리가 이미 알려져 있고 어떤 실용 문제가 아직 남는지 가려내는 데 성공했다. 다음 단계에서 현실적인 contact-model ambiguity와 calibrated counterfactual prediction을 입증하지 못한다면, 이 toy만 계속 확장하는 데 많은 시간을 쓰지는 않겠다.


---

재현 artifact: `results/main/metrics_*.csv`, `results/analysis/*.json`, `results/audit/*.json`, `logs/`, `provenance/`. Source와 artifact hash는 `artifact_manifest.json`에 기록한다.
