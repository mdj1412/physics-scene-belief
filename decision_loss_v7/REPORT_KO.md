# 새로운 worlds와 반복 observation noise에서의 decision-loss 검증 — v7

검증일: 2026-11-11. Server에서 실행한 결과와 local statistical analysis를 정리했다. 설명은 한글, technical terms는 English로 유지했다.

**Primary comparison에서 tempered bridge가 factual-bank SNIS4096보다 decision loss를 낮춘다는 증거는 확보하지 못했다.**

Primary `bridge − SNIS4096`는 0.000110049 `(m/s)^2`, world-cluster bootstrap 95% CI [-0.00199844, 0.00259593]다. 음수는 bridge에 유리하다. 사전 지정한 12 comparisons 중 Bonferroni-level interval 전체가 음수인 비교는 0개, 전체가 양수인 비교는 0개다. Interval에 0이 포함되면 우열 미확정으로 해석하며 equivalence의 증거로 바꾸지 않는다.

다만 **corner family + precise view + equal cost**는 후속 검증할 조건으로 남았다. 이 16-world descriptive subgroup에서는 bridge MSE가 0.000374449, SNIS4096이 0.00282173, SNIS16384가 0.000992342였다. Paired differences의 95% CI는 각각 [-0.00501239, -0.000608381]와 [-0.00132653, -6.81232e-05]다. 이 family subgroup은 사전 계획상 descriptive analysis이며, 위 사전 지정 12-condition comparison family와 다르다. 작은 subset의 발견을 전체 superiority나 real-camera 효과로 일반화하지 않는다.

관측 precision을 높였을 때 corner의 SNIS4096 MSE는 0.000399241에서 0.00282173로 오히려 커졌고, bridge는 0.000399604에서 0.000374449로 소폭 낮아졌다. 이는 sharper likelihood에서 finite-bank acquisition이 불안정해질 수 있다는 단서다. **현재 논문 후보는 이 조건을 설명하고 저렴한 baseline으로도 해결되지 않는지 검증하는 데 있으며, tempering 자체의 novelty에는 없다.**

## 1. 이번에 해결하려던 불확실성

V6에서는 tempered bridge가 conditional posterior를 안정화했지만, factual-bank SNIS 대비 actual decision loss의 개선은 확인되지 않았다. 이번에는 기존 sample에 noise를 덧붙이는 데 그치지 않고 새로운 latent parameters와 factual evidence를 생성했다. 또한 SNIS에 쓰는 bank와 hypothetical observations를 생성하는 bank를 분리했다.

여기서 **new worlds는 같은 3개 scene families의 새로운 parameter draws**다. 새로운 scene class로의 generalization을 뜻하지 않는다. World마다 factual measurement는 하나이며, 그 measurement에 따른 posterior를 고정한 채 **추가 observation의 noise만 32회** 반복했다. 48×32개 independent worlds가 아니다.

## 2. Protocol과 비교의 의미

| 항목 | 설정 |
| --- | --- |
| Worlds | 48: corner/gate/deflection 각 16; seed 202611110700 + case |
| Latent parameters | Geometry offset, restitution, friction, obstacle extent, corner rounding의 5D bounded uniform prior |
| Queries / actions | 6 counterfactual shifts / view, left, right, center |
| Factual evidence | 한 번의 Gaussian x-velocity measurement; world 안에서 고정 |
| Inference / generator bank | 16384 / 별도 seed의 8192; bridge와 SNIS4096은 첫 4096 사용 |
| Bridge acquisition | N1024, 12 MH moves/stage, ESS threshold 0.75N, 2 independent inner runs |
| Outer budget | H256 → acquisition-only ambiguity에 따라 H512; 32 worlds 확장 |
| Actual observations | 각 action/precision에서 32 post-observation noise repeats; methods 사이 같은 inputs |
| Common decoder | Prior-start full SMC N2048, 18 MH moves/stage |
| Audits | 6 worlds: full acquisition H128/N2048, decoder N4096; 12 worlds: raster observations R8 |
| Primary statistics | 48 world clusters, 50000 bootstrap draws; primary 95% CI |
| Multiple comparisons | 2 precisions × 3 costs × 2 SNIS sizes; 99.5833% marginal percentile intervals |

Simulator는 radius 20 mm의 sphere와 static rounded rectangles 사이의 2D continuous collision detection, Coulomb impulse와 spin을 사용하며 gravity는 없다. Prior는 geometry offset g∈[−8,8] mm, restitution e∈[0.15,0.85], friction μ∈[0,0.6], d∈[22,34] mm, corner rounding b∈[0,6] mm다. d는 family에 따라 obstacle half-height 또는 gate half-gap을 제어한다. Counterfactual shifts는 [−30,−17,−7,4,15,28] mm이고 target은 t=0.8 s의 instantaneous velocity다. Dynamic measurements는 t=0.75–0.8 s의 finite-exposure mean velocity를 모델링한다.

`native`는 기존 pixel-only calibration의 feature covariance다. `precise`는 **view standard deviation만 0.25배**로 줄인 synthetic continuous-proxy stress condition이다. Dynamic probe noise는 바꾸지 않았다. Native calibration 자체도 geometry offset feature의 standard deviation이 약 0.073–0.089 mm, d feature가 약 0.072 mm로 sharp하다. 따라서 native를 저정밀 observation으로 해석하면 안 된다. 이는 known scene family, known colors, 고정된 image scale의 synthetic sensor calibration이다. 실제 camera resolution이나 raw-image encoder를 개선한 실험은 아니다.

Cost profiles의 action order는 `[view, left, right, center]`이고, `equal=[0,0,0,0]`, `view_cheaper=[0,.002,.002,.002]`, `view_costlier=[.002,0,0,0]`이다. Cost는 MSE와 합산하는 사전 지정 utility 단위다. 실제 장비 비용의 측정값이 아니다. 정확히 한 observation을 선택해야 한다.

각 query의 target velocity를 $z_q$, factual evidence를 $D_f$라 하면 acquisition은 다음 quantity를 근사한다.

$$U_q(a)=\tfrac12\mathbb E_{y_a\mid D_f}\!\left[\|\mathbb E[z_q\mid D_f,y_a]-\mathbb E[z_q\mid D_f]\|^2\right]-c(a).$$

Bridge는 두 independent inner posterior means의 shifts를 cross-product해서 inner sampling variance의 squared-mean contribution을 줄인다. SNIS는 weighted mean의 squared shift를 사용한다. 둘 다 finite factual bank와 finite outer budget의 approximation을 포함하며 unbiased estimator라고 주장하지 않는다. **따라서 이것은 두 acquisition pipelines의 비교이며 tempering만의 causal ablation은 아니다.** Bridge에는 cross-product estimator를, SNIS에는 standard squared-shift estimator를 사용했다. Cross-fitted SNIS utility나 matched-compute의 더 큰 bank를 별도로 비교하지 않았으므로, 관찰되는 차이를 전부 tempering의 효과로 귀속시키지 않는다. Shared current mean도 finite independent generator bank에서 추정한다.

실제 loss는 선택된 observation의 **공통 full SMC decoder**가 만든 velocity prediction과 simulator truth 사이 component-mean squared error에 cost를 더한 값이다. 따라서 primary comparison은 선택 알고리즘의 차이다. SNIS decoder로 바꾸는 효과는 별도 ablation으로 분리했다. 같은 world의 actions/precisions에는 common random numbers를 써 paired variance를 줄였고, noise repeats 사이에는 independent draws를 썼다.

## 3. 실제 decision loss

| Precision | Cost | Baseline | Bridge − baseline | 95% CI | Bonferroni interval | Action 변경: 전체 / active / worlds |
| --- | --- | --- | --- | --- | --- | --- |
| native | equal | snis4096 | 0.000110049 | [-0.00199844, 0.00259593] | [-0.00295757, 0.00403843] | 89/288 · 52/250 · 39/48 |
| native | equal | snis16384 | 0.000233618 | [-0.0018606, 0.0026684] | [-0.00281503, 0.00406662] | 75/288 · 38/250 · 36/48 |
| native | view_cheaper | snis4096 | 0.000561389 | [-0.00115831, 0.00294105] | [-0.00176044, 0.0043352] | 58/288 · 58/250 · 28/48 |
| native | view_cheaper | snis16384 | 0.000586133 | [-0.00112925, 0.00296018] | [-0.0017226, 0.00434383] | 42/288 · 42/250 · 21/48 |
| native | view_costlier | snis4096 | 0.000483723 | [-0.00214608, 0.00354888] | [-0.00326411, 0.00520468] | 93/288 · 56/250 · 40/48 |
| native | view_costlier | snis16384 | 0.000891114 | [-0.00179268, 0.00399714] | [-0.00290526, 0.00566112] | 76/288 · 39/250 · 36/48 |
| precise | equal | snis4096 | -0.000675986 | [-0.00296024, 0.00196831] | [-0.00397104, 0.00347713] | 140/288 · 103/250 · 46/48 |
| precise | equal | snis16384 | -7.14741e-05 | [-0.00220261, 0.00244674] | [-0.0031461, 0.00390947] | 131/288 · 94/250 · 45/48 |
| precise | view_cheaper | snis4096 | 0.000114481 | [-0.00185326, 0.00262655] | [-0.00261115, 0.00408249] | 89/288 · 89/250 · 33/48 |
| precise | view_cheaper | snis16384 | 0.000467224 | [-0.00130779, 0.00288732] | [-0.00192869, 0.00428389] | 84/288 · 84/250 · 32/48 |
| precise | view_costlier | snis4096 | 0.000127697 | [-0.0026022, 0.0033348] | [-0.00373222, 0.00504133] | 109/288 · 72/250 · 43/48 |
| precise | view_costlier | snis16384 | 0.000434062 | [-0.00225638, 0.00360185] | [-0.00337477, 0.00532485] | 99/288 · 62/250 · 42/48 |

![Decision loss](results/analysis/figures/decision_loss.png)

Cost가 있는 rows는 순수 prediction MSE가 아니라 `MSE + cost`다. 아래는 equal-cost에서의 absolute MSE다. Random은 action을 균일하게 뽑았을 때의 expected loss를 계산했다. No-observation은 constrained task 밖의 참고값이다. Empirical oracle는 32개의 actual losses를 본 privileged optimistic lower bound이며 실행 가능한 policy가 아니다. Crossfit oracle는 16 repeats로 고른 action을 반대 16 repeats에서 평가하고 교환했다. 역시 hidden-world outcome를 사용한 진단이다.

| Precision | Policy | Mean MSE | 95% CI |
| --- | --- | --- | --- |
| native | bridge | 0.00735024 | [0.00404854, 0.0116568] |
| native | snis4096 | 0.0072402 | [0.0041534, 0.0110015] |
| native | snis16384 | 0.00711663 | [0.00404854, 0.0108861] |
| native | always_view | 0.00868801 | [0.00562323, 0.0123893] |
| native | probe_only_bridge | 0.0124391 | [0.00633631, 0.0204882] |
| native | random | 0.0188688 | [0.0127869, 0.0258317] |
| native | no_observation | 0.0473858 | [0.0352608, 0.0605251] |
| native | empirical_oracle | 0.00307182 | [0.00185023, 0.00460042] |
| native | crossfit_oracle | 0.00315004 | [0.00192014, 0.00468299] |
| precise | bridge | 0.00691907 | [0.00358444, 0.0112661] |
| precise | snis4096 | 0.00759506 | [0.00445925, 0.0114216] |
| precise | snis16384 | 0.00699054 | [0.00387283, 0.010822] |
| precise | always_view | 0.00822799 | [0.00509578, 0.0120279] |
| precise | probe_only_bridge | 0.0124391 | [0.00633631, 0.0204882] |
| precise | random | 0.0187538 | [0.0126768, 0.0257057] |
| precise | no_observation | 0.0473858 | [0.0352608, 0.0605251] |
| precise | empirical_oracle | 0.00274426 | [0.00151917, 0.00429993] |
| precise | crossfit_oracle | 0.00279275 | [0.00155873, 0.00435342] |

### 반복 noise가 추정에 미친 영향

| Precision | Baseline | R | Paired difference | World-cluster 95% CI | Conditional noise SEM |
| --- | --- | --- | --- | --- | --- |
| native | snis4096 | 1 | 0.000121378 | [-0.001945, 0.00259863] | — |
| native | snis4096 | 4 | 0.00018735 | [-0.00192457, 0.00266268] | 9.81954e-05 |
| native | snis4096 | 8 | 6.53127e-05 | [-0.00200806, 0.00246723] | 5.87237e-05 |
| native | snis4096 | 16 | 8.93984e-05 | [-0.00200549, 0.00256096] | 4.74157e-05 |
| native | snis4096 | 32 | 0.000110049 | [-0.00199844, 0.00259593] | 3.21283e-05 |
| native | snis16384 | 1 | 0.000251645 | [-0.00181096, 0.00269041] | — |
| native | snis16384 | 4 | 0.00031765 | [-0.00178182, 0.00275109] | 9.53697e-05 |
| native | snis16384 | 8 | 0.000198425 | [-0.00186992, 0.00256084] | 5.69303e-05 |
| native | snis16384 | 16 | 0.000216395 | [-0.00186517, 0.00263615] | 4.45047e-05 |
| native | snis16384 | 32 | 0.000233618 | [-0.0018606, 0.0026684] | 3.06443e-05 |
| precise | snis4096 | 1 | -0.000610309 | [-0.00290935, 0.00203523] | — |
| precise | snis4096 | 4 | -0.000453896 | [-0.00277259, 0.00218876] | 0.00013164 |
| precise | snis4096 | 8 | -0.000654524 | [-0.00293857, 0.00193134] | 7.6294e-05 |
| precise | snis4096 | 16 | -0.000653233 | [-0.00294161, 0.00198392] | 5.27274e-05 |
| precise | snis4096 | 32 | -0.000675986 | [-0.00296024, 0.00196831] | 3.51713e-05 |
| precise | snis16384 | 1 | -2.09167e-05 | [-0.00214062, 0.00249773] | — |
| precise | snis16384 | 4 | 0.000126775 | [-0.00202824, 0.00265176] | 0.000130738 |
| precise | snis16384 | 8 | -5.75051e-05 | [-0.00219699, 0.00239988] | 7.54477e-05 |
| precise | snis16384 | 16 | -5.33297e-05 | [-0.00218792, 0.00245527] | 5.21027e-05 |
| precise | snis16384 | 32 | -7.14741e-05 | [-0.00220261, 0.00244674] | 3.46757e-05 |

![Noise convergence](results/analysis/figures/noise_convergence.png)

Conditional noise SEM은 현재 48 worlds에 조건을 둔 repeat 평균의 불확실성이다. 각 measurement마다 finite full SMC decoder를 실행했으므로 measurement noise와 inner Monte Carlo variation이 함께 포함된다. Pure measurement-noise variance만을 분리 추정한 값은 아니다. World-to-world uncertainty와 구별해야 한다. R을 늘려도 독립 world 수는 48로 남는다. 유리한 prefix를 선택해서 primary result를 바꾸지 않았다.

R32의 primary conditional repeat SEM은 3.21283e-05이고 world-level SEM은 0.00117933이다. 이번 불확실성은 추가 observation noise 평균화보다 **world 간 heterogeneity**의 영향을 크게 받는다. R만 더 늘리는 것보다 independent worlds를 늘려야 하는 이유다. Bridge model이 예측한 SNIS4096 대비 평균 utility gain은 0.00135316였지만 실제 paired loss가 그만큼 감소했다고 확인되지 않았다. 이 model-based quantity를 actual decision benefit의 증거로 대체하지 않는다.

## 4. 남은 ranking ambiguity

Practical stability는 common-outer bootstrap에서 선택 action의 regret upper95%가 `max(5e-5, 0.05 × max gross EVSI)` 이내인지로 정의했다. Exact top-1 confidence와 다르다. Two actions가 거의 동률이면 top-1이 흔들려도 practical regret는 작을 수 있다. H512 이후에도 21 worlds에 적어도 하나의 unresolved query/precision/cost가 남았다.

분류는 actual outcomes를 보지 않고 고정했다. `stable_agreement`는 bridge와 SNIS 모두 practically stable하고 같은 action을 고르는 경우다. `stable_near_tie_disagreement`는 둘 다 stable하지만 선택이 다르고 bridge utility 기준 gap이 tolerance 안인 경우다. `stable_material_disagreement`는 그 gap이 tolerance보다 큰 경우다. 이것이 실제 truth에서 bridge가 이긴다는 뜻은 아니다.

| Precision / equal cost | Class | Queries | Eligible worlds | 전체 mean에 대한 contribution | Contribution 95% CI |
| --- | --- | --- | --- | --- | --- |
| native | inactive | 38 | 16 | 0 | [0, 0] |
| native | bridge_unresolved | 13 | 12 | 7.73901e-05 | [0, 0.000188517] |
| native | snis_unresolved | 3 | 3 | 0 | [0, 0] |
| native | stable_agreement | 186 | 48 | 0 | [0, 0] |
| native | stable_near_tie_disagreement | 19 | 16 | -0.000530613 | [-0.00118735, -1.45707e-05] |
| native | stable_material_disagreement | 29 | 15 | 0.000563272 | [-0.00128014, 0.00301614] |
| precise | inactive | 38 | 16 | 0 | [0, 0] |
| precise | bridge_unresolved | 13 | 12 | 6.67169e-05 | [-1.12591e-05, 0.000180397] |
| precise | snis_unresolved | 32 | 18 | -0.000360367 | [-0.000837917, -3.247e-05] |
| precise | stable_agreement | 124 | 44 | 0 | [0, 0] |
| precise | stable_near_tie_disagreement | 21 | 18 | -0.000527688 | [-0.00118984, -8.78522e-06] |
| precise | stable_material_disagreement | 60 | 32 | 0.000145352 | [-0.00180833, 0.00266503] |

| Precision | Cost | Active queries | Practically stable | Decisive top-1 | Half-sample agreement |
| --- | --- | --- | --- | --- | --- |
| native | equal | 250 | 237 | 140 | 0.78 |
| native | view_cheaper | 250 | 245 | 155 | 0.816 |
| native | view_costlier | 250 | 236 | 145 | 0.772 |
| precise | equal | 250 | 237 | 141 | 0.804 |
| precise | view_cheaper | 250 | 245 | 164 | 0.844 |
| precise | view_costlier | 250 | 236 | 152 | 0.764 |

![Ranking ambiguity](results/analysis/figures/ranking_ambiguity.png)

표의 contribution은 class 밖의 queries를 0으로 두고 전체 48 worlds × 6 queries에서 평균한 값이다. 모든 classes의 contribution을 합하면 primary difference가 된다. CI도 48 world clusters로 계산했다. Eligible worlds 수는 해당 class가 포함된 worlds 수다. 별도의 eligible-world-balanced subgroup mean은 summary JSON에 저장했으며, query 수에 따른 가중치가 달라 contribution과 부호가 다를 수 있다. Small subgroups의 CI는 descriptive이며 12-condition primary family의 confirmatory evidence로 사용하지 않는다.

이 bootstrap은 주로 finite outer sample uncertainty를 본다. Shared factual bank의 approximation bias, finite MH mixing, shared current mean의 오류까지 보장하지 않는다. 따라서 practical stability가 곧 posterior correctness라는 해석은 잘못이다.

또 하나는 **numerical ambiguity와 hidden-world ambiguity의 차이**다. Acquisition은 factual evidence에 부합하는 posterior worlds 전체에서의 expected loss를 최소화한다. 이번 R32 evaluation은 각 draw된 true world를 고정하고 observation noise를 평균한다. Exact Bayes-optimal action조차 특정 true world에서는 privileged oracle보다 나쁠 수 있다. 개별 case의 actual-loss ranking이 acquisition ranking과 다르다는 사실만으로 inference error라고 단정할 수 없다. Numerical bias는 matched full-SMC audit로, 실제 가치의 일반화는 independent world-cluster comparison으로 판단해야 한다.

### SNIS ESS와 강한 bank baseline

| Precision | Bank | View median ESS | 5th percentile | ESS < 20 | Hypothetical observations |
| --- | --- | --- | --- | --- | --- |
| native | snis4096 | 1.60444 | 1 | 0.999854 | 20480 |
| native | snis16384 | 3.50078 | 1.00797 | 0.972656 | 20480 |
| precise | snis4096 | 1 | 1 | 1 | 20480 |
| precise | snis16384 | 1.00022 | 1 | 1 | 20480 |

ESS < 20은 사전 outcome 열람 전 정한 descriptive diagnostic cutoff다. 최적 threshold로 검증한 값이 아니다. Low ESS는 potential failure의 신호이며, 그 action이 query의 decision에 중요하지 않거나 다른 action과 near-tie이면 actual loss 개선으로 이어지지 않을 수 있다.

| Precision / equal cost | Bank | World group | N worlds | Difference | 95% CI |
| --- | --- | --- | --- | --- | --- |
| native | snis4096 | view_ESS_lt20 | 48 | 0.000110049 | [-0.00199844, 0.00259593] |
| native | snis4096 | view_ESS_ge20 | 0 | — | — |
| native | snis16384 | view_ESS_lt20 | 47 | 0.000238589 | [-0.00188008, 0.0027108] |
| native | snis16384 | view_ESS_ge20 | 1 | 0 | [0, 0] |
| precise | snis4096 | view_ESS_lt20 | 48 | -0.000675986 | [-0.00296024, 0.00196831] |
| precise | snis4096 | view_ESS_ge20 | 0 | — | — |
| precise | snis16384 | view_ESS_lt20 | 48 | -7.14741e-05 | [-0.00220261, 0.00244674] |
| precise | snis16384 | view_ESS_ge20 | 0 | — | — |

| Precision | Cost | Method | View / left / right / center: active queries |
| --- | --- | --- | --- |
| native | equal | bridge | 97 / 32 / 52 / 69 |
| native | equal | snis4096 | 144 / 3 / 39 / 64 |
| native | equal | snis16384 | 133 / 14 / 36 / 67 |
| native | view_cheaper | bridge | 150 / 23 / 40 / 37 |
| native | view_cheaper | snis4096 | 199 / 1 / 25 / 25 |
| native | view_cheaper | snis16384 | 188 / 4 / 23 / 35 |
| native | view_costlier | bridge | 74 / 46 / 54 / 76 |
| native | view_costlier | snis4096 | 123 / 14 / 42 / 71 |
| native | view_costlier | snis16384 | 109 / 29 / 42 / 70 |
| precise | equal | bridge | 98 / 32 / 51 / 69 |
| precise | equal | snis4096 | 197 / 1 / 25 / 27 |
| precise | equal | snis16384 | 192 / 4 / 28 / 26 |
| precise | view_cheaper | bridge | 155 / 22 / 38 / 35 |
| precise | view_cheaper | snis4096 | 242 / 0 / 7 / 1 |
| precise | view_cheaper | snis16384 | 239 / 0 / 7 / 4 |
| precise | view_costlier | bridge | 80 / 42 / 53 / 75 |
| precise | view_costlier | snis4096 | 147 / 2 / 40 / 61 |
| precise | view_costlier | snis16384 | 140 / 7 / 36 / 67 |

위 action counts는 bridge의 gross EVSI 기준 active queries만 포함한다. Inactive queries에서는 사실상 같은 predictions 사이의 floating-point tie도 action identity를 바꿀 수 있어 별도로 제외했다. Loss의 primary average에서는 288 queries를 모두 유지한다.

### Scene family와 영향력이 큰 worlds

| Precision / equal cost | Bank | Family | Bridge MSE | Baseline MSE | Difference | 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| native | snis4096 | corner | 0.000399604 | 0.000399241 | 3.62699e-07 | [-3.05327e-07, 1.39342e-06] |
| native | snis4096 | gate | 0.0181267 | 0.0179437 | 0.000183025 | [-0.00607993, 0.00775411] |
| native | snis4096 | deflection | 0.00352438 | 0.00337763 | 0.000146759 | [-9.16387e-07, 0.000396479] |
| native | snis16384 | corner | 0.000399604 | 0.000402459 | -2.85449e-06 | [-8.56347e-06, 0] |
| native | snis16384 | gate | 0.0181267 | 0.017565 | 0.000561773 | [-0.00569726, 0.00794267] |
| native | snis16384 | deflection | 0.00352438 | 0.00338245 | 0.000141936 | [-2.60472e-05, 0.000407622] |
| precise | snis4096 | corner | 0.000374449 | 0.00282173 | -0.00244728 | [-0.00501239, -0.000608381] |
| precise | snis4096 | gate | 0.0180443 | 0.0177538 | 0.000290484 | [-0.00603298, 0.00789667] |
| precise | snis4096 | deflection | 0.00233844 | 0.00220961 | 0.000128834 | [-0.000280617, 0.00051474] |
| precise | snis16384 | corner | 0.000374449 | 0.000992342 | -0.000617894 | [-0.00132653, -6.81232e-05] |
| precise | snis16384 | gate | 0.0180443 | 0.0177513 | 0.000293053 | [-0.00603047, 0.00789918] |
| precise | snis16384 | deflection | 0.00233844 | 0.00222802 | 0.000110418 | [-0.000298447, 0.000493761] |

| Precision / equal cost | Bank | Leave-one-world-out mean range | 가장 유리한 world | 가장 불리한 world |
| --- | --- | --- | --- | --- |
| native | snis4096 | [-0.000826195, 0.00065932] | 25 | 30 |
| native | snis16384 | [-0.000657233, 0.000785519] | 25 | 30 |
| precise | snis4096 | [-0.00163032, -0.000137652] | 25 | 30 |
| precise | snis16384 | [-0.00101294, 0.000479722] | 25 | 30 |

Family 분석은 각 16 worlds에 불과하다. Leave-one-world-out range는 하나의 world가 결론을 지배하는지 보는 sensitivity diagnostic이며 새로운 CI가 아니다.

### Likelihood concentration의 post-hoc diagnostic

| Precision | SNIS bank | ESS ≤ 1.01 / 768 hypothetical views | 해당 views에서 full-SMC predictive variance median |
| --- | --- | --- | --- |
| native | snis4096 | 149 / 768 | 0.0227054 |
| native | snis16384 | 47 / 768 | 0.0205783 |
| precise | snis4096 | 600 / 768 | 0.0108906 |
| precise | snis16384 | 420 / 768 | 0.0170805 |

Normalized weights에서는 `max(w) ≥ Σw² = 1/ESS`이므로 ESS ≤ 1.01이면 한 particle에 적어도 약 99%의 weight가 놓인다. 그런데 동일 observations의 full SMC reference에는 남은 counterfactual uncertainty가 있었다. 표의 variance는 velocity components와 6 queries를 평균한 conditional variance이며 `(m/s)^2` 단위다. 이는 geometry observation의 sharp likelihood가 finite bank의 다른 predictive uncertainty까지 과도하게 없애는 근사와 일치한다. Bank-induced uncertainty loss와 squared-mean utility estimator의 bias가 observation을 과대평가하는 설명 후보이며, tempering만의 causal effect를 분리해 증명한 것은 아니다. 이 표는 기존 6-world reference outputs의 post-hoc diagnostic이고 새로운 confirmatory subgroup test가 아니다.

![조건별 효과와 world heterogeneity](results/analysis/figures/condition_and_worlds.png)

## 5. Reference, decoder, measurement mismatch audits

### Full SMC acquisition reference

사전 지정한 6 worlds에서 같은 첫 128 hypothetical observations를 사용했다. Main result의 H256/512 choice와 reference H128 choice를 혼동하지 않도록 이 표에서는 양쪽 H128로 맞췄다. Reference는 N2048/18 MH/2 runs이고 exact posterior가 아니다.

| Precision / equal cost | Method | Active queries | Rank agreement | Reference utility regret | Prediction RMSE |
| --- | --- | --- | --- | --- | --- |
| native | bridge | 30 | 1 | 0 | 0.00578853 |
| native | snis4096 | 30 | 0.8 | 0.0018173 | 0.0465885 |
| native | snis16384 | 30 | 0.866667 | 0.00100309 | 0.0325626 |
| precise | bridge | 30 | 1 | 0 | 0.00660399 |
| precise | snis4096 | 30 | 0.633333 | 0.00347866 | 0.0582476 |
| precise | snis16384 | 30 | 0.633333 | 0.0024011 | 0.0521097 |

### Common decoder를 N4096으로 바꾼 audit

| Precision / equal cost | Baseline | N2048 차이: 같은 6 worlds | N4096 차이 | N4096 95% CI | Decoder prediction RMSE |
| --- | --- | --- | --- | --- | --- |
| native | snis4096 | 0.00237006 | 0.00238161 | [3.25499e-05, 0.00667252] | 0.00509425 |
| native | snis16384 | 0.00286161 | 0.00287905 | [0, 0.0085363] | 0.00509425 |
| precise | snis4096 | -0.000590633 | -0.000632446 | [-0.00849674, 0.00647727] | 0.00563032 |
| precise | snis16384 | 0.00280474 | 0.00277719 | [0.000348591, 0.00695664] | 0.00563032 |

이 audit는 selected action을 바꾸지 않는다. 6 worlds의 결과를 전체 48 worlds의 decoder convergence proof로 일반화하지 않는다.

### Selection과 decoding의 분리

| Precision / equal cost | Fixed selector | Decoder | Absolute loss | Common full decoder 대비 차이 |
| --- | --- | --- | --- | --- |
| native | bridge | snis4096 | 0.0130623 | 0.00571209 |
| native | snis4096 | snis4096 | 0.0178815 | 0.0106414 |
| native | snis16384 | snis4096 | 0.0171161 | 0.00999949 |
| native | bridge | snis16384 | 0.0109988 | 0.00364855 |
| native | snis4096 | snis16384 | 0.0149821 | 0.00774189 |
| native | snis16384 | snis16384 | 0.0143129 | 0.0071963 |
| precise | bridge | snis4096 | 0.0141408 | 0.00722175 |
| precise | snis4096 | snis4096 | 0.0200353 | 0.0124402 |
| precise | snis16384 | snis4096 | 0.0195318 | 0.0125412 |
| precise | bridge | snis16384 | 0.0124175 | 0.00549846 |
| precise | snis4096 | snis16384 | 0.0187053 | 0.0111102 |
| precise | snis16384 | snis16384 | 0.0179407 | 0.0109502 |

위 표에서 decoder를 바꿔 생긴 차이는 observation selection의 개선이 아니다. Actual observation에 대한 bridge decoder는 이번 실험에서 따로 실행하지 않았으므로 `bridge selector + full decoder`를 `bridge end-to-end inference`라고 부르지 않는다.

### Raster measurement mismatch

| Native precision | Baseline | Gaussian: 같은 12 worlds | Raster R8 difference | Raster 95% CI |
| --- | --- | --- | --- | --- |
| equal | snis4096 | 0.000640676 | 0.000892418 | [-0.000827372, 0.00348643] |
| equal | snis16384 | 0.000942971 | 0.0011651 | [-0.000832664, 0.00441017] |
| view_cheaper | snis4096 | 0.00105176 | 0.00129064 | [-0.000412875, 0.00397145] |
| view_cheaper | snis16384 | 0.000929733 | 0.00118522 | [-0.000499781, 0.00387889] |
| view_costlier | snis4096 | -0.000487861 | -0.000240683 | [-0.00310133, 0.00284572] |
| view_costlier | snis16384 | 0.000123002 | 0.00027636 | [-0.00307952, 0.00414504] |

이것은 synthetic RGB renderer와 centroid/geometry feature extractor를 사용한 additional-observation mismatch audit다. Factual evidence는 여전히 Gaussian이고 policy도 primary acquisition에서 고정한 것을 썼다. Real video, real contact dynamics, end-to-end visual reconstruction 검증은 아니다. Precise condition에는 구현된 고정밀 image sensor가 없으므로 raster 결과를 붙이지 않았다.

Equal-cost decoder audit에서 N2048→N4096으로 바꾼 paired difference의 최대 변화는 4.18129e-05였다. 이 6-world subset에서는 particle budget을 두 배로 늘려도 비교 양상이 크게 바뀌지 않았다. Native raster audit의 bridge − SNIS4096은 0.000892418, 95% CI [-0.000827372, 0.00348643]로 추가 이득이 확인되지 않았다. Corner/precise의 단서는 synthetic precision stress에서 얻은 것이며 이 raster audit로 실증된 것이 아니다.

### 개별 case에서 관찰한 선택 차이

아래 examples는 전체 evaluation 뒤 선택한 가장 유리하거나 불리한 query 사례다. 조건 발견을 위한 descriptive examples이며 confirmatory evidence가 아니다. 한 world의 R32 mean은 해당 hidden parameters에 조건을 둔 값이다.

| Precision / equal cost | World / query | Family | Bridge action | SNIS4096 action | R32 mean difference |
| --- | --- | --- | --- | --- | --- |
| precise | 25 / 1 | gate | left | view | -0.0606435 |
| native | 25 / 1 | gate | left | view | -0.0600627 |
| precise | 0 / 5 | corner | center | view | -0.0538835 |
| native | 17 / 1 | gate | left | view | -0.0523079 |
| precise | 25 / 0 | gate | left | view | -0.0517493 |
| precise | 30 / 1 | gate | left | view | 0.250037 |
| native | 30 / 1 | gate | left | view | 0.249515 |
| precise | 17 / 5 | gate | right | view | 0.154154 |
| native | 17 / 5 | gate | right | view | 0.153657 |
| precise | 22 / 5 | gate | right | view | 0.025002 |


## 6. Compute와 실행 신뢰성

| Precision | Bridge final-H acquisition seconds | SNIS4096 seconds | SNIS16384 seconds |
| --- | --- | --- | --- |
| native | 4266.55 | 1.29954 | 4.00542 |
| precise | 4570.44 | 1.13189 | 3.99954 |

이 표는 최종 선택된 H의 conditional inference timers를 합한 값이다. Initial bank build, 버려진 H256 계산, summaries, planning, evaluation, 전체 I/O를 포함하는 end-to-end timing이 아니다. SNIS는 cached predictions에 대한 weighting을 재사용한다. 따라서 fair matched-compute benchmark라고 표현하지 않는다. 실제 deployment 비용 비교에서는 bank 생성과 cache memory, 한 factual scene에서 처리할 query 수까지 포함해야 한다. 두 precision rows는 shared probe 계산을 각각 포함하므로 합쳐서 unique total compute로 해석하면 안 된다.

| Integrity check | 결과 |
| --- | --- |
| New worlds / fixed policies | 48 unique truth files; independent generator bank; recomputed choices가 frozen choices와 동일 |
| Conditional SMC populations | 303424 |
| Conditional result files | 1178 |
| Unique conditional batches / distinct seeds | 347 |
| Minimum stage ESS / N | 0.75 |
| Maximum final-beta error | 0 |
| Conditional trajectory rollouts | 6.10948e+10 |
| Conditional GPU seconds 전체 합 | 12291.7 |
| Numerical validation | Bounded Gaussian quadrature, independent NumPy SNIS, covariance scaling, batched smoke 통과 |
| Analysis validation | Action/query axis, world clustering, exact zero difference, empty subgroup 통과 |

GPU는 실행 직전 process occupancy가 없는 사용자 allocations만 사용했고 foreign-process guard를 유지했다. 이전 v5/v6 artifacts와 사용자 allocations는 변경하지 않았다. File hashes와 runtime/worker logs는 accompanying artifacts에 저장한다. Standard SMC의 stage ESS 유지가 모든 mode discovery를 보장하지는 않는다.

## 7. 논문 기여로 이어질 가능성

Tempered bridge 자체를 새로운 algorithm으로 주장할 수는 없다. [Del Moral, Doucet, Jasra의 SMC samplers](https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf)는 sequence of distributions, importance weighting, resampling과 Markov moves의 framework를 제시한다. 이번 bridge는 그 계열의 factual-to-hypothetical conditioning 구현이다.

최근 관련 원문도 다시 확인했다. [Nesting Particle Filters for Experimental Design in Dynamical Systems, ICML 2024](https://proceedings.mlr.press/v235/iqbal24a.html)는 sequential dynamical design의 Inside-Out SMC²와 policy amortization을 다룬다. 이번 one-step contact query에 대한 nested conditional SMC audit는 해당 algorithm의 재현이나 우월성 평가가 아니다. [Amortized Bayesian Experimental Design for Decision-Making](https://arxiv.org/html/2411.02064)은 downstream decision utility에 맞춘 design을 이미 다룬다.

[Task-Specific Next-Best-View with Uncertain Geometry, SIGGRAPH 2026](https://arxiv.org/html/2605.05095)은 geometry posterior와 downstream utility를 결합하고 heat diffusion task도 평가한다. 따라서 “uncertain geometry를 유지해 physics에 필요한 view를 고른다”는 설명만으로는 차별화되지 않는다. [Action-BED의 2026-11-06 v2](https://arxiv.org/html/2606.23662v2)는 expected future loss를 이용해 design/action policies를 공동 최적화하며, Appendix G.1.3에서 design과 decoder의 cross-evaluation까지 다룬다. Selection/decoding의 분리 평가 자체도 독점적인 novelty가 아니다.

남을 수 있는 기여는 **contact-induced geometry/material ambiguity에서 posterior approximation의 numerical error가 언제 consequential decision error로 바뀌는가**를 검증 가능하게 설명하는 것이다. 이 경우 reference approximation, held-out worlds, repeated measurement noise, strong cheap baseline, cost, measurement mismatch를 함께 보고해야 한다. ESS나 rank agreement 하나만 제시하는 것보다 이 failure taxonomy가 유용하다. 다만 지금은 2D, 5 parameters, 3 predefined families, known scene family와 calibrated feature model이라는 한계가 크다.

### 다음 진행 판단

아래 판단은 이번 실험 결과에 대한 해석이며, 새로운 experiment를 실행한 결과와 구분한다.


이번 12 predefined comparisons에서 multiple-comparison interval로 확인된 bridge의 추가 이득은 없다. 현재 evidence로 “tempered bridge를 쓰면 더 좋은 observation을 골라 loss를 줄인다”를 main claim으로 삼지 않겠다. 이것은 모든 환경에서 효과가 없다는 결론이 아니라, 이번 budget·world distribution·sensor/cost conditions에서 그 주장을 뒷받침하지 못했다는 뜻이다.


내가 이어간다면 먼저 corner/precise 조건을 새로운 worlds에서 재검증하고, **cross-fitted SNIS utility와 matched-compute bank scaling**을 추가하겠다. 이는 같은 caches를 더 적절하게 사용하는 것만으로 gap이 사라지는지 확인하는 관문이다. 이 검증 없이 새로운 bridge method를 중심 주장으로 삼지 않겠다. 그 이후에 **decision-relevant numerical refinement**를 제한된 후속 후보로 둔다. 먼저 cached factual-bank SNIS로 계산하고, cheap diagnostics가 potential action-value error를 시사하며 그 오류가 action gap을 넘을 때만 추가 conditional inference를 할당하는 방향이다. Generic adaptive SMC나 loss-aware inference와의 구분이 필요하고, 지금 만든 구현은 이 selective allocator를 포함하지 않는다.

계속할 조건은 새로운 held-out scenes에서 SNIS bank scaling, standard SMC, 더 긴 MH와 비교해 **같은 end-to-end simulator-call/latency budget에서 더 낮은 decision loss 또는 같은 loss의 더 낮은 비용**을 얻는 것이다. Low ESS만으로 refine하면 irrelevant action에도 계산을 낭비할 수 있다. 이 관문을 통과하지 못하면 bridge를 연구의 중심에서 내려놓고, 실제 visual collision geometry와 model discrepancy에 따른 counterfactual reliability로 문제를 옮기는 편이 합리적이다. 지금 toy를 더 어렵게 조정해 baseline을 실패시키는 방식은 권하지 않는다.

실제 물리 환경의 추가 measurement가 없으므로 main-conference method paper의 실증 근거가 충분하다고 판단하지 않는다. 이번 결과의 가장 직접적인 가치는 posterior improvement, rank change, practical regret, actual decision loss를 서로 다른 주장으로 분리하는 재현 가능한 audit다.

## 8. 재현 artifacts

- Server: `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/decision_loss_v7`
- [사전 protocol](PROTOCOL.md), [실행 안내](README.md), [실험 코드](experiment.py), [selection freeze](decision.py)
- [통계 summary](results/analysis/summary.json), [integrity](results/analysis/integrity.json), [query outcomes](results/analysis/query_outcomes.csv), [acquisition classes](results/analysis/acquisition_classes.csv)
- [분석 코드](analyze.py), [figures 생성 코드](plot.py), [GPU resource log](results/resource_state.json), [종료 GPU 상태](results/gpu_final.json), [SHA-256 manifest](server_run_manifest.json)

이전 v6 report의 결과를 덮어쓰지 않았다. 모든 수치 표는 이번 saved artifacts에서 생성했다.
