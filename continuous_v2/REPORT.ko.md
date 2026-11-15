# Continuous geometry와 contact boundary ambiguity: 원격 실행 보고서 v2

작성일: 2026-11-04, Asia/Seoul. 첫 four-world physics pilot을 continuous latent parameter와 grazing contact로 확장했다. 본문의 primary 결과는 **physics timestep 1/3840초**로 다시 생성한 data에 기반한다. 먼저 실행한 1/960초 결과는 numerical sensitivity 비교용으로 보존했다.

**같은 query type 안에서도 view·probe·stop의 최선 선택이 달라지는 현상을 확인했다. Plug-in geometry의 실패도 남았다. 그러나 561-point grid, 256-point quadrature, 512-particle baseline이 dense joint와 매우 가까운 성능을 보였으므로, 이 결과로 복잡한 새 belief algorithm의 필요성을 주장할 수는 없다.**

## 1. 이번 실험에서 실제로 바꾼 것

이전 실험은 네 hidden world와 세 query template을 사용했다. 이번에는 wall offset g와 restitution e를 continuous distribution에서 sampling하고, 모든 query를 다음 하나의 type으로 통일했다.

```text
translate_wall_y(delta): wall의 y 위치를 g에서 g + delta로 이동한다.
```

공·wall·camera·factual 상태를 고정한 채 numeric delta만 바꾼다. Delta는 family마다 12개 interval에서 연속값으로 sampling했다. Predictor가 받는 query는 type label과 numeric delta이며, head-on/boundary/miss 같은 정답 regime label을 주지 않는다.

| 항목 | 구현 |
|---|---|
| Continuous geometry | g ~ Truncated Normal(μg, σg²), support μg ± 4σg |
| Geometry prior | μg ∈ [−25, 25] mm, σg ∈ [3, 12] mm, family마다 sampling |
| Continuous restitution | e ~ Truncated Normal(μe, σe²), support [0.1, 0.9] |
| Material prior | μe ∈ [0.3, 0.7], σe ∈ [0.08, 0.23] |
| Query coverage | intervention 뒤 prior mean offset μg + delta가 −90…90 mm를 지나도록 stratified continuous sampling |
| 공 / wall | radius 20 mm, mass 1 kg / 고정 box 30 × 60 × 80 mm |
| Counterfactual initial state | ball x=−0.15 m, wall x=0.20 m, incoming speed=0.6 m/s |
| Physics | PyBullet, zero gravity/friction/damping, planar rigid-body motion |
| Primary timestep / horizon | 1/3840초 / 1.2초 |
| Nominal contact boundary | abs(g + delta) ≈ 50 mm |
| Contact event의 ground truth | rollout 중 normal force > 1e−7 N인 contact가 한 번 이상 발생 |
| Observation noise | family마다 view·probe 각각 σ ∈ [0.5, 2] px |

Camera, object identity, primitive dimensions, prior hyperparameter는 알려 준다. Continuous truth는 inference grid와 독립적으로 뽑았다. 정답 parameter가 candidate set의 원소라고 가정하지 않는다. 다만 정확한 model family와 prior를 알려 주는 통제 실험이며, image만으로 unknown scene representation을 학습한 것은 아니다.

Prediction target은 outgoing velocity와 contact event다. Full counterfactual video generation은 포함하지 않는다. MSE에는 posterior mean velocity를 사용하며, 이것이 하나의 물리적으로 가능한 video sample이라는 뜻은 아니다.

또한 prior와 현재 observation likelihood는 geometry와 restitution에 대해 factorize한다. **여기서 joint belief는 두 uncertainty를 함께 보존해 outcome으로 propagation하는 것을 뜻한다. Correlated geometry–physics posterior의 학습이나 복원을 입증한 실험은 아니다.**

## 2. Contact boundary에서 생기는 ambiguity

Wall의 앞면 중앙과 부딪히면 outgoing x velocity는 주로 restitution에 의해 달라진다. Wall의 모서리를 스치면 geometry가 contact normal을 바꾸고, 같은 restitution이어도 outgoing x·y velocity가 달라진다. Boundary를 넘으면 collision 자체가 사라진다.

따라서 이번에는 hit/miss uncertainty뿐 아니라 **contact normal uncertainty**도 있다. Geometry posterior가 unimodal이어도 nonlinear contact response 때문에 outcome distribution은 한 개의 fitted world로 충분히 표현되지 않을 수 있다.

![Contact response](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/continuous_v2/precision3840/results/analysis/figures/contact_response.png)

Boundary subset은 true world를 보고 고르지 않고, 알려진 prior/query context에서 `abs(abs(μg + delta) − 0.05) < 0.015 m`로 정의했다. Test의 245개 family–query context가 해당한다. 모든 regional metric은 family 안에서 먼저 평균한 뒤 64개 family를 평균한다.

## 3. Observation과 data

RGB는 상위 pilot의 exact orthographic primitive renderer로 만들었다. 여기서는 pre-contact frontal frame이 hidden world마다 같은지 검사했다. RGB color centroid를 추출한 뒤 Gaussian detector noise를 추가한다. Learned detector나 photorealistic renderer를 사용하지 않았다.

| Action | 실제 measurement | 비용 |
|---|---|---:|
| view | 원래 scene의 top-view RGB에서 wall vertical centroid | 1 |
| probe | 같은 contact pair를 별도 reset한 head-on collision의 두 RGB frame에서 ball displacement | 1 |
| repeat | 원래 frontal centroid를 독립 noise로 재측정. 이 조건에서는 latent information 없음 | 1 |
| stop | 현재 prediction 제출 | 0 |

Probe는 wall x=0.08 m, ball x=−0.08 m, speed=0.4 m/s의 별도 calibration experiment다. Target counterfactual을 대신 실행해 보여 주지 않는다. 두 frame의 centroid noise가 독립이므로 displacement noise는 √2σ다.

128개 family를 development 32, validation 32, held-out test 64로 고정했다. Family마다 32개 continuous world와 12개 query가 있으므로 **direct target rollout은 49,152개**다. 각 world–query에 detector-noise realization 두 개를 사용한다. Test에서는 24,576개 world–query episode와 49,152개 noisy realization을 평가하지만, CI의 독립 단위는 **64개 scene family**다.

모든 방법은 동일한 world, numeric query, action별·visit별 noise draw를 공유한다. Policy에는 현재 belief와 query, candidate likelihood만 들어간다. True parameter, target ground truth, 선택하지 않은 action의 realized measurement를 넘기지 않는다.

CI는 scene-family 단위 paired bootstrap 10,000회로 계산했다. 한 family의 world·query·noise realization은 함께 resample한다. 여러 contrast에 대한 multiplicity adjustment는 하지 않았으므로 exploratory CI로 읽어야 한다.

Truth outcome은 직접 PyBullet rollout으로 생성했다. Predictor는 별도로 계산한 continuous physics lookup을 interpolation해 사용한다. 이렇게 truth가 inference grid에 맞춰지는 문제를 피했지만, lookup approximation error가 생긴다. 아래 numerical validation에서 그 크기를 공개한다.

## 4. 비교한 baseline과 objective

Primary comparison은 같은 EVSI와 같은 cost penalty를 사용한다.

```text
z = outgoing vx / incoming vx
risk = Var[z | observation history, query]
gain(a) = risk − expected posterior risk after action a
action = argmax gain(a), if max gain > 0.005; otherwise stop
```

Observation expectation은 257-point integration으로 계산한다. Budget은 최대 0·1·2이며, budget 2도 greedy myopic policy다. 최적의 non-myopic policy라는 의미의 “optimal”로 사용하지 않는다. 본문의 Bayes design reference는 candidate forward/observation model 아래의 approximate one-step reference다.

| Method | 의미 |
|---|---|
| Joint grid 129×33 | geometry와 restitution uncertainty를 유지하는 기본 numerical reference, 4,257 hypotheses |
| Joint grid 257×65 | 16,705 hypotheses의 추가 convergence reference |
| Joint grid 33×17 | 561 hypotheses의 작은 grid baseline |
| Prior quadrature 16×16 | prior CDF 공간의 Gauss–Legendre quadrature, 256 hypotheses |
| Importance particles 512 | prior에서 독립 sampling한 512 particles를 likelihood로 reweight. Resampling 없는 importance sampling이며 SMC 공식 재현이 아님 |
| Plug-in refit + EVSI | 관찰 이력 전체로 geometry marginal MAP을 재계산하되 planning·prediction에서는 한 geometry로 collapse |
| Plug-in + exploration | 첫 action에서 25% 확률로 view를 강제하고, 나머지는 plug-in EVSI |
| Variance rule | total variance를 geometry와 material 기여로 나누어 큰 쪽의 measurement를 선택. Sensor SNR에 따른 실제 expected reduction을 계산하지 않는 단순 baseline |
| Query-type rule | numeric delta나 현재 observation 값에 적응하지 않는 고정 action sequence를 validation에서 선택 |
| Parameter EIG / random | query-irrelevant information 획득 및 무작위 선택의 control |

Plug-in refit은 4,257-point model을 이용해 geometry MAP을 찾으므로 poor local optimization 때문에 실패하는 baseline으로 만들지 않았다. 새 view가 주어지면 geometry를 다시 바꿀 수 있다. Strong baseline들은 같은 shared physics lookup과 measurement model을 사용하며, candidate 수는 다르게 두고 명시했다. 같은 wall-clock compute budget을 맞춘 비교라고 주장하지 않는다.

Query-type rule의 budget 1 후보는 view, probe, stop이다. Budget 2에서는 vv, vp, pv, pp, vs, ps, ss의 일곱 sequence 중 validation의 `MSE + 0.005×cost`가 가장 작은 것을 고른다. 최종 선택은 **budget 1: static_vv, budget 2: static_vv**였다. Test 결과로 이 선택을 바꾸지 않았다.

## 5. 같은 query type에서 observation이 실제로 달라졌는가

기본 joint reference의 test 768개 family–query context에서 첫 선택은 **view 286개, probe 113개, stop 369개**였다. Repeat를 선택한 경우는 0개다.

64개 test family 모두 같은 family 안에서 두 종류 이상의 action이 나타났고, **30개 family에서는 view·probe·stop 세 가지가 모두 나타났다.** 따라서 query type label만으로 reference action을 일관되게 결정할 수 없다.

![Query value와 observation 선택](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/continuous_v2/precision3840/results/analysis/figures/action_map.png)

Factual context가 같아도 collision 가능성이 높은 head-on 영역에서는 restitution probe가 유용할 수 있고, grazing contact 영역에서는 geometry view가 유용하다. Contact가 일어날 가능성이 작거나 남은 prediction risk가 observation 비용보다 작으면 stop이 유리하다. Head-on이라고 항상 probe를 고르는 것도 아니다. Material prior와 sensor noise에 따라 기대 이득이 작으면 stop한다.

![같은 family의 numeric query 비교](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/continuous_v2/precision3840/results/analysis/figures/same_family_queries.png)

이 그림은 “세 action이 모두 나타난 첫 test family”라는 고정된 규칙으로 선택했다. 가장 큰 성능 차이를 보이는 사례를 사후에 고른 것이 아니다. 최선 action은 실제 noisy observation 결과를 미리 보는 clairvoyant oracle이 아니라, 관찰 전 expected utility를 기준으로 정했다.

## 6. Held-out test: prediction과 observation cost

아래 MSE는 normalized outgoing vx에 대한 dimensionless squared error다. B는 허용되는 최대 budget이다. Actual cost가 다른 방법들을 같은 관찰 횟수로 실행했다고 표현하지 않는다.

**Budget 1**

| Method | Hypotheses | MSE | Actual cost | MSE + 0.005×cost | Brier |
|---|---:|---:|---:|---:|---:|
| Joint grid 129×33 | 4,257 | 0.042138 | 0.520 | 0.044735 | 0.034226 |
| Joint grid 257×65 | 16,705 | 0.042126 | 0.520 | 0.044723 | 0.034210 |
| Joint grid 33×17 | 561 | 0.042121 | 0.522 | 0.044731 | 0.034505 |
| Prior quadrature 16×16 | 256 | 0.042199 | 0.518 | 0.044790 | 0.034363 |
| Importance particles 512 | 512 | 0.042100 | 0.529 | 0.044743 | 0.034231 |
| Variance rule + joint prediction | 4,257 | 0.041668 | 0.719 | 0.045262 | 0.033821 |
| Validation-selected query-type rule | 4,257 | 0.043177 | 1.000 | 0.048177 | 0.033795 |
| Plug-in refit + EVSI | 4,257 | 0.074177 | 0.201 | 0.075179 | 0.066203 |
| Plug-in + 25% view exploration | 4,257 | 0.068267 | 0.404 | 0.070287 | 0.062764 |
| Parameter EIG | 4,257 | 0.044393 | 1.000 | 0.049393 | 0.034803 |
| Random acquisition | 4,257 | 0.059450 | 1.000 | 0.064450 | 0.042728 |

Joint의 MSE는 plug-in보다 43.19% 낮았다. Paired difference `plug-in − joint`는 **0.032039 [95% CI 0.025410, 0.038773]**다. 같은 objective를 사용해도 geometry collapse 때문에 필요한 view를 평가하지 못하는 실패가 continuous case에서도 남았다.

Query-type rule과 비교한 MSE difference는 **0.001039 [95% CI 0.000599, 0.001492]**다. 이 차이는 plug-in과의 차이보다 작다. 하지만 query-type rule이 평균 cost 1을 쓰는 동안 joint는 **0.520**만 사용했다. Penalized loss difference는 **0.003441 [95% CI 0.003047, 0.003833]**다. 이 실험에서는 prediction 개선과 불필요한 observation 감소를 함께 보아야 한다.

**Budget 2**

| Method | MSE | Actual cost | MSE + 0.005×cost |
|---|---:|---:|---:|
| Joint grid 129×33 | 0.032646 | 0.832 | 0.036805 |
| Joint grid 257×65 | 0.032627 | 0.831 | 0.036783 |
| Joint grid 33×17 | 0.032642 | 0.838 | 0.036830 |
| Importance particles 512 | 0.032642 | 0.838 | 0.036832 |
| Variance rule + joint prediction | 0.031296 | 1.370 | 0.038147 |
| Validation-selected query-type rule | 0.033762 | 2.000 | 0.043762 |
| Plug-in refit + EVSI | 0.073969 | 0.249 | 0.075214 |

Query-type rule 대비 budget 2의 penalized loss difference는 **0.006957 [95% CI 0.006358, 0.007535]**다. Budget 2에서의 결론은 greedy policy들 사이의 비교이며, 두 step 전체를 jointly optimize한 정책과의 비교가 아니다.

![Baseline 비교](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/continuous_v2/precision3840/results/analysis/figures/baseline_comparison.png)

## 7. Strong baseline이 알려 주는 한계

561-point grid의 budget 1 MSE difference `small grid − dense joint`는 **-0.000017 [95% CI -0.000093, 0.000054]**다. 512-particle baseline의 차이는 **-0.000038 [95% CI -0.000262, 0.000192]**다. 이 CI와 effect size를 기준으로, 이번 pilot에서 dense representation의 분명한 실용적 이점을 확인하지 못했다. Formal equivalence test를 한 것은 아니다.

256-point quadrature의 MSE도 0.042199로 dense reference와 가깝다. 이 benchmark는 아직 두 개의 unknown scalar와 알려진 prior/likelihood를 사용한다. Generic approximation만으로 충분한 조건이므로, learned proposal이나 복잡한 scene-belief model이 필요하다는 주장은 현 단계에서 성립하지 않는다.

Variance rule은 budget 1에서 MSE 0.041668, actual cost 0.719를 기록했다. Dense EVSI보다 raw MSE가 낮아질 수 있지만 더 많은 관찰을 사용한다. `variance rule − joint`의 raw MSE difference는 **-0.000469 [95% CI -0.000678, -0.000278]**, penalized loss difference는 **0.000527 [95% CI 0.000324, 0.000724]**다. 어느 metric에서도 EVSI가 항상 최고라고 주장하지 않는다.

![Actual cost와 risk](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/continuous_v2/precision3840/results/analysis/figures/cost_risk.png)

Design regret은 chosen first action의 expected utility와 dense reference의 best expected utility 차이다. Budget 1 평균은 query-type rule 0.003414, plug-in 0.023632, small grid 0.000001, particles 0.000042다. 이 값은 test truth를 사후 선택해 얻은 oracle error가 아니다. `fine_design_regret`도 raw JSON에 저장해 reference grid 자체의 근사 영향을 확인할 수 있게 했다.

## 8. Same-information control: 이번에는 refitting만으로 완전히 회복되지 않았다

두 방법에 동일한 view를 주고, plug-in에 geometry refitting을 허용했다. Budget 1 MSE는 joint **0.043177**, plug-in **0.048138**였다. Paired difference는 **0.004961 [95% CI 0.003899, 0.006056]**다. Boundary subset의 차이는 **0.011631 [95% CI 0.009302, 0.014022]**다.

Budget 2에서 동일한 view→probe를 준 뒤의 MSE는 joint **0.040673**, plug-in **0.045618**이고, 차이는 **0.004945 [95% CI 0.003875, 0.006056]**였다. 이 비교는 선택 action, 실제 measurement와 cost를 모두 맞췄으므로 prediction 시 geometry uncertainty를 버리는 효과를 더 직접적으로 보여 준다.

이전 four-world pilot에서는 top view가 두 geometry mode를 거의 완벽히 구별해 refitting만으로 격차가 대부분 사라졌다. 이번에는 measurement noise가 있는 continuous geometry여서 view 뒤에도 수 mm 수준의 uncertainty가 남는다. 그 uncertainty가 contact normal과 contact 여부를 바꾸므로 하나의 geometry MAP만으로 prediction하는 손실이 남는다.

이 해석을 넘어서 joint belief가 모든 continuous scene에서 필수라고 말할 수는 없다. 관찰이 충분히 정확하거나 contact-sensitive region에서 멀면 point approximation도 충분할 수 있다.

## 9. Boundary uncertainty와 calibration

Budget 1 결과다.

| Method / condition | MSE | Posterior variance | Brier | 90% coverage | Interval width |
|---|---:|---:|---:|---:|---:|
| Joint / all | 0.042138 | 0.040538 | 0.034226 | 93.73% | 0.436272 |
| Plug-in / all | 0.074177 | 0.007374 | 0.066203 | 76.77% | 0.200524 |
| Joint / boundary | 0.092475 | 0.089451 | 0.099795 | 94.10% | 0.748755 |
| Plug-in / boundary | 0.176490 | 0.003266 | 0.201408 | 47.39% | 0.123412 |
| Joint / same view / boundary | 0.092294 | 0.089164 | 0.098922 | 94.21% | 0.745151 |
| Plug-in / same view / boundary | 0.103925 | 0.004099 | 0.144714 | 55.71% | 0.133079 |

Plug-in은 boundary에서 작은 posterior variance와 짧은 interval을 보고하지만 실제 error와 coverage는 그 확신을 뒷받침하지 못한다. View를 강제로 제공한 뒤에도 residual geometry uncertainty를 없애 버리면 uncertainty를 과소평가한다.

Joint의 interval과 proper score는 개선됐지만 nominal 90% coverage와 empirical coverage가 정확히 일치하지는 않는다. Outcome에 no-contact point mass가 있고, finite grid·finite sample·lookup approximation도 존재한다. Brier와 CRPS, interval width를 함께 보고 calibration을 판단해야 한다. 이번 reference는 vx MSE를 optimize하므로 contact-event Brier를 직접 optimize한 최선 policy라는 의미도 아니다.

## 10. Numerical validation과 physics timestep 수정

개발 단계에서 primary physics lookup과 direct rollout의 normalized velocity component RMSE는 **0.000639**, 99th-percentile absolute error는 **0.003432**, 최대는 **0.022852**였다. Contact disagreement는 **0.000%**였다.

View/probe centroid lookup의 RMSE는 각각 **0.0189 px / 0.0553 px**, 최대 오차는 **0.2438 px / 0.2828 px**였다. Noise가 0.5–2 px인 상황에 비해 작지만 0은 아니다. 따라서 완벽한 likelihood match라고 표현하지 않는다.

257-point와 513-point observation integration의 최대 gain 차이는 8.19e-14였다. Test에서 129×33과 257×65 grid의 첫 action agreement는 **100.000%**였다. 최대 gain 차이는 0.002602였다. Finer grid의 budget 1 MSE difference는 **-0.000012 [95% CI -0.000021, -0.000004]**다. 작은 수치 차이는 남으므로 reference를 정확한 continuous Bayes optimum이라고 부르지 않는다.

처음 실행한 timestep 1/960초의 grazing-contact curve에는 작은 계단 모양이 보였다. 363개 별도 contact case에서 timestep을 줄여 확인한 결과는 다음과 같다.

| Timestep 비교 | Normalized velocity component RMSE | 최대 absolute error | Contact disagreement |
|---|---:|---:|---:|
| 1/960 → 1/3840 s | 0.007394 | 0.031040 | 0.00% |
| 1/3840 → 1/7680 s | 0.001335 | 0.005952 | 0.00% |
| 1/960 → 1/7680 s | 0.008220 | 0.036191 | 0.00% |

이에 따라 **1/3840초로 direct truth, physics/sensor lookup, main suite, finer-grid reference와 analysis를 모두 다시 실행했다.** 본문의 primary table은 이 재실행 결과다. 1/7680초는 위 363개 case의 diagnostic에만 사용했으며 전체 benchmark를 그 timestep으로 실행했다고 주장하지 않는다. Finite-timestep physics error가 완전히 제거된 것도 아니다.

모든 development truth가 inference grid 밖에 있는지, 모든 family의 frontal RGB가 hidden world 간 동일한지, repeat의 information gain이 없는지 확인했다. 1/960초와 1/3840초 결과를 덮어쓰지 않고 각각 보존했으며 source hash가 각 runtime manifest와 일치하는지도 검증했다.

## 11. 서버 실행과 재현

Remote project:

```text
/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/continuous_v2
```

Primary artifacts:

```text
continuous_v2/precision3840/data
continuous_v2/precision3840/results/main
continuous_v2/precision3840/results/refined
continuous_v2/precision3840/results/analysis
```

`datacenter3` → `yonsei-ai-login3`를 통해 기존 사용자 allocation 네 개를 사용했다. 실행 전 해당 GPU에 compute process가 없음을 확인했다. Physics는 compute node의 CPU에서 네 shard로 생성했고, posterior·acquisition은 각 allocation의 CUDA device에서 실행했다. 기존 job을 취소하거나 allocation을 변경하지 않았다.

| Shard | Host | Slurm job | GPU UUID prefix | Main runtime | Peak allocated |
|---|---|---|---|---:|---:|
| 0 | node55 | 2642474 | 58f3772e-d982… | 11.54 s | 318.9 MiB |
| 1 | node56 | 2642475 | c3c4b83c-668c… | 10.06 s | 318.9 MiB |
| 2 | node56 | 2642476 | f211384e-b912… | 10.63 s | 318.9 MiB |
| 3 | node55 | 2642473 | e5775416-8193… | 10.22 s | 318.9 MiB |

표의 runtime은 cached physics lookup을 사용하는 main evaluation loop와 metric 저장 시간이며 physics 생성·environment 설치·network 지연을 포함하지 않는다. Peak allocated는 PyTorch tensor allocation이며 driver가 보여 주는 전체 GPU memory와 같지 않다. Neural model training은 하지 않았다. GPU는 float64 Bayes update와 acquisition integration에 사용했다.

Python 3.9.18, PyBullet 3.2.7, PyTorch 2.7.1+cu128를 사용했다. CUDA compatibility와 timing 방식은 [PyTorch 공식 version 안내](https://pytorch.org/get-started/previous-versions/)와 [CUDA semantics](https://docs.pytorch.org/docs/main/notes/cuda.html)를 확인했으며, method timing 전후에 CUDA synchronization을 수행했다.

## 12. 연구 방향에 대한 판단

이번 확장으로 “query type만 알면 action이 정해지는 toy”의 한계는 줄였다. Numeric intervention과 geometry uncertainty에 따라 유용한 observation이 바뀌고, boundary에서는 관찰 뒤의 residual uncertainty도 prediction에 영향을 준다는 evidence를 얻었다.

그러나 **현재 evidence는 dense joint model이나 새로운 learned acquisition algorithm을 제안해야 할 이유를 충분히 제공하지 않는다.** 작은 generic grid와 importance particles가 이미 강하기 때문이다. 이 결과를 숨기고 plug-in과의 큰 격차만 강조하면 method contribution을 과장하게 된다.

다음 질문은 “joint가 point estimate보다 좋은가?”에서 **“어떤 uncertainty를 유지해야 observation decision과 contact-sensitive prediction을 보존할 수 있는가?”**로 좁히는 것이 타당하다. 다음 단계의 우선순위는 아래와 같다.

1. **Correlated posterior:** factual grazing motion을 넣어 geometry와 restitution이 서로 보상하는 ambiguity를 만든다. 지금의 factorized posterior보다 어려운 실제 식별 문제를 만든다.
2. **Unknown contact structure:** wall orientation·extent, occluded geometry 또는 여러 object를 추가해 두 scalar만의 grid로 풀리지 않는 조건을 만든다. 그때도 generic SMC/mixture와 compute budget을 맞춘다.
3. **Decision-preserving compression:** 256/512 hypotheses가 이미 충분한 현재 조건을 출발점으로, 어떤 scene에서 작은 approximation이 action ranking을 바꾸는지 측정한다. 새 방법은 실제 acquisition regret을 줄이는 경우에만 기여가 된다.
4. **Observation realism:** RGB detector, occlusion, camera error, probe reset uncertainty를 실제로 측정하고 likelihood mismatch를 별도 축으로 평가한다. 지금의 known prior와 calibrated primitive renderer를 real perception 성능으로 확대 해석하지 않는다.

현재 단계에서 가능한 논문 기여는 continuous contact ambiguity를 분리해 평가하는 protocol과 failure diagnosis다. 새로운 algorithm의 기여를 주장하려면 이번에 강했던 small grid, quadrature, importance-particle baseline이 어려워지는 조건과 그때의 개선을 추가로 보여야 한다.

## 13. 산출물

- [재현 설명](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/continuous_v2/README.md)
- [Physics data 생성](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/continuous_v2/prepare.py), [higher-precision wrapper](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/continuous_v2/precision3840.py)
- [Belief와 acquisition 구현](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/continuous_v2/belief.py), [baseline 실행](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/continuous_v2/run.py)
- [전체 primary test 결과표](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/continuous_v2/precision3840/results/analysis/tables.md)
- [Primary summary JSON](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/continuous_v2/precision3840/results/analysis/summary.json)
- [Paired contrast와 CI](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/continuous_v2/precision3840/results/analysis/paired_contrasts.json)
- [Same-information contrast](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/continuous_v2/precision3840/results/analysis/equal_information.json)
- [Validation-only query-type rule 선택](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/continuous_v2/precision3840/results/analysis/query_type_rule_selection.json)
- [Numerical validation](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/continuous_v2/precision3840/results/development_validation.json)
- [Timestep audit](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/continuous_v2/results/timestep_audit/audit.json)
- [실제 dependency lock](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/continuous_v2/requirements.lock.txt)

같은 보고서를 remote project의 `REPORT.ko.md`에도 저장한다. Main raw metrics, test prediction detail, GPU별 runtime·source hash, direct physics data와 log를 함께 남겼다.
