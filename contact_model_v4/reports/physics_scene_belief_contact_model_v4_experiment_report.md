# Contact-Model Ambiguity: 서버 실험과 연구 진행 판단

실행·검토일: 2026-11-05. 기존 correlated v3의 §5–6에서 제안한 A/B/C 방향을 비교했다. 논문별 연구 흐름과 가상 지도 feedback은 별도 research review, 실제 원문 확인 범위는 reading ledger에 정리했다.

## 1. 이번 결과가 말하는 것

**현재 연구를 계속할 이유는 있지만, 새로운 inference algorithm이나 calibrated real-world physics method를 얻었다고 말할 단계는 아니다.** 세 scene layout에서 query-dependent observation selection을 구현했고, 작은 particle bank의 reweighting을 full SMC refit으로 대체한 뒤에도 fixed rule과 비교했다. 단순 branch compression과 threshold-triggered expansion은 strong baseline을 대체할 근거가 약했다.

Primary comparison에서 task EVSI의 MSE는 0.012678, validation으로 정한 query-value rule은 0.017502였다. 평균 차이의 상대 크기는 27.6%이며, world 단위 paired difference는 -0.004823, 95% bootstrap CI [-0.008309, -0.001573]다. 이 수치는 지정된 synthetic sensor/action pool에서의 결과이며 실제 camera·robot에서의 성능을 뜻하지 않는다.

가장 중요한 수정은 ‘틀린 model을 판별하는 새 방법’이라는 주장을 늦추는 것이다. Sharp/rounded proxy는 여기서는 동일한 impact law 아래의 collision geometry 차이다. Continuous rounding SMC까지 추가해, discrete support가 부족해서 생긴 실패를 별도로 확인했다. 이 baseline은 초기 결과를 본 뒤 추가한 **post-hoc robustness audit**이며 사전등록된 실험으로 제시하지 않는다.

## 2. 실제 실행한 범위

| 항목 | 실행 내용 |
|---|---|
| Compute | datacenter3 경유 node56, Slurm 2642475/2642476, RTX PRO 6000 Blackwell 두 GPU |
| Main worlds | 72개: corner / gate / deflection 각 24개 |
| Split | 각 family의 첫 8개 validation, 나머지 16개 test; 전체 24 / 48 |
| Query | 같은 source-offset intervention type, world당 6개 offsets; test 288 queries |
| Initial inference | adaptive-tempering SMC-768, stage당 6회 MH rejuvenation moves |
| Observation 후 검증 | 72 worlds × 4 actions의 full SMC-1536 refit, 12 moves |
| Compression | 64 / 128 support, systematic / branch-stratified, 3 resampling seeds |
| Model checking | 독립 sharp-null calibration 192 worlds, family당 64 |
| Strong proxy audit | 72 worlds의 continuous-rounding SMC-1536 및 privileged known-proxy SMC |
| Numerical checks | analytic check, 18,432 random query rollouts, PyBullet 72 rollouts |
| Posterior convergence | 미리 지정한 validation 9 worlds × 2 particle counts × 2 seeds |
| Public real data | 한 cloth, 3 interactions × 2 cameras; 2,214 frames decoding |

Neural model training은 하지 않았다. GPU에서 vectorized simulation과 posterior inference를 실행했다. 사용 중이던 다른 allocation의 job을 변경하지 않았다. 두 GPU만으로 이번 검증을 수행했으며 세 번째 GPU가 비기를 기다려 추가 작업을 남겨 두지 않았다.

## 3. Physics와 observation의 정확한 정의

Sphere radius는 0.02 m, mass는 1, initial speed는 1 m/s이며 gravity는 없다. Solid-sphere inertia와 instantaneous Coulomb tangential impulse를 적용하고 spin을 내부 상태로 유지한다. Static rounded rectangles의 flat faces와 circular corners에 대해 continuous collision detection을 계산한다. 최대 32 contact events를 허용하며 overflow를 확인한다.

Continuous latent parameter는 geometry translation g ∈ [−8,8] mm, restitution e ∈ [0.15,0.85], friction μ ∈ [0,0.6], shape parameter d ∈ [22,34] mm다. Broad model의 rounding b는 {0,1,3} mm, truth regimes는 0 / 3 / 5 mm다. 따라서 `outside`는 **원래 discrete model support 밖의 5 mm truth**다. 추가 continuous baseline은 b ∈ [0,6] mm도 추정하므로 이 truth를 포함한다. Known-proxy baseline은 true b를 제공받는 privileged diagnostic이다.

Corner는 한 obstacle의 grazing impact, gate는 양쪽 obstacle 사이의 통과·충돌·반복 충돌, deflection은 두 obstacle을 거치는 sequence를 만든다. **동일한 rigid law의 세 layout**이므로 ‘서로 다른 세 물리계’를 검증한 것은 아니다. Pose uncertainty도 translation으로 제한했고 rotation은 포함하지 않았다. Occluded support, articulated contact, deformable contact를 구현하지 않았다.

Target query는 initial source y-offset을 −30, −17, −7, 4, 15, 28 mm로 바꾼 후 0.8 s 시점의 vx/vy다. Geometry/material은 factual과 공유한다. Observation action은 top view와 source offset −22 / 22 / 9 mm의 left / right / center probe다. Target query와 probe offset은 서로 다르다. 각 probe는 동일 latent world를 reset한 별도 interaction으로 정의한다.

Factual은 outgoing vx 하나만 제공한다. View는 [g,d], probe는 [vx,vy]를 측정한다. Localization jitter, rasterization, blur를 거친 RGB에서 color centroid와 bounding box를 추출했다. View noise는 좌표당 1.5 mm, dynamic likelihood의 velocity noise는 좌표당 0.025 m/s다. Dynamic image의 quantization step은 약 0.0222 m/s이므로 Gaussian likelihood는 sensor의 근사다.

**이 RGB setting의 중요한 제한:** known camera, known object color, known scene family를 사용한다. 또한 view feature가 [g,d]만 남기므로 raw silhouette의 corner curvature가 줄 수 있는 rounding 정보를 버린다. 실제 visual observer가 같은 action ranking을 보인다고 일반화할 수 없다. General perception 및 raw-image geometry uncertainty gate는 아직 통과하지 못했다.

## 4. Observation selection: strong posterior update 뒤에도 남는가?

Policy는 factual posterior와 query만 보고 action을 선택한다. 이후 공개되는 observation의 값이나 target truth를 selection에 사용하지 않는다. Offline evaluation을 위해 4개 observation outcomes를 모두 저장했지만 각 policy는 선택한 observation 하나만 받는다. 모든 policy의 추가 observation budget은 1이며 main comparison에 stop은 없다. View와 physical probe를 같은 cost로 세는 것은 이번 diagnostic의 가정이다.

EVSI는 squared prediction loss의 expected reduction을 posterior-mean 변화로 계산한다. Parameter EIG는 joint latent parameter/model에 관한 information gain을 계산한다. Initial acquisition은 SMC-768과 256 hypothetical observations를 사용했다. Full refit 단계에서는 선택을 바꾸지 않고 factual+selected observation으로 SMC-1536을 새로 돌렸다.

Query-type rule은 family별 fixed action이다. 더 강한 query-value rule은 family와 6개 query offsets 각각에 대해 validation MSE가 가장 낮은 action을 정한다. Refined 결과의 rule은 **validation의 refined estimator**로 선택했다. Test truth로 action을 고르지 않았다. Uniform random은 4개 actions의 expected score를 계산했다.

| Method | MSE ↓ | CRPS ↓ | Marginal 90% coverage | Mean interval width |
|---|---:|---:|---:|---:|
| task_EVSI | 0.012678 | 0.037562 | 88.19% | 0.1751 |
| query_type_rule | 0.028791 | 0.062520 | 88.54% | 0.2941 |
| query_value_rule | 0.017502 | 0.046458 | 86.46% | 0.2000 |
| parameter_EIG | 0.024378 | 0.050468 | 87.85% | 0.2388 |
| uniform_random | 0.024027 | 0.056882 | 88.85% | 0.2883 |

MSE는 vx/vy squared error의 평균이며 unit은 (m/s)²다. CRPS는 두 marginal의 평균이다. Coverage는 두 scalar central 90% intervals의 empirical coverage이며 joint 2D region coverage가 아니다.

| Paired comparison, MSE difference | Mean 및 95% bootstrap CI |
|---|---|
| EVSI − query-type rule | -0.016112, 95% bootstrap CI [-0.024848, -0.008375] |
| EVSI − query-value rule | -0.004823, 95% bootstrap CI [-0.008309, -0.001573] |
| EVSI − parameter EIG | -0.011699, 95% bootstrap CI [-0.020222, -0.004525] |
| EVSI − random | -0.011349, 95% bootstrap CI [-0.016131, -0.007235] |

Negative difference는 앞 method의 MSE가 낮다는 뜻이다. CI는 48 worlds를 bootstrap resampling해 계산했으며 같은 world의 6개 queries를 독립 표본으로 세지 않았다. 10,000 bootstrap replicates, fixed seed 512를 사용했다. 여러 comparison의 multiplicity correction은 하지 않은 exploratory 결과다.

CI는 이번 validation으로 정한 rule에 조건부인 test uncertainty다. Family당 validation worlds 8개로 rule을 고르는 과정의 불확실성을 다시 bootstrap한 것은 아니다. 더 큰 validation set과 새로운 scene seeds에서 replication이 필요하다.

| Family, test 16 worlds씩 | EVSI MSE | Query-value rule MSE | Parameter EIG MSE |
|---|---:|---:|---:|
| corner | 0.002295 | 0.002868 | 0.002371 |
| gate | 0.027612 | 0.034706 | 0.061401 |
| deflection | 0.008128 | 0.014930 | 0.009361 |

전체 argmax action count는 view/left/right/center 순으로 [7, 112, 59, 110]다. 거의 zero-gain query의 arbitrary tie를 제외하기 위해 max estimated EVSI > 1e−4인 query만 보았을 때 count는 [5, 76, 59, 110]다. 서로 다른 useful action을 고른 world는 43/48개다. 이 threshold는 diversity를 설명하기 위한 descriptive cutoff이며 새로운 성능 metric의 사전등록된 기준은 아니다.

Initial bank를 단순 reweighting했을 때 평균 ESS는 EVSI 138.5, EIG 28.0였다. Likelihood를 좁히는 observation을 선택한 EIG가 작은 bank에서 더 손해 볼 가능성이 있어 full refit을 추가했다. 따라서 primary 결과는 위 full refit table이다. Factual-only joint MAP와 no-observation 결과는 원본 summary에 보존했지만 observation budget이 다른 이들을 main policy baseline으로 비교하지 않는다.

EIG의 test-world action은 corner에서 center 16회, gate에서 right 9회/left 7회, deflection에서 left 12회/center 4회였다. EIG는 한 world에서 latent parameter 전체에 대한 information을 기준으로 action 하나를 선택한다. EVSI는 같은 world라도 query가 요구하는 response에 따라 action을 달리한다. 이 objective 차이는 관측된 결과의 해석이며, 이 작은 실험만으로 모든 EIG formulation보다 우월하다고 일반화하지 않는다.

![refined_comparison](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/contact_model_v4/results/analysis/refined_comparison.png)

![posterior_and_selection](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/contact_model_v4/results/analysis/posterior_and_selection.png)

위 posterior plot은 test에서 보기 좋은 사례를 고른 것이 아니라 validation case 1/25/49를 사용했다. Red star는 evaluation truth이며 policy에 전달하지 않는다. Heatmap의 color scale은 family별로 다르다. Action argmax의 변동과 실제 성능 개선은 별개의 증거로 해석한다.

## 5. Branch-preserving compression: 현재는 주력 method로 선택하지 않는다

같은 SMC-768 posterior에서 64 또는 128 atoms를 유지한다. Systematic resampling과, public query pool의 contact sequence signature로 stratify하는 단순 method를 비교했다. Branch마다 최소 한 atom을 유지하고 남는 budget을 mass에 비례해 배분한다. Branch 수가 budget을 넘으면 systematic으로 fallback한다. 세 resampling seeds를 world 내부에서 평균했다.

| Method | MSE ↓ | CRPS ↓ | Marginal 90% coverage | Mean interval width |
|---|---:|---:|---:|---:|
| systematic_64 | 0.022766 | 0.059856 | 63.89% | 0.0861 |
| branch_64 | 0.021999 | 0.058364 | 64.06% | 0.0881 |
| systematic_128 | 0.020004 | 0.053784 | 72.28% | 0.1131 |
| branch_128 | 0.019360 | 0.053171 | 72.16% | 0.1122 |

- 64 support의 branch − systematic MSE: -0.000766, 95% bootstrap CI [-0.001578, +0.000006].
- 128 support의 branch − systematic MSE: -0.000644, 95% bootstrap CI [-0.001671, +0.000251].

두 CI 모두 zero를 포함한다. 작은 평균 이득만으로 contact-aware inference의 새 contribution을 주장하지 않겠다. Coverage도 full posterior보다 낮다. 한 branch 안에서도 continuous parameter가 크게 다르면 같은 contact sequence만 유지하는 것으로 observation likelihood와 target outcome을 보존하기 어렵다.

추가 branch audit에서 test world의 joint query-signature 수는 1–167개였다. 64 support에서는 15/48 worlds, 128 support에서는 14/48 worlds가 budget을 넘어 systematic으로 fallback했다. 따라서 이 결과를 ‘모든 branch-preserving method가 무효’라고 일반화할 수 없다. 원본 posterior에서 누락된 signature mass는 systematic/branch 각각 64에서 0.1257/0.1184, 128에서 0.0774/0.0723이었다. 현 구현은 여러 query의 signature를 한꺼번에 유지하려다 support budget이 부족해진다. Query별 grouping이나 decision-aware merging을 연구하려면 upstream 및 query별 추가 비용까지 포함해 다시 비교해야 한다.

이 비교는 **같은 upstream posterior와 retained support budget**에서의 compression test다. Upstream SMC, branch signature 계산, acquisition 비용을 무료로 간주한 end-to-end efficiency 비교가 아니다. MCMC, adaptive quadrature, learned surrogate까지 동일 total cost로 비교한 단계도 아니다. 따라서 후보 B는 더 큰 model로 확장하기보다 여기서 보류한다.

## 6. Predictive check와 model expansion

먼저 sharp model로 factual+view를 fit한다. 아직 보지 않은 center probe의 predictive mean/covariance와 measurement noise로 Mahalanobis residual score를 계산한다. 독립 sharp-null calibration 64 worlds/family의 62nd order statistic을 threshold로 사용했다. Corner/gate/deflection threshold는 각각 6.6082, 10.4532, 6.9257다.

Threshold는 test data로 정하지 않았다. Exchangeability 아래의 nominal null acceptance를 위한 calibration일 뿐, counterfactual coverage 또는 OOD detection을 보장하지 않는다. Probe는 center로 고정했으며 diagnostic probe selection 자체를 최적화한 실험은 아니다. Sequential multi-round closed loop도 아니다.

Probe 이후에는 모두 같은 factual+view+center를 사용해 SMC-768로 refit했다. Sharp는 b=0 유지, broad는 세 b 값을 포함, noise inflation은 sharp model에서 observation σ를 2배로 늘린 baseline이다. Adaptive는 score가 threshold를 넘으면 broad, 아니면 sharp refit을 선택한다. 별도로 구현된 refit 중 해당 결과를 고른 정확한 offline evaluation이다.

| Method | MSE ↓ | CRPS ↓ | Marginal 90% coverage | Mean interval width |
|---|---:|---:|---:|---:|
| sharp_refit | 0.028101 | 0.063513 | 79.34% | 0.1775 |
| broad_refit | 0.015283 | 0.041069 | 87.85% | 0.1934 |
| noise_inflation | 0.026239 | 0.060364 | 86.81% | 0.2583 |
| adaptive_expand | 0.018987 | 0.049198 | 83.33% | 0.1807 |

Broad − sharp MSE는 -0.012818, 95% bootstrap CI [-0.021957, -0.005814]다. 반면 adaptive − always-broad MSE는 +0.003703, 95% bootstrap CI [+0.000761, +0.007679]다. **현재 threshold rule은 always-broad보다 불리하다.** Model expansion 자체의 이득을 adaptive rule의 이득으로 바꾸어 말하면 안 된다.

| Truth regime | Sharp MSE | Broad MSE | Sharp coverage | Broad coverage |
|---|---:|---:|---:|---:|
| sharp | 0.006718 | 0.007314 | 97.22% | 97.78% |
| rounded | 0.031652 | 0.011614 | 76.67% | 94.44% |
| outside | 0.042962 | 0.024982 | 66.67% | 74.07% |

Detection은 sharp 0/15, rounded 5/15, outside 9/18이었다. False positive가 이 작은 표본에서 없었다는 것과 detection이 충분하다는 것은 다르다. Rounded의 2/3, outside의 절반을 놓쳤다. 특히 gate에서는 diagnostic probe가 model 차이를 충분히 드러내지 못했다.

Likelihood용 forward observation evaluation의 test 평균은 always-broad 138,240회, adaptive 203,328회였다. Adaptive는 initial sharp fit 비용도 필요해 약 47% 더 많았다. Acquisition와 rendering까지 포함한 end-to-end cost는 아니지만, 현재 구현에서 효율적이라고 주장할 근거가 없다.

## 7. Continuous proxy라는 더 단순한 반론

앞 결과는 b를 세 값으로만 표현한 영향일 수 있다. 그래서 같은 3개 observations에서 b까지 continuous parameter로 추정하는 five-dimensional SMC-1536을 추가했다. Proposal, tempering 원리는 동일하며 stage당 12 moves다. 아래 broad와 known-proxy도 SMC-1536/12이므로 particle와 rejuvenation budget을 맞췄다. Tempering stage 수와 forward evaluation은 달라질 수 있다.

| Method | MSE ↓ | CRPS ↓ | Marginal 90% coverage | Mean interval width |
|---|---:|---:|---:|---:|
| three_obs_broad | 0.014648 | 0.040431 | 88.72% | 0.1974 |
| three_obs_continuous | 0.012462 | 0.035695 | 94.27% | 0.2128 |
| three_obs_known_proxy | 0.010475 | 0.030781 | 95.31% | 0.1853 |

| Truth regime | Discrete broad MSE | Continuous b MSE | Known b MSE | Continuous b coverage |
|---|---:|---:|---:|---:|
| sharp | 0.007481 | 0.010256 | 0.006623 | 93.89% |
| rounded | 0.010952 | 0.010351 | 0.010027 | 96.67% |
| outside | 0.023700 | 0.016058 | 0.014059 | 92.59% |

Continuous − discrete broad MSE: -0.002186, 95% bootstrap CI [-0.004326, -0.000261]. Continuous − known proxy MSE: +0.001986, 95% bootstrap CI [+0.000812, +0.003251].

Continuous baseline의 overall improvement는 특히 outside-discrete-support subgroup에서 크다. 반대로 sharp subgroup의 MSE는 discrete broad보다 높다. Uniform continuous b prior는 sharp endpoint에 별도의 point mass를 주지 않으므로 가능한 bias를 구분해야 한다. 따라서 continuous model을 모든 상황의 winner라고 선언하지 않는다. 또한 이 3-observation table과 §4의 query-selected one-additional-observation table은 정보와 selection setting이 달라서 전체 method ranking으로 합치지 않는다.

이 실험은 ‘새로운 contact law가 필요하다’와 ‘현재 shape parameterization이 너무 거칠다’를 구분하기 위한 것이다. 모든 model은 같은 rigid impact law를 사용한다. 따라서 여기서 관측된 improvement를 일반적인 solver mismatch나 deformable physics correction으로 해석하지 않는다. Continuous prior는 exact sharp b=0에 point mass를 두지 않아 sharp regime에는 불리할 수도 있다. Model class가 넓다고 모든 subgroup에서 우월하다는 보장도 없다.

Known-proxy가 남기는 prediction error는 다른 latent parameter의 ambiguity와 finite inference error를 함께 포함한다. Exact Bayes oracle이나 달성 가능한 최저 위험으로 부르지 않는다. Continuous baseline도 post-hoc 한 번의 audit이므로 별도의 seed/scene replication 전까지 새 확정적 주장을 세우지 않는다.

Three-observation fit의 72 worlds 평균 forward observation evaluations는 discrete broad 549,888, continuous 547,584, known-proxy 529,152회였다. Particle count와 move count를 맞춘 비교이지 exact total-forward-call budget을 일치시킨 비교는 아니다. 비용별 performance frontier를 입증한 것으로 표현하지 않는다.

## 8. Numerical audit와 posterior convergence

Frictionless corner의 analytic solution과 구현의 max absolute velocity error는 9.26e−14 미만이었다. 1,024 random worlds/family × six queries에서 event overflow를 발견하지 않았고 translational kinetic energy가 초기값을 넘지 않았다. 이 검사는 spin을 포함한 완전한 conservation proof가 아니다.

PyBullet comparison은 family당 12 worlds, μ=0/0.3, b=0/1/3 mm, 두 timestep으로 수행했다. 최초 설정은 default contact ERP/slop와 우리 instantaneous impulse model이 맞지 않아 큰 오차를 냈다. 그 결과도 `cross_engine.json`에 보존했다. `contactERP=0`, `contactSlop=0`, restitution velocity threshold=0 등 solver 설정을 맞춘 별도 audit에서는 아래와 같았다.

| Family | Mean velocity vector error, dt=1/7680 s | Mean error, dt=1/15360 s | Max error, finer dt |
|---|---:|---:|---:|
| corner | 0.001403 | 0.000736 | 0.003689 |
| gate | 0.006826 | 0.003189 | 0.021850 |
| deflection | 0.000912 | 0.000456 | 0.001991 |

Unit은 m/s이며 2D vector norm이다. Timestep을 줄이면 평균 오차가 감소하지만 gate worst case는 여전히 sensor noise scale에 근접한다. Main experiment의 truth와 inference는 custom engine을 공유한다. 따라서 이 audit는 cross-engine generalization benchmark를 통과했다는 뜻이 아니다. Numerical configuration mismatch를 물리적 발견이나 method gain으로 해석하지 않는 것이 핵심이다. [Bullet solver implementation](https://github.com/bulletphysics/bullet3/blob/master/src/BulletDynamics/ConstraintSolver/btSequentialImpulseConstraintSolver.cpp)

![cross_engine_convergence](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/contact_model_v4/results/analysis/cross_engine_convergence.png)

Validation 9 cases에서 SMC-768과 SMC-3072의 두 seeds의 평균 predictive mean 간 RMS difference는 0.00209–0.00676 m/s였다. 가까운 utility의 action ranking은 일부 seed에서 달라졌다. 이는 exact optimal policy를 구한 것이 아님을 뜻한다. Main estimator를 practical strong baseline이라고 부르고 oracle이라고 부르지 않는다. Full refit의 uniform weights/큰 ESS만으로 mode mixing이 완전함을 증명할 수도 없다.

## 9. 실제 dataset에서 완료한 일과 미완료인 일

사용자의 지정에 따라 `/scratch2/mdj1412/datasets/deform360_contact_belief_audit`에 공개 [Deform360](https://huggingface.co/datasets/brownu/deform360)의 작은 subset을 저장했다. Official revision은 `5ea8c5d3fc7b4a7b4f9f921f2ceb1de24610f6a4`이며 각 파일의 URL/size/SHA256을 manifest에 남겼다. Official card의 license는 MIT다.

대상은 `008-pink-cloth`의 lift corner / lift edge / lift center, 두 cameras다. 6 MP4, 6 timestamp files, metadata, 3 calibration dictionaries, 총 16 files / 22,615,329 bytes다. 1280×720, nominal 30 fps 영상의 총 2,214 frames가 모두 decode됐고 frame count와 timestamp count가 일치했다. Calibration dictionary에는 36 cameras의 항목이 있었다.

| Episode | Nearest timestamp median [ms] | p95 [ms] | max [ms] |
|---|---:|---:|---:|
| 0 | 15.594 | 16.561 | 18.724 |
| 1 | 15.509 | 16.668 | 17.991 |
| 2 | 15.454 | 16.672 | 18.279 |

약 반 frame의 timing difference가 남아 있어 frame index만 맞추는 것으로 충분하지 않다. 초기 cloth shape와 grasp도 episode마다 달랐다. 같은 물체의 다른 interaction이라는 조건은 만족하지만, initial condition까지 같은 paired counterfactual이라는 조건은 만족한다고 선언할 수 없다.

![Recorded interactions](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/contact_model_v4/results/real_data_audit/recorded_interactions.jpg)

Simple HSV component의 area/centroid를 추출했으나 이는 learned segmentation이나 GT material point가 아니다. 서로 다른 camera의 silhouette centroid를 같은 3D point로 triangulation하지 않았다. 실제 physics fitting, material identification, held-out physical intervention prediction은 수행하지 않았다. **이 부분은 data feasibility audit이며 real-world pilot validation gate는 미통과다.**

## 10. 후보별 결정과 논문 가능성

| 후보 | 이번 판단 | 근거 |
|---|---|---|
| A. Benchmark / failure taxonomy | 가장 유망한 중심축으로 유지 | Finite support, sensor approximation, proxy support, solver setting을 분리한 reusable diagnostic이 생겼다. |
| B. Branch-preserving efficient inference | 현 구현은 보류 | Compression CI가 zero를 포함하고 end-to-end cost advantage가 없다. |
| C. Predictive-check-driven expansion | 현 rule은 채택하지 않음 | Detection misses가 많고 always-broad보다 정확도·비용이 불리하다. |
| Query-aware acquisition | A의 평가 component로 유지 | Full-refit 비교에서 fixed rule 대비 효과를 확인했지만 EVSI 원리 자체는 선행연구다. |
| Calibrated real-world method | 아직 주장하지 않음 | General perception과 physical held-out intervention 검증이 없다. |

지금 논문 기여로 제출할 수 있다고 자신 있게 말할 부분은 아직 없다. 다만 **잘못된 이유로 좋아 보이는 결과를 제거할 수 있는 실험 기반**은 마련했다. EVSI의 알려진 원리, generic Bayesian model checking, geometry/material uncertainty 자체를 새 contribution으로 쓰지 않는다. [Task-specific NBV](https://arxiv.org/html/2605.05095), [Action-BED](https://arxiv.org/html/2606.23662), [IDTD](https://arxiv.org/html/2610.10905), [Inside-Out SMC²](https://arxiv.org/html/2402.07868)와의 차별화는 이 지점에서 엄격해야 한다.

내가 선택할 다음 연구는 **실제 visual reconstruction에서 자연스럽게 발생한 collision-boundary uncertainty의 측정과 counterfactual reliability 검증**이다. 인위적인 discrete b support를 더 복잡하게 만드는 방향은 선택하지 않겠다. Raw image의 boundary evidence를 버리지 않는 geometry baseline, continuous-proxy SMC, calibrated sensor model이 먼저다. 이 셋으로도 남는 failure가 확인되어야 새로운 correction mechanism의 필요성이 생긴다.

네 연구자의 공개 논문을 바탕으로 만든 가상 지도 질문은 다음으로 모인다. Yale Song의 관점에서는 intervention/task의 유효성, Seon Joo Kim의 관점에서는 object identity와 observation evidence, Seungryong Kim의 관점에서는 correspondence와 representation의 정보 손실, Lingjie Liu의 관점에서는 simulation representation과 실제 물성 식별의 차이를 검증해야 한다. 실제 발언이나 개인적인 지도 방식을 안다고 주장하는 것이 아니다.

## 11. 다음 검증을 구체적으로 제한한다

1. **Visual gate:** 현재 raw silhouette에서 corner rounding을 추정하는 baseline부터 추가한다. [g,d]만 쓰는 sensor와 비교해 view selection 결론이 유지되는지 본다. 같은 images에서 extracted boundary likelihood를 calibrate하고, known color/family 가정을 단계적으로 제거한다.
2. **Real-data gate:** Deform360의 official processed tracks/calibration과 grasp trajectory를 점검해 initial-state registration과 material-point correspondence가 가능한 object만 선정한다. Cloth와 현재 rigid engine은 물리계가 다르므로 단순 parameter 복사로 실험하지 않는다. Deformable setting을 선택할 경우 PhysTwin/appropriate elastic baseline을 별도로 구성해야 한다.
3. **Counterfactual gate:** 같은 object의 한 interaction으로 fit하고 다른 interaction은 prediction에만 사용한다. 초기 state/action을 conditioning variable로 명시한다. Additional observed episode를 선택할 때 target episode를 보지 않는다. Test 분리는 object 단위로 둔다.
4. **진행 기준:** Continuous geometry와 strong SMC를 포함한 baseline보다 proper score와 coverage가 object별로 일관되게 개선되어야 한다. Pipeline artifact나 mismatched observation budget으로 설명되면 method claim을 중단한다. 충분한 objects가 확보되지 않으면 benchmark paper라는 범위도 축소한다.

이번 결과만으로 다음 학회의 acceptance나 novelty를 예측하지 않겠다. 현재는 mechanism과 evaluation을 명확히 만든 단계이며, real visual evidence로 넘어가는 데 성공하는지가 main-conference 수준으로 발전할지 결정한다.

## 12. 재현과 산출물

Remote project: `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/contact_model_v4`.

- [실행 절차와 가정](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/contact_model_v4/README.md)
- [Primary refined results](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/contact_model_v4/results/analysis/refined_summary.json)
- [Equal-budget model / compression results](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/contact_model_v4/results/analysis/summary.json)
- [원본 case별 결과](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/contact_model_v4/results/main)
- [Independent calibration](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/contact_model_v4/results/calibration)
- [Real-data audit](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/contact_model_v4/results/real_data_audit/audit.json)

`calls`는 likelihood용 forward observation evaluations이며 view도 포함한다. Rendering, acquisition, 모든 query prediction을 포함하는 total physics-call budget이 아니다. GPU workloads가 겹친 상태의 wall time으로 speedup을 주장하지 않았다. Seeds, observation, truth, costs, posterior samples를 보존했고 초기 solver audit의 실패 결과도 남겼다. Source/result SHA256 manifest와 report copy는 remote project에 함께 저장한다.
