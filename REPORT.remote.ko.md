# 첫 physics 실험 실행 보고서: plug-in geometry와 joint scene belief

작성일: 2026-11-03, Asia/Seoul. 앞선 v4 topic 1 설계의 첫 simulator pilot을 실제 원격 서버에서 구현하고 실행한 결과다.

**이번 통제 실험에서는 joint scene belief가 필요한 geometry observation을 선택했고, plug-in geometry는 초기 geometry 선택을 확신해 그 관찰을 생략했다. 그러나 같은 top view를 제공하고 geometry refitting을 허용하면 두 방법의 차이는 거의 사라졌다. 단순한 query-type rule도 budget 1에서는 joint EVSI와 같은 성능을 냈다.** 따라서 현재 결과가 뒷받침하는 것은 geometry collapse의 실패 메커니즘이며, 새로운 acquisition algorithm의 우수성은 아니다.

## 1. 어디에서 무엇을 실행했는가

원격 프로젝트는 다음 경로에 생성했다.

```text
/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief
```

`datacenter3`로 SSH 접속한 실제 host는 `yonsei-ai-login3`였다. Login node에서는 GPU driver에 접근할 수 없어 Slurm allocation과 compute node를 확인했다. 초기 조회에서는 아래 네 GPU 모두 memory usage 0 MiB, utilization 0%, compute process 없음으로 나타났다. 이 상태는 조회 시점의 snapshot이며 미래의 사용 가능성을 보장하지 않는다.

| Slurm job | Compute node | 해당 node의 physical GPU index | 초기 조회 |
|---|---|---:|---|
| 2642474 | node55 | 5 | idle |
| 2642473 | node55 | 6 | idle |
| 2642475 | node56 | 2 | idle |
| 2642476 | node56 | 4 | idle |

실제 실험은 `2642474` allocation 안의 `node55`에서 CPU 1개로 실행했다. `CUDA_VISIBLE_DEVICES=""`와 BLAS/OpenMP thread 1개를 설정했다. **GPU 사용량은 0이며 neural model training이나 learned model inference를 실행하지 않았다.** Bayesian posterior 계산은 CPU에서 수행했다. 기존 사용자 job과 pending job은 취소하거나 변경하지 않았고, 실행 후에도 기존 네 allocation이 유지됨을 확인했다.

Python 3.9.18, PyBullet 3.2.7, NumPy 1.26.4를 프로젝트 전용 `.venv`에 설치했다. 실제 측정한 physics 생성 시간은 **13.96초**, 전체 belief/policy 비교 시간은 **23.47초**다. 환경 설치, 개발 중 실패한 검증, 분석·figure 생성, 파일 전송 시간은 이 수치에서 제외한다. 이처럼 짧은 실행이 가능한 이유는 네 candidate world의 forward simulation을 미리 계산하고 모든 방법이 같은 table을 사용하기 때문이다. Real-video model의 inference 속도가 아니다.

## 2. 기존 설계를 어떻게 구체화했는가

### 2.1 Physics와 hidden world

공 하나와 고정된 box 하나를 PyBullet에서 simulation했다. x는 진행 방향, y는 깊이 방향, z는 높이다. 중력, friction, damping을 0으로 둔 ideal planar-motion 조건이다. 실제 table 위의 rolling dynamics를 재현한 것이 아니다.

| 항목 | 구현값 |
|---|---|
| 공 | radius 0.02 m, mass 1 kg |
| 고정 box | 크기 0.03 × 0.06 × 0.08 m, infinite mass |
| 공의 초기 위치 | x = −0.25 m, y = 0, z = 0.04 m |
| wall x | family마다 Uniform(0.18, 0.22) m |
| geometry candidate | g=0: wall y ∼ Uniform(0.075, 0.095) m로 collision 없음. g=1: wall y=0으로 collision 발생 |
| restitution candidate | low ∼ Uniform(0.15, 0.30), high ∼ Uniform(0.65, 0.85), 각 family 안에서 두 값 고정 |
| prior | 2 geometry × 2 restitution, 네 world에 각각 0.25 |
| factual clip | incoming speed ∼ Uniform(0.23, 0.27) m/s, 길이 0.4초, collision 이전 종료 |
| physics timestep | 1/960초 |
| target rollout | intervention 뒤 2초, trajectory는 0.05초마다 저장 |

**각 family의 두 geometry 값과 두 restitution 값 자체는 model에 알려 준다. 모르는 것은 그중 실제 world가 무엇인지다.** 따라서 정답이 포함된 finite candidate set에서의 belief 비교이며, unknown continuous geometry·material을 RGB만으로 찾아낸 실험은 아니다. Wall x, radius, mass, camera, initial state도 알려진 값이다.

### 2.2 Visual observation과 information access

PyBullet의 실제 simulated state를 별도 CPU renderer가 128 × 128 RGB로 그린다. Renderer는 exact orthographic projection과 단색 sphere/box를 사용하며 4배 supersampling을 적용한다. Sensor가 반환하는 것은 image에서 color thresholding으로 추출한 centroid이고, predictor에 true y나 restitution 숫자를 직접 반환하지 않는다. Simulator state 접근은 renderer, candidate forward simulation, evaluation에 한정한다.

추출한 centroid에 Gaussian detector noise를 추가했다. Standard deviation은 0.5, 1, 2 px다. 이는 실제 detector를 학습하고 noise를 calibration한 결과가 아니라 **정의한 synthetic measurement model**이다. Likelihood도 동일한 Gaussian model을 사용한다. Renderer와 likelihood mismatch가 없는 조건이다.

| Action | 획득하는 measurement | 원래 hidden world와 공유하는 것 |
|---|---|---|
| `view` | 원래 scene의 top-view RGB에서 얻은 wall의 vertical centroid + noise | 원래 geometry |
| `probe` | 별도 calibration collision의 0.5초·0.6초 RGB에서 얻은 ball horizontal centroid 차이 + noise | 공–wall contact pair의 restitution |
| `repeat` | 기존 frontal view에서 wall vertical centroid를 다시 측정 | 이 조건에서는 unknown geometry와 restitution을 구별하지 못함 |
| `stop` | 추가 measurement 없음 | 현재 belief로 예측 |

`probe`는 공 x=−0.08 m, wall x=0.08 m, wall y=0, incoming speed 0.4 m/s에서 별도로 reset한 experiment다. Target query의 원래 상태와 혼합하지 않는다. 원래 wall y를 노출하지 않고, target counterfactual 자체를 실행해 반환하지도 않는다. 두 frame의 centroid noise가 독립이므로 displacement likelihood의 standard deviation은 √2σ다.

아래는 noise를 넣기 전 실제 생성한 RGB다. 같은 family의 네 world는 factual view에서는 구분되지 않고 top view와 probe가 서로 다른 latent factor를 구별한다.

![실제 생성한 observation](/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/results/pilot/figures/observations.png)

### 2.3 같은 factual observation에 주는 세 query

| Query | Counterfactual intervention | 필요한 정보 |
|---|---|---|
| Q1 | 공의 speed를 family별 Uniform(0.5, 0.7) m/s로 변경 | 원래 geometry가 collision을 일으키는지, collision 시 restitution이 얼마인지 |
| Q2 | wall y를 0으로 이동, 공의 speed는 유지 | 원래 geometry는 무관하고 restitution이 필요 |
| Q3 | wall 제거 | 이 제한된 model에서는 geometry와 restitution 모두 무관 |

Primary target은 `z = outgoing x velocity / incoming x velocity`다. No collision이면 z=1이고 head-on collision이면 z=−e다. 이 관계를 단순 수식으로 target에 대입하지 않고 실제 simulator output으로 저장했으며, 별도로 그 관계가 성립하는지 검증했다. Prediction은 posterior mean, uncertainty는 네 simulator outcome의 weighted distribution으로 표현한다. Posterior mean trajectory가 하나의 물리적으로 가능한 video라는 뜻은 아니다.

## 3. 공정한 비교를 위해 고정한 조건

### 3.1 Belief의 차이

**Joint scene belief**는 네 geometry–restitution 조합을 모두 유지하고 선택한 measurement의 likelihood로 posterior를 update한다.

**Plug-in fixed**는 첫 geometry MAP을 선택한 뒤 그 geometry를 계속 유지한다. 같은 geometry 안에서는 restitution posterior를 유지한다.

**Plug-in refit**은 실제로 받은 measurement의 이력 전체로 geometry MAP을 매번 다시 구할 수 있다. 다만 다음 observation의 가치를 계산할 때와 prediction을 제출할 때는 하나의 geometry로 collapse한다. Refitting 때 original prior와 모든 candidate의 likelihood를 다시 사용하므로, view를 실제로 받으면 처음에 버렸던 geometry를 복구할 수 있다. 이는 fixed baseline보다 강하지만, **planning 시 geometry uncertainty를 버리는 approximation**이다. 일관된 full Bayesian filter와 동일하지 않다.

초기 factual view는 geometry에 대해 완전히 ambiguous하므로 MAP tie가 발생한다. Family index가 짝수이면 no-collision geometry, 홀수이면 collision geometry를 선택하도록 정했고, test의 20개 family에서 각각 10개씩 포함된다. Hidden world를 보고 tie를 고르지 않는다. 두 tie 방향 모두 별도로 확인했다.

### 3.2 Acquisition objective와 budget

Primary comparison에서는 세 belief에 동일한 myopic EVSI를 사용한다.

```text
risk(p, q) = Var[z_q | p]
gain(a) = risk(p, q) − E_y[risk(posterior(p, a, y), q)]
```

각 action의 cost는 1이다. 최대 budget 0, 1, 2를 비교하며, 최대 gain이 0.005보다 크면 해당 action을 선택한다. 그렇지 않으면 stop한다. 따라서 허용 budget이 같아도 실제 사용한 cost는 다를 수 있다. 이를 숨기지 않기 위해 `MSE + 0.005 × actual cost`도 함께 집계했다.

Expectation은 Gaussian likelihood 아래 32-point Gauss–Hermite quadrature로 계산한다. Budget 2에서도 매번 한 단계의 risk reduction을 보는 **greedy myopic policy**다. 최적의 non-myopic policy나 learned GO-CBED/Action-BED 구현으로 표기하지 않는다.

추가로 parameter EIG, query EIG, fixed view, fixed probe, random action, query-type rule을 비교했다. EIG는 expected information gain이 1e−9 nats를 넘으면 관찰하며, MSE 단위의 0.005 penalty를 EIG에 그대로 적용하지 않았다. 따라서 EIG와 EVSI의 stop/cost 차이는 objective 및 stopping rule 차이를 함께 반영한다. Query-type rule은 Q1에서 view→probe, Q2에서 probe→probe, Q3에서 stop한다.

Known geometry와 known restitution도 privileged-information diagnostic으로 실행했다. 이를 정상적인 inference method의 성능 순위에 섞지 않는다. 전체 방법 수 13개에는 이 두 diagnostic이 포함된다. Budget 0은 각 방법의 no-acquisition baseline이다.

### 3.3 Data, compute, randomization

- 60개 scene family, family당 네 hidden world와 세 query: 720개 world–query episode.
- Family 0–19는 development, 20–39는 validation, 40–59는 held-out test다.
- World마다 32개 detector-noise realization, noise scale 세 가지를 사용했다. Test의 query별 표는 20 × 4 × 32 = 2,560개 realization의 평균이지만 **독립 표본 수는 20개 family**다.
- 같은 family·world·action·visit에는 모든 방법과 query가 같은 noise draw를 사용한다. 선택하지 않은 action의 실제 noise realization은 policy에 주지 않는다.
- 모든 방법은 같은 네 candidate forward output과 observation likelihood를 사용한다. Joint만 추가 simulator call이나 더 좋은 visual encoder를 받은 것이 아니다. Online calculation 시간은 방법별로 별도 측정했다.
- Main table의 noise 2 px 외에도 0.5, 1 px 결과를 모두 저장했다. Test를 본 후 policy parameter를 조정하지 않았다. Figure의 axis 표현만 최종 검토 후 수정했다.

Data와 test family는 seed로 고정했지만 사전 등록된 외부 benchmark는 아니다. Test parameter 범위도 development와 같으며, OOD test라고 부르지 않는다.

## 4. Counterfactual prediction 결과

아래는 held-out 20개 family, centroid noise 2 px 결과다. MSE는 dimensionless normalized velocity의 squared error다. B는 최대 observation budget이며 actual cost는 별도다.

| Query | Method | MSE, B=0 | MSE, B=1 | MSE, B=2 | Actual cost, B=2 |
|---|---|---:|---:|---:|---:|
| Q1 | Joint + EVSI | 0.587748 | **0.036021** | **0.024094** | 1.504 |
| Q1 | Plug-in refit + EVSI | 1.141240 | 1.147092 | 1.146232 | 0.897 |
| Q1 | Plug-in refit + fixed view | 1.141240 | 0.036952 | 0.034255 | 2.000 |
| Q1 | Joint + query-type rule | 0.587748 | 0.036021 | 0.025171 | 2.000 |
| Q1 | Joint + random | 0.587748 | 0.402074 | 0.270250 | 2.000 |
| Q2 | Joint + EVSI | 0.068510 | 0.048058 | 0.036571 | 1.805 |
| Q2 | Plug-in refit + EVSI | 0.068510 | 0.048058 | 0.036571 | 1.805 |
| Q2 | Joint + parameter EIG | 0.068510 | 0.068510 | 0.048130 | 2.000 |
| Q2 | Joint + query-type rule | 0.068510 | 0.048058 | 0.036424 | 2.000 |
| Q3 | Joint + EVSI | ≈0 | ≈0 | ≈0 | 0.000 |
| Q3 | Plug-in refit + EVSI | ≈0 | ≈0 | ≈0 | 0.000 |

Q3의 일부 method에서 10⁻³³ 수준의 floating-point residual이 발생하므로 numerical zero로 해석한다. Figure에서도 이 residual을 큰 차이처럼 확대하지 않았다.

![Risk와 budget 비교](/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/results/pilot/figures/risk_budget.png)

Q1, B=1에서 joint의 MSE는 plug-in refit보다 **96.86% 낮았다**. Paired scene-family bootstrap 10,000회로 구한 `plug-in MSE − joint MSE`는 **1.111071**, 95% CI는 **[1.088012, 1.134905]**다. B=2의 차이는 1.122138, 95% CI [1.099978, 1.145276]다. 이 CI는 이 synthetic family distribution에 대한 것으로, real-world effect size의 CI가 아니다. 여러 contrast에 대한 multiplicity adjustment는 하지 않았으므로 exploratory comparison으로 읽어야 한다.

다만 B=0부터 plug-in이 불리했다. Geometry를 하나로 고정한 point prediction 자체가 posterior mean보다 MSE가 크기 때문이다. **B=1의 전체 차이를 observation selection만의 효과라고 해석하면 안 된다.** Acquisition mechanism은 다음 절의 same-information control과 함께 판단해야 한다.

Q2에서 두 belief의 EVSI 결과는 numerical tolerance 안에서 완전히 같았다. Intervention이 geometry를 고정했으므로 geometry uncertainty를 유지할 이점이 사라진 것이다. Q3에서는 모든 prediction이 사실상 정확했다. Joint가 모든 query에서 무조건 우수하다는 설명과 맞지 않는, 필요한 negative control이다.

Q1의 trajectory 결과도 같은 방향이었다. B=1에서 coordinate-wise trajectory RMSE는 joint 0.0492 m, plug-in refit 0.2765 m였고, trajectory energy score는 각각 0.01691과 0.19004였다. 여기서 RMSE는 40개 time sample × 3 coordinate에 대한 평균 squared error의 square root다. 통상적인 Euclidean position RMSE와 normalization이 다르므로 수치를 섞지 않는다.

## 5. Observation selection에서 무엇이 달라졌는가

| Method | Q1 첫 action | Q2 첫 action | Q3 첫 action |
|---|---|---|---|
| Joint + EVSI | view 100% | probe 100% | stop 100% |
| Plug-in refit/fixed + EVSI | probe 50%, stop 50% | probe 100% | stop 100% |
| Joint + parameter EIG | view 100% | view 100% | view 100% |
| Joint + query EIG | view 100% | probe 100% | stop 100% |
| Joint + query-type rule | view 100% | probe 100% | stop 100% |

![Observation selection 결과](/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/results/pilot/figures/selection.png)

Plug-in이 no-collision geometry를 골랐으면 Q1의 결과는 restitution과 무관하다고 믿는다. 남은 posterior 안에서는 prediction variance가 0이므로 stop한다. 실제 world가 collision world여도 그 가능성이 belief에 없으므로 관찰 필요성을 계산할 수 없다.

반대로 collision geometry를 골랐으면 restitution을 알아내기 위한 probe를 고른다. 하지만 원래 scene의 collision 여부는 여전히 모른다. Probe를 더 정밀하게 읽는 것만으로 잘못 고정한 geometry를 복구할 수 없다. B=1 Q1의 MSE를 tie 방향별로 나누면 no-collision tie 1.1404, collision tie 1.1538이었다. 한쪽 tie만 유리하게 선택해서 생긴 결과가 아니다.

Joint는 Q1에서 먼저 collision 여부를 구별하는 view를 선택한다. B=2에서는 첫 view 이후 남은 risk를 보고 추가 probe 또는 view를 선택하거나 stop한다. 평균 cost는 1.504로, 항상 두 action을 쓰는 query-type rule의 2.000보다 작았다. `MSE + 0.005 × cost`는 joint 0.03161, query-type rule 0.03517이었다.

Parameter EIG는 query와 무관한 전체 latent uncertainty를 줄이므로 Q2와 Q3에서도 geometry view를 먼저 골랐다. Q2, B=1에서 query-conditioned EVSI와의 MSE 차이는 0.020452, paired 95% CI [0.016333, 0.024887]였다. 이는 task-relevant information의 중요성을 확인하지만 이미 알려진 goal-oriented experimental design 원리를 다시 확인하는 결과다.

**Query EIG와 query-type rule도 B=1에서는 EVSI와 동일했다.** B=2에서도 query EIG의 Q1 MSE는 0.024105로 EVSI 0.024094와 매우 가깝다. Q2에서는 query-type rule과 query EIG가 cost 2를 모두 사용해 EVSI보다 raw MSE가 조금 낮았다. EVSI가 모든 metric에서 최고라는 주장 대신, cost-sensitive stopping과 representation 차이를 분리해 설명해야 한다.

## 6. 가장 중요한 ablation: 같은 정보를 주면 어떻게 되는가

Q1, B=1에서 두 방법에 top view를 똑같이 제공했다.

| Prediction method | MSE | Brier | Actual cost |
|---|---:|---:|---:|
| Joint + fixed view | 0.036021 | 0.000799 | 1 |
| Plug-in refit + fixed view | 0.036952 | 0.001172 | 1 |

MSE 차이는 **0.000932**, paired 95% CI **[−0.000409, 0.002736]**로 이 pilot에서 뚜렷한 차이라고 할 수 없다. Top view를 받으면 geometry MAP을 다시 fitting할 수 있고 geometry ambiguity가 대부분 해소되기 때문이다.

따라서 이번 실험의 가장 설득력 있는 해석은 다음과 같다. **Geometry uncertainty를 planning 시 버리면 그 uncertainty를 해소할 관찰의 가치를 평가하지 못한다. 그러나 필요한 관찰을 외부에서 제공하면, observation 뒤에 geometry를 다시 fitting하는 단순한 방법도 대부분 회복한다.** Joint posterior를 모든 단계에 유지해야만 좋은 prediction이 가능하다는 강한 결론은 지지하지 않는다.

Known-geometry diagnostic에서는 Q1의 Brier가 모든 budget에서 0이었다. B=0 MSE는 0.034255로 restitution uncertainty만 남았고, B=1은 0.023461이었다. Known-restitution diagnostic은 Q1에서 view를 고르고, B=1 MSE 0.001741만 남았다. Q2에서는 known restitution이면 아무것도 관찰할 필요가 없었다. Geometry와 material을 구별한 실험 의도와 일치한다.

## 7. Uncertainty는 얼마나 믿을 수 있었는가

Q1, B=1, noise 2 px 결과다.

| Method | 실제 MSE | 평균 posterior variance | CRPS | 90% interval coverage | 평균 interval width |
|---|---:|---:|---:|---:|---:|
| Joint + EVSI | 0.036021 | 0.035381 | 0.066050 | 99.96% | 0.263181 |
| Plug-in refit + EVSI | 1.147092 | 0.024252 | 0.744077 | 50.00% | 0.247028 |
| Plug-in refit + fixed view | 0.036952 | 0.034352 | 0.066585 | 99.88% | 0.260577 |

Plug-in은 작은 posterior variance를 보고하지만 실제 error는 훨씬 컸다. Interval도 거의 같은 폭인데 정답을 약 절반만 포함했다. 이는 material uncertainty의 크기를 조금 잘못 추정하는 문제가 아니라, 가능한 geometry mode 하나를 support에서 제거한 데 따른 overconfidence다.

Joint의 실제 MSE와 평균 posterior variance는 가깝고 proper score도 개선됐다. 그러나 **90% interval coverage가 약 100%이므로 perfectly calibrated interval이라고 말할 수 없다.** 네 점의 discrete support에 central quantile interval을 적용해 conservative coverage가 나온다. Continuous parameter 실험에서는 coverage뿐 아니라 sharpness, CRPS, reliability를 다시 검증해야 한다.

![Uncertainty 결과](/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/results/pilot/figures/uncertainty.png)

## 8. Noise sensitivity와 검증 결과

| Centroid noise | Q1 joint MSE, B=1 | Q1 joint MSE, B=2 | Q1 plug-in refit MSE, B=1 | Q2 joint/plugin MSE, B=1 |
|---|---:|---:|---:|---:|
| 0.5 px | 0.034255 | 0.000619 | 1.142301 | 0.001751 |
| 1 px | 0.034255 | 0.010412 | 1.147749 | 0.021924 |
| 2 px | 0.036021 | 0.024094 | 1.147092 | 0.048058 |

낮은 noise에서는 top view가 geometry를 거의 완벽하게 알려 준다. Q1, B=1에 남은 MSE는 대부분 restitution uncertainty다. B=2에서 probe를 추가하면 그 error가 줄어든다. Noise가 커지면 probe의 짧은 displacement measurement가 불확실해져 Q2와 B=2 Q1의 error가 커진다. Plug-in의 geometry 누락은 noise를 줄이는 것만으로 해결되지 않았다.

검증한 항목은 다음과 같다.

- 모든 60개 family에서 factual sampled frame의 RGB가 네 hidden world 간 pixel 단위로 같았다. Maximum difference는 0이다.
- 모든 factual rollout에서 contact가 없고, target contact 여부는 Q1의 geometry 및 Q2/Q3 intervention과 일치했다.
- Probe와 target에서 측정한 normalized outgoing velocity와 ideal restitution 관계의 최대 오차는 **2.22 × 10⁻¹⁶**이었다.
- Probe measurement mean은 원래 geometry에 따라 달라지지 않았다. Original geometry의 leakage를 검사했다.
- Q2의 joint/plugin EVSI 결과 동일성, Q3 numerical-zero error, known-geometry Brier 0, nonnegative proper score 검사를 통과했다.
- Query relevance, stop, geometry refitting, degenerate correct prediction의 proper score, quadrature에 관한 5개 unit test를 통과했다.
- Development/validation family의 다양한 posterior에서 32-point와 64-point quadrature를 비교했다. 최대 EVSI gain 차이는 **1.71 × 10⁻⁴**, 11,880개 비교에서 action 불일치는 0이었다. 모든 가능한 posterior에 대한 수학적 정확성 보장은 아니다.

구현 중 발견한 두 문제도 기록한다. TinyRenderer의 custom orthographic projection은 빈 image를 반환했다. Distant perspective 대안은 일부 family에서 depth-dependent pixel 차이를 남겼으므로 폐기했다. 최종 결과는 exact orthographic CPU renderer에서만 생성했다. 또한 초기 PyBullet contact 설정에서는 impact 전에 velocity가 줄어 설정 restitution이 실현되지 않았다. `restitutionVelocityThreshold=0`, `contactERP=0`, `contactSlop=0`, 각 body의 `contactProcessingThreshold=0`을 사용하고 실제 outgoing velocity로 검증했다. 공 restitution=e, wall restitution=1이다. 이 설정은 이상화된 head-on collision을 통제하기 위한 것이며 복잡한 resting contact에 대한 권장 default라고 주장하지 않는다. PyBullet의 rendering·dynamics API는 [공식 Quickstart Guide](https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstart_guide/PyBulletQuickstartGuide.md.html)와 [공식 restitution 예제](https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/restitution.py)를 참고했다.

## 9. 이 결과로 논문에서 주장할 수 있는 범위

현재 evidence로는 다음과 같은 제한된 주장이 가능하다.

> 같은 ambiguous visual observation, forward model, likelihood, acquisition objective 아래에서도 geometry를 하나로 고정하는 approximation은 필요한 geometry observation의 가치를 과소평가할 수 있다. 이 실패는 counterfactual prediction error와 overconfidence를 동시에 만들며, 같은 정보를 외부에서 제공하고 geometry refitting을 허용하면 크게 완화된다.

이 주장은 실제 simulator·rendered measurement·paired control로 확인했다. 앞선 exact four-world 계산을 실행한 것보다 한 단계 나아갔지만, 단순한 finite-state model의 예상 가능한 실패를 재현한 측면이 크다. 현재만으로 충분한 algorithmic novelty가 생긴 것은 아니다.

특히 다음을 주장할 수 없다.

1. Real video에서 geometry와 physics를 함께 복원했다는 주장. Camera, object identity, shape family, candidate parameter를 알려 줬다.
2. 기존 GO-CBED, Action-BED, VideoPhysEdit, ATW보다 좋다는 주장. 해당 원 구현을 실행하지 않았고, 단순 baseline의 공식 재현으로 표기하지 않았다.
3. 새로운 EVSI 원리를 제안했다는 주장. EVSI와 query EIG는 기존 criterion을 이 작은 문제에 적용했다.
4. 네 candidate를 유지하는 것이 compute 제한 아래서 최선이라는 주장. Shared forward table과 같은 quadrature order로 비교했지만, online runtime 자체가 완전히 같지는 않다. Joint EVSI는 전체 평가에서 2.72초, plug-in refit EVSI는 2.57초였다.
5. 모든 continuous geometry나 contact topology에서 같은 큰 효과가 난다는 주장. 지금은 hit/miss가 분명한 두 mode이며 정답이 model support 안에 있다.

## 10. 다음 연구 방향은 어떻게 좁히는 것이 좋은가

이번 결과를 보면 다음 논문 질문은 “joint belief가 plug-in보다 좋은가?”보다 더 좁고 어려워져야 한다.

**추천 질문은 “언제 geometry uncertainty를 유지해야 하고, 언제 하나로 collapse해도 downstream observation decision을 보존할 수 있는가?”다.** 같은 top view 뒤에는 refit과 joint가 거의 같았다는 결과가 이 질문을 직접 뒷받침한다. Prediction에 영향을 주지 않는 mode까지 계속 유지하기보다, contact outcome과 acquisition ranking을 바꾸는 mode를 제한된 compute 안에서 보존하는 방법이 연구 대상이 될 수 있다.

다음 단계는 continuous wall offset을 contact boundary 주변에 놓고 같은 query type 안에서 view, probe, stop의 최선 선택이 달라지게 만드는 것이다. Query-type rule이 계속 풀 수 있는 benchmark라면 learned acquisition의 필요성을 주장하기 어렵다. 이후 geometry–restitution correlation, ambiguous initial velocity, camera error를 하나씩 추가해야 한다.

Baseline도 강화해야 한다. 이번 plug-in refit 외에 multi-start fitting, 작은 mixture, generic SMC, occasional exploration을 사용하는 plug-in을 같은 forward-call budget에서 비교해야 한다. 새로운 방법의 기여는 이 강한 baseline보다 적은 candidate로 acquisition regret을 줄이는 데서 찾아야 한다. Four-state exact reference를 능가하는 결과를 목표로 삼으면 안 된다.

이번에는 unequal action cost, learned policy, neural video model, real sensor noise calibration, continuous geometry fitting, model mismatch, 대규모 compute sweep을 실행하지 않았다. GPU가 필요해지는 후속 단계는 사용자의 allocation과 실제 process 사용 상태를 다시 확인한 뒤 진행해야 한다. 모든 GPU가 사용 중이면 model training/inference는 시작하지 않는다는 조건을 프로젝트 지침에도 저장했다.

## 11. 산출물과 재현 경로

- [원격 실행 코드와 README의 로컬 사본](/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/README.md)
- [Physics generator](/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/physics.py)
- [Belief·acquisition 비교 코드](/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/experiment.py)
- [분석 코드](/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/analyze.py)
- [전체 test 결과표](/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/results/pilot/tables.md)
- [전체 test aggregate JSON](/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/results/pilot/summary.json)
- [Paired bootstrap contrast](/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/results/pilot/paired_contrasts.json)
- [실행 설정과 source/data hash](/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/results/pilot/config.json)
- [Physics 검증과 실제 runtime](/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/data/pilot/physics_validation.json)
- [전체 비교 runtime](/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/results/pilot/runtime.json)

동일한 보고서는 원격 프로젝트의 `REPORT.ko.md`에도 저장한다. Python source, dependency lock, seeds, raw family metrics, 주요 test episode detail, figure, 실행 log를 함께 남겼다. 이번 결과는 실행하지 않은 추정치나 예정된 실험 결과가 아니라 위 artifact에서 다시 확인할 수 있는 실제 원격 실행 결과다.
