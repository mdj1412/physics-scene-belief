# v27 · 여러 world의 factual evidence로 observation 선택을 어디까지 바꿀 수 있는가

**두 proxy 중 profile–curvature 관계는 추정할 수 있었고, velocity–friction 관계는 이 실험의 factual evidence만으로 식별할 수 없었다.** 8개 unseen shift에서 estimated-prior K256의 held-out Brier는 **0.11083**이었다. 기존 prior를 쓰는 K256은 **0.19296**, 모집단의 두 shift를 실제로 아는 known-prior K256은 **0.09765**였다. 그러나 어떤 calibration worlds도 보지 않고 두 관계를 중립으로 둔 neutral K256도 **0.11468**에 도달했다. Estimated−neutral 차이는 −0.00385, simultaneous 95% interval은 [−0.01163, +0.00392]였다. 관측하지 않고 얻을 수 있는 개선을 먼저 제하면 factual batch 자체가 제공한 decision 이득의 크기는 아직 불확실하다.

같은 known-prior decoder로 평가해도 estimated policy는 source K256보다 −0.02674 낮은 Brier를 기록했다. 이 차이는 observation 선택에서 나온다. 반면 always-query와 동일 decoder에서 비교한 차이는 −0.00274 [−0.00918, +0.00371]였다. GPU p90은 estimated 3.343 ms, source K256 3.313 ms, always-query 0.737 ms였다. Brier 개선과 속도 개선은 다른 결론을 준다.

## 1. 데이터와 배포 조건

각 world에서 한 번의 factual collision으로 noisy boundary profile 9개와 outgoing velocity 2개를 읽는다. 정책은 같은 world에서 query 위치의 정밀 geometry를 한 번 더 볼지(query view), 정해진 다른 충돌의 velocity를 측정할지(probe) 고른다. 그 한 번의 추가 관측으로 다른 위치·입사속도에서 접촉이 stick할 확률을 예측한다. 평가값은 binary stick label과 이 확률의 squared difference인 Brier loss다. 작은 값이 좋다.

V26과 같은 geometry, Coulomb contact law, sensor noise, action pair, M2048 acquisition을 썼다. αc는 profile offset의 중심값이 curvature spread에, αf는 restitution의 중심값이 friction spread에 정렬되는 정도다. 각 관계에서 중심을 반대로 쓰는 확률은 (1−α)/2다. 1은 source 관계, 0은 독립, −1은 완전 반전이다. 학습된 v25 RBF gate와 cutoff를 hash로 확인했다. V26의 여섯 단계 grid에서 직접 시험하지 않았던 8개 조합을 먼저 정했고, (1,1), (0,−1), (0,+1)은 별도 controls로 남겼다. 여기서 unseen은 동일한 simulator family에서의 새로운 coupling strength를 뜻한다.

각 조합에서 독립 cohort 3개를 생성했다. 각 cohort의 calibration 1,024개 worlds와 평가 128개 worlds는 서로 다른 난수 stream으로 iid 추출했다. Hidden cell 수를 강제로 균형화하지 않았다. 각 평가 world는 32회 추가 observation noise를 두 action에 대해 갖는다. Primary는 24개 cohort, 3,072개 평가 worlds이며 controls까지 총 33개 cohort, 4,224개 평가 worlds다. 같은 replicate index의 numerical bank pair를 환경 사이에 공유해 비교를 짝지었다. 추가 observation이나 label을 생성하기 전에 정책이 정한 action과 factual inputs를 별도 hash 파일로 동결했다.

실행 전 사용자 allocation 세 장의 process·memory·utilization을 조회했다. 두 장에 다른 Python 작업이 있었고, node55의 RTX PRO 6000 Blackwell Server Edition 한 장(0 MiB, 0%, Slurm 2744468)에서 모든 GPU 실험을 순차적으로 실행했다. 통계와 분석적 simulation은 할당된 compute-node CPU에서 처리했다. [실험 설계](design.json) · [사전 protocol](PROTOCOL.md) · [GPU worker 종료 기록](gpu_worker/results/worker_exit.json)

## 2. 무엇이 식별되고 무엇이 남는가

Factual collision에서는 incoming tangential velocity가 0이고 모든 후보가 stick한다. Local slope를 s, restitution을 e라 두면 outgoing velocity는 (−e/(1+s²), es/(1+s²))가 된다. Friction coefficient가 식에서 빠진다. 넓은/좁은 friction 집단은 등확률이고 geometry와 독립이다. 두 집단 사이에서 restitution 중심을 바꿔도 e의 marginal과 geometry와의 joint law가 유지된다. 따라서 임의 크기의 iid factual batch에 대해 αf의 likelihood는 평평하다. 방향이 반대인 두 환경에 같은 prior probability를 주면, factual-only classifier의 최선의 평균 방향 판별 정확도도 50%다.

Profile에는 다른 신호가 있다. M을 9점 profile mean, T를 quadratic curvature의 least-squares coefficient로 두면 대칭 좌표와 독립 Gaussian noise 아래 E[M(T²−E[T²])] = 0.00115 × (0.340²−0.022²) × αc / 2다. 이 값은 αc가 변하면 달라진다. 즉 관측 marginal 전체가 보존되는 friction과 달리, 눈에 띄기 어려운 profile 내부의 고차 관계는 남아 있다. [물리 식](../within_calibration_v25/physics25.py) · [moment와 독립 power 검산](diagnostics27.py)

추정기는 실제 simulator에서 8,192개 후보로 aligned/swapped profile에 대한 두 factual likelihood를 계산한다. Friction spread를 쌍으로 구성해 αf 반전이 같은 candidate signals의 순열이 되게 했다. 각 calibration world에서 두 값을 αc에 따라 혼합하고, 401개 지점에서 log likelihood를 더해 uniform hyperprior에 대한 posterior mean과 구간을 구한다. αf의 posterior는 처음 정한 uniform hyperprior 그대로이며 mean 0을 사용한다. **αf=0은 실제로 shift가 없다는 측정값이 아니다.** 두 독립 coupling의 one-world predictive mixture가 각 축에 선형이므로 이 설정에서 posterior mean을 사용하면 marginal predictive prior와 일치한다. 이후 K256은 이를 finite bank로 근사한다. 이 방법은 contact law와 sensor calibration을 안다고 가정한다. [Estimator 구현](estimator27.py)

## 3. Held-out Brier와 GPU p90

Frozen gate는 source RBF score로 관측을 직접 고르고 낮은 confidence에서는 K1024로 넘긴다. 고정 K256은 source prior (1,1), neutral K256은 (0,0), known-prior K256은 실제 환경의 (αc,αf)를 사용한다. Known-prior의 특권 정보는 집단의 law까지이며 개별 world의 latent나 label은 숨겼다. Always-query는 source decoder로 query view만 읽는다. Native 비교에서는 각 정책이 배포할 decoder를 사용했다. 공통 decoder 열에서는 모든 정책이 고른 action을 같은 known-prior decoder로 다시 채점했다. 따라서 공통 열의 차이는 action 변화에 한정된다.

| 정책 | Native Brier | 공통 decoder Brier | Probe 선택률 | Warm GPU p50 ms | Warm GPU p90 ms | 128개에 setup 배분한 p90 ms |
|---|---|---|---|---|---|---|
| Estimated-prior K256 | 0.11083 | 0.10581 | 10.2% | 3.256 | 3.343 | 3.769 |
| Frozen gate | 0.19324 | 0.13216 | 27.5% | 0.895 | 3.869 | 4.089 |
| 고정 K256 | 0.19296 | 0.13255 | 27.2% | 3.235 | 3.313 | 3.542 |
| Known-prior K256 | 0.09765 | 0.09765 | 13.7% | 3.255 | 3.335 | 3.653 |
| Always-query | 0.13735 | 0.10855 | 0.0% | 0.727 | 0.737 | 0.957 |
| Neutral K256 | 0.11468 | 0.10855 | 0.0% | 3.262 | 3.340 | 3.653 |

| Estimated에서 뺀 비교 대상¹ | Native Δ [simultaneous 95%] | 공통 decoder Δ [simultaneous 95%] | Estimated와 다른 선택 | Warm p90 ratio [pointwise 95%] |
|---|---|---|---|---|
| Frozen gate | -0.08241 [-0.09858, -0.06625] | -0.02635 [-0.03511, -0.01759] | 29.7% | 0.864 [0.856, 0.871] |
| 고정 K256 | -0.08214 [-0.09826, -0.06601] | -0.02674 [-0.03447, -0.01900] | 29.6% | 1.009 [1.003, 1.017] |
| Known-prior K256 | +0.01318 [+0.00681, +0.01955] | +0.00816 [+0.00196, +0.01437] | 12.7% | 1.002 [0.998, 1.005] |
| Always-query | -0.02653 [-0.03847, -0.01459] | -0.00274 [-0.00918, +0.00371] | 10.2% | 4.535 [4.517, 4.546] |
| Neutral K256 | -0.00385 [-0.01163, +0.00392] | -0.00274 [-0.00918, +0.00371] | 10.2% | 1.001 [0.996, 1.004] |

¹ Δ = estimated loss − 비교 대상 loss다. 8개 target laws에 같은 가중치를 주고, 3개 replicate/bank blocks를 함께 재추출한 뒤 해당 block·law 안의 test worlds를 재추출했다. 방법 간 짝은 유지했다. 10,000회 bootstrap으로 native/common의 총 10개 평균 차이에 대해 studentized scale의 최대 절댓값을 써서 simultaneous bands를 구했다. Timing은 2,000회 paired world/block bootstrap의 pointwise interval이다. 세 bank의 반복으로 모집단 전체에 대한 오차를 정밀하게 측정하기는 어렵다. 각 block을 한 번씩 제외해도 native estimated−source K256은 −0.08628에서 −0.07837, estimated−neutral은 −0.00643에서 −0.00091이었다. 절댓값이 작은 후자에서 simultaneous interval은 0을 포함한다.

Source K256에서 estimated로 바꾸면 action이 29.6% 달라진다. Neutral K256은 primary의 모든 world에서 query view를 골랐다. 그래서 common decoder를 쓰면 neutral과 always-query의 Brier가 정확히 같다. Estimated는 10.2%에서 probe를 사용했다. 이 변경에 고유한 평균 이득은 통계적으로 확정되지 않았다. Known-prior가 고른 action과도 12.7%에서 달랐다. Known-prior K256 역시 근사 계산이므로 Bayes lower bound로 취급할 수 없다.

Timing은 batch-1 raw factual input부터 실제로 고른 observation의 전송과 최종 확률 계산까지 CUDA synchronization을 포함했다. 환경별 무작위 16 worlds × 2 repeats × 24 cohorts로 정책당 768회다. 여섯 정책의 실행 순서는 섞었다. Estimated의 평균 calibration fit은 21.39 ms, 환경별 bank 생성까지 합하면 61.83 ms다. 마지막 열은 각 방법의 환경별 setup을 128개 평가 world에 균등 배분한 계산값이다. Cold static estimator library 생성 229.6 ms는 별도 1회 비용이다. GPU p90에는 실물 sensing, 1,024개의 calibration world 수집, 물체 reset 시간이 들어 있지 않다. Always-query도 동일 engine 생성 비용으로 계상해 setup 추정이 전용 구현보다 클 수 있다. [원시 GPU 실행 및 timing](runner27.py) · [통계 요약](results/summary.json)

## 4. 어느 환경에서 이득이 사라지는가

| (αc, αf) | 용도 | 평균 α̂c | Estimated | Frozen gate | 고정 K256 | Known K256 | Always-query | Neutral K256 |
|---|---|---|---|---|---|---|---|---|
| (-0.8, 1) | target | -0.306 | 0.10687 | 0.23112 | 0.22827 | 0.08392 | 0.13552 | 0.11447 |
| (1, -0.8) | target | +0.543 | 0.09418 | 0.16602 | 0.16704 | 0.07274 | 0.15138 | 0.12139 |
| (-0.65, -0.85) | target | -0.611 | 0.10512 | 0.25494 | 0.25443 | 0.09289 | 0.16059 | 0.10707 |
| (-0.65, 0.85) | target | -0.301 | 0.10754 | 0.20488 | 0.20496 | 0.08833 | 0.11500 | 0.11594 |
| (0.35, -0.65) | target | +0.049 | 0.10787 | 0.16414 | 0.16505 | 0.10017 | 0.12908 | 0.10614 |
| (0.35, 0.65) | target | +0.409 | 0.13236 | 0.12554 | 0.13073 | 0.10973 | 0.11788 | 0.11999 |
| (-0.2, -0.35) | target | -0.161 | 0.11646 | 0.20814 | 0.20995 | 0.11633 | 0.14662 | 0.11698 |
| (-0.2, 0.35) | target | -0.167 | 0.11621 | 0.19114 | 0.18327 | 0.11707 | 0.14277 | 0.11545 |
| (1, 1) | control | +0.576 | 0.10511 | 0.09016 | 0.09308 | 0.09308 | 0.10432 | 0.11516 |
| (0, -1) | control | +0.152 | 0.11478 | 0.22850 | 0.22493 | 0.10173 | 0.16607 | 0.11263 |
| (0, 1) | control | -0.278 | 0.10416 | 0.14348 | 0.13915 | 0.09423 | 0.10548 | 0.10500 |

두 관계가 아직 원래 방향에 남은 (+0.35,+0.65)에서 estimated의 Brier 0.13236은 frozen gate 0.12554와 always-query 0.11788보다 높았다. Source control에서도 frozen gate 0.09016에 비해 estimated 0.10511이었다. 변화가 없는 환경에서도 αf를 중립화하는 대가가 나타난다. 환경별 수치는 독립 calibration batch 세 개의 평균이며, 약한 shift를 모두 검출할 수 있다는 증거로 해석하기 어렵다.

## 5. Batch에서 실제로 알아낸 정보

| Calibration worlds | Profile MAE | Profile 부호 정확도 | 부호 posterior probability >95%인 cohort | 평균 posterior KL (nats) |
|---|---|---|---|---|
| 64 | 0.481 | 62.5% | 0/24 | 0.076 |
| 256 | 0.279 | 87.5% | 1/24 | 0.243 |
| 1024 | 0.299 | 83.3% | 7/24 | 0.484 |

서로 같은 batch의 prefixes를 비교했다. 1,024개에서 profile 부호는 20/24회 맞았고, continuous alternative 대 source (αc=1)의 Bayes factor가 10을 넘은 target cohort는 14/24개였다. 후자는 profile 방향 판별과 다른 지표다. 95% equal-tail posterior interval의 평균 폭은 1.266이었다. αc=1은 연속 uniform prior의 경계값이므로 중앙 equal-tail interval로는 이 endpoint를 포함할 수 없다. 24개 전체를 한데 묶어 명목 coverage를 주장할 수 없다. αf의 posterior information gain은 모든 batch 크기에서 정확히 0이다.

별도 CPU power 실험에서는 αc=−0.8, 0, +0.8마다 새 iid batch 96개를 만들었다. (M,T)의 projected likelihood에 대한 bounded MLE를 사용했다. 이는 GPU 배포 estimator와 독립적인 식별력 검산이다. 아래 power는 αc=0에 대한 nominal 5% two-sided score test가 귀무가설을 기각한 비율이다. 0행은 false positive rate다.

| αc | N | MLE 평균 | SD | 부호 정확도 | αc=0 기각률 |
|---|---|---|---|---|---|
| -0.8 | 1024 | -0.733 | 0.270 | 99.0% | 38.5% |
| -0.8 | 4096 | -0.773 | 0.172 | 100.0% | 97.9% |
| -0.8 | 16384 | -0.796 | 0.093 | 100.0% | 100.0% |
| 0.0 | 1024 | +0.004 | 0.444 | — | 6.2% |
| 0.0 | 4096 | -0.027 | 0.236 | — | 7.3% |
| 0.0 | 16384 | -0.007 | 0.106 | — | 6.2% |
| 0.8 | 1024 | +0.705 | 0.311 | 93.8% | 39.6% |
| 0.8 | 4096 | +0.805 | 0.177 | 100.0% | 97.9% |
| 0.8 | 16384 | +0.793 | 0.104 | 100.0% | 100.0% |

αc의 부호가 맞을 가능성과 0에서 유의하게 떨어졌다고 말할 수 있는 확률은 다르다. 예컨대 +0.8, N=1,024에서 각각 93.8%와 39.6%였다. [전체 표본수·moment·quadrature 검산](results/identifiable_power.json)

Bayes factor >10을 profile-shift 경보로 해석한 탐색적 대조에서, 실제 αc가 1인 source control은 0/3회, friction만 반전한 target (1,−0.8)은 0/3회 경보를 냈다. αc가 source와 다른 나머지 target cohort에서는 14/21회였다. 표본이 작고 Bayes factor는 policy를 켜고 끄는 threshold로 사용하지 않았다. 추정 posterior는 경보 여부와 무관하게 K256의 prior에 반영됐다.

## 6. Friction이 보이지 않는 것이 decision에도 영향을 주는가

반대 αf를 가진 두 환경의 world를 일대일로 짝지었다. Friction spread bit와 coupling flip을 함께 반전하면 같은 restitution과 geometry 및 factual noise를 유지하면서 서로 다른 friction을 만들 수 있다. 네 쌍의 환경에 16,384개씩 생성했을 때 모든 factual input의 최대 차이는 4.45×10⁻¹⁶ 이하였다. 같은 두 world의 query view는 동일했지만 probe response와 query stick label은 달라질 수 있었다. 이는 한 가지 명시적 coupling에서 관찰한 label 변경률이며 임의로 뽑은 두 world 사이 오차율을 뜻하지 않는다.

| 고정 αc | 서로 반대인 αf의 절댓값 | 최대 factual 차이 | 평균 noiseless probe L2 차이 | Query label 변경 |
|---|---|---|---|---|
| -0.65 | 0.85 | 4.44e-16 | 0.2157 | 17.9% |
| 0.35 | 0.65 | 4.44e-16 | 0.2157 | 17.6% |
| -0.2 | 0.35 | 4.44e-16 | 0.2159 | 17.2% |
| 0.0 | 1.0 | 4.44e-16 | 0.2157 | 17.5% |

이어 실제 test의 동일한 x와 동일 numerical seeds에서 두 privileged K256에 서로 반대의 αf를 제공했다. 아래는 각 방향·replicate를 같이 평균한 탐색적 수치다. 오른쪽 열은 실제 환경의 known decoder로 양쪽 action을 채점한 값이다.

| (αc, 반대 방향의 αf) | 평가 worlds | 두 known-prior 선택 불일치 | 반대 αf의 action을 사용한 Brier 증가 |
|---|---|---|---|
| (-0.65, ±0.85) | 768 | 26.2% | +0.01835 |
| (0.35, ±0.65) | 768 | 12.6% | +0.00948 |
| (-0.2, ±0.35) | 768 | 2.0% | -0.00033 |
| (0, ±1) | 768 | 0.9% | +0.00091 |

총 3,072개 test worlds에서 같은 factual input에 대해 known-prior 선택이 10.4% 달랐다. 실제 환경 대신 반대 방향으로 관측을 고른 뒤 실제 환경 decoder로 채점하면 평균 Brier가 +0.00710 변했다. 이는 이 finite K256에서의 측정값이고, 최적 decision regret에 대한 보편적 lower bound는 아니다. [쌍을 이룬 물리 worlds](results/twins.json) · [동일 input의 action 검증](results/twin_actions.json)

## 7. Calibration worlds를 16배로 늘린 보조 실험

(−0.65, ±0.85), (+0.35, ±0.65)의 네 환경만 사용했다. 12개 cohort의 test worlds 1,536개와 bank를 그대로 두고 독립 calibration 16,384개를 새로 생성했다. 따라서 이 수치는 주 실험 8개 환경 평균과 직접 비교할 수 없다. αc만 정확히 알고 αf를 0으로 둔 privileged control도 평가했다.

| 정책 | Native Brier | 같은 known decoder Brier |
|---|---|---|
| Estimated (1,024) | 0.11322 | 0.10617 |
| Estimated (16,384) | 0.10991 | 0.10288 |
| Known profile only | 0.10711 | 0.10105 |
| Neutral | 0.11229 | 0.10408 |
| Known both | 0.09778 | 0.09778 |

| 탐색적 대비 | Native Δ [동시 95%] | 공통 decoder Δ [동시 95%] |
|---|---|---|
| 16,384 − 1,024 | -0.00331 [-0.02117, +0.01454] | -0.00329 [-0.01742, +0.01083] |
| 16,384 − neutral | -0.00238 [-0.01166, +0.00690] | -0.00120 [-0.00944, +0.00704] |
| Known profile − neutral | -0.00518 [-0.01194, +0.00159] | -0.00303 [-0.00972, +0.00366] |
| 16,384 − known profile | +0.00280 [-0.00349, +0.00909] | +0.00183 [-0.00301, +0.00667] |
| Known profile − known both | +0.00933 [+0.00178, +0.01688] | +0.00327 [-0.00260, +0.00914] |

Profile MAE는 0.308에서 0.088로 줄었다. 큰 batch fit에는 평균 178.1 ms가 들었다. 같은 test에서 native Brier도 0.11322에서 0.10991로 내려갔으나 paired interval은 0을 포함했다. Profile을 정확히 알려 줘도 αf까지 아는 comparator보다 native loss가 +0.00933 높았다. 알려진 family 안에서 한 축을 더 잘 식별하는 일과 전체 decision loss를 낮추는 일을 별도로 검증해야 한다. [보조 실험 코드](context27.py) · [같은 test의 비교 결과](results/extended.json)

![Native and common-decoder risk, GPU latency and context adaptation](results/v27.png)

## 8. 논문 기여에 대한 판단

**현재 자료로 주장할 수 있는 기여는 어떤 population shift가 관측 선택에 필요하면서도 factual evidence에서 보이지 않는지 실측한 것이다.** Profile은 고차 관계를 통해 약하게 보인다. Friction은 대칭성 때문에 완전히 가려진다. 실제로 동일한 factual input을 받는 known-prior K256 두 개에서 action이 달라졌고, 반대 환경의 action에는 양의 평균 비용이 있었다. Source-only 정책과 비교해 selection 자체가 개선된다는 점도 공통 decoder 대조에서 확인했다.

**새 적응 방법이 neutral policy보다 낫다는 주장은 보류한다.** Neutral 대비 Brier interval이 0을 포함하고, 원 배포 환경에서는 적응에 비용이 있었다. 또한 오랜 관측이나 다른 instance를 사용해 다음 design을 바꾸는 방법과 decision-aware experimental design은 이미 연구됐다. 이 결과는 하나의 analytic simulator, 한 sensor calibration, 두 actions, 알려진 parametric shift family에 한정된다. [Huang et al.](https://proceedings.neurips.cc/paper_files/paper/2024/file/c59f05d7ab3638b138cc61f32e1a7cd1-Paper-Conference.pdf) · [Zhao et al.](https://arxiv.org/html/2611.11512v1)

**다음으로 판별력이 높은 실험은 과거 world 하나에 유료 calibration probe를 허용하는 것이다.** 그 관측이 현재 world에는 주어지지 않도록 나눈 뒤, probe 0·1·4·16회에서 αf posterior와 새로운 world의 Brier를 측정한다. 같은 sensing budget의 random calibration, neutral K256, known-prior K256을 포함하고, unseen contact law에서 반복한다. 추가 측정으로 αf의 방향과 neutral 대비 selection 이득을 함께 회복하는지, probe 시간을 상쇄할 만큼 환경이 오래 유지되는지를 별도 평가해야 한다. 이것이 입증되면 식별 한계 진단을 넘어 구체적인 observation policy 기여로 발전할 여지가 있다.

## 선행연구와 논문 기여의 경계

### 1. 논문별 근거 비교

아래 다섯 원문을 대상으로 했다. 서로 다른 population에서 얻은 수치를 합산하지 않았다. 논문에서 직접 검증한 결과는 표에, v27에 적용하며 추가한 가정과 추론은 이어지는 절에 제시한다.

| 연구 | Research question · methodology | Sample · data source | 주요 결과 · 결론 | v27으로 옮길 때의 한계 |
|---|---|---|---|---|
| Lipton, Wang & Smola, *Detecting and Correcting for Label Shift with Black Box Predictors* (ICML 2018) | Source의 labeled confusion matrix와 target의 unlabeled predicted frequencies로 class proportions를 복구할 수 있는가? BBSE의 consistency와 error bound를 증명하고 BBSD의 detection power를 측정했다. | MNIST detection에서는 training·validation·test를 각 20,000장으로 구성해 digit 5의 비율을 조작했다. | Stable class-conditional law, source support와 invertible confusion matrix 아래에서는 biased classifier로도 추정할 수 있다. 실험에서는 분류기가 제공하는 정보에 따라 detection power가 달랐다. | V27의 mixture component는 물리적 coupling의 flip 상태다. Stick outcome의 prevalence만 바뀐다는 모형으로 치환할 근거는 없다. [원문 §§3–6](https://proceedings.mlr.press/v80/lipton18a/lipton18a.pdf) |
| Garg, Wu, Balakrishnan & Lipton, *A Unified View of Label Shift Estimation* (NeurIPS 2020) | MLLS와 moment matching의 consistency 조건과 finite-sample 오차는 어떻게 다른가? 이론과 비교 실험으로 calibration의 영향을 분해했다. | GMM, MNIST, CIFAR-10. Fig. 1의 shift-strength 비교에서 target samples는 GMM 1,000개, image tasks 각각 5,000개였다. | 시험한 조건에서 MLLS의 shift-weight MSE가 낮았다. Confusion matrix 수준으로 calibration을 coarse-grain하면 차이가 작아졌다. 충분한 class separation과 calibration의 역할을 각각 분석했다. | Weight MSE는 decision loss가 아니다. V27에서는 정확히 분리되지 않는 물리 축을 따로 판정해야 한다. [원문 §§4–6, Fig. 1](https://proceedings.neurips.cc/paper/2020/file/219e052492f4008818b8adb6366c7ed6-Paper.pdf) |
| Go & Isaac, *Robust Expected Information Gain for Optimal Bayesian Experimental Design Using Ambiguity Sets* (UAI 2022) | Prior가 부정확할 때 어느 design을 선택해야 하는가? KL ambiguity set과 affine relaxation을 이용해 robust EIG를 계산했다. | 두 binary tests의 분석 예시와 A/B, preference, pharmacokinetic 세 simulation benchmarks. Fig. 3의 일부 비교는 posterior samples 30/100/1,000개를 사용했다. | Binary 예시에서는 prior probability가 0.5인 지점을 지나면 EIG 순위가 뒤집혔다. Sampling과 ambiguity radius의 설정에 따라 design score의 안정성이 달라졌다. | 목적은 EIG다. 이 결과는 unlabeled shift recovery나 실제 stick prediction의 Brier 개선을 보장하지 않는다. [원문 §§3, 7](https://proceedings.mlr.press/v180/go22a/go22a.pdf) |
| Huang, Guo, Acerbi & Kaski, *Amortized Bayesian Experimental Design for Decision-Making* (NeurIPS 2024) | 다음 design과 downstream decision을 함께 amortize하면 실현 utility를 개선할 수 있는가? Transformer Neural Decision Process로 두 작업을 학습했다. | Hyperparameter optimization의 네 meta-datasets. Retrosynthesis planning 사례에서는 training 1,500 molecules × 70 routes, 평가 50 molecules였다. | 네 meta-datasets의 초기 10 queries에서 utility gain이 컸고, retrosynthesis 사례에서는 random search보다 높은 top-3 utility를 보고했다. | Query head와 inference head를 함께 학습한 결과다. V27에서 얻은 개선을 acquisition에 귀속하려면 decoder를 고정한 비교가 필요하다. [원문 §§5–7, Appendix G.3](https://proceedings.neurips.cc/paper_files/paper/2024/file/c59f05d7ab3638b138cc61f32e1a7cd1-Paper-Conference.pdf) |
| Zhao, Huang, Wen & Kaski, *Amortized Bayesian Experimental Design with In-Context Knowledge Conditioning* (2026년 11월 8일 공개본) | 과거 experiment instances와 외부 prior가 현재 design에 유용한지 판단할 수 있는가? IMBUE는 previous-instance tokens를 Transformer로 처리하고 learned reliability filter를 사용한다. | Location Finding, CES, GP active learning, psychometric model. CES의 Table 11은 5개 과거 instances, 10 rounds, 2,000 rollouts다. | CES의 Gaussian-noise 조건에서 relevant instances의 sPCE는 13.93, 추가 instance 없이 13.46, irrelevant instances에서 13.14였다. Deployment에서 과거 관측을 사용하는 선례다. | Relevant instance의 parameter를 현재 parameter 주변에서 생성한다. V27의 구별점은 독립적으로 추출한 world들이 공유하는 두 population coupling과 one-step stick decision을 다루는 데 있다. [원문 §§3–4, Appendix E.6](https://arxiv.org/html/2611.11512v1) |

### 2–3. Literature review와 knowledge map

**Unlabeled evidence에서 무엇을 복구할 수 있는가.** BBSE는 component별 관측 특성이 구분될 때 target mixture proportions를 추정한다. MLLS 연구는 관측을 coarse summary로 줄일 때 정보가 손실될 수 있음을 보였다. 이 결과를 v27에 적용하려면 환경마다 유지되는 것이 무엇인지 명시해야 한다. V27의 가정은 simulator가 제공하는 flip별 factual likelihood이고, 추정 대상은 두 flip probabilities다. 이 구성은 synthetic component model을 사용할 수 있는 조건에서만 의미가 있다. [Lipton et al.](https://proceedings.mlr.press/v80/lipton18a/lipton18a.pdf) · [Garg et al.](https://proceedings.neurips.cc/paper/2020/file/219e052492f4008818b8adb6366c7ed6-Paper.pdf)

**추정과 의사결정은 언제 연결되는가.** Go & Isaac은 prior가 design ranking을 바꿀 수 있음을 보였고 Huang 등은 downstream utility에 직접 맞춰 design을 학습했다. Zhao 등은 이전 instances에서 얻은 관측이 유용한지에 따라 실현 정보량이 달라짐을 보였다. 따라서 v27의 knowledge map은 **population coupling → factual joint distribution → coupling posterior → query/probe 선택 → 선택한 관측으로 갱신한 stick probability → held-out Brier**다. Setup latency와 online p90은 Brier와 별도의 측정 경로다. 화살표들은 v27의 평가 구조이며 선행연구가 이 전체 경로를 실제 물체에서 검증했다는 뜻은 아니다. [Go & Isaac](https://proceedings.mlr.press/v180/go22a/go22a.pdf) · [Huang et al.](https://proceedings.neurips.cc/paper_files/paper/2024/file/c59f05d7ab3638b138cc61f32e1a7cd1-Paper-Conference.pdf) · [Zhao et al.](https://arxiv.org/html/2611.11512v1)

### 4. Research gaps와 v27에서의 식별 조건

두 proxy를 모두 복구한다는 주장은 이 simulator에서 성립하지 않는다. Factual impact는 중심 위치, incoming tangential velocity 0, stick regime이다. Slope를 s, restitution을 e라 두면 noiseless outgoing velocity는 (−e, es)/(1+s²)다. Friction coefficient가 식에서 사라진다. 두 friction-spread 집단이 같은 비율이고 딴 물성으로부터 독립이면 coupling을 뒤집어도 marginal restitution과 factual vector의 population law가 보존된다. 따라서 같은 αc, 다른 αf를 갖는 환경의 **i.i.d. factual batch likelihood가 같다**. αf에 대한 posterior가 특정 값에 강하게 집중하면 finite-bank artifact, coupling에 관한 hyperprior 또는 조작되지 않은 다른 정보원을 검사해야 한다. 이 명제는 동일 latent group 수를 강제한 채 group membership을 공개하거나 특정 material을 추적하는 관측 설계까지 포괄하지 않는다. [V25 factual generator](../within_calibration_v25/physics25.py) · [impact 식](../factual_tail_v23/physics23.py) · [V26 coupling 조작](../proxy_shift_v26/physics26.py)

Profile 축에는 반대 방향의 분석 근거가 있다. 각 world의 offset O와 signed curvature C는 9개 profile 위치에서 quadratic regression으로 추정할 수 있다. 같은 분산의 zero-mean Gaussian measurement error와 고정된 위치에서 coefficient errors는 joint zero-mean Gaussian이다. Population에서 E[O]=E[C]=0이므로 E[Ô Ĉ²]=E[O C²]가 된다. V25 parameterization에서 이 moment는 **0.00115 × αc × (0.340²−0.022²)/2**다. 두 curvature 집단이 같은 확률이고 within-cell jitter의 분산이 같다. 따라서 full profile에는 αc를 population 수준에서 식별할 정보가 있다. 드문드문 모은 noisy worlds로 이 작은 moment를 정밀하게 추정할 수 있는지는 empirical question이다. 이 유도는 v27 simulator의 성질이며 선행 논문에서 보고한 식이 아니다. [V25 parameterization](../within_calibration_v25/physics25.py) · [quadratic profile](../factual_tail_v23/physics23.py) · [V26 center swap](../proxy_shift_v26/physics26.py)

### 5–7. 상충하는 결과, methodology comparison, evidence synthesis

| 비교 | 달라지는 측정·가정 | 증거로부터 가능한 판단 |
|---|---|---|
| BBSE는 uncalibrated predictor로도 consistent할 수 있다. MLLS 실험에서는 granular calibration이 MSE를 낮췄다. | 전자는 invertible moment equation의 asymptotic 조건, 후자는 finite target sample에서 추정 효율을 비교한다. | 같은 질문에 대한 모순으로 볼 근거가 없다. V27의 구조적 identifiability와 유한 world 수의 추정 정확도는 따로 측정해야 한다. [Lipton](https://proceedings.mlr.press/v80/lipton18a/lipton18a.pdf) · [Garg](https://proceedings.neurips.cc/paper/2020/file/219e052492f4008818b8adb6366c7ed6-Paper.pdf) |
| Robust EIG, downstream utility, previous-instance sPCE 모두 observation selection을 다룬다. | Target·평가 단위·목적 함수가 각각 다르다. EIG는 parameter information, Brier는 probabilistic outcome prediction을 채점한다. | 추정된 α의 오차와 Brier의 상관관계만으로 선택이 개선됐다는 인과 주장을 할 수 없다. 같은 held-out world에서 두 실제 잠재 관측을 평가하고 decoder를 고정해 선택의 차이를 검사한다. [Go & Isaac](https://proceedings.mlr.press/v180/go22a/go22a.pdf) · [Huang](https://proceedings.neurips.cc/paper_files/paper/2024/file/c59f05d7ab3638b138cc61f32e1a7cd1-Paper-Conference.pdf) |
| Previous instances가 어떤 조건에서는 도움이 되고 다른 조건에서는 성과가 나빠진다. | IMBUE는 현재 parameter와 가깝게 뽑은 과거 instances 및 reliability supervision을 사용한다. V27의 calibration worlds는 environment를 공유하고 개별 정답은 공개하지 않는다. | Target environment를 공유한다는 사실만으로 모든 물성을 식별할 수 없다. Factual law가 같고 query value가 다른 두 환경은 직접적인 negative control이 된다. [Zhao](https://arxiv.org/html/2611.11512v1) · [V26 simulator](../proxy_shift_v26/physics26.py) |

**근거의 강도.** 주어진 component laws의 분리 가능성이 mixture recovery를 좌우한다는 근거는 이론적이다. 가까운 previous instances가 design을 개선한다는 근거는 해당 benchmark에 한정된 실험적 근거다. V27의 현장 로봇에서 얻을 수 있는 Brier 개선량, 필요한 독립 물체 수, 실제 probe 수행 시간에는 이 다섯 연구에서 직접 대입할 수 있는 수치가 없다. [Garg et al.](https://proceedings.neurips.cc/paper/2020/file/219e052492f4008818b8adb6366c7ed6-Paper.pdf) · [Zhao et al.](https://arxiv.org/html/2611.11512v1) · [Huang et al.](https://proceedings.neurips.cc/paper_files/paper/2024/file/c59f05d7ab3638b138cc61f32e1a7cd1-Paper-Conference.pdf)

### 8. 후속 Research questions 10개

| 질문 · 왜 필요한가 | 변수와 검증 방법 | 근거 |
|---|---|---|
| 1. 몇 개의 calibration worlds부터 profile shift 방향이 재현되는가? Finite-sample detection 한계를 정한다. | 독립 batches의 크기를 바꾸고 sign error, interval coverage, paired Brier를 측정한다. | [Garg, Fig. 1](https://proceedings.neurips.cc/paper/2020/file/219e052492f4008818b8adb6366c7ed6-Paper.pdf) |
| 2. Geometry moment만으로 full likelihood를 대체할 수 있는가? 저렴한 식별 통계의 실용성을 본다. | Ô Ĉ²와 full profile로 각각 αc를 추정해 같은 worlds의 error, GPU p90, setup을 비교한다. | [Garg, §5](https://proceedings.neurips.cc/paper/2020/file/219e052492f4008818b8adb6366c7ed6-Paper.pdf) · [profile 식](../factual_tail_v23/physics23.py) |
| 3. 동일 factual law 아래 αf 변화가 실제 선택을 얼마나 바꾸는가? 식별 불가가 결정에도 중요한지 판정한다. | αc를 고정한 두 환경에서 known-prior choices, 공통 decoder의 Brier와 action gap을 비교한다. | [Go & Isaac, §3](https://proceedings.mlr.press/v180/go22a/go22a.pdf) |
| 4. 약한 tangential excitation 한 번이면 friction의 비식별성이 풀리는가? 유료 calibration의 가치가 있다. | Factual incoming velocity와 noise를 바꾸고 slip rate, αf likelihood separation, Brier를 비교한다. | [impact 식](../factual_tail_v23/physics23.py) · [Go & Isaac](https://proceedings.mlr.press/v180/go22a/go22a.pdf) |
| 5. Bank discretization이 friction shift를 검출한 것처럼 보이게 하는가? Numerical artifact를 분리한다. | Scrambled bank size·seed를 바꾸고 완전히 대칭화한 likelihood에서의 null과 비교한다. | [Go & Isaac, §7](https://proceedings.mlr.press/v180/go22a/go22a.pdf) · [V26 simulator](../proxy_shift_v26/physics26.py) |
| 6. 같은 batch 안에 서로 다른 환경이 섞이면 어디서 손해가 발생하는가? Homogeneous-context 가정의 비용이다. | Contamination fraction·batch age를 조작해 context posterior, query/probe action, Brier를 비교한다. | [Zhao, §§4.3–4.5](https://arxiv.org/html/2611.11512v1) |
| 7. Unidentified 축을 평균내는 선택과 worst-case 선택은 어떻게 다른가? 추정할 수 없는 물성의 대응을 결정한다. | αf hyperprior·ambiguity radius를 고정한 뒤 mean Brier, worst-environment Brier, action fractions를 평가한다. | [Go & Isaac, §§4–5](https://proceedings.mlr.press/v180/go22a/go22a.pdf) |
| 8. Prior adaptation의 효과가 selection과 decoding 중 어디에 집중되는가? Method claim의 귀속 근거다. | Source/estimated selector와 두 decoders를 교차하고 각 효과를 paired worlds로 추정한다. | [Huang, §§3–5](https://proceedings.neurips.cc/paper_files/paper/2024/file/c59f05d7ab3638b138cc61f32e1a7cd1-Paper-Conference.pdf) |
| 9. Deployment에서 환경이 얼마나 유지돼야 calibration이 경제적인가? 빠른 online p90이 setup 비용을 가릴 수 있다. | Environment당 처리 world 수와 reset 빈도를 바꿔 cold, warm latency와 sensor 시간을 별도로 측정한다. | [Zhao, Appendix E.8](https://arxiv.org/html/2611.11512v1) |
| 10. 다른 contact law나 실제 표면에서도 추정한 profile coupling이 유효한가? 알려진 component family 바깥의 일반화 문제다. | 표면·object identity 기준 holdout과 독립 contact model에서 likelihood misspecification, Brier, calibration을 평가한다. | [Lipton, §3](https://proceedings.mlr.press/v80/lipton18a/lipton18a.pdf) · [Zhao, Appendix E.6](https://arxiv.org/html/2611.11512v1) |

### 9. Research assistant review

이 문헌에서 이미 확립한 내용에는 unlabeled target batch로 mixture weight를 추정하는 방법, 과거 관측을 반영해 acquisition을 바꾸는 방법, downstream decision utility에 맞춘 design 학습이 있다. V27이 주장할 수 있는 차별점은 **동일한 factual contact에서 한 coupling은 식별 가능하고 다른 coupling은 식별 불가능한 이유를 물리적으로 유도한 뒤, 남은 정보만으로 회복할 수 있는 선택과 Brier의 범위를 측정하는 것**이다. Known-prior K256도 finite bank approximation을 사용하므로 exact Bayes optimum으로 부를 수 없다. Context를 공유하는 worlds는 같은 adaptation error를 물려받는다. 한 calibration batch 아래의 여러 world나 반복된 observation noise를 각각 독립 환경으로 세면 uncertainty가 지나치게 좁아질 수 있다. 여러 배치의 독립 replication, common-decoder 비교 및 setup을 포함한 timing이 이 주장의 근거다. [Lipton](https://proceedings.mlr.press/v80/lipton18a/lipton18a.pdf) · [Go & Isaac](https://proceedings.mlr.press/v180/go22a/go22a.pdf) · [Huang](https://proceedings.neurips.cc/paper_files/paper/2024/file/c59f05d7ab3638b138cc61f32e1a7cd1-Paper-Conference.pdf) · [Zhao](https://arxiv.org/html/2611.11512v1)


## 재현 자료

서버 프로젝트는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/population_shift_v27`이다. GPU guard를 이용해 runner27.py, context27.py, twin_actions27.py를 각각 별도 worker directory에서 같은 GPU에 순차 실행했다. CPU에서 twins27.py, analyze27.py, diagnostics27.py, extended_stats27.py, report27.py를 실행했다. 각 cohort에 calibration/test factual inputs, 동결된 선택의 SHA-256, 두 potential observations에 대한 예측값·Brier, timing samples, numerical seeds를 보존했다. [사전 설정](design.json) · [paired bootstrap](analyze27.py) · [동일 input 검산](twin_actions27.py) · [large-context 통계](extended_stats27.py) · [primary 수치](results/summary.json) · [추가 GPU 종료 기록](context_worker/results/worker_exit.json) · [동일 input GPU 종료 기록](twin_worker/results/worker_exit.json)
