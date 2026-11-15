# v20 · Factual evidence로 observation을 바꿀 world를 고를 수 있는가

**Factual evidence로 probe를 다른 view로 바꾸면 손해가 큰 집단은 구분했다. 실제로 바꿔서 이득을 얻는 집단은 이번 held-out test에서 확인하지 못했다.** Validation이 정한 하위 score 구간에서는 ridge가 제안한 view를 사용하면 Brier loss가 평균 0.03919 증가했다. 상위 구간에서는 평균 0.00363 감소했지만 95% CI가 0을 포함했다. 상위 구간의 평균 예측 이득은 0.02174로 실제 이득을 과대평가했다. 여기서 집단은 factual evidence로 정의하며, hidden geometry나 실제 intervention outcome으로 고른 집단이 아니다.

새로운 1,024 matched worlds에서 routed policy는 756회 selector를 호출하고 448번 실제 observation을 바꿨다. Mean Brier loss는 routed 0.074226, always-probe 0.074164, fixed global mixture 0.071584였다. Routed minus fixed global은 +0.002642였으며 두 primary comparisons에 대해 보정한 97.5% CI는 [+0.000991, +0.004453]이었다. GPU p90은 각각 4.91, 1.70, 7.06 ms였다. Fixed global보다 짧은 latency를 얻는 대가로 loss가 증가했고, probe와 비교하면 시간만 늘고 loss 개선은 확인되지 않았다.

**독립적으로 바꾼 세 요인 중 matched 조건에서 가장 분명한 차이는 label probability의 출처였다.** 다른 두 축을 균등 평균하면 compact에서 independent full bank로 label source를 바꿀 때 선택 후 Brier loss가 0.001513 감소했다. Secondary 95% CI는 [−0.002665, −0.000403]이다. Observation 생성 law만 바꾼 차이는 −0.000115, sampled label을 conditional mean으로 바꾼 차이는 −0.000306으로 두 CI 모두 0을 포함했다. Shifted contact law에서는 label-source 효과도 −0.000597 [−0.001701, +0.000513]로 불확실했다. 한 seed와 M512에서 여러 exploratory contrasts를 비교했으므로 이 결과는 replication이 필요한 단서다.

**논문 판단.** 이번 결과는 deployable switch policy가 두 baseline을 이긴다는 주장을 지지하지 않는다. 논문으로 발전시킬 만한 부분은 (1) 동일한 hypothetical observation에서 predictive marginal과 conditional label law를 분리해 어떤 근사가 실제 선택에 손실을 더하는지 측정한 실험, (2) factual evidence의 harm 예측과 profitable switch 예측이 어떻게 다른지 보여준 검증이다. 다음 우선순위는 여러 bank seeds와 unseen contact laws에서 label-source 효과를 재현하고, full external label posterior의 GPU 비용을 줄였을 때 fixed global과 같은 latency에서 이득이 남는지 확인하는 것이다.


## 무엇을 독립적으로 바꿨는가

Factual evidence는 경계 주변에서 얻은 9개 boundary coordinates와 충돌 후 속도 2개다. 같은 evidence를 설명하는 scene hypotheses로부터 query 지점에서 stick이 일어날 확률을 예측한다. 연구자가 추가로 고를 수 있는 observation은 query view, factual view, probe 세 가지다. Probe를 기준으로 다른 view를 선택했을 때의 이득을 **probe의 Brier loss에서 그 view의 Brier loss를 뺀 값**으로 정의했다. 양수이면 바꾼 편이 낫다.

C는 replica당 256개 hypotheses를 resampling한 compact posterior mixture, E는 독립적인 16,384-hypothesis external posterior bank다. 첫 축은 hypothetical observation Y의 marginal predictive law다. 두 번째 축은 같은 Y를 주었을 때 binary label의 조건부 확률 p를 제공하는 source다. 두 축에 C 또는 E를 적용하므로 CC·EE는 일관된 joint distribution, CE·EC는 의도적으로 교차한 distribution이다. 세 번째 축은 p로 Bernoulli label을 하나 뽑아 점수를 계산하는 sample 방식과 두 label을 해석적으로 적분하는 mean 방식이다.

모든 cell은 같은 compact decoder의 출력 q를 채점한다. Sample 방식은 (Z−q)², mean 방식은 p(1−q)²+(1−p)q²를 사용한다. Source가 C이면 p=q여서 mean score는 q(1−q)가 된다. E이면 매 hypothetical Y에 대해 external posterior의 Gaussian likelihood update로 p를 계산한다. E에서 observation만 뽑는 비용과 E에서 conditional probability까지 계산하는 비용이 다르다. 세 action이 outer random coordinates를 공유하고 sampled cells는 공통의 독립 IID uniforms를 사용했다. V19의 sampled label은 generating component에서 읽었지만 이번에는 지정한 source의 p로부터 다시 뽑아 교차 조작을 가능하게 했다.

각 replica에 4 scrambles × 128 outer samples, 독립적인 compact representatives 두 벌을 썼다. V18 fixed global mixture는 기존의 K128/M4096 direct kernel로 다시 실행했다. 새 여덟 cell은 M512이므로 factorial 내부에서 한 축을 바꾼 비교가 그 요인의 효과다. Fixed global과 비교하면 budget·compression 세부 구현·scoring이 함께 달라진다. 모든 방법은 선택 후 같은 full bank decoder로 실제 synthetic observation을 처리한다.


## 어떤 world에서 바꿀지 예측하는 방법

Train 1,024 worlds에서는 각 world에 세 observation을 모두 가한 simulated potential outcomes를 16개 noise repeats로 평가했다. 각 view의 target은 해당 world에서 probe를 사용했을 때와의 평균 Brier 차이다. 실제 적용 때 입력은 observation 전 factual evidence, 이 evidence로 계산한 query context, posterior stick probability, ESS, contact-law masses, 여섯 continuous parameters의 posterior mean과 SD뿐이다. Hidden world parameter, actual intervention 결과, unselected observation은 inference 시 입력에 들어가지 않는다.

Ridge regression은 public features와 제곱항으로 두 view의 gain을 추정한다. Train 통계로 표준화한 값을 ±8에 제한하고 penalty를 0.05×N으로 고정했다. 두 예측 중 큰 값을 gate score로 사용한다. Direct는 score가 threshold를 넘으면 예측한 view로 바로 바꾼다. Routed는 같은 score로 factorial selector를 호출할지 정하고, 호출한 selector가 probe를 다시 선택할 수도 있다. 나머지 worlds에서는 probe를 수행한다.

Validation 384개에서 후보 threshold와 selector 조합의 mean Brier loss를 비교했다. Direct는 모든 world에 probe를 쓰는 규칙을, routed는 E-C-mean과 validation score 하위 quartile 경계를 선택했다. Routed의 threshold는 −0.022512이며 validation에서 288/384번 호출했다. Validation은 threshold와 selector를 고른 자료이므로 성능 평가는 새 worlds에서 진행했다. Gate를 먼저 고정하고 matched와 shifted 조건의 test actions를 모두 저장한 후 private test outcomes를 열었다.

Held-out 1,024 worlds 각각에 실제 noise를 16번 반복했다. Shifted condition은 같은 worlds와 noise에서 flexible contact law만 바꾸며 inference는 nominal로 유지했다. Primary endpoint는 matched stick에서 routed minus baseline의 paired mean loss difference다. 두 baseline에 대한 family-adjusted 97.5% CI를 world 단위 bootstrap 20,000회로 계산했다. 음수면 routed가 유리하다. Secondary 95% intervals는 exploratory 비교에 사용했다.


## Held-out Brier loss와 online compute

| Policy 또는 Y–label–mode | Matched Brier ↓ | Shifted Brier ↓ | Matched query/factual/probe | GPU mean / p50 / p90 ms |
|---|---|---|---|---|
| probe | 0.074164 | 0.069624 | 0 / 0 / 1024 | 1.66 / 1.66 / 1.70 |
| fixed_global | 0.071584 | 0.069699 | 465 / 14 / 545 | 6.96 / 6.99 / 7.06 |
| direct | 0.074164 | 0.069624 | 0 / 0 / 1024 | 2.29 / 2.29 / 2.34 |
| routed | 0.074226 | 0.070552 | 353 / 95 / 576 | 4.26 / 4.82 / 4.91 |
| C-C-sample | 0.074234 | 0.071129 | 442 / 143 / 439 | 2.70 / 2.70 / 2.77 |
| C-C-mean | 0.073394 | 0.070013 | 373 / 79 / 572 | 2.74 / 2.74 / 2.80 |
| C-E-sample | 0.072565 | 0.070348 | 470 / 119 / 435 | 12.84 / 12.87 / 13.27 |
| C-E-mean | 0.071923 | 0.069737 | 275 / 85 / 664 | 12.86 / 12.86 / 13.32 |
| E-C-sample | 0.073263 | 0.070404 | 415 / 141 / 468 | 4.09 / 4.11 / 4.17 |
| E-C-mean | 0.074022 | 0.070514 | 387 / 100 / 537 | 4.13 / 4.14 / 4.20 |
| E-E-sample | 0.072437 | 0.069703 | 467 / 91 / 466 | 8.55 / 8.55 / 8.77 |
| E-E-mean | 0.071936 | 0.069885 | 260 / 88 / 676 | 8.59 / 8.61 / 8.83 |

| Routed minus comparator | Matched 97.5% CI | Shifted 95% CI |
|---|---|---|
| probe | +0.000062 [-0.002484, +0.002594] | +0.000927 [-0.001210, +0.003126] |
| fixed_global | +0.002642 [+0.000991, +0.004453] | +0.000852 [-0.000873, +0.002580] |

GPU timing은 RTX PRO 6000 Blackwell에서 held-out 첫 32 matched worlds를 3회씩 측정했다. 순서를 섞고 batch 1·float64에서 GPU synchronization을 time boundary에 적용했다. Factual fit, 필요한 full/compact acquisition, gate feature의 CPU 전송·ridge prediction, 선택된 observation update를 포함한다. Static bank 및 QMC plan 생성, offline simulator supervision·ridge fitting, 실제 sensing/reset 시간은 제외했다. 표의 비용은 online per-world 비용이다. Timing에서 산출한 모든 action은 사전에 고정한 actions와 일치했다.

![Held-out Brier loss, GPU latency and gain calibration](results/v20_results.png)

## Factorial contrasts

각 행은 8개 cell의 **선택 후 실제 held-out loss**에서 계산했다. Main effect는 다른 축을 균등 평균하고 E−C 또는 mean−sample을 취했다. Interaction은 대응하는 차이의 차이다. 음수는 해당 방향으로 바꿀 때 loss가 낮아지는 경향이다. 한 bank와 한 acquisition randomization에 조건부인 secondary intervals다.

| Factor contrast | Matched paired 95% CI | Shifted paired 95% CI |
|---|---|---|
| Y law E-C | -0.000115 [-0.001192, +0.001062] | -0.000180 [-0.001241, +0.000985] |
| label source E-C | -0.001513 [-0.002665, -0.000403] | -0.000597 [-0.001701, +0.000513] |
| Y law E-C x label source E-C | +0.000114 [-0.001339, +0.001541] | -0.000137 [-0.001529, +0.001229] |
| mean-sample | -0.000306 [-0.001253, +0.000556] | -0.000359 [-0.001286, +0.000487] |
| Y law E-C x mean-sample | +0.000870 [-0.000756, +0.002634] | +0.001010 [-0.000485, +0.002698] |
| label source E-C x mean-sample | -0.000532 [-0.001843, +0.000729] | +0.000289 [-0.001041, +0.001593] |
| Y law E-C x label source E-C x mean-sample | -0.001459 [-0.004029, +0.001147] | -0.000433 [-0.003126, +0.002379] |

| 한 축만 바꾼 transition | Matched 선택 변경 /1024 | Shifted 선택 변경 /1024 |
|---|---|---|
| C-C-sample -> E-C-sample | 283 | 278 |
| C-C-sample -> C-E-sample | 297 | 293 |
| C-C-sample -> C-C-mean | 237 | 236 |
| C-C-mean -> E-C-mean | 173 | 172 |
| C-C-mean -> C-E-mean | 384 | 412 |
| C-E-sample -> E-E-sample | 269 | 268 |
| C-E-sample -> C-E-mean | 373 | 413 |
| C-E-mean -> E-E-mean | 333 | 364 |
| E-C-sample -> E-E-sample | 256 | 263 |
| E-C-sample -> E-C-mean | 224 | 210 |
| E-C-mean -> E-E-mean | 381 | 411 |
| E-E-sample -> E-E-mean | 394 | 441 |

## Factual score가 예측한 switch value

Validation score quartile 경계를 test에 그대로 적용해 각 구간의 gain을 측정했다. Direct-view gain은 gate threshold를 적용하기 전 ridge가 제안한 view를 항상 택할 때의 이득이다. Selector gain은 선택된 E-C-mean을 그 구간 모두에서 호출할 때의 이득이다. 배포한 routed policy는 gate를 통과한 world에만 이 selector를 쓴다. 아래 구간별 값은 subgroup 평균으로, 한 world의 intervention 성공을 보장하지 않는다.

| 조건 / score 구간 | n | 평균 predicted gain | 실측 direct-view gain [95% CI] | 실측 selector gain [95% CI] | 실제 switches | Median posterior minority mass |
|---|---|---|---|---|---|---|
| matched Q1 | 268 | -0.052319 | -0.039190 [-0.058676, -0.019556] | +0.000777 [-0.002985, +0.004666] | 0 | 0.23703 |
| matched Q2 | 268 | -0.011699 | -0.008255 [-0.019358, +0.002062] | -0.002824 [-0.008801, +0.003040] | 129 | 0.05039 |
| matched Q3 | 244 | +0.001788 | +0.001942 [-0.006943, +0.011121] | +0.000790 [-0.004211, +0.005790] | 146 | 0.01636 |
| matched Q4 | 244 | +0.021739 | +0.003632 [-0.001796, +0.009718] | +0.002050 [-0.002381, +0.006946] | 173 | 0.00749 |
| shifted Q1 | 255 | -0.052696 | -0.029265 [-0.049293, -0.009694] | +0.000150 [-0.004575, +0.004690] | 0 | 0.23204 |
| shifted Q2 | 253 | -0.011232 | -0.011523 [-0.022373, -0.001506] | -0.003106 [-0.008666, +0.001978] | 122 | 0.01553 |
| shifted Q3 | 268 | +0.001799 | +0.002075 [-0.006113, +0.010615] | -0.001800 [-0.007323, +0.003168] | 162 | 0.00952 |
| shifted Q4 | 248 | +0.021775 | +0.003141 [-0.001958, +0.009124] | +0.001285 [-0.002852, +0.005870] | 183 | 0.00370 |

Matched에서 routed selector를 756/1024번 호출하고 실제로 448번 probe를 다른 observation으로 바꿨다. 변경한 worlds의 paired benefit는 -0.000142 [-0.005322, +0.004986]였다. Factual score로 ridge 제안 view의 realized gain이 양수인지 판별한 AUROC는 0.462였고, Q4−Q1 direct-view gain 차이는 +0.042821 [+0.022731, +0.063525]였다. AUROC의 positive label은 16 repeats 후의 strict gain>0이어서 작은 numerical 차이에도 민감하다.

Shifted에서 routed selector를 769/1024번 호출하고 실제로 467번 probe를 다른 observation으로 바꿨다. 변경한 worlds의 paired benefit는 -0.002033 [-0.006931, +0.002637]였다. Factual score로 ridge 제안 view의 realized gain이 양수인지 판별한 AUROC는 0.432였고, Q4−Q1 direct-view gain 차이는 +0.032406 [+0.011823, +0.053228]였다. AUROC의 positive label은 16 repeats 후의 strict gain>0이어서 작은 numerical 차이에도 민감하다.

## 추가 독립 검증 · 호출할 selector의 gain을 직접 학습하면 달라지는가

주 실험을 완료한 뒤 두 가지 한계를 발견했다. 첫째, ridge가 예측한 값은 direct view의 이득인데 routed rule은 다른 selector를 호출한다. 둘째, primary threshold grid에는 always-probe가 포함됐지만 모든 worlds에서 selector를 호출하는 always-call endpoint는 빠져 있었다. Ungated selector를 test baseline으로 별도 보고했으나 validation에서 routing 자체의 가치가 과대평가될 여지가 있다.

추가 protocol을 먼저 기록하고, 같은 development 자료에서 E-C-mean과 fixed global 각각의 실제 train gain을 target으로 ridge 두 출력을 학습했다. Threshold 후보에 always-call과 never-call을 모두 넣었다. 기존 matched test와 겹치지 않는 새 1,024 worlds(16 repeats)를 생성해 맞춤 gate 두 개를 평가했다. 이 추가 분석의 설계는 주 결과를 본 뒤 정했고, 추가 test의 choices는 그 outcomes를 열기 전에 고정했다. Target과 threshold grid를 함께 바꿨으므로 두 수정의 효과를 각각 추정한 비교는 아니다.

| 새 worlds의 policy | Mean Brier loss | Switches /1024 | Matched GPU p50 / p90 ms |
|---|---|---|---|
| probe | 0.058467 | 0 | 1.69 / 1.74 |
| fixed_global | 0.058392 | 486 | 7.07 / 7.16 |
| E-C-mean | 0.058105 | 495 | 4.20 / 4.30 |
| original_routed | 0.058891 | 468 | 4.79 / 4.96 |
| aligned_E-C-mean | 0.058105 | 495 | 4.87 / 4.98 |
| aligned_fixed_global | 0.058392 | 486 | 7.74 / 7.82 |

| Matched selector target / comparator | Loss difference [98.75% family-adjusted CI] |
|---|---|
| aligned_E-C-mean minus probe | -0.000362 [-0.003599, +0.002999] |
| aligned_E-C-mean minus fixed_global | -0.000288 [-0.002279, +0.001694] |
| aligned_fixed_global minus probe | -0.000074 [-0.003175, +0.003108] |
| aligned_fixed_global minus fixed_global | +0.000000 [+0.000000, +0.000000] |

Validation에서 두 gate 모두 모든 worlds에서 selector를 호출하는 endpoint를 선택했다. 새 test에서 E-C-mean gate는 probe 대비 −0.000362, fixed global 대비 −0.000288이었고 두 adjusted CI가 모두 0을 포함했다. Fixed-global gate는 원래 fixed global과 정확히 같은 observation을 골랐다. Gate feature 계산을 실제로 포함한 p90은 E-C-mean에서 4.98 ms, fixed global에서 7.82 ms였다. 이처럼 gate가 일정하면 배포 시 해당 계산을 생략할 수 있다. 표에는 실제 측정한 경로의 비용을 기재했다. 주 test와 추가 test는 표본이 달라 각 policy의 절대 loss가 달라질 수 있으며 비교는 동일한 worlds 안에서만 수행한다.

주 test에서 Q1/Q4의 posterior minority mass median은 약 0.237/0.0075였다. 강한 harm은 query event가 더 불확실한 하위 score 집단에서 관찰됐다. 그러나 단일 feature가 그 차이를 일으켰다고 판단할 수 없다. Matched test에서 direct view의 realized gain이 양수인지를 판별한 AUROC도 0.462였다. 몇 건의 큰 손해가 집단 평균을 움직일 수 있으므로, gain의 부호를 잘 맞히는 것과 평균 손실을 줄이는 것을 각각 확인해야 한다.

## 검증 범위와 재현

원본 simulator의 Gaussian kernels를 physical coordinates에서 다시 계산해 validation 한 world의 세 action·여덟 cell risk를 비교했다. 최대 절대 차이는 2.394e-16였다. Source·gate·test-choice freeze hashes를 확인했다. 양 측정 조건은 같은 test worlds를 공유하며 bank seed와 acquisition seed는 고정됐다. 실제 카메라 입력·full trajectory·robot control loss는 이 실험에서 직접 측정되지 않았다. Hindsight best action은 같은 repeats를 보고 최솟값을 고르므로 배포 가능한 성능으로 해석할 수 없다.

서버 폴더는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/factual_switch_v20`다. `PROTOCOL.md`, `policy.py`, `experiment.py`, `analyze.py`가 사전 설계와 구현이며 `audit.py`는 독립 계산과 추가 strata 진단이다. `results/ridge.npz` 및 `gate.json`에는 fitted policy, choices_0/1에는 public-side decisions, loss_0/1에는 held-out per-action losses가 있다. 같은 상대 경로에 v18/v19 dependencies를 둔 새 폴더에서 할당된 idle GPU에 v18 guard를 적용해 experiment.py를 실행하고 compute-node CPU에서 analyze.py, audit.py, report.py 순으로 실행한다.

## 문헌에서 실험 가설까지

이번 literature review의 corpus는 v19에서 선정한 네 원문의 같은 버전이다. 아래의 비교표는 논문에서 관찰한 결과와 v20에서 별도로 검증한 가설의 경계를 명시한다. 원문별 세부 sampling counts와 추가 논의는 직전 [논문 분석표](../conditional_risk_v19/LITERATURE_KO.md)에 있다.

| 연구 | Research question · method | Sample과 data source | Finding · conclusion | Limitation 및 v20에서 바꾼 점 |
|---|---|---|---|---|
| Bartuska et al., *Multilevel randomized quasi-Monte Carlo estimator for nested integration* | Inner approximation과 outer integration의 오차를 목표 tolerance에 맞게 조절할 수 있는가? Multilevel differences 및 randomized QMC로 bias와 statistical error를 분석했다. | Exact/inexact sampling의 두 EIG numerical examples. 첫 예제 pilot에 1,000 randomizations가 쓰였다. | 조건을 만족하는 integrands에서 level 간 variance 감소와 work–tolerance 관계를 보였다. | Regularity에 의존하는 scalar integration 결과다. V20의 binary event와 argmin 선택에 같은 rate를 적용하려면 별도 증명이 필요하다. [원문 §§3–5](https://arxiv.org/html/2412.07723v1#S5) |
| Sorokin et al., *Fast Bayesian Multilevel Quasi-Monte Carlo* | Sample doubling의 예상 이득으로 level별 계산을 배분할 수 있는가? Single low-discrepancy sequence와 GP posterior variance를 사용했다. | 문제·budget별 250 trials. Option pricing, elliptic PDE, Darcy flow 등을 포함한다. | 저자들의 numerical benchmarks에서 replicated MLQMC보다 효율적인 경우를 보였다. | Forecast는 kernel과 GP hyperparameters에 의존한다. V20에서 중요한 추가 endpoint는 선택 후의 paired Brier loss다. [원문 §§2.3, 3](https://arxiv.org/html/2510.24604v1#S3) |
| Dinkel et al., *Efficient Bayesian Optimal Experimental Design for Expensive Computational Models over Finite Design Sets* | 적은 expensive forward calls로 가장 높은 EIG를 갖는 design을 고를 수 있는가? Common random numbers, Rao–Blackwellization과 adaptive elimination을 결합했다. | Synthetic, pharmacokinetic, lung, viscoplastic 사례. 일부 accuracy curves는 1,000 independent runs에 기반한다. | 유망한 designs에 계산을 집중해 높은 선택 정확도에 필요한 forward evaluations를 줄였다. | 비용의 단위가 forward calls이고 utility는 EIG다. V20은 세 cached actions와 end-to-end GPU latency를 측정했다. [원문 §§2–3](https://arxiv.org/html/2607.16933v1#S3) |
| Zhu & Hainy, *Sequential Bayesian Experimental Design for Prediction in Physical Experiments Informed by Computer Models* | Physical measurements를 어디서 취하면 future prediction이 좋아지는가? KOH model, predictive MI와 local complexity, Gaussian mixture compression을 사용했다. | Toy study 및 JAK–STAT5 19개 field time points. 실제 사례에서 4개로 초기화하고 10 rounds를 평가했다. | 실제 사례에서 MI+complexity가 MSE에서, D-optimality가 adaptive design CRPS에서 유리했다. | 보간한 pseudo ground truth를 사용했다. V20의 held-out event probability는 다른 endpoint이며 실측 장치로의 확장은 열려 있다. [원문 §§4–5.4](https://arxiv.org/html/2603.16756v1#S5.SS4) |

**Research knowledge map과 evidence synthesis.** Bartuska의 nested integration과 Sorokin의 Bayesian cubature는 추정 오차에서 계산량으로 이어진다. Dinkel의 design comparison은 추정 오차에서 action ranking으로, Zhu와 Hainy의 predictive design은 action에서 후속 prediction으로 이어진다. 이 가운데 조건부 계산의 수학과 특정 benchmark의 효율에는 각각 이론적·경험적 근거가 있다. 이 네 연구에서 factual evidence로 stick-event의 world별 switch benefit를 예측한다는 직접 근거는 확인되지 않았다. V20에서 직접 측정하는 연결이 이 마지막 단계다. [Bartuska](https://arxiv.org/html/2412.07723v1) · [Sorokin](https://arxiv.org/html/2510.24604v1) · [Dinkel](https://arxiv.org/html/2607.16933v1) · [Zhu & Hainy](https://arxiv.org/html/2603.16756v1)

**불일치와 methodology.** Dinkel의 EIG ranking 개선과 이번 프로젝트의 Brier loss 결과는 서로 다른 objectives와 compute 단위를 사용한다. Zhu와 Hainy의 MSE·CRPS 비교에서는 동일 데이터에서도 criterion 순위가 달랐다. 네 편이 같은 contact population과 동일한 outcome에서 상반된 실험을 했다는 근거는 없다. 수치 적분 trials와 실측 time points를 하나의 표본 수로 합칠 수 없다. V20의 world 내 paired contrasts는 law와 label 계산의 효과를 이 simulator 안에서 평가하고, factual subgroup의 특성은 예측적 연관으로 해석한다. [Dinkel §3](https://arxiv.org/html/2607.16933v1#S3) · [Zhu & Hainy §5.4](https://arxiv.org/html/2603.16756v1#S5.SS4)

**연구 gap과 다음 질문 10개.** 각 행은 논문 또는 자체 실험에 근거한 연구 제안이다.

| 질문 | 남은 gap과 중요성 | 바꿀 변수와 적합한 검증 | 근거 |
|---|---|---|---|
| 1. Label posterior의 이득은 bank가 바뀌어도 유지되는가? | 한 randomization의 matched main effect는 우연한 approximation에 민감할 수 있다. | 여러 inner·outer bank와 acquisition seeds를 교차하고 world와 seed를 각각 resample한다. | V20 factorial contrasts · [Bartuska §5](https://arxiv.org/html/2412.07723v1#S5) |
| 2. Predictive law의 효과는 outer budget에 따라 달라지는가? | M512에서 측정한 marginal effect는 다른 budget을 대표하지 않는다. | C/E와 M을 교차하고 reference regret 및 actual loss를 함께 본다. | V20 protocol · [Bartuska §3](https://arxiv.org/html/2412.07723v1#S3) |
| 3. 호출할 selector의 gain을 직접 학습하면 routing 오류가 줄어드는가? | 주 실험의 ridge target과 실제 호출 policy가 다르다. | 각 teacher의 probe 대비 이득을 train target으로 삼아 독립 worlds에서 평가한다. | V20 추가 실험 · [Zhu & Hainy §3](https://arxiv.org/html/2603.16756v1#S3) |
| 4. 어떤 public features가 harm을 예측하는가? | Quartile과 minority mass의 연관만으로 원인을 알 수 없다. | Profile·velocity·law mass를 feature family별로 ablate하고 held-out gain을 재검증한다. | V20 strata · [Zhu & Hainy §3](https://arxiv.org/html/2603.16756v1#S3) |
| 5. View 이득의 lower confidence bound를 사용하면 과도한 switch가 줄어드는가? | 평균 gain이 양수여도 individual-world outcome은 불확실하다. | Calibration split에서 switching tolerance를 고정하고 subgroup loss와 coverage를 측정한다. | V20 Q4 interval · [Dinkel §2.3](https://arxiv.org/html/2607.16933v1#S2.SS3) |
| 6. 더 저렴한 external label estimator가 full update를 대체하는가? | E label source의 16,384-hypothesis update가 실제 시간을 늘린다. | Weighted subsampling과 teacher distillation의 budget별 bias·p90·Brier 측정 | V20 GPU timing · [Zhu & Hainy §4.1](https://arxiv.org/html/2603.16756v1#S4.SS1) |
| 7. Rare label 지원을 보존하는 것이 언제 필요한가? | Global mixture가 minority support를 잃어도 action loss 증가 크기는 미확정이다. | Validation에서 public minority mass strata를 고정하고 region/global를 교차 비교한다. | [v19 진단](../conditional_risk_v19/REPORT_KO.md) · [Zhu & Hainy §4](https://arxiv.org/html/2603.16756v1#S4) |
| 8. Contact-law shift의 강도가 gate threshold를 얼마나 바꾸는가? | 한 shifted law의 결과를 다른 물성에 일반화하기 어렵다. | 시뮬레이터 shift 강도 및 unseen law를 held-out 단위로 두고 calibration slope·loss를 비교한다. | V20 shifted test · [Zhu & Hainy §2](https://arxiv.org/html/2603.16756v1#S2) |
| 9. Action 수가 늘면 selector 호출 가치가 커지는가? | 현재 세 actions에서 always-probe가 강하다. | 동일한 sensor model에서 view 수만 바꾸고 calls·latency·best fixed rule을 비교한다. | [Dinkel §3.3](https://arxiv.org/html/2607.16933v1#S3.SS3) |
| 10. 물리적 비용이 다를 때 gate가 유효한가? | 실제 probe에는 reset 비용이 들 수 있으나 이번 sensing costs는 동일하다. | 반복 가능한 실물 intervention 자료에서 reset time과 sensor noise를 측정해 utility를 사전 정의한다. | [Dinkel §3](https://arxiv.org/html/2607.16933v1#S3) · [Sorokin §2.3](https://arxiv.org/html/2510.24604v1#S2.SS3) |

**Research assistant review.** Matched analytic simulator의 label-source effect, subgroup 특성, GPU timing은 서로 다른 종류의 근거다. 어느 하나를 전체 active perception의 novelty로 확대할 수 없다. Predictive acquisition 논문을 준비하려면 real-contact multi-intervention 자료, value of information, contextual policy learning에서 사용할 수 있는 직접 comparator를 추가해야 한다. 이 corpus로 분야 전체 선행연구의 부재를 판정하기 어렵다. 이번 연구에서 test outcome을 확인한 뒤 추가한 targeted verification은 주 실험과 구별해 보고했다.

