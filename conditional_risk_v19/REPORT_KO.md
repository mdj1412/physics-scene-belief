# v19 · Conditional Brier risk로 contact observation을 고르는 실험

2026년 11월 14일. 새로 생성한 512 worlds에서 **hypothetical label을 적분하면 고정된 scene hypotheses 안의 numerical noise는 줄었지만, 실제로 더 좋은 observation을 선택한다는 증거는 얻지 못했다.** 같은 compact mixture로 비교한 matched stick-event에서 바뀐 선택은 14/512건이었다. Independent outer bank로 평가 조건을 바꾸면 113/512건이 바뀌었다.

이 보고서의 앞부분은 서버 실험, 뒷부분은 이 실험에 근거를 제공한 네 원문의 비교·knowledge map·gaps·연구 질문 10개·project review다. ‘실제 loss’는 simulator가 감춰 둔 world에서 선택 후 측정한 event-probability Brier loss를 가리킨다.

## 어떤 observation을 고르는가

Factual boundary profile과 충돌 직후 속도를 본 뒤, 멀리 떨어진 지점의 boundary profile을 보는 query view, 가까운 boundary를 정밀 측정하는 factual view, 다른 속도로 접촉시켜 outgoing velocity를 측정하는 probe 중 한 가지를 고른다. 세 observation의 sensing cost는 같다고 두었다. 관심 event는 query 지점의 contact 여부와 접촉 후 stick 여부다. 주 검정은 이전에 순위가 불안정했던 stick에 두었다.

## 실험을 고른 이유와 구현

V18은 action gap에 두 종류의 numerical error가 섞인다는 점을 보였다. Component-only 대비 seed agreement가 높아졌지만, stick GPU median은 adaptive 18.52 ms, fixed region mixture 6.14 ms였다. 이번에는 매번 계산량을 늘리는 대신 하나의 계산을 바꿨다. Dinkel et al.이 finite design EIG 비교에서 사용한 conditional averaging에 착안해 binary predictive Brier risk에서 label을 정확히 평균했다. 이 실험은 Dinkel et al.의 EIG estimator나 전체 ADE 알고리즘을 재현한 것이 아니다. [Dinkel et al., §2.3](https://arxiv.org/html/2607.16933v1#S2.SS3)

Factual posterior에서 replica마다 256개 global representatives를 취해 discrete mixture를 만들었다. 각 hypothetical observation에 대하여 mixture decoder가 계산한 stick probability를 q라 하자. Self-label은 그 observation을 생성한 component의 label Z를 사용해 (Z−q)²를 계산한다. Conditional은 같은 observation에서 Z=0과 Z=1의 두 경우를 합친 q(1−q)를 사용한다. 동일한 mixture가 생성과 예측을 담당할 때 E[(Z−q)² | Y]=q(1−q)이다. 한 pilot case의 모든 해당 entries에서 직접 합산한 값과 공식의 최대 차이는 5.55×10⁻¹⁷이었다. 독립적인 outer law에서 실제 조건부 확률이 p라면 기대 손실은 p(1−p)+(p−q)²다. 이 차이가 self-consistent risk의 적용 범위를 정한다.

두 replicas와 4개의 outer scrambles를 사용했다. M은 replica당 512, 2,048 또는 4,096이며 두 replicas가 outer random coordinates를 공유한다. Conditional/self-label의 representatives와 hypothetical observations가 같으므로 이 비교가 label 적분만 바꾼 ablation이다. External-label은 같은 compact decoder를 사용하되 별도의 16,384-particle outer bank에서 hypothetical scene과 label을 가져온다. 이때 sampling law, targets, 외부 posterior fitting 비용이 함께 바뀐다.

모든 방법은 하나의 observation을 선택하고 공통 16,384-particle full decoder로 예측한다. 이전 global/region fixed mixtures, adaptive allocation, full nested2048을 새 banks와 같은 worlds에 다시 실행했다. 두 fixed rules도 포함했다. Primary acquisition에서 inner/outer bank seeds는 183911/193913이다. 두 acquisition seeds는 213901/243911이다. Matched control의 첫 64 worlds에서 별도 bank pair와 별도 outer seed로 nested4096 reference를 계산했다. 이 reference도 finite-sample estimate다.

## Matched stick 결과

Loss는 synthetic world당 16개의 paired observation-noise repeats를 먼저 평균했다. Agreement는 representatives와 outer scrambles를 함께 바꾼 두 acquisition seeds에서 동일한 action을 고른 world 비율이다. Reference regret는 첫 64개 worlds에서 선택한 action의 reference risk에서 최소 reference risk를 뺀 값의 평균이다. GPU timing은 첫 32개 test worlds에서 방법 순서를 섞어 3회 측정했고, batch 1·float64에서 factual fit, 선택, 선택된 observation update까지 포함한다. Static bank/plan 생성과 physical sensing은 제외했다.

| Method | Brier loss ↓ | Seed agreement ↑ | Reference regret ↓ | GPU p50 / p90 (ms) | 8 ms 초과율 |
|---|---|---|---|---|---|
| conditional/512 | 0.056958 | 81.64% | 0.000469 | 2.68 / 2.75 | 0.0% |
| conditional/2048 | 0.057150 | 85.16% | 0.000436 | 2.85 / 2.94 | 0.0% |
| conditional/4096 | 0.057030 | 85.94% | 0.000436 | 3.03 / 3.14 | 0.0% |
| self_label/512 | 0.056685 | 83.20% | 0.000611 | 2.68 / 2.77 | 0.0% |
| self_label/2048 | 0.057258 | 86.52% | 0.000432 | 2.84 / 2.93 | 0.0% |
| self_label/4096 | 0.057142 | 87.30% | 0.000432 | 3.03 / 3.14 | 0.0% |
| external/512 | 0.056832 | 82.23% | 0.000018 | 4.09 / 4.22 | 0.0% |
| external/2048 | 0.055928 | 88.48% | 0.000017 | 4.29 / 4.37 | 0.0% |
| external/4096 | 0.056039 | 90.04% | 0.000029 | 4.46 / 4.56 | 0.0% |
| v18_global | 0.055903 | 91.21% | 0.000037 | 7.02 / 7.14 | 0.0% |
| v18_region | 0.056014 | 87.11% | 0.000026 | 7.18 / 7.29 | 0.0% |
| v18_adaptive | 0.056018 | 86.33% | 0.000026 | 16.31 / 23.38 | 86.5% |
| nested2048 | 0.057240 | 83.01% | 0.000036 | 31.58 / 32.71 | 100.0% |
| query | 0.069293 | 고정 rule | 0.021364 | 1.74 / 1.78 | 0.0% |
| probe | 0.056670 | 고정 rule | 0.000389 | 1.67 / 1.73 | 0.0% |

별도 pilot 12개 worlds에서 방법별 p90이 8 ms 이하인 가장 큰 M을 미리 선택했다. 세 mixture 방법 모두 M4096을 선택했고 test p90도 8 ms 이내였다. 따라서 같은 시간 상한으로 고른 비교와 M4096 직접 비교는 동일한 관측값이다. 이 실험은 각 방법을 8 ms까지 채웠을 때의 최적 성능을 측정하지 않았다. 세 후보 중 더 큰 M을 시험했을 때의 효과는 열려 있다.

아래 차이는 conditional/4096의 loss에서 baseline의 loss를 뺀 값이다. 4개의 사전 가설에 Bonferroni를 적용한 98.75% paired world-bootstrap CI를 사용했다(20,000 resamples). 같은 상한에서의 비교가 외부 bank M4096과 중복돼 고유한 수치는 세 개다.

| Primary matched stick contrast | Mean difference | Adjusted CI |
|---|---|---|
| conditional/4096 minus self_label/4096 | -0.0001117 | [-0.0003287, +0.0000019] |
| conditional/4096 minus external/4096 | +0.0009915 | [-0.0028478, +0.0058072] |
| conditional/4096 minus nested2048 | -0.0002097 | [-0.0047244, +0.0049193] |

Conditional 대 self-label의 unadjusted 95% interval은 [−0.0002733, −0.0000037]이었으나 사전 family 기준에서는 0을 포함한다. 많은 comparisons 중 한 결과만 고르면 결론이 달라질 수 있다. 두 acquisition seeds의 Top-1 agreement 차이는 −1.37 percentage points였으며 paired 95% CI는 [−3.13, +0.39]였다. Conditional과 always-probe의 loss 차이는 +0.000360, exploratory 95% CI [−0.003286, +0.004454]였다. Always-probe는 GPU p90 1.73 ms여서 이 환경에서 특히 중요한 baseline이다.

![Matched stick loss, GPU latency and conditional numerical variability](results/tradeoff.png)

왼쪽 그림은 point estimates다. 오른쪽은 아래의 추가 진단 결과다. Hypothetical sample 수를 늘려도 realized Brier loss의 순위가 단조롭게 변하지 않았다.

## 어디에서 차이가 생겼는가

**고정된 representatives에서의 outer integration.** 결과를 본 뒤 추가한 진단으로 첫 64개 matched stick worlds에서 representatives를 고정하고 별도의 outer plans 8개를 사용했다. 미리 선택했던 probe-minus-query risk gap에 대해 각 world 안의 sample SD를 구하고 그 RMS를 보고한다. M512에서 self-label 0.003822가 conditional 0.001854로, M4096에서는 0.000533이 0.000257로 줄었다. M4096에서 같은 world의 8개 seeds로 만들 수 있는 28쌍의 평균 Top-1 agreement는 오히려 self-label 97.77%, conditional 97.21%였다. 고정된 gap 하나의 변동 감소와 세 action 중 최선의 선택은 다른 측정값이다.

**Mixture가 지운 label support.** Matched stick의 122/512 worlds에서는 양쪽 replicas 각각에서 representatives의 event label이 한 값뿐이었다. 이 122개가 conditional의 세 action risk가 모두 같아진 122개와 정확히 일치했다. Full factual bank의 minority probability는 이들에서 median 0.000023, 최대 0.00613이었다. 12개만 0.001을 넘었고 0.01을 넘는 case는 없었다. 고정된 tie rule은 query view를 선택했다. Support loss가 실제로 발생했지만, 이 수치만으로 평균 decision loss의 주원인이라고 판정할 수 없다.

**어떤 변경이 action을 바꿨는가.** Same-mixture label 적분으로 14개가 바뀌었고 external generation으로 바꾸면 113개, v18 fixed global로 바꾸면 119개가 달라졌다. 첫 64 worlds의 independent-bank nested reference에서 conditional/4096 regret는 0.000436, external/4096은 0.000029였다. 별도 approximation을 사용하는 reference인 만큼 reference agreement와 realized Brier loss를 함께 해석해야 한다. 이 결과는 다음 검증에서 representative support와 predictive law를 먼저 점검할 이유를 준다.

## Contact control과 shifted contact law

| Method | Matched contact | Matched stick | Shifted-law stick · nominal inference |
|---|---|---|---|
| conditional/4096 | 0.047161 | 0.057030 | 0.056384 |
| self_label/4096 | 0.047161 | 0.057142 | 0.056211 |
| external/4096 | 0.047021 | 0.056039 | 0.057333 |
| v18_global | 0.047030 | 0.055903 | 0.057155 |
| v18_region | 0.047030 | 0.056014 | 0.057309 |
| v18_adaptive | 0.047030 | 0.056018 | 0.057309 |
| nested2048 | 0.047030 | 0.057240 | 0.057707 |
| query | 0.047000 | 0.069293 | 0.060890 |
| probe | 0.147412 | 0.056670 | 0.057708 |

Matched contact에서 conditional과 self-label은 512개 모두 같은 action을 선택했다. Conditional은 508건에서 query view를 골랐다. Shifted stick에서는 conditional 대 self-label의 차이가 +0.000174였고 secondary 95% CI는 [−0.000016, +0.000540]이었다. 두 조건은 같은 hidden worlds와 noise를 공유하는 paired conditions다. Shifted 결과에서 새로운 독립 표본 512개를 추가로 얻은 것으로 계산하지 않았다.

## 논문 기여와 다음 결정

**현재 뒷받침되는 주장은 좁고 구체적이다.** 이 simulator에서 binary conditional averaging은 고정된 compact belief 안에서 지정한 action-gap estimate의 outer 변동을 약 절반으로 줄였다. Strong fixed mixture 또는 always-probe 대비 실제 decision loss 개선은 확인되지 않았다. Conditional과 self-label의 계산 시간도 거의 같아, external 방법과의 속도 차이를 label 적분 자체의 효과로 귀속할 수 없다. Simulator, bank, device가 달라진 v18과 v19 사이의 절대 loss 차이를 성능 향상으로 해석하지 않는다.

**다음으로 선택할 질문은 “어느 world에서 다른 observation을 고를 만큼 predictive law가 달라지는가?”다.** Public factual evidence로 minority mass와 compact/external disagreement를 정의하고 disjoint pilot에서 threshold를 고정한다. Held-out worlds에서 global·label-stratified·independent-bank generation을 동일한 GPU budget으로 비교하고, 관측 전 예측한 이득과 관측 후 actual loss의 관계를 측정한다. Primary endpoint는 always-probe와 strong fixed mixture 양쪽에 대한 paired Brier loss difference로 둔다. 고정된 action-gap의 variance 감소만 재현된다면 numerical diagnostic으로 정리하는 편이 현재 증거에 맞다.

이 판단은 analytic local impulse model, 같은 비용의 세 observation, nominal inference, batch 1 GPU에서 얻었다. 실제 영상의 geometry estimation, 장치의 reset cost, robot intervention 성공률은 별도의 검증 대상이다. Acquisition replication은 두 seeds, independent-bank reference와 고정 representatives 진단은 64 worlds로 제한했다. 8 ms는 공학적 비교를 위해 선택한 상한이다.

## 실행 기록과 재현

서버 디렉터리는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/conditional_risk_v19`이다. 처음 node56에서 pilot 이후 다른 작업이 시작돼 guard가 이번 worker를 중단했다. 기록을 `interrupted_node56/`에 보존했다. 최종 표의 pilot·test·timing은 node55, allocation 2744468, RTX PRO 6000 Blackwell 한 장에서 처음부터 다시 측정했다. 완료된 worker의 종료 코드는 0이다.

`selector.py`는 public evidence를 받아 action을 정한다. `run.py`는 data generation, pilot, policy freeze, 실측 timing과 공통 decoder evaluation을 실행한다. `analyze.py`는 사전 contrasts, `diagnose.py`와 `outer_audit.py`는 exploratory diagnostics를 담당한다. `PROTOCOL.md`, `results/sources.json`, `results/freeze.json`에 분석 계획과 hash를 저장했다. 재현할 때는 새 디렉터리에 같은 상대 경로의 v18 dependencies를 두고 할당된 GPU에서 v18의 guard로 run.py를 실행한 뒤 compute-node CPU에서 analyze.py와 diagnose.py를 실행한다. Outer audit은 같은 GPU에서 별도 guard 디렉터리로 실행한다.

## 문헌 검토와 연구 질문

이 검토의 연구 질문은 “factual evidence를 본 뒤 하나의 observation을 고를 때, 추정 과정에서 label noise를 줄이면 같은 GPU budget에서 stick-event의 실제 Brier loss가 낮아지는가?”이다. 이번 요청에는 새로운 논문 목록과 [TOPIC] 값이 지정되지 않아 v18에서 이 계산 문제와 직접 연결한 네 원문의 arXiv 버전을 corpus로 선택했다. 아래에서 논문에 관한 근거는 이 네 편으로 한정했다. v18과 v19의 측정값은 별도의 자체 실험 근거다.

## 1. 논문별 비교

| 논문 | Research question | Methodology | Sample과 data source | Key finding | Limitation과 conclusion |
|---|---|---|---|---|---|
| Bartuska et al., *Multilevel randomized quasi-Monte Carlo estimator for nested integration* | Inner integral과 forward model을 근사하면서 목표 tolerance에 도달하는 비용은 얼마인가? | Scrambled QMC, multilevel differences, bias와 statistical error 분석. Inner sample 수와 discretization level을 조정 | Exact/inexact sampling의 EIG 수치 예제 2개. 첫 예제의 pilot variance 분석에 1,000 randomizations, 여러 tolerance에서 consistency 평가에 100 runs | 첫 예제에서 level-difference variance의 감소를 확인. 점근적 work rate가 관찰되는 tolerance 구간도 제한됨 | Regularity와 noise truncation 조건이 이론에 들어간다. 저자들이 보인 결론은 조건을 만족하는 nested integral의 work–accuracy 관계다. Contact boundary에 적용하려면 해당 조건을 다시 확인해야 한다. [§§3–6, Fig. 1–2](https://arxiv.org/html/2412.07723v1#S5) |
| Sorokin et al., *Fast Bayesian Multilevel Quasi-Monte Carlo* | Level마다 여러 randomizations를 쓰는 비용을 줄이면서 어디에 samples를 추가할지 예측할 수 있는가? | GP Bayesian cubature, single low-discrepancy sequence, sample doubling의 예상 posterior variance 감소와 비용 비교 | Problem·budget·method 조합당 250 independent trials. 32-dimensional single-level test functions 및 option pricing, elliptic PDE, Darcy flow. 주 비교의 RQMC는 8 randomizations | 저자들의 benchmark에서 제안 방법이 replicated MLQMC보다 효율적이었다. Kernel과 point set의 구조를 이용해 GP 계산 비용을 낮췄다 | Forecast는 추정한 GP hyperparameters와 kernel에 의존한다. 적분 정확도 결과를 observation 선택 후의 event loss에 옮기는 데에는 별도 실험이 필요하다. [§§2.3, 3–4](https://arxiv.org/html/2510.24604v1#S3) |
| Dinkel et al., *Efficient Bayesian Optimal Experimental Design for Expensive Computational Models over Finite Design Sets* | Forward evaluations가 비쌀 때 유한한 designs 중 EIG가 가장 큰 것을 얼마나 적은 호출로 찾는가? | Sample reuse, common random numbers, Rao–Blackwellization, paired bootstrap 기반 adaptive design elimination. 반복적으로 열등한 design만 제거 | Analytic synthetic, pharmacokinetic, lung, viscoplastic 사례 4개. Accuracy curves는 예를 들어 synthetic 및 material에서 각각 1,000 independent runs. Lung 사례는 4개 designs | Material 사례에서 높은 선택 정확도에 도달하려면 fixed allocation이 대응하는 adaptive 방식보다 ANMC/LRNMC에서 약 2.5배의 model evaluations를 사용했다 | 주요 비용 단위는 expensive forward calls다. 우리와 같이 cached action 세 개를 GPU에서 평가할 때의 latency는 이 실험으로 결정할 수 없다. [§§2.3–2.4, 3–4](https://arxiv.org/html/2607.16933v1#S3) |
| Zhu & Hainy, *Sequential Bayesian Experimental Design for Prediction in Physical Experiments Informed by Computer Models* | 다음 physical measurement를 어디에서 얻어야 미래 physical response의 예측이 좋아지는가? | Kennedy–O’Hagan calibration, predictive MI와 local complexity를 결합한 criterion, Gaussian mixture compression, precomputation | Toy simulation과 JAK–STAT5 observations. 실제 사례는 19 field time points 중 4개로 초기화, simulator 60 inputs, 10 sequential rounds, 100 prediction points | 실제 사례에서 MI+complexity가 MSE로 좋았고 adaptive design에서 D-optimality가 CRPS로 좋았다. Toy 사례보다 criteria 간 차이가 작았다 | 희소한 실측을 보간한 pseudo ground truth로 평가했다. Prediction metric에 따라 선택한 criterion의 장점이 달라진다. [§§4–5.4, Fig. 7–8](https://arxiv.org/html/2603.16756v1#S5.SS4) |

## 2. 문헌의 세 흐름

**적분 자체의 정확도.** Bartuska et al.은 계산을 어느 level에 배분해야 주어진 tolerance를 만족할지 bias와 variance로 분석했다. Sorokin et al.은 다음 doubling 후의 uncertainty를 GP로 예측해 비용에 반영했다. 둘을 비교할 때는 “현재 error가 큰 곳”과 “계산을 추가하면 error가 가장 많이 줄어들 곳”을 구분해야 한다. 두 논문이 제공하는 근거는 각각의 integrand 가정과 numerical benchmark에 대응한다. [Bartuska et al., §3](https://arxiv.org/html/2412.07723v1#S3) · [Sorokin et al., §2.3](https://arxiv.org/html/2510.24604v1#S2.SS3)

**후보 간 차이를 추정하는 정확도.** Dinkel et al.의 실험 단위는 하나의 integral뿐 아니라 두 designs의 EIG 차이다. Common random numbers가 후보 간 차이의 변동을 줄일 수 있고, 평균 EIG의 bias도 후보 간 차이를 키우거나 줄일 수 있다. 이 결과는 우리 문제에서도 각 action의 risk와 pairwise risk gap을 각각 평가할 이유가 된다. [Dinkel et al., §§2.3–2.4](https://arxiv.org/html/2607.16933v1#S2.SS3)

**선택 후 예측의 정확도.** Zhu와 Hainy는 선택에 사용한 criterion과 선택 후 평가하는 MSE·CRPS를 구분했다. 우리의 질문에서도 먼저 numerical ranking을 평가하고, 선택한 observation을 공통 decoder에 주었을 때 event probability가 얼마나 정확해지는지 측정해야 한다. 후자는 논문에서 가져온 결과가 아니라 그 구분에서 도출한 우리의 실험 설계다. [Zhu & Hainy, §§5.1, 5.4](https://arxiv.org/html/2603.16756v1#S5)

## 3. Research knowledge map

| 출발 개념과 연구자 | 연결 관계 | 도착점 | 근거가 닿는 범위 |
|---|---|---|---|
| Nested integration · Bartuska et al. | Inner approximation과 outer integration을 정량화 | Error tolerance에 대한 work | Assumptions 아래 이론과 두 예제. Event decision까지의 연결은 열려 있다. [원문](https://arxiv.org/html/2412.07723v1) |
| Bayesian cubature · Sorokin et al. | GP가 다음 refinement의 uncertainty를 예측 | Cost-aware level allocation | GP와 point set의 조건 및 benchmark에서 평가. Per-scene observation latency는 미측정이다. [원문](https://arxiv.org/html/2510.24604v1) |
| Paired comparison · Dinkel et al. | Randomness를 공유하고 일부 변동을 적분 | Finite design의 EIG ranking | Analytic reference와 expensive-model 예제가 있다. Event probability의 Brier score는 다른 endpoint다. [원문](https://arxiv.org/html/2607.16933v1) |
| Predictive design · Zhu & Hainy | Mixture compression으로 predictive criterion 계산 | 관측 이후 MSE와 CRPS | Toy와 biochemical time series에서 평가. Contact transition과는 data-generating process가 다르다. [원문](https://arxiv.org/html/2603.16756v1) |

V19는 세 번째 행의 **conditional averaging**과 네 번째 행의 **prediction endpoint**를 연결한다. 이 연결에서 확인할 값은 action gap의 numerical error, seed가 바뀔 때의 선택, 실제 observation 이후의 loss, GPU latency다. 각 값의 측정 단위가 달라 하나로 다른 값을 대체할 수 없다.

## 4–7. Gaps, 불일치, 방법 비교와 evidence synthesis

| Evidence 수준 | 근거를 대조한 결과 | 우리 연구에서 남은 검증 |
|---|---|---|
| 강함 · 범위가 한정된 이론 | Inner/outer error를 다루는 multilevel construction이 존재하고, GP 기반으로 refinement의 예상 비용 대비 이득을 비교하는 방법도 제시되어 있다. 두 결과의 가정과 error 정의가 다르다. [Bartuska](https://arxiv.org/html/2412.07723v1#S3) · [Sorokin](https://arxiv.org/html/2510.24604v1#S2) | 두 error를 나누었다는 사실만으로 novelty를 주장하기 어렵다. Discontinuous event에서 paired action-gap error를 별도로 확인해야 한다. |
| 중간 · 독립 반복을 둔 수치 근거 | Dinkel et al.은 expensive-model evaluations를 측정하고 Sorokin et al.은 integration benchmark에서 반복 trials를 사용했다. 둘 다 computational efficiency를 평가하지만 최적화하는 대상과 비용 단위가 다르다. [Dinkel](https://arxiv.org/html/2607.16933v1#S3) · [Sorokin](https://arxiv.org/html/2510.24604v1#S3) | GPU milliseconds, target error, physical simulator calls가 서로 바뀌는 조건을 같은 사례에서 측정해야 한다. 3-action cached simulator는 특히 저렴한 fixed baseline이 강할 수 있는 조건이다. |
| 측정값에 따른 순위 역전 | Zhu & Hainy의 같은 실측 사례에서도 MSE와 CRPS의 선두 criterion이 달랐다. 실제 관측 수가 적고 보간한 기준을 사용했다. 이는 같은 endpoint의 논문 간 직접 충돌과 구별된다. [Zhu & Hainy, §5.4](https://arxiv.org/html/2603.16756v1#S5.SS4) | Ranking, actual Brier loss, tail latency를 동시에 보고해야 한다. 어느 하나의 개선이 나머지 개선을 유발했다는 주장은 별도 근거가 필요하다. |
| 겉보기 불일치 | Dinkel et al.에서는 adaptive elimination이 forward calls를 줄였다. v18의 stick GPU timing에서는 fixed mixture가 adaptive보다 빨랐다. 전자는 여러 expensive designs의 EIG, 후자는 세 cached actions의 Brier risk와 batch-1 GPU latency다. 후보 수, forward cost, sequential launches가 모두 다르므로 원인을 한 가지로 특정할 수 없다. [Dinkel, §§3.3–3.4](https://arxiv.org/html/2607.16933v1#S3) · [v18 자체 실험](../adaptive_budget_v18/REPORT_KO.md) | 동일한 models와 objectives에서 후보 수·forward cost를 각각 바꿔야 explanation을 검증할 수 있다. |
| 불충분 · 대상 환경 | 네 편에서 직접 조사한 조건은 EIG numerical examples, cubature benchmarks, finite calibration designs, predictive time series다. 그 범위에서 **같은 compact posterior로 hypothetical labels를 적분한 뒤, 실제 contact observation을 골라 fixed GPU budget에서 stick loss를 평가한 비교**는 확인되지 않았다. [Bartuska](https://arxiv.org/html/2412.07723v1#S5) · [Sorokin](https://arxiv.org/html/2510.24604v1#S3) · [Dinkel](https://arxiv.org/html/2607.16933v1#S3) · [Zhu & Hainy](https://arxiv.org/html/2603.16756v1#S5) | 이번 simulator의 matched/shifted contact law와 드문 stick event가 해당 gap을 부분적으로 다룬다. Real contact, camera occlusion, intervention cost로의 확장은 실물 근거가 필요하다. |

네 편 중 동일한 contact dataset, 동일한 predictive loss, 동일한 budget에서 서로 반대 결론을 낸 두 논문은 찾지 못했다. 특히 1,000 numerical runs, 250 algorithm trials, 19 physical time points는 서로 다른 실험 단위다. 이 수들을 합친 meta-analysis나 공통 effect size를 산출할 근거가 없다. [Dinkel, §3](https://arxiv.org/html/2607.16933v1#S3) · [Sorokin, §3](https://arxiv.org/html/2510.24604v1#S3) · [Zhu & Hainy, §5.4](https://arxiv.org/html/2603.16756v1#S5.SS4)

## 8. 열 개의 후속 연구 질문

아래는 원문에서 보고된 결과와 이번 연구자가 제안하는 가설을 구별하기 위한 설계표다. 마지막 열의 원문은 해당 질문을 떠올리게 한 근거다.

| # | Research question | Gap과 중요한 이유 | 독립·종속 변수 및 검증 방법 | 근거 |
|---|---|---|---|---|
| 1 | Hypothetical event label을 조건부 평균하면 stick 선택이 개선되는가? | Pairwise numerical noise가 실제 prediction에 미치는 영향이 미측정이다. | Sampled/conditional label만 교체, 같은 samples에서 seed agreement·Brier loss·latency 비교. **V19 실행** | [Dinkel §2.3](https://arxiv.org/html/2607.16933v1#S2.SS3) |
| 2 | Outer predictive law를 같은 compact belief에서 생성하면 어떤 bias가 생기는가? | 정확한 conditional average도 부정확한 posterior에 대해 계산할 수 있다. | Self/external generation을 교체하고 independent reference regret와 actual loss 비교. **V19 실행**, self/external은 여러 요소가 함께 바뀌는 비교 | [Zhu & Hainy §4.1](https://arxiv.org/html/2603.16756v1#S4.SS1) |
| 3 | 세 개에서 수십 개로 action 수가 늘면 elimination이 유리해지는가? | 후보 수가 작으면 early elimination 기회가 적다. | View grid의 크기와 sensor redundancy를 조절하고 동일 hardware에서 forward calls와 p90 측정 | [Dinkel §3.3](https://arxiv.org/html/2607.16933v1#S3.SS3) |
| 4 | 현재 error보다 예상 regret 감소/시간이 좋은 refinement criterion인가? | 큰 proxy가 계산 한 단위의 이득을 의미하는지 불확실하다. | Disjoint pilot에서 marginal GPU cost와 regret 감소를 추정. Held-out에서 fixed·balanced 비교 | [Sorokin §2.3](https://arxiv.org/html/2510.24604v1#S2.SS3) |
| 5 | 허용 가능한 regret에서 멈추면 near ties의 비용이 감소하는가? | Exact Top-1이 필요한 정도는 application loss에 따라 달라진다. | 사전 지정 regret tolerance를 바꿔 expected regret–p95 frontier 측정 | [Bartuska §3](https://arxiv.org/html/2412.07723v1#S3) · [Dinkel §2.4](https://arxiv.org/html/2607.16933v1#S2.SS4) |
| 6 | Rare stick posterior에서 region-preserving compression이 유리한가? | Global compression의 평균 결과가 minority-label 실패를 가릴 수 있다. | 독립 bank의 public event mass로 strata 고정 후 global/stratified/GMM reduction 비교 | [Zhu & Hainy §4.1](https://arxiv.org/html/2603.16756v1#S4.SS1) |
| 7 | Discontinuous event에서 boundary smoothing이 work–bias tradeoff를 바꾸는가? | Multilevel error rate가 integrand regularity에 의존한다. | Smoothing bandwidth와 sample budget을 교차시키고 hard event loss를 동일하게 평가 | [Bartuska §§3–4](https://arxiv.org/html/2412.07723v1#S3) |
| 8 | Independent bank를 바꾸면 stable ranking도 뒤집히는가? | 한 bank에서의 replication은 posterior approximation 전체를 재현하지 않는다. | Bank·representatives·outer scramble을 crossed design으로 바꾸고 각 gap variance를 추정 | [Sorokin §2.2](https://arxiv.org/html/2510.24604v1#S2.SS2) · [Bartuska §3](https://arxiv.org/html/2412.07723v1#S3) |
| 9 | Contact-law shift 때 geometry view와 probe 중 무엇이 model discrepancy를 드러내는가? | Predictive design은 simulator discrepancy에 영향을 받는다. | Known/unknown law와 observation 종류를 교차. 동일 decoder의 calibration·Brier loss를 비교 | [Zhu & Hainy §§2–3](https://arxiv.org/html/2603.16756v1#S2) |
| 10 | 측정 비용을 포함하면 실제 장치에서도 observation 선택 순위가 유지되는가? | 계산 시간과 physical acquisition 시간이 다른 비용이다. | 동일 물체·여러 개입의 반복 가능한 접촉 자료에서 sensing/reset cost, prediction error, 실제 action utility를 함께 기록 | [Dinkel §3](https://arxiv.org/html/2607.16933v1#S3) · [Zhu & Hainy §5.4](https://arxiv.org/html/2603.16756v1#S5.SS4) |

## 9. 현재 연구에 대한 검토

- **주장 범위.** V18에서 두 error proxy를 분리해 component-only보다 seed agreement가 높아졌지만 강한 global mixture의 GPU latency와 agreement가 모두 더 좋았다. Proposed contribution을 “adaptive allocation이 우수하다”로 쓰면 자체 결과와 충돌한다. 논문에는 global baseline, task별 timing, actual loss의 paired interval을 본문에 함께 제시하는 것이 적절하다. [v18 자체 실험](../adaptive_budget_v18/REPORT_KO.md)
- **약한 논증.** Rao–Blackwellization이 scalar integrand의 조건부 변동을 없앤다는 점만으로 discrete action 선택이나 최종 Brier loss 개선을 결론 내릴 수 없다. Common random numbers가 만든 action 간 covariance와 model mismatch가 남는다. V19의 same-mixture ablation과 외부 평가를 분리한 이유다. [Dinkel §2.3](https://arxiv.org/html/2607.16933v1#S2.SS3)
- **직접 비교가 필요한 방법.** Published adaptive design elimination, GP-based cost allocation, Gaussian mixture reduction은 서로 다른 objectives로 개발됐다. 현재 구현과의 명칭 유사성으로 비교를 갈음할 수 없다. Method 논문을 준비한다면 같은 predictive Brier objective에 적용 가능한 각 구성요소를 지정하고, 원래 objective의 reproduction과 adaptation 결과를 구별해야 한다. [Dinkel](https://arxiv.org/html/2607.16933v1#S2) · [Sorokin](https://arxiv.org/html/2510.24604v1#S2) · [Zhu & Hainy](https://arxiv.org/html/2603.16756v1#S4)
- **추가 근거가 필요한 영역.** 이번 corpus는 numerical integration과 experimental design에 집중돼 있다. 전체 분야의 novelty 판단에는 active physical perception, expected value of sample information, value-directed belief compression, real contact intervention benchmark 문헌을 별도로 검토해야 한다. 여기서는 이 영역에 선행 방법이 없다는 주장을 하지 않는다. 네 편의 범위는 위 비교표와 같다.

