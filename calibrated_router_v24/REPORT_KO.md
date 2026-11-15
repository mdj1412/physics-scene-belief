# v24 · 독립적인 curvature와 friction에서 factual gate의 noninferiority와 GPU p90

**동시 판정은 충족이다.** Camera-favouring, probe-favouring, balanced sensor calibration 세 조건에서 absolute Brier 증가의 one-sided 95% upper bound를 +0.003과 비교하고, 같은 입력의 GPU p90 ratio upper bound를 0.95와 비교했다. Brier 기준은 3/3개, p90 기준은 3/3개 calibration에서 통과했다. 대상은 새로운 synthetic worlds 1,152개, 별도의 bank pairs 5쌍과 한 장의 RTX PRO 6000이다.

| Held-out calibration | Gate Brier | Always-preserve Brier | Gate−reference · 95% upper | Gate / reference p90 (ms) | p90 ratio · 95% upper | 동시 판정 |
|---|---|---|---|---|---|---|
| camera | 0.100762 | 0.099453 | +0.001310 · +0.002673 | 3.517 / 4.744 | 0.741 · 0.744 | 충족 |
| probe | 0.116045 | 0.116045 | +0.000000 · +0.000000 | 3.871 / 4.620 | 0.838 · 0.840 | 충족 |
| balanced | 0.105862 | 0.105330 | +0.000532 · +0.001171 | 3.502 / 4.656 | 0.752 · 0.754 | 충족 |

## 1. V23에서 어떤 가정을 바꿨는가

V23은 curvature로 stick을 판단해야 하는 family와 friction을 봐야 하는 family의 factual slope 분포가 달랐다. V24는 그 family를 제거했다. Quadratic boundary의 curvature 중심 −0.29, 0, +0.29와 Coulomb friction coefficient 중심 0.12, 0.32를 모두 교차했다. 각 셀에서 curvature에는 독립 Uniform(−0.06,+0.06), friction에는 독립 Uniform(−0.0375,+0.0375) jitter를 주었다. Offset, local slope, restitution도 별도로 추출했다. 셀마다 192개 world를 같은 가중치로 평가했다. 새로운 worlds는 v23·v24 development에서 관찰한 loss로 선별하지 않았다.

Factual evidence는 중심부 boundary profile의 관측값 9개와 incoming velocity (1,−0.10)인 충돌 직후의 velocity 2개다. 선택지는 query view, factual view, incoming velocity (1,−0.85)의 probe다. Target은 y=0.025, incoming velocity (1,0.425)에서의 stick이다. 공통 16,384-hypothesis inner decoder에 선택한 observation을 한 번 넣고, 실제 stick label에 대한 (p−label)²를 평가했다. Simulator와 inference는 같은 Coulomb contact law를 사용했다. Calibration 간에는 같은 world, label 및 표준화한 observation noise를 공유한다.

실제 held-out continuous curvature와 friction의 Pearson correlation은 -0.0108였다. 독립성은 이 sample correlation의 크기에서 추론한 것이 아니라 generator의 factorial assignment와 독립 jitter에서 온다.

| Calibration | Factual profile σ | Factual velocity σ | Query view σ | Factual view σ | Probe σ |
|---|---|---|---|---|---|
| Development · camera | 0.0021 | 0.12 | 0.00040 | 0.0010 | 0.060 |
| Development · probe | 0.0033 | 0.18 | 0.00130 | 0.00075 | 0.023 |
| Held-out · camera | 0.0018 | 0.15 | 0.00030 | 0.0012 | 0.075 |
| Held-out · probe | 0.0037 | 0.11 | 0.00165 | 0.00055 | 0.018 |
| Held-out · balanced | 0.0027 | 0.21 | 0.00076 | 0.00090 | 0.042 |

모든 policy는 해당 센서의 올바른 noise model을 받는다. 여기서 unseen calibration은 두 development 설정과 다른 noise 표준편차 조합을 뜻한다. 잘못 보정된 센서나 새로운 contact law를 시험한 실험은 별도로 필요하다.

## 2. 저비용 gate를 어떻게 만들었는가

**첫 구현(A).** Factual feature에서 작은 K와 큰 K의 realized Brier 차이를 예측하는 ridge를 학습했다. K768–2048에서 training labels가 모든 budget pair에서 0이었다. Validation 두 calibration에서도 모든 입력이 high로 갔다. 이 결과만 확인한 시점에 A를 고정 K 대조로 남겼다. A의 held-out loss를 열기 전에 B의 training target·budget 후보·seeds와 판단 기준을 별도 protocol로 고정했다. A와 B는 test worlds·bank pairs를 공유하지 않는다.

**독립 확인 실험(B).** Ridge는 full-reference K2048과 low K에서 세 action의 acquisition risk 차이를 계산한 뒤, action 공통 offset을 제외한 RMS error의 log를 학습한다. Target은 log(error+10⁻⁷)이다. 실제 stick label 없이도 candidate posterior로 만들 수 있다. Budget 선택에는 별도의 validation Brier를 사용했다. Training 288개, validation 240개, test 1,152개의 worlds가 독립적이다.

Online에서 gate는 아직 GPU posterior를 만들기 전에 host에 있는 factual profile에서 선형 slope와 quadratic curvature를 추정한다. 두 factual velocity, log(query-view σ / probe σ), factual-profile σ까지 6개 features를 만들고, 표준화한 값·제곱·calibration interactions를 합한 20개 항에 ridge coefficients를 적용한다. Gate는 hidden physics, 셀 번호, query label, 추가 observation을 받지 않는다. 두 분기 모두 kernel·M·decoder는 같고 K만 다르다. K의 3/4은 큰 factual weights를 그대로 유지하고, 1/4은 stick/non-stick label별 residual mass에 배정한다. Reference는 이를 매번 K2048/M2048로 실행한다.

K 후보 쌍은 (64, 768), (128, 1024), (256, 1536)이었다. Development의 40·60·80 percentile cutoffs 중 양쪽 validation calibrations에서 high 빈도 10–90%를 우선하고 worst validation Brier 증가가 +0.0015 이하인 후보에서 평균 K를 최소화했다. 선택된 low/high는 K128 / K1024, 각각 tail slots 32 / 256개다. Cutoff는 training score의 80 percentile(-4.2537)이고 validation high 빈도는 0.0%, 45.0%였다. Threshold나 budget을 test 결과로 다시 맞추지 않았다.

양쪽 validation calibration 모두에서 high route 비율 10–90%인 후보가 없었다. 사전 fallback 기준으로 위 구성을 선택했다. 이는 두 개발 sensor에서 적응성이 고르게 학습됐다는 해석을 제한한다.

## 3. Noninferiority와 GPU p90의 동시 판정

각 calibration에서 Δ = Brier(gate)−Brier(always-preserve), R = p90(gate)/p90(always-preserve)로 정의했다. Δ의 one-sided 95% upper bound < +0.003이고 동시에 R의 upper bound < 0.95여야 통과한다. 세 calibrations가 전부 통과해야 전체 성공이다. 이 intersection–union test에서는 여섯 조건의 결합만 primary claim으로 검정한다. +0.003은 v23의 runner-up 비용 약 0.150의 2%로 정한 연구용 tolerance이며 실제 사용자가 허용할 비용은 측정하지 않았다. GPU 기준은 측정 오차를 넘는 5% 이상의 실질적 p90 절감이다.

같은 world에서 32번 반복된 noise로 action별 Brier를 평균한 후 비교했다. Availability of additional observations는 simulation evaluation에만 허용했다. 모든 test choices를 먼저 hash로 고정한 뒤 private labels와 observation을 열었다. Brier uncertainty는 6개 cells 안에서 worlds를 resample하고 5개 banks를 교차 resample한 paired bootstrap 10,000회로 계산했다. Timing은 cell마다 12 worlds, 4 repeats, 5 banks로 방법·calibration당 1,440회다. 동기화한 batch-1, float64, 무작위 interleaved 측정에서 raw factual host input의 전송, gate, support 생성, acquisition, 선택한 observation의 전송·update를 포함한다. Bank 생성·학습, physical sensor 취득 시간은 제외했다. 동일한 bank, world, repeat block을 paired resample한 5,000개 p90 ratios의 95 percentile을 upper bound로 사용했다. 한 서버·한 GPU에 대한 조건부 결과다.

| 민감도 · 통과한 calibration 수 / 3 | Δ < +0.001 | Δ < +0.002 | Δ < +0.003 | Δ < +0.005 | R < 1.00 | R < 0.95 |
|---|---|---|---|---|---|---|
| 95% upper 기준 | 1 | 2 | 3 | 3 | 3 | 3 |

## 4. Routing이 어디에서 도움이 되었는가

| Calibration | High 사용률 | 평균 K | 다른 action · bank-world % | 다른 action이 있었던 고유 worlds / 1152 | 다른 선택에서 평균 Δ Brier | High / low route에서 low−high Brier |
|---|---|---|---|---|---|---|
| camera | 0.0% | 128.0 | 2.6% | 110 | +0.050980 | 해당 없음 / +0.001310 |
| probe | 95.1% | 979.7 | 0.0% | 0 | 해당 없음 | +0.000095 / +0.000000 |
| balanced | 1.0% | 136.6 | 0.6% | 32 | +0.095829 | +0.000000 / +0.000538 |

Online decoder가 같아 action이 같으면 두 정책의 실현 Brier도 같다. 따라서 전체 Δ는 다른 action을 골랐을 비율과 그때의 평균 Δ의 곱이다. High/low 집단의 단순 loss 차이는 gate-on 효과에 대한 인과 추정치가 아니다. 비교에는 같은 world에서 강제로 분기를 고정한 두 결과를 사용했다.

| 비교 방법 | Camera Brier / p90 ms | Probe Brier / p90 ms | Balanced Brier / p90 ms |
|---|---|---|---|
| Factual gate | 0.100762 / 3.517 | 0.116045 / 3.871 | 0.105862 / 3.502 |
| Always-preserve K2048 | 0.099453 / 4.744 | 0.116045 / 4.620 | 0.105330 / 4.656 |
| Always-low K128 | 0.100762 / 3.395 | 0.116135 / 3.320 | 0.105862 / 3.380 |
| Always-high K1024 | 0.099453 / 3.850 | 0.116045 / 3.754 | 0.105330 / 3.814 |
| Fixed budget K256 | 0.099967 / 3.457 | 0.116045 / 3.378 | 0.105330 / 3.435 |
| Fixed global mixture | 0.099340 / 미측정 | 0.116045 / 미측정 | 0.105330 / 미측정 |
| Blind routing | 0.100762 / 미측정 | 0.116049 / 미측정 | 0.105857 / 미측정 |

| Gate−control · paired 95% exploratory CI | Camera | Probe | Balanced |
|---|---|---|---|
| Always-low K128 | +0.000000 [-0.000000, -0.000000] | -0.000090 [-0.000392, +0.000146] | +0.000000 [-0.000000, -0.000000] |
| Always-high K1024 | +0.001310 [-0.000232, +0.002987] | +0.000000 [-0.000000, -0.000000] | +0.000532 [-0.000208, +0.001315] |
| Fixed budget K256 | +0.000795 [-0.000689, +0.002470] | +0.000000 [-0.000000, -0.000000] | +0.000532 [-0.000208, +0.001315] |
| Blind routing | +0.000000 [-0.000000, -0.000000] | -0.000004 [-0.000019, +0.000007] | +0.000005 [-0.000002, +0.000013] |

같은 primary 기준을 사후에 정적 controls에도 적용했을 때 세 calibration을 모두 통과한 구성은 Always-low K128, Always-high K1024, Fixed budget K256이었다. Fixed budget은 validation 평균 K에 가장 가까운 사전 후보를 골랐다. Blind routing은 calibration별 동일 high 빈도로 무작위 배정했을 때의 기대 손실이다. 별도 kernel로 실행하지 않았다. Fixed global은 추가 accuracy 대조이며 이번 primary latency comparator는 always-preserve다.

| Calibration | Calibration 다수 분기 | 같은 분기인 worlds / 1152 | 다수 분기 Brier | Gate−다수 분기 · 95% CI |
|---|---|---|---|---|
| camera | low | 1152 | 0.100762 | +0.000000 [+0.000000, +0.000000] |
| probe | high | 1095 | 0.116045 | +0.000000 [+0.000000, +0.000000] |
| balanced | low | 1141 | 0.105862 | +0.000000 [+0.000000, +0.000000] |

이 추가 대조는 각 calibration의 test factual inputs에서 더 자주 쓰인 분기를 그 calibration 전체에 적용했다. Labels를 보지 않는 transductive diagnostic이다. 실제로 calibration metadata만으로도 비슷한 결과가 나오는지 점검하며, 사전 확정된 deployable primary baseline과는 구분한다.

| Calibration | Gate action 횟수 Q / F / P | Reference action 횟수 Q / F / P | 항상 query view / factual view / probe Brier |
|---|---|---|---|
| camera | 5610 / 32 / 118 | 5758 / 0 / 2 | 0.099572 / 0.217356 / 0.206887 |
| probe | 5760 / 0 / 0 | 5760 / 0 / 0 | 0.116045 / 0.199760 / 0.208963 |
| balanced | 5728 / 7 / 25 | 5760 / 0 / 0 | 0.105330 / 0.217149 / 0.210320 |

Q, F, P는 각각 query view, factual view, probe다. 한 calibration당 5,760개 bank–world choices를 셌다. 하나의 action이 지배적인 조건이라면 관측된 noninferiority는 budget 선택이 어려운 worlds에서의 성과와 구별해야 한다.

### Gate만의 비용과 근사 오차

동일한 factual input에 대해 cheap host-side score와 v23의 posterior top-512 score를 따로 timing했다. p50은 15.0 / 430.8 µs, p90은 19.8 / 452.5 µs였다(cheap / v23). V23에는 이미 fitted state를 제공했고 GPU 결과를 host에서 분기 가능하게 반환하는 synchronization을 포함했다. 위 end-to-end primary 측정과 별개의 추가 진단이다. Factual input이 처음부터 GPU에만 있는 시스템에서는 host gate 비용이 달라질 수 있다.

| Calibration | Score와 관찰된 low risk error의 Spearman ρ | Low / high centered risk error 평균 | 은행별 Δ 범위 | 은행별 p90 ratio 범위 |
|---|---|---|---|---|
| camera | 0.136 | 0.01079 / 0.00221 | 0.00044–0.00226 | 0.739–0.742 |
| probe | -0.035 | 0.01887 / 0.00453 | 0.00000–0.00000 | 0.837–0.839 |
| balanced | 0.040 | 0.01464 / 0.00357 | -0.00016–0.00098 | 0.751–0.753 |

Spearman은 timing에 사용한 72 worlds와 5개 bank의 descriptive association이다. 추가 observation의 실제 label이나 효과를 score의 입력으로 쓰지 않았다.

### Factorial cell별 결과

| Curvature / friction 중심 | Camera Δ / high % | Probe Δ / high % | Balanced Δ / high % |
|---|---|---|---|
| -0.29 / 0.12 | +0.000773 / 0.0% | +0.000000 / 92.2% | +0.002657 / 0.0% |
| -0.29 / 0.32 | +0.003488 / 0.0% | +0.000000 / 95.3% | +0.000733 / 2.6% |
| +0.00 / 0.12 | -0.001870 / 0.0% | +0.000000 / 95.3% | -0.000611 / 1.0% |
| +0.00 / 0.32 | +0.000479 / 0.0% | +0.000000 / 96.9% | -0.000420 / 1.0% |
| +0.29 / 0.12 | +0.000787 / 0.0% | +0.000000 / 96.9% | +0.000118 / 0.5% |
| +0.29 / 0.32 | +0.004203 / 0.0% | +0.000000 / 93.8% | +0.000718 / 0.5% |

Cell별 Δ는 exploratory point estimate다. Primary 판정은 여섯 cells의 동일 가중 평균에 적용했다. 개별 물체나 모든 cell에 noninferiority가 보장된다는 뜻으로 확대할 수 없다.

### 최초 gate(A)의 독립 기록

| A · calibration | High 빈도 | Gate−reference Δ · upper | p90 ratio · upper |
|---|---|---|---|
| camera | 100.0% | -0.000051 · +0.000000 | 0.954 · 0.959 |
| probe | 100.0% | +0.000000 · +0.000000 | 0.955 · 0.959 |
| balanced | 100.0% | +0.000000 · +0.000000 | 0.956 · 0.961 |

A의 development 결과는 K768 / K1536였지만 모든 test 입력이 같은 분기를 탔다. A의 속도 또는 loss 결과를 adaptive routing의 근거로 합산하지 않았다.

![Calibration별 Brier noninferiority와 GPU p90 ratio](results/v24.png)

## 5. 논문 기여 가능성과 판단

**사전 joint test는 통과했다.** 세 calibration 모두에서 full K2048 always-preserve에 대한 Brier 증가의 95% upper bound가 +0.003보다 작았고, GPU p90을 5% 이상 줄였다는 기준도 통과했다. Camera가 가장 아슬아슬했다. Point estimate는 +0.001310, upper bound는 +0.002673이므로 margin을 +0.002로 낮추면 여기서는 noninferiority를 주장할 수 없다. 한 camera cell의 loss 증가 점추정은 +0.004203이었다. 전체의 성공을 모든 물체에서의 보장으로 확대해서는 안 된다.

**실제로 시간을 줄인 요인은 support 크기였다.** Validation 평균 K에 가장 가까워 미리 정한 고정 K256은 세 calibration 모두에서 gate보다 빨랐다. Camera·probe·balanced의 p90은 gate가 각각 3.517·3.871·3.502 ms, 고정 K256은 3.457·3.378·3.435 ms였다. Brier 역시 고정 K256이 camera와 balanced에서는 낮고 probe에서는 같았다. 강제 K1024는 세 조건 모두 reference와 같은 평균 Brier를 냈고 p90도 줄였다. 현재 배포 대안을 고른다면 K256이 간단하고, reference의 loss 재현을 우선하면 K1024가 합리적이다. 이 판단은 이 synthetic workload에서의 측정에 한정된다.

**개별 world마다 routing할 필요성은 입증하지 못했다.** High budget 비율은 camera 0.0%, probe 95.1%, balanced 1.0%였다. 각 calibration의 다수 분기를 모든 world에 적용해도 평균 Brier 차이는 세 조건에서 모두 0이었다. 특히 ‘probe’는 probe sensor의 상대적 precision이 좋은 설정이라는 이름이다. 이 조건에서 gate와 reference는 모두 query view를 선택했다. 세 설정에서 reference가 query view를 벗어난 경우를 합쳐도 17,280개 bank–world–calibration 선택 중 2건이다. 따라서 이 benchmark에서는 대부분의 world에서 action을 유지하는 것만으로 noninferiority에 도달할 수 있다. 차이 0에서 bootstrap upper bound가 0으로 나온 조건은 표본 내 경험적 일치이며, 보지 않은 world에서의 완전 동등성을 보장하지 않는다.

**논문으로 남길 수 있는 기여는 검증 설계와 적용 범위다.** Curvature와 friction을 독립화하고 factual·additional sensor noise를 함께 바꾼 상황에서, 예측 허용오차와 tail latency를 묶어 prospective test로 판정했다. 최초 training target이 0으로 퇴화한 기록과 정적 K·blind routing·calibration-only 대조를 공개해, 큰 reference를 줄인 효과와 입력별 routing 효과를 분리했다. 현재 결과만으로 adaptive gate의 algorithmic advantage를 핵심 기여로 삼기는 어렵다. 다음 검증에서 가장 먼저 바꿀 것은 gate의 복잡도가 아니라 **같은 calibration 안에서도 두 개 이상 actions가 실제로 유리한 world 분포**다. Factual evidence의 분포는 겹치되 probe가 friction을 식별해 stick decision을 바꾸는 물리 조건을 독립 development에서 확인하고, 고정 K와 always-query를 미리 지정한 별도 test에서 비교해야 한다.


## 문헌 검토와 연구 설계에 대한 평가

### 1. 원문별 비교

| 연구 | Research question과 방법 | Sample·data source | 결과와 결론 | V24에 적용할 때의 한계 |
|---|---|---|---|---|
| Teerapittayanon, McDanel & Kung, *BranchyNet* | 쉬운 입력에서 추론을 끝낼 수 있는가? 여러 깊이에 side classifiers를 두고 confidence로 early exit한다. | MNIST·CIFAR-10, LeNet·AlexNet·ResNet 계열 | 실험한 네트워크에서 많은 입력이 일찍 종료됐고 inference time을 줄일 수 있었다. | 최종 image classification과 one-step observation selection은 목표가 다르다. 입력별 분기가 V24 p90을 줄이는지는 별도 측정이 필요하다. [원문](https://bradmcdanel.com/wp-content/uploads/2016/10/2016-icpr-teerapittayanon-mcdanel-kung.pdf) |
| Douc, Cappé & Moulines, *Comparison of Resampling Schemes for Particle Filtering* | Particle resampling 방식에 따라 conditional variance가 달라지는가? 이론적 비교와 counterexample을 제시했다. | Weighted empirical measures, asymptotic analysis. 실제 장비 표본 없음 | Residual 및 stratified resampling은 제시한 조건에서 multinomial보다 낮거나 같은 variance를 갖는다. Systematic resampling에는 해당 ordering의 반례가 있다. | Particle 통계량의 variance 보장을 action 선택 후의 Brier noninferiority로 옮길 수 없다. [원문](https://arxiv.org/pdf/cs/0507025) |
| Dinkel et al., *Efficient Bayesian Optimal Experimental Design for Expensive Computational Models over Finite Design Sets* | 제한된 forward evaluations로 가장 유용한 design을 찾는가? Common random numbers, sample reuse, adaptive elimination을 비교했다. | Synthetic, pharmacokinetic, lung, viscoplastic 네 사례. 일부 ranking curves는 1,000 independent runs | Estimator와 design별 budget 배정에 따라 selection accuracy를 높이는 데 드는 평가 수가 달라졌다. | EIG 순위와 model evaluations를 비교한다. 실행시간 분포의 p90 및 physical query의 예측 오차는 다른 측정량이다. [원문 §3](https://arxiv.org/html/2607.16933v1#S3) |
| Zhu & Hainy, *Sequential Bayesian Experimental Design for Prediction in Physical Experiments Informed by Computer Models* | 어떤 physical observation이 미래 예측의 불확실성을 더 줄이는가? Predictive mutual information과 local complexity adjustment, Gaussian mixture compression을 사용했다. | Toy simulation, JAK–STAT5 시계열. 실제 사례에서는 초기 4 time points에서 10 rounds를 평가 | Field prediction에서 predictive criterion과 평가 지표에 따라 방법의 상대적 순위가 달라졌다. | Continuous time-series와 보간한 reference를 사용했다. 새 sensor의 binary contact prediction에 적용하려면 다른 검증이 필요하다. [원문 §§4–5](https://arxiv.org/html/2603.16756v1#S5) |
| Piaggio et al., *Reporting of Noninferiority and Equivalence Randomized Trials* | Noninferiority 주장을 해석 가능하게 보고하려면 무엇을 밝혀야 하는가? CONSORT reporting extension과 예시를 제시했다. | 임상시험 보고 방법에 관한 원문. 새 GPU나 물리 실험 cohort가 없다. | Margin의 사전 명시와 근거, confidence interval 방향, active comparator를 명시하도록 제시한다. | 여기서 차용한 것은 판정·보고 원칙이다. +0.003이나 5%를 권장하는 논문은 아니다. [원문](https://jamanetwork.com/journals/jama/fullarticle/1487502) |

### 2–3. Literature review와 knowledge map

**예측 목표와 계산 배정.** Zhu·Hainy는 관측이 최종 예측에 주는 가치를 다루고, Dinkel 등은 여러 관측 후보에 대한 numerical ranking 비용을 다룬다. BranchyNet은 입력별로 추가 계산이 필요한지를 앞단에서 판단한다. V24는 같은 action 후보에 대해 public factual evidence를 읽고 posterior support의 크기를 정한다. 이 세 문헌은 목표와 계산이 만나는 위치를 설명하지만, 동일한 contact simulator의 Brier와 p90을 함께 평가하지는 않았다. [Zhu & Hainy](https://arxiv.org/html/2603.16756v1#S3) · [Dinkel et al.](https://arxiv.org/html/2607.16933v1#S2) · [BranchyNet](https://arxiv.org/abs/1709.01686)

**Knowledge map.** Factual evidence → cheap approximation-error predictor → support budget → resampled label posterior → action별 expected Brier → selected observation → realized Brier로 이어진다. 별도 경로인 budget → GPU latency distribution은 같은 입력에서 측정한다. Weighted resampling의 일부 variance 성질과 margin을 사용한 판정은 알려져 있다. Error predictor가 새로운 calibration에서도 decision-relevant worlds를 찾아내는지, routing overhead를 포함해 p90을 줄이는지는 경험적으로 검증해야 한다. [Douc et al.](https://arxiv.org/pdf/cs/0507025) · [Dinkel et al.](https://arxiv.org/html/2607.16933v1#S2) · [Piaggio et al.](https://jamanetwork.com/journals/jama/fullarticle/1487502)

### 4–7. Research gaps, 상충하는 결과, methodology, evidence synthesis

| Evidence와 비교 축 | 남은 gap 및 해석 |
|---|---|
| BranchyNet의 classification accuracy와 inference time, Dinkel 등의 design 선택 정확도와 forward calls는 sample, hardware, endpoint가 다르다. [BranchyNet](https://arxiv.org/abs/1709.01686) · [Dinkel et al.](https://arxiv.org/html/2607.16933v1#S3) | 평균 계산량 감소와 p90 감소가 같은 결론을 내릴 필요는 없다. 두 연구가 같은 지표에 대해 충돌한다는 근거는 없다. V24에서는 gate와 always-high의 비용을 같은 호출 경로에서 잰다. |
| Douc 등은 estimator의 반복 변동을 다루고, Zhu·Hainy는 design 선택 후 다른 metric에서의 predictive performance를 비교한다. [Douc et al.](https://arxiv.org/pdf/cs/0507025) · [Zhu & Hainy](https://arxiv.org/html/2603.16756v1#S5) | Posterior approximation 개선과 binary loss 개선의 전달 과정에 대한 직접 근거는 약하다. Continuous risk distortion과 실제로 action이 달라진 횟수, 그때의 loss를 따로 측정한다. |
| Piaggio 등의 margin은 특정 적용에서 허용 가능한 저하와 관련된다. 문헌끼리 trial, algorithm run, empirical measure의 표본 단위가 다르다. [Piaggio et al.](https://jamanetwork.com/journals/jama/fullarticle/1487502) · [Dinkel et al.](https://arxiv.org/html/2607.16933v1#S3) | 이 corpus의 효과를 하나의 평균으로 합치지 않았다. 새 장비와 contact law에 허용 가능한 차이는 불확실하다. V24의 crossed bootstrap은 simulator의 world·bank 변동을 평가한다. |

### 8. 다음 연구 질문 10개

| 질문과 연구가 필요한 이유 | Variables와 적합한 검증 | 근거 |
|---|---|---|
| 1. 실제 high-budget 배정이 같은 사용 빈도의 blind routing보다 유용한가? 저렴한 branch 자체와 배정 가치를 분리해야 한다. | Firing rate를 유지해 기대 Brier를 비교한다. | [BranchyNet](https://arxiv.org/abs/1709.01686) |
| 2. 같은 평균 support를 전부에게 주면 더 나은가? Adaptive computation의 overhead를 반영해야 한다. | 정적 K와 gated K를 같은 M과 decoder에서 interleaved timing한다. | [Dinkel et al.](https://arxiv.org/html/2607.16933v1#S2) |
| 3. 값이 작은 action-risk error가 실제로 중요한가? Ranking과 loss는 다를 수 있다. | Centered risk error, action flip, paired Brier를 함께 추적한다. | [Dinkel et al.](https://arxiv.org/html/2607.16933v1#S2) |
| 4. Tail의 label별 slots 대신 sensor likelihood별 slots를 배정할 수 있는가? 같은 크기로도 남는 bias가 다를 수 있다. | Label stratification과 response clustering을 같은 K에서 비교한다. | [Douc et al.](https://arxiv.org/pdf/cs/0507025) · [Zhu & Hainy](https://arxiv.org/html/2603.16756v1#S4) |
| 5. Curvature와 friction의 joint effect가 특정 셀에 피해를 집중시키는가? 전체 평균으로는 가려질 수 있다. | Factorial cell별 Brier와 action 빈도를 보고, 독립 object sample에서 interaction을 재검증한다. | [Zhu & Hainy](https://arxiv.org/html/2603.16756v1#S5) |
| 6. Sensor calibration을 잘못 알면 gate는 얼마나 민감한가? 알려진 noise scale을 사용하는 검증만으로는 부족하다. | 실제 noise와 inference calibration을 독립적으로 바꿔 calibration-error sweep을 한다. | [Zhu & Hainy](https://arxiv.org/html/2603.16756v1#S2) |
| 7. Lab에서 정한 noninferiority margin이 실제 사용 비용에 맞는가? | 재촬영·reset·실패 비용과 Brier 차이를 측정해 새 margin을 실험 전에 확정한다. | [Piaggio et al.](https://jamanetwork.com/journals/jama/fullarticle/1487502) |
| 8. Training에 보지 않은 contact law에서도 방법이 성립하는가? Matched prior 실험으로는 충분하지 않다. | Coulomb, speed-dependent friction, damping을 train/test에서 교차한다. | [Zhu & Hainy](https://arxiv.org/html/2603.16756v1#S2) |
| 9. GPU-resident vision encoder를 쓰면 host-side routing이 여전히 싼가? Raw input의 시작 위치가 달라진다. | Host/device input, batch size, device-to-host synchronization을 요인으로 p90을 다시 잰다. | [BranchyNet](https://arxiv.org/abs/1709.01686) |
| 10. One-step에서 줄인 비용이 여러 관측을 고를 때 누적되는가? | Sequential horizon, posterior updates, trajectory loss와 wall-clock budget을 함께 평가한다. | [Zhu & Hainy](https://arxiv.org/html/2603.16756v1#S3) |

### 9. Research assistant review

입력별 계산 예산 조절, predictive experimental design, stratified resampling은 각각 선례가 있다. 이 다섯 편은 V24의 방법과 측정에 직접 관련된 corpus이며 포괄적인 novelty search는 아니다. 알고리즘 기여를 주장하려면 강한 정적 budget과의 차이, route가 달라졌을 때의 이득, 새로운 물체에서의 재현을 보여야 한다. Simulator에서 동등한 sensing 비용을 가정한 end-to-end latency는 물리 센서 동작까지 포함한 robot latency로 해석할 수 없다. [BranchyNet](https://arxiv.org/abs/1709.01686) · [Dinkel et al.](https://arxiv.org/html/2607.16933v1#S3) · [Zhu & Hainy](https://arxiv.org/html/2603.16756v1#S5)


## 재현 자료

서버의 확인 실험 폴더는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/calibrated_router_v24`이다. [사전 protocol](PROTOCOL.md), [독립 simulator](../cheap_tail_v24/physics24.py), [factual features와 ridge](../cheap_tail_v24/gate24.py), [최종 실행 및 branch 배정](router24.py), [통계 분석](../cheap_tail_v24/stats24.py), [gate 비용 진단](overhead24.py), [후보별 validation](results/development.json), [primary 통계 및 각 cell 결과](results/summary.json), [최초 실행 결과](../cheap_tail_v24/results/summary.json)를 보존했다. Raw choices, losses, timing blocks, seeds, source·plan hashes는 각 실행의 results 아래에 있다. 같은 sibling directories와 기존 v18–v23 dependencies를 유지하고 빈 Slurm GPU에서 v18 guard로 run24.py(A), router24.py(B)를 각각 실행한 후 각 폴더에서 stats24.py를 compute-node CPU로 실행한다. B의 overhead24.py는 별도 guard log 폴더에서 실행한다. 실제 sensor 동작 시간를 포함하는 real-world 실험 자료는 없다.
