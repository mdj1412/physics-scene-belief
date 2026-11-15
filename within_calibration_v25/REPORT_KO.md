# v25 · 하나의 sensor calibration에서 query view와 probe의 선택 가치

**같은 calibration에서 world에 따라 최적 observation이 바뀌는 조건을 만들었다.** 새로 추출한 1,280개 worlds에서 동일 stick query를 사용했다. 32회 observation noise를 둘로 나눴을 때 두 half 모두에서 query view가 0.02 Brier 넘게 유리한 world는 773개(60.4%), probe가 유리한 world는 373개(29.1%)였다. 좁은 curvature·넓은 friction에서는 probe의 Brier가 query view보다 평균 0.23371 낮았고, 넓은 curvature·좁은 friction에서는 query view의 Brier가 평균 0.21908 낮았다.

**Factual gate는 always-query를 이겼지만, 고정 K256까지 이긴다는 사전 superiority 기준을 충족하지 못했다.** Gate, K256, always-query의 held-out Brier는 각각 0.076195, 0.076308, 0.106122였다. Gate−always-query 차이는 −0.029927 [95% CI −0.035475, −0.024266], gate−K256 차이는 −0.000113 [−0.002066, +0.002067]이었다. K256 자체가 전체 선택의 25.3%에서 probe를 고른 강한 adaptive observation selector였다.

**P90에서는 K256이 더 빨랐다.** Gate의 GPU p50은 0.926 ms로 K256의 3.421 ms보다 짧았지만 p90은 4.023 ms 대 3.503 ms였다. 81.0%의 입력에서 gate가 action을 직접 선택하고, 나머지 19.0%에서 K1024 acquisition을 실행한 결과와 일관된다. Always-query의 p90은 0.757 ms였다. 이 비용은 GPU에서 선택 후 posterior를 갱신하기까지의 시간이며 실제 probe 동작 시간은 포함하지 않는다.


## 1. 무엇을 독립적으로 조작했는가

센서는 하나로 고정하고 **불확실한 물리량**을 바꿨다. Quadratic boundary의 curvature spread와 Coulomb friction coefficient의 spread를 독립적인 두 요인으로 두어 네 셀을 만들었다. Curvature의 크기 중심은 0.022 또는 0.340이고, 부호는 무작위다. Friction coefficient는 0.251에 ±0.011 또는 ±0.138을 더한다. 두 부호와 두 연속 jitter는 서로 독립이고, 같은 셀에서도 물체의 정확한 물리 상태는 다르다. Factual sensor noise σ는 boundary profile 0.0016, 충돌 후 velocity 0.095다. Query view, factual view, probe의 σ는 각각 0.00030, 0.00120, 0.027이며 모든 split과 world에 공통이다.

**왜 다른 관측이 필요한가?** Factual 충돌은 y=0에서 incoming velocity (1,0)으로 일어나 항상 stick한다. 이때 관측된 velocity에는 friction coefficient의 실제 크기가 드러나지 않는다. Probe는 같은 곳에서 incoming velocity (1,−1.10)로 충돌해 항상 slide하고, 충돌 후 속도는 friction에 반응한다. Query view는 y=0.025 주변의 boundary profile을 측정한다. 어느 쪽을 고르든 예측할 대상은 y=0.025에 incoming velocity (1,0.425)를 가했을 때의 stick 여부로 같다.

Query 위치의 slope를 s_q, restitution을 e라고 하면 stick을 가르는 friction threshold는 **µcrit = |s_q + 0.425| / [(1+e)(1−0.425s_q)]**다. 넓은 curvature·좁은 friction 셀에서 음의 curvature면 µcrit≤0.078, 양이면 µcrit≥0.483이다. 이 셀의 실제 µ는 0.2325–0.2695 안에 있으므로 curvature의 부호가 label을 결정한다. 좁은 curvature·넓은 friction 셀에서는 µcrit가 0.183–0.299 안에 있고, low µ≤0.1205, high µ≥0.3815여서 friction의 high/low가 label을 결정한다. 이 식은 **무엇이 label을 결정하는지** 보여준다. Noise가 있는 observation 중 어느 쪽이 더 낮은 Brier를 주는지는 별도의 paired test로 판정했다.

**정책에 보이는 단서.** Boundary offset은 넓은/좁은 curvature에 따라 평균이 +0.00115/−0.00115이고 독립적인 ±0.0017 jitter가 있다. Restitution은 넓은/좁은 friction에 따라 평균이 0.795/0.605이고 ±0.09 jitter와 velocity measurement noise가 더해진다. 따라서 factual profile과 velocity에는 spread를 짐작할 불완전한 단서가 있다. 어느 부호가 실현됐는지, 실제 셀 번호나 정답은 제공하지 않는다. 이 두 proxy와 spread의 관계는 generator에 의도적으로 넣은 가정이다.

## 2. Factual gate와 비교 정책

모든 방법은 **선택한 observation을 동일한 16,384-hypothesis inner-bank decoder로 업데이트**해 stick probability를 낸다. Brier는 (p−label)²이고 작을수록 좋다. 달라지는 것은 observation 선택이다. Primary action set은 query view와 probe 두 가지다. 두 numerical selectors는 기존 세 action용 acquisition kernel을 실행하고 Q/P 두 risk만 비교한다. 세 방법 모두 같은 물리 연산을 할 필요가 없으므로 고정 action이나 직접 선택이 가능하면 outer posterior와 acquisition을 건너뛴다. 별도의 unrestricted K256 대조에서 factual view도 허용했다.

V24의 factual gate는 support 크기만 정했다. V25 primary는 새로운 **action-value gate**다. 훈련 때 두 observation 모두를 독립적으로 시뮬레이션해 stick Brier를 얻고, 두 training bank pairs에서 평균한 advantage L(probe)−L(query)를 regression target으로 쓴다. 배포 때는 factual profile의 mean, slope, quadratic curvature, residual RMS, 충돌 직후 velocity의 두 성분만 읽는다. Advantage가 양수이면 query, 음수이면 probe로 직접 선택할 수 있다. 예측 절댓값이 작으면 K1024/M2048 posterior acquisition으로 돌아간다. 직접 선택은 새 observation의 결과나 test labels를 사용하지 않는다.

이 변경을 따로 평가하기 위해 ① 새 factual confidence로 support만 K128 또는 K1024로 고르는 control, ② v24-B 계수와 cutoff를 그대로 가져와 현재 calibration의 public metadata만 넣은 frozen gate, ③ 직접 선택만 사용하는 policy, ④ 항상 K1024 및 항상 K128인 policy를 실행했다. Static K256/M2048은 v24와 같은 label-stratified tail preservation이다. 따라서 고정된 것은 K이며, observation은 world에 따라 바꿀 수 있다.

Training 3,072, validation 1,024, held-out 1,280개 worlds를 분리했다. 각 split은 네 셀의 수가 같고 seeds는 각각 2511001, 2512001, 2513001이다. Pair별 16,384개 inner와 별도의 16,384개 external hypotheses를 사용했다. 2개 model families × 3 ridge penalties × 4 confidence thresholds의 24개 후보 중 validation Brier가 가장 작은 rbf 모델을 선택했다. Penalty 0.001, cutoff 0.04780, validation 직접 선택 비율 80.7%였다. RBF 모델은 quadratic features와 training set에서 고정 추출한 96개 Gaussian centers를 결합한다. Training target 생성, validation, 모델 선택까지 서버 wall time은 20.5초였다. 이는 offline 비용이다.

설정을 고정한 validation에서 gate/K256/always-query의 Brier는 0.069368 / 0.072614 / 0.104040였다. Test는 새로운 bank pairs 5쌍에 같은 독립 worlds와 noise를 적용했다. 모든 policy choices의 SHA-256을 먼저 고정하고, 그 뒤에 evaluator가 held-out labels와 potential observations를 열었다.

## 3. 같은 calibration에서 실제로 최적 observation이 바뀌었는가

동일 world에서 Q/P 각각에 32회 observation noise를 적용했다. 16회씩 나눠 절반에서 더 낮은 Brier를 주는 action을 정하고, 다른 절반에서 그 선택의 loss를 측정한 다음 방향을 바꿔 평균했다. 두 절반 모두에서 같은 action이 0.02 넘게 유리하면 bank 평균 기준으로 그 world를 stable winner로 셌다. 사전 기준은 Q와 P의 stable worlds가 각각 10% 이상이고, 방향을 미리 정한 두 셀의 평균 Brier advantage에 대한 one-sided 95% bound가 각각 0.03을 넘는 것이다. 이는 고정된 hidden world에 여러 관측을 반복할 수 있는 evaluator의 판단이다. Factual evidence만 있는 policy가 그 winner를 미리 안다는 의미는 아니다.

| Curvature / friction spread | Q Brier | P Brier | P−Q Brier | Stable Q / P worlds (각 320) | K256 Q / P 선택률 | V25 gate Q / P 선택률 |
|---|---|---|---|---|---|---|
| 좁음 / 좁음 | 0.156158 | 0.214917 | +0.058759 | 201 / 65 | 73.0% / 27.0% | 70.6% / 29.4% |
| 좁음 / 넓음 | 0.268330 | 0.034618 | -0.233712 | 9 / 308 | 35.7% / 64.3% | 35.3% / 64.7% |
| 넓음 / 좁음 | 0.000000 | 0.219085 | +0.219085 | 302 / 0 | 98.3% / 1.7% | 98.4% / 1.6% |
| 넓음 / 넓음 | 0.000000 | 0.215084 | +0.215084 | 261 / 0 | 91.8% / 8.2% | 92.5% / 7.5% |

사전 물리 검증은 **충족**이다. 넓은 curvature·좁은 friction의 P−Q lower bound는 +0.203672, 좁은 curvature·넓은 friction의 upper bound는 -0.221766였다. Stable Q 773개, P 373개, 나머지 134개였다. 개별 bank–world에서 noise halves의 winner가 일치한 비율은 98.9%다.

| 안정성에 요구한 Brier gap | 두 half 모두 Q 우세 | 두 half 모두 P 우세 | 미분류 worlds |
|---|---|---|---|
| 0.00 | 876 | 392 | 12 |
| 0.01 | 817 | 386 | 77 |
| 0.02 | 773 | 373 | 134 |
| 0.05 | 656 | 347 | 277 |

Cross-fitted Q/P winner의 Brier는 0.040362, always-query 대비 실현 이득은 +0.065760 [+0.062059, +0.069658]이다. 나머지 noise half에서 다시 평가한 winner별 이득은 Q 우세 group에서 +0.189030 [+0.177215, +0.200983], P 우세 group에서 +0.212611 [+0.201373, +0.224819]였다. 셀 효과는 10,000회 within-cell world 및 bank bootstrap으로 계산했다.

## 4. Held-out Brier와 GPU p90

| 방법 | Brier ↓ | Q / factual view / P 횟수 | GPU p50 / p90 ms | Gate−방법 Brier [paired 95% CI] |
|---|---|---|---|---|
| V25 factual gate | 0.076195 | 4750 / 0 / 1650 | 0.926 / 4.023 | 기준 |
| 고정 K256 | 0.076308 | 4781 / 0 / 1619 | 3.421 / 3.503 | -0.000113 [-0.002066, +0.002067] |
| Always-query | 0.106122 | 6400 / 0 / 0 | 0.744 / 0.757 | -0.029927 [-0.035475, -0.024266] |
| Always-probe | 0.170926 | 0 / 0 / 6400 | 0.668 / 0.680 | -0.094731 [-0.103979, -0.085721] |
| 직접 선택만 | 0.078154 | 4615 / 0 / 1785 | 0.911 / 0.945 | -0.001959 [-0.005281, +0.001251] |
| 고정 K1024 | 0.075422 | 4802 / 0 / 1598 | 3.802 / 3.896 | +0.000774 [-0.000191, +0.001924] |
| Support-only gate | 0.076272 | 4792 / 0 / 1608 | 3.596 / 4.015 | -0.000077 [-0.001481, +0.001306] |
| 고정 K128 | 0.076216 | 4782 / 0 / 1618 | 3.363 / 3.439 | -0.000020 [-0.002351, +0.002507] |
| Fixed global mixture | 0.075100 | 4811 / 0 / 1589 | 미측정 | +0.001095 [-0.000402, +0.002857] |
| 고정 K256 · 세 action | 0.076306 | 4781 / 1 / 1618 | 미측정 | -0.000111 [-0.002065, +0.002075] |
| V24 frozen gate | 0.076216 | 4782 / 0 / 1618 | 3.447 / 3.500 | -0.000020 [-0.002351, +0.002507] |

| Primary 비교 | Gate−baseline Brier | One-sided 95% upper | 다른 action 비율 / 고유 worlds | 다른 선택에서 Δ Brier | GPU p90 ratio [paired 95% CI] |
|---|---|---|---|---|---|
| 고정 K256 | -0.000113 | +0.001691 | 3.3% / 104 | -0.003416 | 1.148 [1.128, 1.163] |
| Always-query | -0.029927 | -0.025227 | 25.8% / 349 | -0.116079 | 5.316 [5.222, 5.377] |

Gate가 두 baseline보다 낮은 Brier를 준다는 사전 공동 기준은 **미충족**이다. 두 upper bounds가 모두 0 미만이어야 한다. GPU p90은 별도로 보고하며, 항상 같은 observation을 쓰는 경우의 저렴한 실행 비용도 포함했다. 같은 action을 고르면 decoder와 결과가 같으므로 전체 loss 차이는 ‘선택이 다른 비율 × 다른 선택에서의 평균 loss 차이’다.

각 policy에 대해 CUDA synchronization을 포함한 float64 batch-1 wall latency를 측정했다. 각 셀 20 worlds × 4 interleaved repeats × 5 banks로 방법당 1,600회다. Raw factual host input, 필요한 bank fit과 acquisition, 선택된 observation의 host→GPU transfer, posterior decoding을 포함한다. Sensor의 실제 노출·이동·접촉 시간과 정적 bank 구축은 제외했다. GPU는 allocation 2744468, node55의 RTX PRO 6000 Blackwell Server Edition 한 장이다. P90 intervals는 bank·cell별 world·repeat block의 pairing을 유지한 5,000회 bootstrap에서 구했다. Loss와 elapsed time에 사용한 물리적 world 수는 각각 1,280개와 80개다. Noise와 bank는 독립적인 물체로 세지 않았다.

| Bank pair | Gate−K256 | Gate−always-query |
|---|---|---|
| 1 | -0.000778 | -0.030853 |
| 2 | +0.000424 | -0.029125 |
| 3 | -0.001698 | -0.030098 |
| 4 | -0.000027 | -0.029797 |
| 5 | +0.001517 | -0.029760 |

## 5. 어떤 구성에 이득을 귀속할 수 있는가

| Support-only policy | Policy−K256 [95% CI] | Policy−always-query [95% CI] |
|---|---|---|
| Support-only gate | -0.000036 [-0.002034, +0.002267] | -0.029850 [-0.035307, -0.024201] |
| V24 frozen gate | -0.000092 [-0.002278, +0.002120] | -0.029906 [-0.035085, -0.024623] |

V25의 direct branch는 held-out의 81.0%에서 실행됐다. V24 frozen gate의 high-budget 비율은 0.0%였다. 같은 direct 실행률이지만 world와 독립인 coin으로 직접 선택과 K1024를 섞은 blind routing의 기대 Brier는 0.077635이고 실제 gate−blind 차이는 -0.001440 [-0.004146, +0.001178]다. Blind routing은 계산한 기대값이며 p90을 별도로 측정하지 않았다.

| 분기 | Worlds | Gate Brier | K256 Brier | 항상 직접 선택 Brier | 항상 K1024 Brier | Gate−K256 |
|---|---|---|---|---|---|---|
| confident | 1037 | 0.063619 | 0.063485 | 0.063619 | 0.062664 | +0.000134 |
| uncertain | 243 | 0.129863 | 0.131028 | 0.140182 | 0.129863 | -0.001164 |

두 branch의 worlds는 gate의 confidence로 선별돼 있다. 집단 간 평균 Brier를 비교해 gate의 인과효과로 해석할 수 없다. 같은 world에 서로 다른 정책을 강제로 적용한 paired comparison이 branch 선택의 근거다.

### 생성한 proxy에 얼마나 의존했는가

추가 분석에서는 전체 모델에서 두 가지 제한 모델을 만들었다. 하나는 profile mean과 수평 velocity만 쓰고, 다른 하나는 두 값을 제거한 나머지 네 features를 쓴다. 각 제한 모델을 원래 training/validation worlds에서만 24개 후보로 다시 학습한 뒤, 고정된 test K1024 fallback과 같은 decoder로 평가했다. 별도 prior width rule은 observation의 두 proxy를 generator의 offset 및 restitution 분포에서 유도한 근사 log-odds로 변환해 Q/P를 고른다. 이 rule은 실제 cell이나 label을 보지 않으나 generator 관계식을 사용한다. 이 표는 primary 계획 이후의 CPU replay로, 추가 p90은 측정하지 않았다.

| 모델 | Held-out Brier | 직접 선택률 | 모델−전체 gate [95% CI] | 모델−K256 [95% CI] | 전체 gate와 달랐던 고유 worlds |
|---|---|---|---|---|---|
| 전체 factual features | 0.076195 | 81.0% | +0.000000 [+0.000000, +0.000000] | -0.000113 [-0.002066, +0.002067] | 0 |
| Profile mean 및 수평 velocity | 0.075975 | 60.4% | -0.000221 [-0.001296, +0.000829] | -0.000333 [-0.002168, +0.001741] | 12 |
| 위 두 features 제거 | 0.080588 | 19.8% | +0.004392 [+0.001386, +0.007361] | +0.004280 [+0.000989, +0.007730] | 66 |
| 훈련 없는 prior width rule | 0.093122 | 100% | +0.016927 [+0.010569, +0.023504] | +0.016814 [+0.010682, +0.023268] | 306 |

| 전체 모델의 score 진단 | 고유 worlds | Spearman ρ · score와 실현 P−Q | Score 부호와 실현 winner 일치 | 평균 score / 실현 P−Q |
|---|---|---|---|---|
| train | 3072 | 0.582 | 80.0% | +0.0694 / +0.0694 |
| validation | 1024 | 0.549 | 79.5% | +0.0665 / +0.0585 |
| test | 1280 | 0.525 | 79.7% | +0.0699 / +0.0648 |

| Test score quintile | 평균 예측 advantage | 평균 실현 advantage | Probe 우세 worlds |
|---|---|---|---|
| 1 | -0.1311 | -0.1351 | 76.6% |
| 2 | +0.0099 | +0.0344 | 41.0% |
| 3 | +0.0912 | +0.0909 | 23.4% |
| 4 | +0.1522 | +0.1427 | 12.1% |
| 5 | +0.2272 | +0.1912 | 2.7% |

두 noise halves에서 0.02 이상 안정적으로 Q가 유리했던 worlds 중 직접 선택한 667개에서 Q를 놓친 경우는 39개였다. P가 안정적으로 유리했던 worlds 중 직접 선택한 278개에서는 P를 놓친 경우가 99개였다. Held-out score는 선택 전의 예측이고, 여기서 대조한 winner는 실제 hidden state와 반복 observation에 조건화된 realized loss로 정해진다.


![Within-calibration observation value and paired held-out Brier](results/v25.png)

## 6. 논문 기여 가능성과 적용 범위

**논문에 방어할 수 있는 주장은 관측 가치의 전환이다.** V24의 reference는 17,280개 선택 중 두 번만 query view를 벗어났다. V25는 동일한 query와 calibration에서 friction의 정보를 얻는 probe가 유리한 경우와 geometry를 읽는 view가 유리한 경우를 따로 예측하고, 독립적인 worlds에서 두 방향을 모두 검증했다. 비교 대상 K256도 probe를 활용했으므로 이 현상은 새 학습 모델의 고유한 동작이 아니다. V24 수치는 당시 prior·동역학·sensor가 달랐던 실험의 설명이며 두 버전 사이의 loss 차이를 인과효과로 해석하지 않았다.

**현재 gate를 성능 우위의 핵심 기여로 쓰기는 어렵다.** Always-query와 선택이 달랐을 때 gate는 평균 0.11608 Brier를 줄였다. 반면 K256과 선택이 달랐을 때 차이는 −0.00342였고, 전체 비교의 upper bound가 0을 넘었다. Fixed global mixture의 Brier 점추정은 0.075100, 고정 K1024는 0.075422로 gate보다도 낮았다. 새 confidence로 support만 배정한 policy와 v24 계수를 고정한 policy 역시 K256 대비 superiority를 보여주지 못했다. V24 frozen gate는 이 calibration에서 모두 K128을 택했다. 제공된 simulator supervision으로 직접 선택을 학습하는 방법과 factual support를 압축하는 방법은 서로 다른 algorithmic 주장이다. Fixed global의 p90은 이번에 측정하지 않아 속도 순위는 판단할 수 없다.

**설계한 proxy가 실제로 무엇을 설명하는지도 한계로 남는다.** Profile mean과 수평 velocity만 남긴 gate의 Brier는 0.075975였다. 두 값을 제거한 gate는 0.080588로 전체 gate보다 +0.004392 [95% CI +0.001386, +0.007361] 높았다. Factual signals에 observation 가치를 예측할 정보가 있다는 근거이면서, 특정 offset–curvature 및 restitution–friction 관계에 의존한다는 근거다. 나머지 features만 학습해도 fallback은 full factual record를 본다는 점을 고려해야 한다. 이 ablation은 gate 입력의 역할에 한정된다. Privileged cross-fitted Q/P winner의 Brier 0.040362는 unseen hidden state의 반복 결과를 이용하므로, gate와의 차이 전체를 알고리즘이 회수할 수 있다고 주장할 수 없다.

**다음 검증의 우선순위는 proxy와 probe의 실제 가치가 분리되는 상황이다.** Training은 고정하고, test에서 offset–curvature 및 restitution–friction 관계의 세기만 단계적으로 약화하거나 반전시킨다. Spread·query·calibration은 유지하고 K256, 직접 선택만 하는 0.945 ms p90 모델, always-query를 사전에 비교 대상으로 둔다. 이때도 gate가 K256을 이기는지와, probe를 실제로 구동하는 시간을 더해도 그만한 decision loss 절감이 남는지를 함께 봐야 한다. 같은 물체·같은 contact surface에서 반복 intervention을 측정한 자료가 있어야 simulated calibration 수준의 근거를 실물 성능으로 확장할 수 있다.

수치 비교에서 K256과 fallback은 기존 세 action kernel의 risk를 계산한 뒤 Q/P 두 값만 비교한다. 두 action 전용으로 최적화한 구현의 p90은 추가 실험이 필요하다. 다섯 bank가 달라도 acquisition sample plan의 seed는 공통이며, 결과는 이 simulator prior와 한 GPU에서 측정한 값이다.


## 문헌 검토와 v25의 기여 범위

### 1. 연구 질문, 방법, 표본, 결과, 한계

아래 다섯 원문은 물리적 정보를 얻는 방법, 다음 observation을 선택하는 기준, 그 기준을 근사하는 비용을 각각 다룬다. V25와 가장 가까운 비교는 실물 마찰 측정과, 같은 sensing 장치에서 object별 policy와 고정 policy를 직접 대조한 연구다.

| 연구 | Research question과 methodology | Sample과 data source | Key finding과 conclusion | V25로 옮길 때의 한계 |
|---|---|---|---|---|
| Sundaralingam & Hermans, *In-Hand Object-Dynamics Inference using Tactile Fingertips* | 손끝의 접촉력으로 friction을 측정할 수 있는가? 물체를 고정하고 fingertip을 움직여 slip 직전의 force와 surface normal을 관측했다. | YCB에서 고른 실물 8개, 물체당 10 trials. 두 용기는 plastic과 label 표면을 따로 측정했다. Tilt experiment로 reference를 얻었다. | 전체 friction coefficient 추정의 median error는 0.0056이었다. 힘을 받을 때 변형된 표면에서 오차가 커졌다. | Known geometry와 fingertip contact를 사용한다. 충돌 후 velocity를 읽는 v25 probe에서는 별도의 식별성 검증이 필요하다. [원문 §§IV–V](https://arxiv.org/pdf/2003.13165#page=6) |
| Dutta, Burdet & Kaboli, *Predictive Visuo-Tactile Interactive Perception Framework for Object Properties Inference* | 여러 시점의 형상과 push/pull 결과를 결합하면 물리 속성을 더 적은 상호작용으로 추정할 수 있는가? Next-best view, N-step information gain, dual differentiable filter를 사용했다. | Real robot. Homogeneous·heterogeneous·articulated object configurations가 각 40개로 총 120개다. Everyday objects에서도 별도로 시연했다. | Homogeneous 조건의 overall normalized error는 active GNN이 0.11, random GNN이 0.16이었다. 개별 parameter에서는 다른 방법이 앞선 사례가 있었다. | 여러 차례 update한 뒤의 parameter error다. 같은 query를 위해 view와 probe 중 하나를 사는 one-step decision은 이 수치로 평가할 수 없다. [원문 §IV, Table I](https://arxiv.org/html/2411.09020v1#S4) |
| Smith et al., *Active 3D Shape Reconstruction from Vision and Touch* | 지금까지의 관측으로 다음 touch 위치를 정하는 policy가 dataset 전체의 고정 탐색 순서보다 나은가? Haptic simulator에서 mesh reconstruction과 touch selection을 결합했다. | ABC에서 추출한 CAD models 40,000개 중 26,545개가 전처리를 통과했고, test set은 1,000개였다. | 다섯 actions 후의 Chamfer distance를 비교했다. 두 touch-only 설정에서는 dataset-specific 고정 순서 LEBA, 두 vision+touch 설정에서는 object-specific nearest-neighbor policy가 가장 좋은 비특권적 결과를 냈다. | 형상을 복원하는 simulation이다. Friction이나 stick을 평가하지 않았다. [원문 §§5.1·5.4, Table 2](https://proceedings.neurips.cc/paper/2021/file/8635b5fd6bc675033fb72e8a3ccc10a0-Paper.pdf#page=7) |
| Roy & McCallum, *Toward Optimal Active Learning through Monte Carlo Estimation of Error Reduction* | 다음 label 획득이 미래 예측 오차에 얼마나 도움이 되는지 직접 근사할 수 있는가? 후보 label을 적분하고 업데이트된 classifier의 future error를 추정했다. | 첫 Newsgroups 실험은 문서 2,000개를 train/test 1,000개씩 분리하고 10 trials를 평균했다. | 첫 실험에서 약 77.2% accuracy에 entropy-reducing 방법은 16 queries, disagreement 기준은 68 queries가 필요했다. | Label을 사서 classifier를 재학습한다. 단일 물체의 binary outcome, probe 시간, GPU p90은 측정하지 않았다. [원문 §§2·5](https://groups.csail.mit.edu/rrg/papers/icml01.pdf#page=5) |
| Dinkel et al., *Efficient Bayesian Optimal Experimental Design for Expensive Computational Models over Finite Design Sets* | 제한된 forward calls로 최선의 design을 구별할 수 있는가? Common random numbers, sample reuse, adaptive design elimination을 비교했다. | Synthetic, pharmacokinetic, lung, viscoplastic 네 사례. Synthetic 조건에서는 10개 designs의 exact EIG를 알고 1,000 independent runs에서 선택 정확도를 비교했다. | Synthetic 조건에서 sample reuse와 estimator 선택이 필요한 model evaluations를 줄였고, 경쟁력이 낮은 design을 조기 제외해 추가 절감을 얻었다. | EIG ranking의 정확도와 forward calls가 endpoint다. Held-out binary Brier 및 GPU latency는 별도로 측정해야 한다. [원문 §§2–3](https://arxiv.org/html/2607.16933v1#S3) |

### 2–3. Literature review와 knowledge map

**무엇을 측정해야 하는가.** Surface geometry는 특정 위치를 관측해 정밀하게 만들 수 있고, 마찰은 충분한 tangential excitation을 가해 관측 가능한 반응을 만들 수 있다. Smith et al.은 shape에 대한 시각 정보가 다음 touch의 가치를 바꿀 수 있음을 보였다. Sundaralingam & Hermans는 slip 직전의 실제 접촉 반응에서 friction을 얻었다. V25에서 두 결과를 연결하는 가설은 한 가지다. 같은 sensor noise에서도 최종 stick 예측을 좌우하는 불확실성이 world마다 다르면 더 유용한 observation도 달라질 수 있다. 이는 v25에서 검증할 가설이다. [Smith et al.](https://proceedings.neurips.cc/paper/2021/file/8635b5fd6bc675033fb72e8a3ccc10a0-Paper.pdf#page=8) · [Sundaralingam & Hermans](https://arxiv.org/pdf/2003.13165#page=7)

**어떻게 고를 것인가.** Dutta et al.은 물리 속성 추정의 uncertainty를, Roy & McCallum은 후속 예측의 expected error를 action 선택에 연결했다. Dinkel et al.은 유한한 후보들의 예상 가치를 어떤 계산 예산으로 비교할지 다뤘다. V25의 knowledge map은 **physical world → factual evidence → 가능한 query/probe 결과의 분포 → 각 관측 뒤의 stick probability → action별 expected Brier → 선택된 action의 실현 Brier**다. 이 중 **factual evidence → support budget**은 계산 경로이고, **support budget → 추정 순위 → 실제 선택**을 거쳐야 최종 손실에 영향을 준다. [Dutta et al.](https://arxiv.org/html/2411.09020v1#S3) · [Roy & McCallum](https://groups.csail.mit.edu/rrg/papers/icml01.pdf) · [Dinkel et al.](https://arxiv.org/html/2607.16933v1#S2)

### 4–7. 연구 공백, 상충하는 결과, methodology와 evidence synthesis

| 쟁점 | 비교 가능한 증거와 제한 | v25에서 구분할 주장 |
|---|---|---|
| **Object-specific policy와 고정 policy 중 어느 쪽이 유리한가?** | Smith et al.의 같은 simulator에서도 visual input을 추가하면 가장 좋은 정책이 바뀌었다. 측정 대상은 동일한 Chamfer distance였다. 저자들은 visual prior가 물체별 차이를 식별하기 쉽게 한 것으로 해석했다. 이는 입력 정보의 유용성이 달라지는 사례이며 접촉계의 보편 법칙은 아니다. [Smith et al., §5.4](https://proceedings.neurips.cc/paper/2021/file/8635b5fd6bc675033fb72e8a3ccc10a0-Paper.pdf#page=8) | World별 최적 observation의 다양성, 관측 전에 그 차이를 알아낼 수 있는지, 줄어든 loss가 계산 비용을 감당하는지는 별개다. Fixed K256도 world마다 action을 바꿀 수 있다. |
| **Probe가 friction을 알아내면 stick 예측도 개선되는가?** | 실제 slip을 유도한 8개 물체에서는 friction 추정이 가능했다. 다른 실물 연구에서 shape와 동역학을 함께 추정하자 복잡한 물체에 더 많은 상호작용이 필요했고, 개별 parameter 순위와 전체 오차 순위도 달랐다. Known surface normal을 사용하는 단일 접촉과 여러 미지 속성을 동시 추정하는 실험 간 수치는 합산할 수 없다. [Sundaralingam & Hermans](https://arxiv.org/pdf/2003.13165#page=7) · [Dutta et al., Tables I–III](https://arxiv.org/html/2411.09020v1#S4) | 실물 friction이 관측 가능하다는 근거는 있다. 같은 물체에서 query view와 probe 중 무엇이 새 velocity의 stick을 더 잘 예측하는지는 직접 측정해야 한다. |
| **Ranking과 compute는 최종 평가를 대신할 수 있는가?** | Exact EIG가 있는 synthetic case에서는 정답 design 식별 빈도와 model calls를 측정할 수 있다. Text active learning에서는 구입한 labels가 같은 수일 때 후속 accuracy를 측정했다. 두 수치는 서로 다른 모집단과 endpoint를 사용한다. [Dinkel et al.](https://arxiv.org/html/2607.16933v1#S3) · [Roy & McCallum](https://groups.csail.mit.edu/rrg/papers/icml01.pdf#page=5) | 같은 calibration, 동일 factual evidence·decoder, 동일 held-out worlds에서 paired Brier와 GPU p90을 각각 기록해야 비교가 가능하다. |

이 원문들에서 비교적 강한 근거가 있는 부분은 특정한 physical excitation으로 friction 정보를 얻을 수 있다는 점, 그리고 static 정책도 충분히 강한 경쟁자가 될 수 있다는 점이다. 그 범위는 각각 해당 실물 물체와 simulation task다. 서로 다른 논문의 sample을 합쳐 일반적인 adaptive gate 효과를 추정할 수 없다. 특히 **동일 calibration 안의 oracle switch가 factual gate의 K256 대비 우위로 이어진다**는 직접 증거는 이 corpus에 없다. [Sundaralingam & Hermans](https://arxiv.org/pdf/2003.13165#page=7) · [Smith et al., Table 2](https://proceedings.neurips.cc/paper/2021/file/8635b5fd6bc675033fb72e8a3ccc10a0-Paper.pdf#page=8)

### 8. 검증 가능한 연구 질문 10개

| 질문과 중요한 이유 | Variables와 검증 방법 | 근거가 되는 원문 |
|---|---|---|
| 1. 같은 calibration에서 실제 probe 우세 world와 view 우세 world가 모두 재현되는가? 하나가 지배적이면 adaptive selection을 평가하기 어렵다. | 한쪽 noise repeats에서 고른 winner를 남은 repeats에서 검증하고 두 방향의 빈도와 loss gap을 제시한다. | [Smith et al.](https://proceedings.neurips.cc/paper/2021/file/8635b5fd6bc675033fb72e8a3ccc10a0-Paper.pdf#page=8) |
| 2. Factual evidence에 winner를 구별할 정보가 있는가? Hidden world만 아는 oracle은 배포할 수 없다. | Factual features를 world 간 섞은 control과 실제 features를 비교하고 action별 realized Brier를 측정한다. | [Smith et al.](https://proceedings.neurips.cc/paper/2021/file/8635b5fd6bc675033fb72e8a3ccc10a0-Paper.pdf#page=8) |
| 3. 어떤 probe direction에서 friction ambiguity가 해소되는가? Stick 안쪽에 머무는 excitation은 competing friction hypotheses에 유사한 반응을 줄 수 있다. | Tangential velocity를 바꿔 competing hypotheses의 response separation과 Brier를 함께 추적한다. | [Sundaralingam & Hermans](https://arxiv.org/pdf/2003.13165#page=3) |
| 4. Diagnostic touch와 실제 query contact의 재질이 다르면 gain이 유지되는가? 같은 물체 안에서도 마찰이 달라질 수 있다. | Region별 friction 및 probe/query 위치 일치 여부를 교차하고 independent objects에서 평가한다. | [Sundaralingam & Hermans](https://arxiv.org/pdf/2003.13165#page=7) |
| 5. Geometry uncertainty의 어떤 성분이 물리 결과를 바꾸는가? 형상 복원이 정확해져도 query 위치의 tangent가 개선된다는 보장은 없다. | Global shape error, local normal error, stick Brier를 같은 worlds에서 비교한다. | [Smith et al.](https://proceedings.neurips.cc/paper/2021/file/8635b5fd6bc675033fb72e8a3ccc10a0-Paper.pdf#page=7) · [Sundaralingam & Hermans](https://arxiv.org/pdf/2003.13165#page=6) |
| 6. Expected information gain과 expected Brier reduction은 같은 action을 고르는가? 예측에 무관한 parameter를 잘 식별할 수도 있다. | 같은 bank와 sensor draws에서 두 criteria를 교체하고 다른 선택에서의 paired loss를 측정한다. | [Dutta et al.](https://arxiv.org/html/2411.09020v1#S3) · [Roy & McCallum](https://groups.csail.mit.edu/rrg/papers/icml01.pdf) |
| 7. 저비용 support가 큰 loss를 낳는 worlds를 미리 찾을 수 있는가? 쉬운 worlds에서 계산만 줄여도 평균 속도는 빨라질 수 있다. | 같은 mean budget의 static K, blind routing과 비교하고 disagreement subset 및 전체 loss를 함께 보고한다. | [Dinkel et al.](https://arxiv.org/html/2607.16933v1#S3) · [Smith et al.](https://proceedings.neurips.cc/paper/2021/file/8635b5fd6bc675033fb72e8a3ccc10a0-Paper.pdf#page=8) |
| 8. 접촉 상실이나 가림이 calibration을 바꾸는 경우에도 견디는가? 고정한 장비에서도 noise가 상태에 의존할 수 있다. | State-dependent noise를 추가하고 알려진 noise와 잘못 명시한 noise를 교차한다. | [Dutta et al., Fig. 11](https://arxiv.org/html/2411.09020v1#S4) |
| 9. Probe 취득 시간과 reset 비용까지 더하면 정책 선택이 달라지는가? Computation 절감과 물리적 비용은 다른 예산이다. | Sensor acquisition, reset, GPU p90을 분리 측정해 Brier–cost 곡선을 작성한다. | [Dutta et al.](https://arxiv.org/html/2411.09020v1#S4) · [Dinkel et al.](https://arxiv.org/html/2607.16933v1#S3) |
| 10. 실제 물체에서도 같은 query/probe crossover가 반복되는가? Synthetic 관측은 재질 변형이나 contact drift를 누락할 수 있다. | 같은 물체·위치·입사 조건을 reset한 뒤 두 observations를 교차 수집하고, object-level holdout으로 확인한다. | [Sundaralingam & Hermans](https://arxiv.org/pdf/2003.13165#page=7) · [Dutta et al.](https://arxiv.org/html/2411.09020v1#S4) |

### 9. 연구 프로젝트에 대한 검토

심사에서 가장 먼저 구분해야 할 비교는 **gate 대 always-query**와 **gate 대 K256**이다. 전자는 관측을 바꿀 가치, 후자는 이미 관측을 고를 수 있는 수치 근사와 비교한 gate의 추가 가치를 시험한다. 설계한 worlds에서 두 action이 각각 이기는 것을 보였더라도 deployable policy가 둘을 구별했다는 주장은 별도 결과가 있어야 한다. 승자 선정에 쓴 동일 noise로 이득을 계산하면 유리한 noise까지 함께 선택된다. 독립 repeats를 사용하고, 한 방법이 이긴 subset만 강조하지 말고 전체 worlds의 paired Brier와 p90을 함께 제시하는 것이 적절하다. 근접 선례도 fixed sequence와 object-specific policy의 순위가 입력 조건에 따라 바뀌는 것을 보여준다. [Smith et al.](https://proceedings.neurips.cc/paper/2021/file/8635b5fd6bc675033fb72e8a3ccc10a0-Paper.pdf#page=8) · [Dinkel et al.](https://arxiv.org/html/2607.16933v1#S3)

이 다섯 편으로 view selection, physical probing, posterior risk minimization 각각의 신규성을 주장하기 어렵다. 기여 후보는 **같은 query·calibration을 유지하며 world별 observation 가치가 교차하는 조건을 독립 설계하고, 그 가치가 실제 factual observations와 제한된 online compute에서 얼마나 회수되는지 검정한 것**이다. Synthetic benchmark의 positive result를 real robot의 성능 향상으로 일반화하려면 matched physical interventions, raw sensing에서의 uncertainty, reset 비용에 관한 별도 자료가 필요하다. [Dutta et al.](https://arxiv.org/html/2411.09020v1#S4) · [Sundaralingam & Hermans](https://arxiv.org/pdf/2003.13165#page=7) · [Roy & McCallum](https://groups.csail.mit.edu/rrg/papers/icml01.pdf)


## 재현 자료

서버의 프로젝트 폴더는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/within_calibration_v25`이다. [사전 protocol](PROTOCOL.md), [simulator와 sensor calibration](physics25.py), [factual gate](gate25.py), [실행 코드](run25.py), [primary 통계](stats25.py), [proxy ablation](diagnostics.py), [검증 결과](results/summary.json), [실측 선택 고정 기록](results/choices_freeze.json)에 구현과 수치를 보존했다. 이전 v18–v24 sibling dependencies를 유지해야 한다. 새 Slurm output directory에서 v18 guard로 run25.py를 실행한 뒤, 같은 directory의 compute-node CPU에서 stats25.py, diagnostics.py를 실행한다. results/eval_*.npz는 양쪽 intervention과 세 번째 view의 realized losses, interleaved timing blocks를 담고 있다.
