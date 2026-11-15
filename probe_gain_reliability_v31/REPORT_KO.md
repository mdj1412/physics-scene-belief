# v31 · 같은 관측에서 출발해도 추가 probe의 가치는 얼마나 재현되는가?

**추가 calibration probe의 가치는 어느 deployment batch에서 재느냐에 크게 좌우됐다.** 같은 과거 접촉을 본 환경마다 독립적인 접촉 경로 4개와 미래 물체 집단 4개를 교차했다. 세 contact law, 144개 cohort에서 각 환경의 다음 접촉이 다른 환경보다 가치 있는지를 두 번 독립적으로 판단했다. 아래에서 두 수치를 나란히 적을 때는 probe를 2→4회, 4→8회로 늘린 순서다.

**이미 factual frame을 보유한 물체에 probe만 더 샀을 때**, 실제 held-out Brier gain은 평균 **-0.00025·-0.00057**였다(pointwise 95% interval [-0.00075, +0.00022], [-0.00093, -0.00023]). Gain의 variance 중 calibration future의 main effect는 3.8%·2.5%, deployment batch의 main effect는 47.7%·50.7%, 두 축의 interaction은 48.4%·46.7%였다. 같은 접촉 결과가 어떤 배포 집단에서는 선택을 개선하고 다른 집단에서는 악화시키는 변동까지 interaction에 포함된다.

두 환경 중 어느 쪽에 추가 probe를 주는 게 나은지, 양쪽 난수를 모두 바꿔 다시 매긴 **순서 일치율은 49.1%·50.3%**였다(simultaneous 95% interval 47.8–50.3% 및 49.0–51.6%, 무작위 기준 50%). 새 물체에 factual과 probe를 함께 추가한 별도 panel에서도 49.5%·49.8%였다. 한 번의 gain으로 선별한 네 환경을 그 자료에서 다시 채점하면 무작위 배분 대비 +0.00283·+0.00286 Brier만큼 좋아 보였으나, 같은 선택을 독립 반복에서 채점한 이득은 -0.00004·+0.00002였다. 전체 실험은 같은 prefix에서 접촉의 가치가 재현되는지 평가하며, 온라인에서 이 선별에 쓴 미래 정답을 이용할 수는 없다.

## 1. 같은 질문을 독립적으로 반복하는 방법

**어떤 gain인가.** 각 world는 geometry·friction·restitution을 모르는 충돌 장면이다. 미래 물체에서는 충돌 위치의 정밀 boundary profile인 query view, 또는 다른 입사속도로 접촉한 뒤의 2D outgoing velocity인 probe 중 하나를 얻는다. 목표는 그 물체의 별도 충돌에서 stick이 일어날 확률이다. Brier loss는 예측 확률과 simulator에서 확인한 binary stick label의 차이 제곱이다. 미래 batch 128개 중, 현재 selector가 추정한 risk(query)−risk(probe)가 가장 높은 13개에만 probe를 쓴다. 나머지 115개에는 query view를 쓴다.

**고정하는 것.** 환경마다 이미 관측한 1,024개 calibration worlds의 factual evidence와 그중 무작위로 고른 물체들의 유료 접촉 결과를 고정했다. 2→4 stage는 처음 2회의 응답을, 4→8 stage는 같은 처음 2회를 포함한 4회의 응답을 paid prefix로 한다. 각 stage의 추가 접촉은 그 stage에서 독립적으로 다시 생성한다. 둘째 stage는 모두 4회까지 이미 관측했다고 가정한 비교다. v30에서 특정 정책이 처음 2회 뒤 계속하기로 선택한 환경만을 모은 결과와 구분한다. 같은 환경 내에서 posterior 계산법, acquisition의 K256/M2048 설정 및 수치 bank도 고정한다.

**서로 독립적으로 다시 뽑는 것.** 각 prefix에서 4개 calibration futures가 갈라진다. 각 future에는 같은 환경 분포에서 뽑은 **새 물체** 2개 또는 4개와 그 물체의 새로운 factual frame과 noisy probe response가 들어간다. 이미 본 물체의 동일한 숨은 상태에 noise만 다시 더한 반복이 아니다. 각 추가 물체의 factual과 probe는 한 쌍의 joint likelihood로 한 번만 반영했다. 그러므로 primary gain은 *새 물체를 만나 factual과 probe를 함께 받은 효과*다. 추가 factual만 보여 주는 비교는 §5에서 확인했다. V30에서는 추가 probe를 과거 factual pool의 물체에서 구입했다는 차이가 있다.

**평가하는 것.** 각 환경에서 calibration과 겹치지 않는 deployment batches를 4개 생성했다. Batch마다 물체 128개, 각 action의 sensor noise 32회가 새로 생성된다. 같은 열의 batch 안에서는 물체·label·sensor draws를 모든 future와 전후 선택에 공유한다. 서로 다른 환경은 같은 index의 batch나 future라 해도 물체를 공유하지 않는다. 예를 들어 두 번째 calibration future의 갱신 결과를 첫 번째 배포 집단과 네 번째 배포 집단에 모두 적용하면, 해당 접촉이 바꾼 선택이 어떤 물체에서 유효했는지 비교할 수 있다. 각 cell의 gain은 동일 batch에서 **prefix에서 멈췄을 때 Brier − 추가 접촉 후 Brier**다. 양수가 실제 예측의 개선이다.

**어떤 물리 환경에서 실험했나.** Source Coulomb, static threshold와 kinetic impulse의 계수를 나눈 static–kinetic split, tangential speed에 따라 계수를 바꾼 speed weakening을 사용했다. 각 law 안의 8개 환경은 geometry proxy와 curvature spread, velocity proxy와 friction spread의 결합 강도·방향이 다르다. 법칙마다 이 여덟 조건을 섞은 독립 block을 6번 만들었다. 총 18 blocks, 144개 환경 cohort, 288개 stage별 paid prefixes, 73,728개 독립 deployment worlds, 4,608개 paired gain cells이다. 2개 stage와 4개 미래에서 같은 batch를 재사용했으므로 물체 수를 중복 계수하지 않았다.

Common decoder에는 해당 환경의 population coupling을 알려주되 물체 고유의 숨은 물성과 새 contact law를 알려주지 않았다. 같은 world·action에는 항상 같은 확률 예측을 쓴다. 두 shift laws에서 추론과 decoding은 Coulomb law를 계속 가정한다. 실제 배포에서 population coupling을 정확히 알고 있다고 할 수 없으므로 이는 관측 선택의 영향을 재는 privileged common decoder이다. 미래 관측과 stick label을 열기 전에 각 stage·future·batch에서 고른 action을 저장하고 hash로 고정했다. V31에서는 새 scorer를 학습하지 않았다. [설계와 seeds](design.json) · [실행 코드](runner31.py) · [두 추가 물리 법칙](../learned_probe_value_v30/contact30.py) · [v30 보고서](../learned_probe_value_v30/REPORT_KO.md)

## 2. 어느 축이 실제 Brier gain을 흔드는가?

각 paid prefix에서 4×4 gain matrix를 만들었다. 같은 행의 네 cell은 calibration 결과가 같고, 같은 열의 네 cell은 평가하는 물체·관측·정답이 같다. Row mean과 column mean으로 설명한 뒤 남는 교차 변동이 interaction이다. 항을 먼저 각 prefix 안에서 추정한 뒤 여덟 환경, 독립 blocks, 세 law에 동일 가중치를 주었다. 환경 자체의 물성 분포 차이는 이 표의 variance에 섞지 않았다.

| Stage | 변동 요인 | Variance ×10⁶ Brier² [95%] | 전체 변동의 비중 | Component SD (Brier) |
|---|---|---|---|---|
| 2→4회 | Calibration future | 1.76 [-0.15, 3.80] | 2.8% | 0.00133 |
| 2→4회 | Deployment batch | 29.05 [24.23, 34.62] | 46.4% | 0.00539 |
| 2→4회 | Future × batch | 31.85 [29.53, 34.51] | 50.8% | 0.00564 |
| 4→8회 | Calibration future | 1.24 [-0.97, 3.83] | 1.8% | 0.00111 |
| 4→8회 | Deployment batch | 35.18 [30.14, 40.49] | 50.4% | 0.00593 |
| 4→8회 | Future × batch | 33.43 [30.27, 36.55] | 47.9% | 0.00578 |

| Stage | Batch − future variance ×10⁶ [simultaneous 95%] | 전체 1×1 SD | 4×4 평균 SD | 평균 signed gain [pointwise 95%] |
|---|---|---|---|---|
| 2→4회 | 27.29 [20.83, 33.74] | 0.00792 | 0.00311 | -0.00008 [-0.00053, 0.00038] |
| 4→8회 | 33.95 [25.65, 42.25] | 0.00836 | 0.00335 | -0.00077 [-0.00128, -0.00026] |

Component SD는 각 variance의 제곱근이므로 서로 더할 수 없다. 마지막 표의 SD는 한 환경에서 반복 표집한 gain 또는 그 평균의 조건부 표준편차를 variance 평균의 제곱근으로 요약한 값이다. 전체 실험의 평균 gain에 대한 confidence interval과는 단위가 같아도 질문이 다르다. 한 future를 고정하고 batch만 바꿔 생기는 변동에는 batch main effect와 interaction이 모두 들어간다. Future를 바꿀 때도 같은 interaction이 들어간다. 따라서 interaction을 양쪽에 중복 배분하거나 임의로 반분하지 않았다.

### 새 반복 하나를 더 구입한다면

Crossed random-effects 모형에서 Var(mean gain) = V_F/n_F + V_B/n_B + V_I/(n_F·n_B)이다. 아래는 동일한 수의 future×batch cell을 조사하면서 배치를 다르게 한 경우다. 각 열의 값은 조건부 RMS 오차(Brier) 추정치다. 한 batch는 128개 world, 한 future는 2개 또는 4개의 새로운 probe이므로, sensor·GPU 비용이 같다는 뜻은 아니다.

| (Futures, batches) | 2→4 RMS | 4→8 RMS |
|---|---|---|
| 1 × 1 | 0.00792 | 0.00836 |
| 2 × 1 | 0.00677 | 0.00725 |
| 1 × 2 | 0.00568 | 0.00596 |
| 2 × 2 | 0.00483 | 0.00515 |
| 4 × 1 | 0.00612 | 0.00662 |
| 1 × 4 | 0.00412 | 0.00429 |
| 4 × 4 | 0.00311 | 0.00335 |

## 3. 환경 순위도 재현되는가?

같은 law의 여덟 환경에서 둘을 골라 어느 쪽의 gain이 더 높은지 판단한 후 다른 반복에서 순서가 유지되는지 계산했다. 각 환경은 독립적인 future 및 batch를 가지므로 환경 쌍마다 허용되는 반복 조합을 독립적으로 열거했다. 동률은 0.5, 무작위 순서의 기준은 50%다. 아래의 |Δgain|과 0.003 초과율은 한 prefix에서 반복 평가가 서로 얼마나 다른지를 나타낸다.

| 반복 비교 | 2→4 순서 일치 % [95%] | 평균 절대 gain 차이 | 0.003 초과 | 4→8 순서 일치 % [95%] | 평균 절대 gain 차이 | 0.003 초과 |
|---|---|---|---|---|---|---|
| 한 future 고정, 새 batch | 50.4 [49.2, 51.7] | 0.00860 | 77.4% | 49.9 [48.0, 52.1] | 0.00914 | 77.4% |
| 한 batch 고정, 새 future | 65.1 [64.0, 66.1] | 0.00640 | 69.2% | 66.9 [65.5, 68.3] | 0.00642 | 68.8% |
| 두 축 모두 새로 추출 · 1×1 | 49.5 [47.9, 51.0] | 0.00872 | 77.8% | 49.8 [48.4, 51.3] | 0.00925 | 78.8% |
| 독립 1×2 평균끼리 | 49.7 [47.2, 52.3] | 0.00624 | 68.3% | 49.5 [47.3, 51.6] | 0.00661 | 70.9% |
| 독립 2×1 평균끼리 | 49.1 [47.4, 50.9] | 0.00749 | 74.0% | 49.9 [48.2, 51.6] | 0.00802 | 75.4% |
| 독립 2×2 평균끼리 | 49.4 [46.0, 53.0] | 0.00532 | 63.7% | 49.8 [46.6, 52.8] | 0.00575 | 67.2% |

두 축이 모두 새로운 1×1 순서 비교와 앞 절의 variance 차이, 각 두 stage를 묶어 총 4개 primary contrast에 simultaneous 95% interval을 계산했다. 나머지는 별도 표기 없는 한 pointwise interval이다. 세 law에서 독립 block 여섯 개씩을 law 내에서 paired bootstrap 10,000회 재표집했다. 28개 환경 쌍, noise 32회, 반복 조합들은 별도의 독립 실험이 아니다. 법칙당 block 수가 여섯 개여서 특히 law별 interval의 꼬리는 조심해서 해석해야 한다. [통계 계획](STATISTICAL_PLAN_KO.md) · [통계 코드](stats31.py) · [전체 추정치와 구간](results/crossed_stats31.json)

| Contact law | Stage | 평균 gain [95%] | Future ×10⁶ | Batch ×10⁶ | Interaction ×10⁶ | 독립 1×1 순서 일치 % [95%] |
|---|---|---|---|---|---|---|
| Coulomb | 2→4회 | -0.00002 [-0.00090, 0.00082] | 0.89 | 28.79 | 29.60 | 49.0 [47.3, 51.4] |
| Coulomb | 4→8회 | -0.00025 [-0.00104, 0.00041] | 0.56 | 43.29 | 28.60 | 48.2 [47.0, 49.5] |
| Static–kinetic split | 2→4회 | -0.00003 [-0.00060, 0.00079] | -1.72 | 23.62 | 30.47 | 50.5 [48.5, 53.1] |
| Static–kinetic split | 4→8회 | -0.00089 [-0.00186, 0.00011] | 0.26 | 27.01 | 28.23 | 50.3 [48.3, 52.2] |
| Speed weakening | 2→4회 | -0.00018 [-0.00096, 0.00051] | 6.13 | 34.75 | 35.48 | 48.8 [46.9, 50.8] |
| Speed weakening | 4→8회 | -0.00117 [-0.00210, -0.00025] | 2.88 | 35.26 | 43.45 | 51.0 [48.6, 53.5] |

표 안의 음수 variance component가 있다면 제한을 두지 않은 method-of-moments 추정의 유한 표본 오차다. 추정치를 0으로 바꿔 우세한 축을 결정하지 않았다.

**같은 평가 물체에서 다시 채점하면 순위가 실제보다 안정적으로 보였다.** 새 물체를 받아 factual과 접촉을 함께 더한 panel에서, deployment batch를 공유하고 calibration future만 바꿀 때 순위 일치는 65.1%와 66.9%였다. 반대로 calibration future가 같더라도 batch를 바꾸면 50.4%와 49.9%였다. Batch와 future를 모두 바꾼 1×1 결과의 절대 차이는 평균 0.00872·0.00925 Brier였다. V30의 실용적 참조 크기인 0.003보다 두 반복이 더 크게 달랐던 비율은 77.8%·78.8%였다. 반면 32회 sensor-noise 평균을 사용하면서 남은 조건부 Monte Carlo SE의 RMS는 0.00051·0.00051였다. 이 SE는 같은 물체·label·action을 고정했을 때 관측 noise 평균을 근사하는 오차다. 나머지 모든 변동을 실제 환경 차이로 돌릴 수는 없다. [독립 world별 검산](results/cell_replay31.json)

**한 번 큰 이득을 본 환경을 다시 고를 근거는 약했다.** Fresh-object cell의 gain이 0.003을 넘은 비율은 33.6%·31.1%였다. 첫 독립 반복에서 이미 0.003을 넘은 환경으로 한정해도, 다음 반복에서 같은 기준을 넘은 비율은 33.3%·30.5%였다. 또한 법칙 하나씩을 제외해도 batch − future variance의 point estimate는 두 stage에서 모두 양수였고, 1×1 순위 일치는 48.9–50.7% 범위였다. 문턱 초과와 law 제외 결과는 기술적 보조 분석이다. [문턱 초과 추정치](results/practical31.json) · [Leave-one-out 결과](results/sensitivity31.json)

4×4를 평균한 뒤 여덟 환경 사이에서 관측된 variance는 8.99·10.61 ×10⁻⁶ Brier²였다. 여기에는 각 환경 평균을 16개 cell로 추정한 오차도 들어 있다. 해당 오차를 빼면 between-prefix signal의 추정치는 −0.71 [−2.89, 1.81] 및 −0.58 [−2.74, 1.63] ×10⁻⁶이었다. 실재하는 variance가 음수라는 의미가 아니다. 이 크기의 표본으로는 환경별 기대 gain의 차이를 양수로 분리하지 못했다. 따라서 다음 실험에 필요한 반복 수를 현재의 rank만으로 확정할 수 없다. [보정 전후 추정치](results/crossed_stats31.json)

## 4. 그 순위로 실제 probe를 배분하면

각 stage에서 후보 8곳 중 4곳에만 같은 수의 probe를 더 준다고 가정했다. 표는 무작위로 4곳을 고를 때의 정확한 기대 gain에 **더해지는** Brier 감소량이다. 선택한 4곳에서 얻은 gain을 전체 8곳으로 나눴다. 양수면 같은 비용에 이익이다. 1×1 또는 2×2로 표시한 행은 미래 결과를 먼저 본 사후 진단이다. 반면 frozen scorer와 uncertainty-only는 paid prefix만 보고 순위를 정한다. 이는 두 stage를 각각 떼어 본 실험이어서 v30 전체 stopping policy의 Brier로 읽을 수 없다.

| 네 환경을 선택한 방법과 평가 자료 | 2→4회 추가 배분 이득 [95%] | 4→8회 추가 배분 이득 [95%] |
|---|---|---|
| 1×1 · 선택에 사용한 결과에 재평가 | 0.00280 [0.00265, 0.00296] | 0.00296 [0.00283, 0.00308] |
| 1×1 · 양쪽이 독립인 결과에 평가 | -0.00004 [-0.00016, 0.00010] | -0.00005 [-0.00017, 0.00007] |
| 2×2 · 선택에 사용한 평균에 재평가 | 0.00170 [0.00157, 0.00184] | 0.00182 [0.00172, 0.00192] |
| 2×2 · 양쪽이 독립인 평균에 평가 | -0.00003 [-0.00023, 0.00018] | -0.00002 [-0.00020, 0.00016] |
| V30 frozen scorer · 4×4 평균으로 채점 | 0.00006 [-0.00021, 0.00033] | 0.00027 [-0.00001, 0.00055] |
| Uncertainty-only · 4×4 평균으로 채점 | 0.00002 [-0.00020, 0.00020] | -0.00015 [-0.00041, 0.00013] |

독립 반복으로 고른 상위 네 환경의 평균 겹침 비율은 1×1에서 49.6%·49.6%, 2×2에서 48.8%·50.3%였다(2→4·4→8 순서). 서로 무작위로 네 곳을 고르면 평균 50%가 겹친다. [사전 보조 분석 설계](PORTFOLIO_PLAN_KO.md) · [재현 코드](portfolio31.py) · [보조 결과](results/portfolio31.json)

별도 질문으로, 같은 population coupling이라도 factual pool·paid prefix·numerical bank를 모두 새로 뽑아서 순위를 매겼다. 각 독립 block 안에서 4×4 gain을 평균하고 같은 여덟 coupling을 대조했다. 아래는 6개 독립 block을 나눈 기술 통계다. 새 prefix까지 일반화하는 질문이라 앞 절의 동일 prefix 재현성과 구분한다.

| 동일 coupling을 새 prefix에서 재측정 | 2→4회 순서 일치율 | 4→8회 순서 일치율 |
|---|---|---|
| 1개 block 대 독립 1개 | 50.6% | 52.5% |
| 3개 평균 대 겹치지 않는 3개 평균 | 49.0% | 56.0% |

## 5. 새 물체의 factual frame과 접촉 응답은 각각 무엇을 바꾸는가?

새 물체의 factual frame을 X, 유료 접촉의 응답을 Y라고 두었다. 원래 prefix의 loss에서 X만 추가한 loss를 뺀 값이 factual gain이다. X만 본 loss에서 X와 Y를 모두 본 loss를 뺀 값이 conditional probe gain이다. 같은 물체·batch·decoder에서 계산했으므로 두 signed gain의 합은 primary gain과 정확히 같다. 아래 값은 미리 지정한 각 law의 첫 block, 총 24개 cohort에서 얻은 기술 통계다.

| 추가한 evidence | 2→4 평균 signed gain | 4→8 평균 signed gain | 2→4 평균 절대 gain | 4→8 평균 절대 gain |
|---|---:|---:|---:|---:|
| X와 Y 함께 | −0.00026 | −0.00153 | 0.00582 | 0.00592 |
| X만 | −0.00065 | −0.00124 | 0.00477 | 0.00522 |
| 같은 X에서 Y 추가 | +0.00039 | −0.00029 | 0.00561 | 0.00594 |

Y를 추가했을 때 cell별 변화의 크기는 평균 약 0.006 Brier였다. 방향을 포함해 평균하면 두 stage에서 부호가 달랐다. 같은 Y의 환경별 gain 순서가 두 축을 전부 새로 뽑아도 유지된 비율은 50.8%와 51.7%였다. Factual frame만 추가해도 선택이 바뀌므로 fresh-object 실험 전체의 변동을 모두 유료 접촉의 효과로 돌릴 수 없다. 절대 gain은 부호가 상쇄되기 전 크기여서 두 항의 절댓값을 더해 전체 절댓값을 복원할 수 없다. 이 control은 law당 한 block이므로 interval을 부여하거나 law 우열을 판정하지 않았다. [Control 결과](results/control31.json) · [재계산 코드](control31.py)

### 이미 관측한 물체에 접촉만 추가하면

이 보충 실험은 v30과 같은 구입 절차를 사용한다. 위와 **같은 144개 cohort, paid prefix, numerical bank, 73,728개 deployment worlds, common decoder**를 다시 썼다. 각 future는 원래 1,024개 factual pool에서 그 stage에 이미 접촉한 물체를 제외하고 서로 다른 2개 또는 4개를 고른다. 보관된 실제 물체의 latent state에서 새로운 noisy contact response만 받는다. 새 물체의 factual 정보는 추가되지 않는다. 서로 다른 future가 같은 물체를 뽑을 수 있다. 독립성의 대상은 고정한 물체 집단에서의 무작위 선택과 sensor noise다. Fresh-object control을 본 뒤 이 검증을 추가했고, secondary 결과를 확인하기 전에 설계를 기록했다. [보충 검증 설계](MATCHED_PLAN_KO.md)

| 추가 관측 | Stage | Signed Brier gain [95%] | Future ×10⁶ | Batch ×10⁶ | Interaction ×10⁶ | Batch − future ×10⁶ [simultaneous 95%] |
|---|---|---|---|---|---|---|
| 새 물체 · X+Y | 2→4 | -0.00008 [-0.00053, 0.00038] | 1.76 | 29.05 | 31.85 | 27.29 [20.83, 33.74] |
| 새 물체 · X+Y | 4→8 | -0.00077 [-0.00128, -0.00026] | 1.24 | 35.18 | 33.43 | 33.95 [25.65, 42.25] |
| 기존 물체 · Y만 | 2→4 | -0.00025 [-0.00075, 0.00022] | 2.49 | 31.08 | 31.53 | 28.59 [22.41, 34.77] |
| 기존 물체 · Y만 | 4→8 | -0.00057 [-0.00093, -0.00023] | 1.64 | 32.82 | 30.24 | 31.18 [24.85, 37.51] |

| 추가 관측 | Stage | 독립 1×1 순서 일치 % [simultaneous 95%] | 독립 2×2 순서 일치 % [pointwise 95%] | 교체한 probe 수 / 13 | 교체 0건인 cell |
|---|---|---|---|---|---|
| 새 물체 · X+Y | 2→4 | 49.5 [47.9, 51.0] | 49.4 [46.0, 53.0] | 6.43 | 0.0% |
| 새 물체 · X+Y | 4→8 | 49.8 [48.4, 51.3] | 49.8 [46.6, 52.8] | 6.54 | 0.2% |
| 기존 물체 · Y만 | 2→4 | 49.1 [47.8, 50.3] | 48.5 [44.9, 52.0] | 6.49 | 0.0% |
| 기존 물체 · Y만 | 4→8 | 50.3 [49.0, 51.6] | 50.0 [46.6, 53.1] | 6.50 | 0.1% |

교체 수는 추가 calibration 후 128개 미래 물체 중 원래 probe를 받았던 13개에서 빠진 물체 수다. 같은 수의 다른 물체가 새로 probe를 받는다. 따라서 전체 batch Brier가 거의 변하지 않은 이유가 애초에 action이 같았기 때문인지 확인할 수 있다. 두 panel 모두 저장된 world별 두 action의 Brier 차이와 교체 명단으로 전체 gain을 복원했다. [Fresh-object action 검산](results/swap31.json) · [기존 물체 action 검산](results_pool/swap31.json)

| 추가 관측 | Stage | 한 반복으로 선택·재평가 | 독립 반복에 평가 | V30 frozen scorer |
|---|---|---|---|---|
| 새 물체 · X+Y | 2→4 | 0.00280 [0.00265, 0.00296] | -0.00004 [-0.00016, 0.00010] | 0.00006 [-0.00021, 0.00033] |
| 새 물체 · X+Y | 4→8 | 0.00296 [0.00283, 0.00308] | -0.00005 [-0.00017, 0.00007] | 0.00027 [-0.00001, 0.00055] |
| 기존 물체 · Y만 | 2→4 | 0.00283 [0.00273, 0.00293] | -0.00004 [-0.00016, 0.00007] | 0.00006 [-0.00011, 0.00022] |
| 기존 물체 · Y만 | 4→8 | 0.00286 [0.00273, 0.00299] | 0.00002 [-0.00008, 0.00012] | 0.00025 [-0.00002, 0.00052] |

마지막 표는 여덟 환경 중 네 곳의 접촉 예산을 배정해 균등 무작위 배분보다 얻은 Brier gain이다. 두 panel이 같은 prefix와 배포 물체를 공유하므로 관측 절차 자체의 차이에 대한 검정은 block을 짝지어 계산했다. [보충 실험의 전체 추정치](results_pool/crossed_stats31.json) · [동일 비용 배분](results_pool/portfolio31.json)

| Contact law | 추가 물체 | 2회 Brier | 4회 Brier(2에서 계속) | 4회 Brier(고정 prefix) | 8회 Brier(4에서 계속) |
|---|---|---|---|---|---|
| Coulomb | 새 물체 | 0.10675 | 0.10678 | 0.10627 | 0.10652 |
| Coulomb | 기존 물체 | 0.10675 | 0.10712 | 0.10627 | 0.10718 |
| Static–kinetic split | 새 물체 | 0.13953 | 0.13956 | 0.13860 | 0.13949 |
| Static–kinetic split | 기존 물체 | 0.13953 | 0.13943 | 0.13860 | 0.13892 |
| Speed weakening | 새 물체 | 0.12714 | 0.12732 | 0.12660 | 0.12778 |
| Speed weakening | 기존 물체 | 0.12714 | 0.12762 | 0.12660 | 0.12708 |

표의 앞뒤 Brier는 같은 stage에서 같은 128개 물체와 decoder로 평가했다. 두 4회 열은 서로 다른 calibration history다. 첫 열의 4회는 2회 prefix에서 독립적으로 분기한 future를 평균했고, 다음 열의 4회는 원래 고정해 둔 실제 4회 prefix로 시작한다.

### 같은 paid prefix에서 나온 실제 4×4 표

파일 순서상 첫 cohort(Coulomb, block 0, environment 0, coupling -0.65·+0.85)의 2→4회 예다. 모든 행은 보유한 물체에 접촉만 추가한다. 같은 행에서 B1과 B2의 차이는 같은 접촉 결과가 다른 배포 집단을 만났을 때의 차이다. 같은 열에서 F1과 F2를 비교하면 어느 물체에 추가 접촉했는지만 달라진다.

| 독립 calibration future | Batch 1 | Batch 2 | Batch 3 | Batch 4 | 행 평균 |
|---|---|---|---|---|---|
| F1 | +0.00642 | -0.00512 | -0.00052 | -0.00192 | -0.00029 |
| F2 | +0.00369 | -0.00998 | -0.00678 | -0.00171 | -0.00370 |
| F3 | +0.00791 | -0.00585 | -0.00340 | -0.00165 | -0.00075 |
| F4 | +0.00701 | -0.00356 | -0.00622 | +0.00254 | -0.00006 |
| 열 평균 | +0.00626 | -0.00613 | -0.00423 | -0.00069 | -0.00120 |

표의 단위는 signed Brier gain이다. 같은 행의 네 값이나 같은 열의 네 값은 하나의 축을 공유한다. [해당 cohort의 raw gain](results_pool/coulomb/block_000/metrics.npz)

### 두 구입 방식의 paired 차이

| Stage | 비교 지표 | 기존 물체 − 새 물체 [8-contrast simultaneous 95%] |
|---|---|---|
| 2→4 | 평균 Brier gain | -0.00017 [-0.00062, 0.00028] |
| 2→4 | Batch − future variance ×10⁶ | 1.30 [-4.22, 6.82] |
| 2→4 | 독립 1×1 순서 일치율(%p) | -0.41 [-2.16, 1.35] |
| 2→4 | 독립 2×2 순서 일치율(%p) | -0.86 [-6.19, 4.47] |
| 4→8 | 평균 Brier gain | 0.00020 [-0.00027, 0.00067] |
| 4→8 | Batch − future variance ×10⁶ | -2.77 [-10.46, 4.93] |
| 4→8 | 독립 1×1 순서 일치율(%p) | 0.50 [-0.76, 1.76] |
| 4→8 | 독립 2×2 순서 일치율(%p) | 0.13 [-3.61, 3.87] |

이 차이의 interval이 0을 포함하면 관측 방식의 차이를 검출하지 못했다는 뜻이다. 두 방식이 동등하다고 판정하려면 실용적 margin을 먼저 정하고 equivalence를 검정해야 한다. [분석 계획](POOL_STATS_PLAN_KO.md) · [Paired 추정 및 sensitivity](results_pool/paired_pool31.json)

### 어떤 환경의 순위가 다시 측정해도 유지되는가?

기존 물체에서 Y만 사는 조건에서 같은 prefix에 독립적인 future와 batch를 다시 적용했다. 각 반복마다 같은 law 안의 여덟 환경을 실제 gain 순으로 1–8위로 정렬했다. 아래는 모든 법칙·block을 같은 가중치로 평균한 결과다. 각 순위는 cell 하나에서 측정한 realized gain으로 정했다. 예를 들어 2위에서 5위가 되면 절대 이동은 3이다. 같은 cohort에서 양쪽 난수가 모두 다른 반복끼리 비교했고, 동률에는 평균 순위를 부여했다.

| Geometry coupling | Velocity–friction coupling | 2→4 평균 순위 | 평균 절대 이동 | 3순위 이상 이동 | 4→8 평균 순위 | 평균 절대 이동 | 3순위 이상 이동 |
|---|---|---|---|---|---|---|---|
| -0.65 | -0.85 | 4.72 | 2.53 | 45.1% | 4.36 | 2.55 | 46.0% |
| -0.65 | +0.85 | 4.55 | 2.73 | 49.2% | 4.90 | 2.50 | 43.5% |
| -0.20 | -0.35 | 4.40 | 2.56 | 46.4% | 4.40 | 2.69 | 48.2% |
| -0.20 | +0.35 | 4.36 | 2.48 | 43.8% | 4.31 | 2.68 | 46.5% |
| +0.35 | -0.65 | 4.21 | 2.73 | 47.8% | 4.61 | 2.51 | 42.4% |
| +0.35 | +0.65 | 4.58 | 2.77 | 49.2% | 4.62 | 2.60 | 47.6% |
| +0.72 | -0.92 | 4.54 | 2.70 | 47.8% | 4.57 | 2.67 | 46.6% |
| +0.72 | +0.92 | 4.64 | 2.75 | 48.5% | 4.22 | 2.67 | 47.6% |

두 coupling은 각각 profile mean–curvature spread와 horizontal velocity–friction spread 관계의 simulator 계수다. 각 행에는 같은 계수로 생성한 18개 서로 다른 prefix가 들어 있다. 순위 이동은 매번 같은 물체를 다시 배치한 결과가 아니라, 같은 prefix에서 새 접촉과 새 deployment worlds를 생성한 결과다. 완전히 무작위이고 서로 독립인 1–8위 순위의 평균 절대 이동은 2.625이고, 3순위 이상 움직일 확률은 46.875%다. [환경별 수치와 law별 결과](results_pool/envrank31.json) · [계산 코드](envrank31.py)

![추가 calibration probe의 변동 요인과 환경별 순위 재현성](results/v31.png)

## 6. 논문 기여 가능성, 한계와 다음 결정

**환경 순위를 맞히는 문제의 실질적 크기.** 단일 stage에서 여덟 환경에 대한 실제 기대 gain을 μ, 네 곳에 probe를 주는 indicator를 w라 두면 균등 무작위 기준을 넘는 기대 이득은 여덟 환경에 대한 평균 (w−0.5)(μ−평균 μ)다. Cauchy–Schwarz로 이는 0.5 × √[(7/8) × 환경 간 sample variance] 이하다. Block·law에 같은 가중치를 주고 Jensen 부등식을 적용해도 성립한다. 이때의 variance는 finite grid noise를 뺀 기대 gain의 환경 간 분산이다.

그 분산의 pointwise 95% bootstrap interval 상단을 대입한 **탐색적 상한**은 새 물체 0.00063·0.00060 / 기존 물체 0.00044·0.00054 Brier였다(2→4·4→8). 제한된 block에서 추정한 interval에 의존하므로 엄밀한 보장이나 모든 stopping policy의 상한으로 쓸 수 없다. 완벽한 기대 순위를 알면 이득이 반드시 이 상한에 도달한다는 뜻도 아니다. 실현된 단일 cell에서 상위 네 곳을 고른 낙관적 이득이 왜 그대로 배포 목표가 되기 어려운지를 크기까지 설명하는 진단이다. [분산과 상한](results/finite_gain_ceiling31.json) · [계산 코드](finite_gain_ceiling31.py)

**무엇을 더 반복할지.** 두 panel 모두 batch main effect가 future main effect보다 컸으므로 같은 cell 수에서 batches를 먼저 늘리는 편이 평균 gain의 sampling variance를 더 줄인다. 다만 interaction도 커서 동일 future만 계속 쓰면 독립 미래 접촉에 대한 검증은 부족해진다. 기존 물체 panel에서 한 prefix의 gain 추정 RMS를 0.002로 낮추려면 이 분산 모형의 plug-in 외삽으로는 각 축을 같은 수로 늘릴 때 10×10 및 10×10 수준이 필요하다. 이는 4×4 밖의 미검증 외삽이며, 물리 로봇이 사용해야 할 probe 수나 순위가 보장되는 표본 수로 해석해서는 안 된다. [RMS 외삽 코드와 결과](results/precision31.json)

**논문에서 검증할 가설.** 추가 접촉이 posterior를 바꿔도, 한 번의 배포 집단에서 관측한 risk reduction이 그대로 다른 집단의 우선순위를 알려 주는지는 별개의 문제다. 특정 calibration 경로에서만 결정을 바꾸는지, 선택이 바뀐 배포 물체에서만 효용이 생기는지, 두 축을 모두 바꿔도 남는 순서가 있는지를 분리해서 보일 수 있다. 이 실험으로 물리 모형 전체의 Bayes-optimal value of information이나 모든 scorer의 성능 한계를 주장하기에는 근거가 부족하다. 현재 frozen scorer의 training과 model class도 제한되어 있다. [V30 실험](../learned_probe_value_v30/REPORT_KO.md) · [EVSI의 정의](https://journals.sagepub.com/doi/pdf/10.1177/0272989X15575286)

**다음 논문 실험의 우선순위.**

1. Source training labels부터 바꾼다. 같은 training GPU 시간에서 (a) prefix를 많이 뽑아 각각 1×1 label을 만드는 방법과 (b) prefix 수를 줄이고 각 label을 여러 futures·batches로 평균하는 방법을 비교한다. 같은 features, ridge family, grouping, tuning budget을 사용하고, 독립 source validation에서만 반복 수를 고른다. 그런 다음 새 contact law와 새 prefixes에서 같은 총 calibration budget의 fixed 4·uncertainty-only를 포함한 **전체 stopping policy**를 비교한다. 평가에는 training 비용, deployment GPU p90, held-out Brier를 모두 포함한다. 기존 0.003의 기대 이득이 목표라면 그 margin과 선택 규칙을 결과를 보기 전에 선언한다.
2. 이번에 관측한 interaction의 원인을 분리한다. 같은 worlds에서 common decoder·quota와 selector의 bank를 하나씩 교체하고, sensor noise만 재표집하는 cell-level replication을 추가한다. 고정한 numerical bank가 posterior 변경에 따라 particle 구성을 불연속적으로 바꿀 수 있다. 더 큰 bank와 부드러운 reweighting을 사용한 acquisition으로도 같은 교차 패턴이 생기는지 확인해야 이를 물리적 ambiguity에 귀속할 수 있다. [현재 predictive bank 구현](../calibration_probe_v28/estimator28.py)
3. 실물 또는 독립 simulator에서 물성·표면을 environment 단위로 고정한 뒤, 이미 관측한 물체에 대한 calibration 반복과 별도 물체에서의 deployment 반복을 분리한다. 같은 표면을 오래 쓰면 drift가 생길 수 있으므로 순서와 시간을 기록한다. 기존 planar pushing data에는 반복 contact가 있지만 v31의 query view 및 probe에 대응하는 두 potential outcomes와 stick label의 존재 여부를 먼저 확인해야 한다. [Yu et al., §§V–VII](https://arxiv.org/pdf/1604.04038)

**논문 진행 결정.** 동일 비용에서 새 policy가 fixed allocation보다 risk를 줄였다는 주장은 별도 held-out 실험에서 확인되어야 한다. 지금의 제출 가능한 주장은 세 synthetic laws, 정해진 여덟 coupling, 128-world batch, quota 13, source common decoder에 한정된 gain-label 재현성 진단이다. 다른 simulator에서도 독립 순위와 배분 손익이 같이 재현되면 evaluation 중심 논문의 근거가 강해진다. 반복 label로 학습한 정책이 추가로 실용적 Brier margin을 넘기면 방법 논문까지 확장할 수 있다.

## 문헌과 연구 기여의 범위

V31의 질문은 이미 확보한 관측인 **paid prefix**를 그대로 두었을 때, 추가 접촉의 가치가 두 번의 실험에서 얼마나 같게 측정되는가이다. 한 번은 추가로 관측할 물체와 접촉 결과를 다시 뽑고, 다른 한 번은 정책이 적용될 미래 물체들을 다시 뽑는다. V30의 source split-half 진단은 이미 정해진 한 calibration future와 한 batch의 quota를 공유했다. V30에서 나타난 낮은 rank correlation만으로 두 변동의 크기를 구별할 수 없었다. 아래 일곱 편은 이 구분에 직접 관련된 방법론·실증 논문을 목적 표집한 결과다. 1963년 논문은 출판사 abstract에서 확인한 주장까지만 사용했고, 나머지는 표에 적은 원문의 방법·결과를 확인했다. [V30 실험 §§4–7](../learned_probe_value_v30/REPORT_KO.md) · [Cronbach et al.](https://doi.org/10.1111/j.2044-8317.1963.tb00206.x)

### 주요 원문 비교

| 연구와 질문 | 방법과 표본 | 저자들이 확인한 결과·결론 | V31에 적용할 때의 한계 |
|---|---|---|---|
| Cronbach, Rajaratnam & Gleser (1963). 어떤 관측 집합으로 일반화할 때 측정값을 신뢰할 수 있는가? | Generalizability theory에서 한 관측과 일반화할 관측의 모집단을 구분했다. 이론 논문이며 여기서는 empirical sample 수를 인용하지 않는다. | 해당 모집단에서 관측을 무작위로 뽑는 가정하에 intraclass correlation과 원하는 generalizability coefficient 사이의 관계를 제시했다. | 다시 뽑는 future, batch, law에 따라 reliability의 의미가 달라진다. 이 원문은 v31의 rank concordance나 그 크기를 예측하지 않는다. [출판사 원문 초록](https://doi.org/10.1111/j.2044-8317.1963.tb00206.x) |
| Janon et al. (2014). 두 독립 입력 중 한 입력이 출력 분산에 기여하는 정도를 어떻게 추정하는가? | Pick-freeze covariance identity를 사용해 두 Sobol index estimators의 점근적 성질을 비교했다. 독립인 세 입력의 Ishigami function과 metamodel에서 수치 검증했다. Table 1의 coverage는 설정당 interval 100회 반복이다. | 한 입력을 공유한 두 출력의 covariance가 그 입력에 대한 conditional mean의 variance와 같다는 Lemma 2.2를 제시했다. Approximation error의 감소 속도도 interval coverage에 영향을 주었다. | 증명은 독립 입력과 명시된 moment 조건을 쓴다. V31의 교차표는 여러 쌍이 행·열을 공유하므로 논문의 independent-pair 표준오차를 그대로 가져올 수 없다. [원문 §§2–5](https://www.esaim-ps.org/articles/ps/pdf/2014/01/ps130040.pdf) |
| Strong et al. (2015). 새 데이터를 얻기 전에 그 데이터의 expected decision value를 계산할 수 있는가? | 두 단계 Monte Carlo와 simulated data summaries에 대한 regression을 비교했다. 모의 의료 의사결정에 세 data collection scenarios를 두었고, 세 번째는 treatment/control 각 200명인 가상 trial이다. | 제시한 case study에서 regression으로 nested calculation보다 적은 계산량에 EVSI를 추정했다. 정의에는 아직 얻지 않은 데이터에 대한 기댓값이 들어간다. | 데이터를 생성할 모형과 regression에 적절한 summary가 필요하다. 이 논문의 의료 결과는 contact Brier의 효과 크기로 옮길 수 없다. [원문 Eqs. 1–5 및 case study](https://journals.sagepub.com/doi/pdf/10.1177/0272989X15575286) |
| Konyushkova, Sznitman & Fua (2017). Label 구입으로 줄어들 generalization error를 학습할 수 있는가? | LAL regressor를 synthetic data 또는 해당 domain의 data로 훈련했다. Synthetic classification과 실제 응용 다섯 종류에서 평가했다. Cold-start 비교에는 세 데이터셋을 사용했다. | 세 cold-start 과제에서 iterative LAL이 uncertainty sampling의 최고 성적에 더 적은 labels로 도달했다. Error reduction을 학습하는 직접 선례다. | 구입한 label로 바로 classifier를 갱신했다. 과거 물체의 접촉이 다른 물체의 관측 선택을 바꾸는 과정, 같은 prefix에서 target이 재현되는 정도는 분리하지 않았다. [원문 §§4–5, Figs. 2–4](https://papers.neurips.cc/paper/7010-learning-active-learning-from-data.pdf) |
| Maier-Hein et al. (2018). 경쟁 알고리즘의 순위는 평가 표본이 바뀔 때 얼마나 유지되는가? | 150개 challenges·549개 tasks를 조사하고, 원자료를 확보한 2015년 segmentation 56개 tasks에서 metric·aggregation·case 구성 등을 재평가했다. Case bootstrap은 1,000회였다. | DSC 기준 모든 bootstrap 표본에서 기존 우승자가 계속 1위였던 과제는 21%였다. 어떤 비교에서는 평균 성능의 유의한 차이와 ranking instability가 함께 관측됐다. | 같은 collection에서 사례를 재추출했다. 미래 population이나 추가 학습 history를 독립 생성하지는 않았다. [원문 Results 및 Statistical methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC6284017/) |
| Bouthillier et al. (2021). Benchmark의 작은 개선이 어떤 무작위 선택에 민감한가? | CIFAR-10, Pascal VOC, MHC, 두 GLUE 과제에서 data sampling, initialization 등 seed를 하나씩 바꾸며 각 200회 반복했다. | 조사한 사례에서 data bootstrap의 영향이 컸고, 원천 하나씩 바꿔 얻은 수치를 단순 합산해서는 전체 분산이 복원되지 않는다고 명시했다. | Pipeline의 절대 성능을 측정했다. Paired Brier 차이에서는 두 정책에 공통인 오차가 상쇄될 수 있어 각 항의 크기와 순서가 달라진다. [원문 §2, Figs. 1–2](https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf) |
| Rossa, Phillips & Rainforth (2026, preprint). Downstream loss를 직접 줄이는 design과 action을 함께 학습하면 어떤 이득이 있는가? | Action-BED는 source location, stochastic pendulum·double pendulum, masked MNIST에서 평가했다. Source location에서 design histories와 action networks를 교차 평가했다. | 좋은 design과 함께 학습된 action network가 다른 정책의 histories에 적용되면 성능이 떨어지는 경우가 있었다. 관측을 어느 decoder에 쓸지가 측정한 가치에 관여한다. | Cross-evaluation은 여러 decoder의 compatibility를 조사한다. 동일 prefix에서 독립 future·batch가 유발하는 gain variance는 별도 문제다. [원문 §6 및 Appendix G.1.3 Tables 19–20](https://arxiv.org/html/2606.23662v1) |

### Knowledge map과 서로 다른 평가 단위

**가치의 정의.** Strong 등의 EVSI, LAL의 error-reduction target, Action-BED의 downstream loss는 최종 결정을 통해 관측의 가치를 정한다. EVSI는 모형 안에서 추가 데이터를 얻기 전 가능한 최적 결정의 가치를 평균한다. V31의 G = Brier(before) − Brier(after)는 고정한 selector·decoder와 특정 future·batch에서 **실현된** 손실 차이다. 잘못된 물리 가정으로 접촉을 해석하거나 유한한 batch에서 예측이 빗나가면 G가 음수일 수도 있다. 이 실험이 추정하는 평균은 해당 정책에 조건부다. [Strong et al., Eq. 4](https://journals.sagepub.com/doi/pdf/10.1177/0272989X15575286) · [LAL](https://papers.neurips.cc/paper/7010-learning-active-learning-from-data.pdf) · [Action-BED, Appendix G](https://arxiv.org/html/2606.23662v1)

**측정의 재현성.** Cronbach의 일반화 대상 지정 → Janon의 입력 고정·교체 → Maier-Hein의 재표집 후 재순위화는 각각 *무엇을 같은 대상으로 보는가*, *어떤 오차를 나눌 수 있는가*, *순위가 다시 맞는가*를 묻는다. Future main effect는 어느 접촉 결과를 얻었느냐에 따른 **batch 평균 gain**의 변화다. Batch main effect는 어느 물체 집단을 처리했느냐에 따른 **future 평균 gain**의 변화다. Interaction은 특정 접촉 결과가 특정 물체 집단에서만 유용해지는 정도다. 한 축만 바꾸고 다른 축 하나를 고정해 관측한 변동에는 그 축의 main effect와 interaction이 함께 들어갈 수 있다. [Cronbach et al.](https://doi.org/10.1111/j.2044-8317.1963.tb00206.x) · [Janon et al., §2](https://www.esaim-ps.org/articles/ps/pdf/2014/01/ps130040.pdf) · [Maier-Hein et al., Statistical methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC6284017/)

**근거 강도와 서로 다른 결과.** 조건부 독립인 두 무작위 입력에 대해서는 main effect와 interaction의 분해가 성립한다. 여러 benchmark에서 data resampling이 성능과 순위를 바꾼다는 경험적 근거도 있다. 다만 Bouthillier 등의 절대 성능, Maier-Hein 등의 우승자, LAL의 학습 효율은 서로 다른 평가량이다. LAL이 유리했던 결과와 V30에서 learned scorer가 baseline을 이기지 못한 결과의 차이를 어느 한 원인으로 환원할 수 없다. Domain, supervision, shared decoder, budget, contact-law mismatch가 함께 달랐다. 이 문헌만으로 어느 물리 법칙·stage에서 접촉 변동 또는 미래 물체 변동이 더 클지, 반복을 늘리면 새 정책이 좋아질지를 판정할 근거는 부족하다. [Janon et al.](https://www.esaim-ps.org/articles/ps/pdf/2014/01/ps130040.pdf) · [Bouthillier et al.](https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf) · [Maier-Hein et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC6284017/) · [LAL](https://papers.neurips.cc/paper/7010-learning-active-learning-from-data.pdf) · [V30의 교차검증 및 실험](../learned_probe_value_v30/REPORT_KO.md)

### 연구 공백에서 도출한 10가지 질문

| 질문 | 공백과 중요성 | 바꿀 변수와 검증 설계 |
|---|---|---|
| 1. 같은 prefix라도 어떤 접촉 응답을 얻느냐에 따라 Brier gain이 달라지는가? | EVSI의 기댓값은 개별 future 사이의 퍼짐을 알려주지 않는다. | Prefix 고정, 독립 futures를 같은 batches에 교차하고 future main effect를 추정한다. [Strong et al.](https://journals.sagepub.com/doi/pdf/10.1177/0272989X15575286) |
| 2. 유용해 보인 probe가 다른 deployment batch에서도 유용한가? | Case resampling이 순위를 바꿀 수 있지만 물리 probe 평가에 대한 수치는 없다. | 같은 future에서 독립 128-world batches의 pairwise concordance와 top-set overlap을 비교한다. [Maier-Hein et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC6284017/) |
| 3. 특정 future가 특정 batch에서만 좋은가? | 한 축씩 바꾼 variance를 더하면 interaction이 중복될 수 있다. | 모든 future와 batch를 교차해 interaction 및 그 interval을 추정한다. [Janon et al.](https://www.esaim-ps.org/articles/ps/pdf/2014/01/ps130040.pdf) · [Bouthillier et al.](https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf) |
| 4. 2→4회의 재현성이 4→8회에도 유지되는가? | Acquisition이 유도한 history에 따라 이용 가능한 정보가 다르다. | Stage별로 같은 설계를 실행하고 별도 실험에서 실제 stopping policy의 survivors만 재검증한다. [Action-BED](https://arxiv.org/html/2606.23662v1) |
| 5. 미래 물체를 더 모을 것인가, 독립적인 추가 접촉 경로를 늘릴 것인가? | 안정성은 다시 뽑는 관측 종류에 따라 달라진다. | Simulator 실측 비용을 고정하고 future/batch 수를 바꾸어 독립 평균 간 concordance를 비교한다. [Cronbach et al.](https://doi.org/10.1111/j.2044-8317.1963.tb00206.x) · [Janon et al.](https://www.esaim-ps.org/articles/ps/pdf/2014/01/ps130040.pdf) |
| 6. 순위가 흔들릴 때 실제 contact budget의 손해도 큰가? | 순위 변화는 후보 사이 loss gap을 반영하지 않는다. | 한 반복의 top-half 배분을 독립 반복에 적용하고 held-out Brier와 regret을 함께 기록한다. [Maier-Hein et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC6284017/) · [Bouthillier et al., §4](https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf) |
| 7. 같은 law에서 reliable한 순위가 law를 바꿔도 대응되는가? | Factual evidence가 같아도 law가 다르면 intervention outcome이 달라질 수 있다. | 동일 latent world를 law 간에 짝지은 별도 panel에서 within-law reliability와 law 간 transfer를 따로 추정한다. [V30의 same-input law 비교](../learned_probe_value_v30/REPORT_KO.md) |
| 8. 동일 future·batch에서 decoder만 바꾸면 variance가 바뀌는가? | Design이 주는 효용은 쌍을 이룬 decoder에도 의존한다. | Actions와 noise draws를 고정하고 source·matched decoder의 gain 분포를 교차 비교한다. [Action-BED, Table 20](https://arxiv.org/html/2606.23662v1) |
| 9. 여러 future·batch로 만든 학습 label이 실제 배분 성능을 높이는가? | 더 안정적인 target이 predictability나 정책의 이득을 보장하지 않는다. | Source labels를 같은 계산 예산으로 재구성하고 동일 scorer를 훈련한 뒤 새로운 laws에서 fixed 4 및 uncertainty-only와 비교한다. [LAL](https://papers.neurips.cc/paper/7010-learning-active-learning-from-data.pdf) · [V30](../learned_probe_value_v30/REPORT_KO.md) |
| 10. 실제 접촉에서도 같은 변동 구조가 남는가? | 선택한 문헌의 ranking 실증은 의료 영상 및 일반 ML pipeline에 집중돼 있다. | 실물 surface별 독립 calibration sequences와 별도 평가 물체 집단을 수집한다. 시간·표면 마모를 block으로 기록하고 simulator와 variance·rank를 비교한다. [Maier-Hein et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC6284017/) · [Bouthillier et al.](https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf) |

### Research assistant review

**실험이 뒷받침할 수 있는 기여.** 같은 evidence에서 시작한 같은 환경의 실현 gain이 왜 달라지는지, 어떤 반복 조합에서 환경 우선순위가 얼마나 회복되는지 보일 수 있다. Crossed variance decomposition과 error-reduction regression 자체에는 직접 선례가 있다. 분야 내 최초성을 주장하려면 별도 systematic search가 필요하다. 실험 범위가 사용한 simulator·law·quota·decoder이므로 실제 물리 환경, 다른 batch size, 다른 acquisition policy로 옮기는 주장은 각각 추가 검증이 필요하다. [Janon et al.](https://www.esaim-ps.org/articles/ps/pdf/2014/01/ps130040.pdf) · [LAL](https://papers.neurips.cc/paper/7010-learning-active-learning-from-data.pdf) · [Action-BED](https://arxiv.org/html/2606.23662v1)

**결과를 판정하는 기준.** 순위 재현성이 낮아도 실제 gain gap이 작으면 배분 손해는 작을 수 있다. 반대로 동일한 편향을 매번 반복하면 순위는 안정적이어도 decoder는 틀릴 수 있다. 같은 prefix에서의 반복은 prefix를 다시 관측했을 때의 재현성까지 포함하지 않는다. 논문에서는 셀 간 의존성을 보존하고 law별 분산 성분, Brier 단위의 불확실성, 독립 반복의 concordance, 실제 배분 성능을 서로 다른 주장에 대응시켜야 한다. [Cronbach et al.](https://doi.org/10.1111/j.2044-8317.1963.tb00206.x) · [Maier-Hein et al., Statistical methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC6284017/) · [V30](../learned_probe_value_v30/REPORT_KO.md)

### 실물 접촉 연구에 비추어 본 외적 타당성

| 연구 질문 | 방법과 표본 | 결과와 한계 |
|---|---|---|
| Yu, Bauza, Fazeli & Rodriguez. 같은 조건의 planar pushing은 어느 정도 반복되는가? | 물체·표면·접촉 위치·방향·속도·가속도를 통제했다. §VII에서 직선 push를 2,000회 반복하는 설정으로 네 표면을 비교했다. | 표면별 translation SD는 3.4–11.7 mm였고, 일부 최종 pose 분포에는 여러 mode가 관측됐다. 실물에서는 같은 조작에서도 구조적인 변동이 생긴다. 이 연구의 출력은 pose여서 v31의 Brier variance 비율을 실물에서 입증하는 자료로 쓸 수 없다. [원문 §VII, Table IV](https://arxiv.org/pdf/1604.04038) |
| Tremblay, Meger, Hogan & Dudek. 접촉할 행동을 학습해 개별 물체의 물성을 더 정확히 추정할 수 있는가? | Bayesian filtering과 information-gathering MPC로 질량·높이·전도 높이의 simulation 세 과제를 평가했다. 실물 높이 과제는 7 cm 범위에서 학습 후 10 episodes로 평가했다. | 50,000 timesteps 학습 뒤 실물 높이 MAE가 1.19 cm였다. 물체 한 개에서 그 물체에 대해 얻는 정보는 실제로 유용할 수 있다. 다른 물체들의 deployment observation을 바꿨을 때의 Brier gain은 별도로 검증해야 한다. [원문 §§4–5.2](https://arxiv.org/pdf/2312.00215) |

V31의 다음 실물 설계에는 두 관찰이 함께 중요하다. 표면에서 연속 실행한 모든 push를 독립 trial로 간주하면 drift가 불확실성에 섞일 수 있다. 또한 동일 물체를 다시 측정해 물성 추정이 좋아지는 것과 접촉을 경험한 뒤 다른 물체의 관측을 더 잘 선택하는 것은 평가 대상이 다르다. 전자는 반복 물체별로, 후자는 calibration 물체 집단과 disjoint deployment 집단을 분리해 측정하는 설계를 제안한다. 이는 위 두 논문을 바탕으로 한 v31의 설계 제안이다. [Yu et al.](https://arxiv.org/pdf/1604.04038) · [Tremblay et al.](https://arxiv.org/pdf/2312.00215)

## 재현 자료

- 서버 프로젝트는 `datacenter3:/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/probe_gain_reliability_v31`이다. 두 panel의 각 law/block 폴더에 choices, gain과 seed·hash를 저장했다. 2개 stage의 prefix가 겹치므로 전체를 독립 표본 288개로 보고 검정하지 않았다. 분석 코드는 아래 순서와 입력 파일 기준으로 다시 실행할 수 있다.
- Fresh-object 실험은 [고정 설정](design.json), [Slurm 실행 스크립트](run_gpu.sh), [world·likelihood·평가 코드](runner31.py), [K256 구현](batch31.py)을 사용했다. 같은 GPU에서 이미 관측한 물체의 접촉만 구입하는 [보충 실험](runner_pool31.py)을 이어서 수행했다. 원래 manifest의 `future_contract` 마지막 부분에는 factual-only control을 평가하지 않는다는 초기 문구가 남아 있다. 같은 manifest의 `extra_factual_control`에 명시하고 실행 전 고정한 대로 각 law의 첫 block에서 이 control을 전부 실행했다. 전체 18개 block에 control이 있는 것으로 집계하지 않았다.
- [독립 crossed 분석](stats31.py), [분산·재표집 검산](check_stats31.py), [quota와 Brier 재현](check_run31.py), [world별 paired action 검산](audit31.py), [top-half 배분](portfolio31.py), [보고서 표·그림 생성](report31.py)을 분리했다. 주 통계와 그림의 입력은 [fresh-object estimates](results/crossed_stats31.json)와 [기존 물체 estimates](results_pool/crossed_stats31.json)이다. 모든 probability와 label은 대조하는 두 action에 대해 같은 world에서 얻었다.
- Fresh-object 실행은 GPU block 계산 약 963.4초였다. 독립 검산에서 scalar K256과 vectorized K256의 1,536개 world별 gap 차이는 최대 3.28×10⁻¹⁵였고, 검증한 batch에서 top-13 action은 완전히 같았다. 54개 raw artifact hash와 1,908개 신규 seed의 중복 여부를 확인했다. 4,608개 gain cell을 원래 확률·label·action으로 다시 계산했다. 이 실행 시간에는 검증용 reference와 controls가 포함되어 있다. 개별 물체의 online GPU latency는 이번 estimand에 포함하지 않았다. [Runner 검증](results/runner_validation.json) · [Cell 검산](results/cell_replay31.json)
- 기존 pool에서 구입한 보충 실험은 GPU block 계산 약 601.8초였다. 두 stage에서 future 간 중복으로 재선택된 object ID 수는 6·15개다(전체 선택 수 1,152·2,304). 별도 CPU에서 RNG와 noise를 재생하여 저장된 contact response를 검산했고, 두 GPU 실행의 종료 기록에서 외부 GPU process와의 충돌은 없었다. [보충 실험 검증](results_pool/runner_validation.json) · [실행 코드](runner_pool31.py) · [Paired 분석](paired_pool31.py)
