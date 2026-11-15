## 교차 반복 실험의 통계 결과

각 stage에서는 각 환경·prefix 안에서 먼저 variance와 gain을 계산하고 세 contact laws에 같은 가중치를 주었다. Gain이 양수이면 추가 관측으로 Brier가 줄었다. Variance는 Brier² 단위이고 SD는 Brier 단위다.

### 주 분석 · 어느 변동이 더 컸고 독립 순위가 재현됐나

| Stage | Batch−future variance ×10⁶ [simultaneous 95%] | 양 축을 모두 바꾼 순위 일치율 [simultaneous 95%] | 평균 signed gain [pointwise 95%] |
|---|---|---|---|
| 2→4 | 27.29 [20.83, 33.74] | 0.495 [0.479, 0.510] | -0.00008 [-0.00053, 0.00038] |
| 4→8 | 33.95 [25.65, 42.25] | 0.498 [0.484, 0.513] | -0.00077 [-0.00128, -0.00026] |

네 대비 전체를 동시에 포함하는 구간이다. Rank concordance의 우연 기준은 0.5다. 표본은 law당 independent blocks 6개이며 각 block에서 8개 환경을 비교했다. Interval은 미리 정한 세 law·여덟 population 조건과 현재 simulator에 조건부다.

### 단일 Brier gain의 variance decomposition

| Stage | Component | Variance ×10⁶ [pointwise 95%] | 단일 관측 분산의 비중 | Component SD (Brier) |
|---|---|---|---|---|
| 2→4 | Calibration future | 1.76 [-0.15, 3.80] | 2.8% | 0.00133 |
| 2→4 | Deployment batch | 29.05 [24.23, 34.62] | 46.4% | 0.00539 |
| 2→4 | Future × batch | 31.85 [29.53, 34.51] | 50.8% | 0.00564 |
| 4→8 | Calibration future | 1.24 [-0.97, 3.83] | 1.8% | 0.00111 |
| 4→8 | Deployment batch | 35.18 [30.14, 40.49] | 50.4% | 0.00593 |
| 4→8 | Future × batch | 33.43 [30.27, 36.55] | 47.9% | 0.00578 |

각 prefix의 F×B 표에서 row/column/grand mean을 구해 두 main effects와 interaction을 나누었다. 어떤 future로 posterior가 갱신됐는지에 따라 *특정 batch*의 action과 이득이 달라질 수 있다. 이 부분은 두 main effects 사이에 임의로 나눠 담지 않았다. Prefix를 고정하고 한 축만 새로 뽑을 때의 재측정 분산은 각각 V_future+V_interaction 및 V_batch+V_interaction이다. 양쪽 모두를 다시 뽑은 한 cell의 분산은 세 성분의 합이다. 별도 반복을 갖지 않는 계산 오차가 존재했다면 interaction과 따로 식별할 수 없다. Main effect의 추정값이 음수가 될 수 있는 unrestricted method-of-moments estimate를 원자료에 그대로 보존했다.

### 같은 환경 여덟 곳의 순위 재현성

| 독립적으로 바꾼 정보 · 두 번째 순위 | 2→4 concordance [95%] | 4→8 concordance [95%] |
|---|---|---|
| 같은 future, 새 batch | 0.504 [0.492, 0.517] | 0.499 [0.480, 0.521] |
| 새 future, 같은 batch | 0.651 [0.640, 0.661] | 0.669 [0.655, 0.683] |
| 새 future, 새 batch | 0.495 [0.483, 0.508] | 0.498 [0.487, 0.510] |
| 둘 다 새로, 각 순위에 1 future × 2 batches | 0.497 [0.472, 0.523] | 0.495 [0.473, 0.516] |
| 둘 다 새로, 각 순위에 2 futures × 1 batch | 0.491 [0.474, 0.509] | 0.499 [0.482, 0.516] |
| 둘 다 새로, 각 순위에 2 futures × 2 batches | 0.494 [0.460, 0.530] | 0.498 [0.466, 0.528] |

환경 쌍의 순서가 같으면 1점, 반대면 0점, 동점은 0.5점이다. 각 환경에서 가능한 disjoint replicate를 서로 독립적으로 열거했다. 같은 future 또는 같은 batch를 쓴 행은 공유한 정보가 있는 상황의 일치율이다. 평균을 내는 세 행에서는 두 순위가 어떤 future도, 어떤 batch도 공유하지 않는다. Averaging은 개별 환경의 population 불변성을 보장하지 않는다. 같은 환경에서 같은 prefix까지 조건을 맞춘 순위다.

### 반복을 어디에 늘리면 환경 평균을 더 정밀하게 추정하는가

| 반복당 futures × batches | 조건부 gain 평균의 예상 SD · 2→4 | 조건부 gain 평균의 예상 SD · 4→8 |
|---|---|---|
| 1 × 1 | 0.00792 | 0.00836 |
| 2 × 1 | 0.00677 | 0.00725 |
| 1 × 2 | 0.00568 | 0.00596 |
| 2 × 2 | 0.00483 | 0.00515 |
| 4 × 1 | 0.00612 | 0.00662 |
| 1 × 4 | 0.00412 | 0.00429 |
| 4 × 4 | 0.00311 | 0.00335 |

이 값은 V_future/F + V_batch/B + V_interaction/(F·B)에 해당하는 variance를 환경과 법칙에 걸쳐 평균한 뒤 square root를 취했다. 독립적으로 얻는 환경 한 곳의 prefix-conditional mean gain label의 sampling uncertainty이다. 전체 논문 평균에 대한 standard error와 다르다. 두 설계의 cell 수가 같아도 센서 구입·reset 비용과 동일 future를 재사용하는 GPU 비용은 다를 수 있다.

| Stage | 독립 두 cell gain의 평균 절대 차이 | 절대 차이 >0.003인 비율 [95%] | 환경 평균 관측 분산 ×10⁶ | 반복오차 보정 후 환경 간 분산 ×10⁶ [95%] |
|---|---|---|---|---|
| 2→4 | 0.00872 | 77.8 [76.3, 79.2]% | 8.99 | -0.71 [-2.89, 1.81] |
| 4→8 | 0.00925 | 78.8 [77.2, 80.4]% | 10.61 | -0.58 [-2.74, 1.63] |

같은 환경의 두 독립 gain 사이 절대 차이는 Brier 감소량 자체의 크기가 아니다. 마지막 열은 한 block 안 여덟 환경의 4×4 평균 사이 표본분산에서 평균 label의 sampling variance를 뺀 추정값이다. 환경당 관측한 서로 다른 paid prefix는 하나이므로 population 특성과 prefix noise를 이 열에서 더 분리할 수 없다.

### 법칙별 결과

| Law | Stage | Future / batch / interaction variance ×10⁶ | 독립 concordance [pointwise 95%] | 평균 signed gain [pointwise 95%] |
|---|---|---|---|---|
| Coulomb | 2→4 | 0.89 / 28.79 / 29.60 | 0.490 [0.473, 0.514] | -0.00002 [-0.00090, 0.00082] |
| Coulomb | 4→8 | 0.56 / 43.29 / 28.60 | 0.482 [0.470, 0.495] | -0.00025 [-0.00104, 0.00041] |
| Static–kinetic split | 2→4 | -1.72 / 23.62 / 30.47 | 0.505 [0.485, 0.531] | -0.00003 [-0.00060, 0.00079] |
| Static–kinetic split | 4→8 | 0.26 / 27.01 / 28.23 | 0.503 [0.483, 0.522] | -0.00089 [-0.00186, 0.00011] |
| Speed weakening | 2→4 | 6.13 / 34.75 / 35.48 | 0.488 [0.469, 0.508] | -0.00018 [-0.00096, 0.00051] |
| Speed weakening | 4→8 | 2.88 / 35.26 / 43.45 | 0.510 [0.486, 0.535] | -0.00117 [-0.00210, -0.00025] |

Estimator 식, 재표집 단위와 사전 판정 기준은 통계 계획에 기록했다. Additive known-effects, tie handling, estimator identity, 축과 index 교환에 대한 검산을 compute node의 CPU에서 통과했다. [사전 통계 계획](STATISTICAL_PLAN_KO.md) · [전체 수치](results/crossed_stats31.json) · [분석 코드](stats31.py) · [검산 코드](check_stats31.py)

### 관측된 개선의 부호와 크기도 재현되는가

| 단일 관측의 개선 조건 | Stage | 임의 첫 cell에서 충족 | 첫 cell 충족 후 두 축이 독립인 다음 cell에서 다시 충족 |
|---|---|---|---|
| Gain > 0 | 2→4 | 49.6% | 49.3% |
| Gain > 0 | 4→8 | 48.3% | 48.8% |
| Gain > 0.003 | 2→4 | 33.6% | 33.3% |
| Gain > 0.003 | 4→8 | 31.1% | 30.5% |

조건부 확률의 분모는 첫 번째 결과가 기준을 충족한 사례다. 여기에 같은 paid prefix를 유지하면서 future와 batch를 모두 독립적으로 뽑은 다음 cell을 대응했다. 유리한 단일 결과를 먼저 본 상황에서 그 판단이 다시 성립하는지 묻는 탐색적 진단이다. [실질 개선의 반복 확률과 구간](results/practical31.json)
