# v32 통계 결과

## Training budget과 실험 단위

두 scorer에서 동일한 15개 features와 λ=3인 standardized ridge를 각 stage에 학습했다. Brier gain은 paid prefix와 추가 probe 뒤의 Brier 차이로 정의했다. Repeated는 각 prefix에서 2개 calibration futures와 2개 deployment batches가 교차하는 네 gain의 평균을 사용했다. Broad는 한 개를 사용했다.

| Source arm | Prefix 수 | Group 수 | Gain cells (두 stages) | Label time (s) | Ridge fit (s) | Label+fit (s) | Read·hash 포함 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| broad | 597 | 75 | 1194 | 539.3290 | 0.0082 | 539.3373 | 540.1149 |
| repeated | 255 | 32 | 2040 | 537.5638 | 0.0029 | 537.5667 | 537.8290 |

작은 arm 시간에 대한 label+fit 차이는 0.33%였다. 사전 기준은 2%다. 두 arm은 independent source bank와 paid prefix를 사용하며, 각 block은 최대 8개 환경의 calibration을 포함한다.

## Held-out Brier와 GPU latency

3개 target laws에 같은 비중을 두었다. 각 law의 16개 independent blocks가 8개 environments를 공유 단위로 묶는다. 총 384 environments, 49,152 deployment worlds를 측정했다. 32회 sensor noise는 같은 world의 반복 측정이다. GPU p90은 law별 첫 6개 blocks에서 같은 네 policy를 interleave해 3회씩 실행한 결과다. Full-call 하나는 8개 environments·1,024개 worlds의 allocation, acquisition과 선택한 observation의 decoding을 포함한다.

| Method | Held-out Brier | 95% pointwise CI | Full GPU p90 (ms) | 95% pointwise CI (ms) | Warm GPU p90 (ms) |
|---|---:|---|---:|---|---:|
| broad | 0.138677 | [0.137107, 0.140143] | 1491.43 | [1485.40, 1496.11] | 951.27 |
| repeated | 0.138911 | [0.137222, 0.140521] | 1489.50 | [1486.51, 1490.89] | 950.34 |
| fixed4 | 0.138774 | [0.137203, 0.140243] | 1418.16 | [1414.47, 1419.13] | 950.34 |
| uncertainty | 0.138868 | [0.137162, 0.140477] | 1488.54 | [1482.91, 1491.06] | 950.50 |

| Contrast A−B | Δ Brier | Simultaneous 95% CI | 해석 (0 기준) | Full p90 A/B | Pointwise 95% CI |
|---|---:|---|---|---:|---|
| repeated − broad | +0.000234 | [-0.000335, 0.000803] | 우열 미결정 | 0.999 | [0.995, 1.002] |
| broad − fixed4 | -0.000097 | [-0.000809, 0.000615] | 우열 미결정 | 1.052 | [1.048, 1.055] |
| repeated − fixed4 | +0.000137 | [-0.000681, 0.000955] | 우열 미결정 | 1.050 | [1.049, 1.052] |
| broad − uncertainty | -0.000191 | [-0.000701, 0.000319] | 우열 미결정 | 1.002 | [0.999, 1.006] |
| repeated − uncertainty | +0.000043 | [-0.000558, 0.000644] | 우열 미결정 | 1.001 | [0.999, 1.003] |

다섯 primary contrasts에 law-stratified paired block bootstrap 10,000회와 simultaneous critical multiplier 2.516을 적용했다. ΔBrier는 음수일수록 앞 policy에 유리하다. ±0.003은 absolute Brier의 실용적 기준이다. P90 interval은 18개 timed blocks를 resample한 pointwise interval이며 preprocessing·sensing·reset·공통 detector 및 decoder bank 생성 시간을 제외한다. Warm은 동일 호출에서 deployment selection·decoding만 잰 값이다.

## Laws에 따른 차이

| Held-out law | Broad Brier | Repeated Brier | Fixed4 Brier | Uncertainty Brier | Repeated−broad | Pointwise 95% CI |
|---|---:|---:|---:|---:|---:|---|
| split_static | 0.141792 | 0.141648 | 0.141714 | 0.141774 | -0.000144 | [-0.000948, 0.000632] |
| speed_weak | 0.129956 | 0.130085 | 0.130126 | 0.130392 | +0.000129 | [-0.000740, 0.001030] |
| directional | 0.144283 | 0.145000 | 0.144482 | 0.144438 | +0.000716 | [0.000174, 0.001305] |
| historical_two | 0.135874 | 0.135866 | 0.135920 | 0.136083 | -0.000007 | [-0.000612, 0.000599] |

`historical_two`는 v30·v31 시점에 이미 조사한 두 contact laws에 같은 비중을 준 값이다. `directional`은 v32에서 사전에 지정한 signed tangential velocity 의존 law다. 세 law 모두 scorer 학습에서 제외했다.

## Source subset sensitivity

각 arm의 생성 순서에서 block index를 2 또는 3으로 나눈 나머지를 이용해 disjoint groups를 만들었다. 각 group에서 동일한 ridge를 새로 fit하고 전체 target block의 frozen stage별 risk로 policy를 replay했다. 각 행의 interval은 해당 두 reduced-budget fits에 조건을 둔다. Modulo 2와 3은 source observations를 서로 공유한다.

| Groups·residue | Broad / repeated prefixes | Label time (s), B / R | Δ repeated−broad | Pointwise 95% CI | Full fit과 allocation이 바뀐 환경 비율 B / R |
|---|---|---|---:|---|---|
| 2·0 | 301 / 128 | 271.2 / 269.9 | +0.000106 | [-0.000404, 0.000592] | 21.1% / 19.3% |
| 2·1 | 296 / 127 | 268.1 / 267.6 | -0.000091 | [-0.000647, 0.000465] | 32.0% / 25.8% |
| 3·0 | 200 / 88 | 180.3 / 185.6 | -0.000211 | [-0.000715, 0.000281] | 22.1% / 34.6% |
| 3·1 | 200 / 87 | 181.5 / 183.3 | +0.000342 | [-0.000290, 0.000991] | 35.7% / 32.6% |
| 3·2 | 197 / 80 | 177.5 / 168.6 | +0.000146 | [-0.000488, 0.000777] | 41.7% / 54.4% |

## 검증 범위

각 artifact의 SHA-256, 학습·평가 seed 분리, 두 ridge fit의 CPU SVD reference, common decoder probabilities, quota, stage별 rank, 실제 paid trajectory를 독립적으로 재현했다. Recorded Brier와 independent replay의 최대 절대 차이는 0.00e+00, scorer output 차이는 0.00e+00였다. 이 primary bootstrap은 첫 full-budget seed pair에 조건을 둔다. 학습 seed 간 차이는 별도 full-budget 재학습 분석에서 평가한다.
