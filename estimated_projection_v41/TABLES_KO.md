## Held-out Brier와 GPU p90

Brier는 양성 사건에 대한 한 항 (p−Y)²이다. GPU p90은 동일 GPU에서 8개 환경×128개 물체를 한 번 처리한 동기화 wall-clock 시간이다. Warm은 query template을 재사용한다. Cold는 group profile이 메모리에 들어온 뒤 방향을 다시 적합하고 template을 만드는 비용까지 포함한다. 두 시간 모두 센서 촬영·충돌·reset, source bank 생성, 모델 학습 시간을 제외한다.

| Decoder | Correction | Held-out Brier ↓ | Warm p90 (ms) | Cold p90 (ms) |
|---|---|---:|---:|---:|
| Original | Fixed 32.5% | 0.125320 | 1441.46 | 1447.52 |
| Original | v38 optimizer 상시 적용 | 0.125329 | 1440.74 | 1446.82 |
| Original | 환경별 selector | 0.124436 | 1442.92 | 1448.88 |
| Oracle projection | Fixed 32.5% | 0.121816 | 1436.56 | 1443.35 |
| Oracle projection | v38 optimizer 상시 적용 | 0.121804 | 1445.44 | 1452.13 |
| Oracle projection | 환경별 selector | 0.120628 | 1441.56 | 1448.25 |
| Estimated projection | Fixed 32.5% | 0.121716 | 1442.40 | 1449.36 |
| Estimated projection | v38 optimizer 상시 적용 | 0.121731 | 1444.71 | 1451.77 |
| Estimated projection | 환경별 selector | 0.120552 | 1445.40 | 1452.41 |

각 panel에서 9개 방법을 섞은 순서로 1회 warm-up 후 3회 측정했다. 방법당 36개 latency를 합친 p90이다. 12개 timing cluster를 함께 재표집한 percentile bootstrap 95% 구간으로 ratio의 불확실성을 평가했다. Warm p90의 estimated selector / estimated fixed는 **1.002× [0.995, 1.233]**, estimated selector / original fixed는 1.003× [0.995, 1.224]였다. Cold에서 같은 두 ratio는 각각 1.002× [0.995, 1.232] 및 1.003× [0.995, 1.223]였다. [원 통계](stats41.json)

**원래 decoder를 보정 없이 사용한 경우.** 같은 576개 block에서 w=0으로 산출한 Brier와, 주 실험 종료 후 같은 GPU 및 같은 catalog panel에서 측정한 세 방법의 timing이다. 이 표 안에서 비교해야 timing session 차이를 통제할 수 있다.

| 방법 | 전체 held-out Brier | 별도 warm p90 (ms) | 별도 cold p90 (ms) |
|---|---:|---:|---:|
| Original, 보정 없음 | 0.123833 | 1413.94 | 1419.59 |
| Original + fixed | 0.125320 | 1422.80 | 1428.64 |
| Estimated + fixed | 0.121716 | 1422.48 | 1429.25 |

보정을 끄면 oracle과 estimated의 Brier는 각각 0.120220, 0.120129였다. [추가 crossover timing](latency_uncorrected41.json) · [원 확률의 재채점](contour_breakdown41.json)

### 같은 decoder에서 정책을 고르는 이득

| Decoder | Selector−fixed, 동시 95% CI | Selector−상시 optimizer, 동시 95% CI | Optimizer 선택 비율 |
|---|---:|---:|---:|
| Original | -0.000883 [-0.001133, -0.000634] | -0.000893 [-0.001096, -0.000690] | 52.1% |
| Oracle projection | -0.001188 [-0.001451, -0.000926] | -0.001176 [-0.001374, -0.000978] | 54.8% |
| Estimated projection | -0.001164 [-0.001421, -0.000907] | -0.001179 [-0.001382, -0.000975] | 54.8% |

Estimated가 주 검정이다. Original 및 oracle은 같은 방식으로 계산한 decoder별 보조 대조다. 각 대조의 음수는 앞의 방법이 유리하다는 뜻이다.

### 보정을 고정했을 때 projection의 효과

Estimated−original은 **-0.003604 [-0.003875, -0.003332]**, estimated−oracle은 **-0.000100 [-0.000280, +0.000080]**였다. Oracle−original의 개별 95% 구간은 -0.003503 [-0.003715, -0.003292]였다.
원래 decoder 대비 oracle의 Brier 감소 중 estimated가 회수한 비율은 102.9% [paired bootstrap 95% CI 99.1%, 106.6%]였다. Denominator는 이 catalog에서 관측된 oracle 이득이다.

| Shape·height | Estimated + fixed | Estimated−original, fixed | Estimated−oracle, fixed | Estimated selector−fixed |
|---|---:|---:|---:|---:|
| flare / 19 mm | 0.133434 | -0.002808 [-0.003073, -0.002542] | +0.000130 [-0.000119, +0.000379] | -0.000789 [-0.001050, -0.000528] |
| flare / 31 mm | 0.111554 | -0.002634 [-0.003045, -0.002223] | +0.000008 [-0.000286, +0.000303] | -0.000843 [-0.001077, -0.000609] |
| roll / 19 mm | 0.134207 | -0.002416 [-0.003181, -0.001651] | -0.000503 [-0.001134, +0.000129] | -0.001811 [-0.002360, -0.001262] |
| roll / 31 mm | 0.107670 | -0.006556 [-0.007037, -0.006075] | -0.000037 [-0.000144, +0.000071] | -0.001213 [-0.001700, -0.000726] |

각 열에서 네 shape·height 대조를 함께 보정했다. 이 표의 4개 행은 같은 catalog panel을 공유한다. 모든 24개 개별 cell과 panel별 원 값은 [전체 결과](stats41.json)에 있다.

### 보정 없음과 직접 비교한 별도 GPU 실행

Estimated decoder를 고정하고 보정 없음·fixed·selector의 순서를 섞어 같은 12개 panel에서 다시 측정했다. 보정을 끈 baseline이 강한 것을 확인한 뒤 실시한 보조 비교다. 위 주 timing에서 규정한 +2% 판정은 그대로 유지한다.

| Estimated decoder의 correction | Held-out Brier | Warm p90 (ms) | Cold p90 (ms) |
|---|---:|---:|---:|
| 없음 | 0.120129 | 1423.73 | 1430.54 |
| Fixed 32.5% | 0.121716 | 1434.78 | 1441.30 |
| Selector | 0.120552 | 1440.56 | 1447.51 |

Selector / 보정 없음의 warm p90 ratio는 1.012× [1.007, 1.013]였다. 서로 다른 timing session의 절대값 차이를 방법 간 효과로 환산하지 않았다. [원 timing](latency_estimated_zero41.json) · [동일 panel Brier 차이](no_correction41.json)
