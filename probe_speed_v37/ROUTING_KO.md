## 같은 환경에서 보정 배정을 바꾼 결과

각 calibration 환경에서 물체, sensor noise, 첫 세 접촉, signed proposal을 고정했다. 세 local regime에 실제 배정한 gate decision을 같은 세 regime 사이에서 무작위로 교환한 경우와 실제 배정을 비교했다. 각 표면은 원래 관측과 deployment 확률, 실제 query outcome을 유지한다. 따라서 물체나 방향의 차이를 이용해 얻은 이득을 표면 강도를 구별한 이득과 분리할 수 있다.

| 마지막 speed | 세 regime 사이 실제 − 재배정 Brier [95%] | Weak와 null 사이 실제 − 재배정 Brier [95%] | 세 regime에서 같은 결정 |
|---|---:|---:|---:|
| 0.50 | -0.00020 [-0.00031, -0.00008] | -0.00005 [-0.00008, -0.00001] | 88.7% |
| 0.75 | -0.00016 [-0.00027, -0.00005] | -0.00001 [-0.00005, +0.00002] | 86.1% |
| 1.10 | -0.00031 [-0.00049, -0.00013] | -0.00003 [-0.00008, +0.00001] | 77.0% |

+0.50에서 +0.75로 속도를 바꿀 때 세 표면의 조건부 배정 효과 변화는 +0.00003 [-0.00007, +0.00013]였으며, weak와 local-null 두 표면만 비교하면 +0.00003 [-0.00001, +0.00008]였다. +1.10에서 기준 속도 대비 세 표면의 변화는 -0.00011 [-0.00028, +0.00005]였다. 이 구간은 두 속도의 기준 대비 변화에 대해 동시에 보정했다.

Δ가 음수이면 같은 빈도로 보정을 적용해도 실제 배정이 더 좋다. 여기서 보정을 켤지는 첫 세 번의 signed proposal z와 실제 해당 속도의 네 번째 velocity로 결정한다. 95% interval은 object–surface와 첫 방향으로 층화한 paired block bootstrap 12,000회로 각 지표에서 세 속도를 동시에 보정했다.

| Speed | 표면별 배정 효과 | Hard randomization penalty | 연속 혼합 − fixed shrinkage | 합계 Gate − fixed shrinkage |
|---|---:|---:|---:|---:|
| 0.50 | -0.00020 | +0.00019 | +0.00067 | +0.00067 |
| 0.75 | -0.00016 | +0.00027 | +0.00037 | +0.00048 |
| 1.10 | -0.00031 | +0.00042 | -0.00050 | -0.00039 |

세 항의 합은 각 paired block에서 관측된 Gate−fixed Brier와 일치한다. 세 표면에서 평균 적용률을 h, 원 probability를 p, full correction probability를 q라고 하자. 같은 h로 켜고 끄는 randomized forecast는 연속 확률 p+h(q−p)보다 h(1−h)(q−p)²만큼 더 큰 기대 Brier를 갖는다. Hard randomization penalty는 이 convexity cost의 평균이다. 이 항등식을 위해 정의한 h에는 같은 환경의 세 반사실적 표면이 모두 필요하며, 배포 가능한 baseline의 성능으로 사용할 수는 없다.

| Speed | Strong > weak · paired ordering | Weak > null · paired ordering | 보정이 유리한 환경 판별 balanced accuracy |
|---|---:|---:|---:|
| 0.50 | 57.0% | 58.5% | 57.4% |
| 0.75 | 57.2% | 59.2% | 58.7% |
| 1.10 | 66.4% | 65.8% | 65.9% |

Paired ordering은 같은 물체·orientation·noise에서 더 강한 표면이 더 큰 posterior strength를 받는 비율이다. 같은 값에는 0.5를 준다. 서로 독립인 두 환경을 비교하는 AUROC와 이 진단은 다른 질문을 다룬다. Balanced accuracy의 positive label은 같은 환경의 독립 deployment 물체 128개에서 full correction Brier가 원래 확률보다 낮았던 경우다. 참측정량의 부호는 유한 표본에서 흔들릴 수 있다.

| 첫 세 접촉의 방향 판독 | 환경 비중 | 0.50 / 0.75 / 1.10의 gate 적용률 | 세 속도의 Gate − fixed shrinkage |
|---|---:|---|---|
| 방향 정답 | 93.9% | 37.6%, 38.4%, 44.3% | +0.00064, +0.00046, -0.00043 |
| 방향 오답 | 6.1% | 39.8%, 48.4%, 39.8% | +0.00110, +0.00071, +0.00012 |

세 local regime에서는 방향 판독에 쓰는 첫 세 고속 관측이 같아 판독의 정오가 속도와 무관하다. Global-null에서는 실제 방향이 정의되지 않아 이 표에 포함하지 않았다. [배정 분석 수치](routing37.json) · [코드](routing37.py)
