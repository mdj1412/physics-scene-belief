# v37 · 마지막 probe의 속도가 slip, 표면 구별, 충돌 예측에 미치는 영향

네 번째 probe의 높이를 23 mm에 고정했을 때, **+0.50, +0.75, +1.10에서 slip은 각각 60.1%, 80.9%, 96.2%**였다. Weak와 local null을 구별하는 실제 posterior의 AUROC는 같은 순서로 0.506, 0.510, 0.522였다. 사전 선택한 +0.75의 AUROC 변화는 +0.004 [-0.003, 0.011]로, 기준 속도와 비교해 변화의 부호를 확정하기 어려웠다.

**주 비교인 +0.75에서 gate − 고정 32.5% shrinkage는 +0.00048 [-0.00012, +0.00107]로 두 방법 사이의 우열은 확정되지 않았다.** 추가 조건 +1.10에서는 -0.00039 [-0.00096, +0.00017]로 두 방법 사이의 우열은 확정되지 않았다. 이 수치는 strong, weak, local null을 같은 비중으로 둔 synthetic target에서의 held-out Brier다. Weak와 local null만 반반으로 두면 각 고속의 차이는 +0.00042, -0.00040였다. 주 비교에서는 구간 하한이 −0.003보다 커서 0.003 이상의 실용적 감소는 지지되지 않았다.

| 네 번째 speed | Slip | Weak 대 local null AUROC | 32.5% Brier ↓ | Gate Brier ↓ | Gate − 32.5% [동시 95%] | GPU p90, 고정 / gate (ms) |
|---|---|---|---|---|---|---|
| +0.50 | 60.1% | 0.506 | 0.11195 | 0.11262 | +0.00067 [+0.00024, +0.00110] | 1,444.8 / 1,446.4 |
| +0.75 | 80.9% | 0.510 | 0.11202 | 0.11250 | +0.00048 [-0.00012, +0.00107] | 1,442.1 / 1,443.5 |
| +1.10 | 96.2% | 0.522 | 0.11187 | 0.11148 | -0.00039 [-0.00096, +0.00017] | 1,445.6 / 1,443.3 |

Local 평균은 세 local regime과 네 object × surface cell에 각각 같은 비중을 준다. Brier는 별도 물체가 다음 충돌에서 stick할 확률로 채점했다. Δ가 음수면 gate가 고정 32.5%보다 유리하다. 95% interval은 같은 world를 대응시켜 계산했다. [전체 수치](stats37.json) · [사전 분석 계획](ANALYSIS_PLAN_KO.md)

## 속도를 높이면 어떤 표면을 더 잘 읽는가

| 표면 | Slip · +0.50 / +0.75 / +1.10 | Gate 적용률 · 같은 순서 | 고속에서 gate 선택 변경 · .75 / 1.10 |
|---|---|---|---|
| Strong | 60.0% / 80.7% / 96.5% | 41.0% / 42.4% / 50.6% | 15.4% / 27.9% |
| Weak | 60.5% / 81.2% / 95.9% | 37.7% / 38.3% / 43.0% | 14.3% / 25.2% |
| Local null | 59.8% / 80.9% / 96.1% | 34.4% / 36.3% / 38.5% | 14.1% / 23.2% |
| Global null | 59.8% / 80.9% / 96.1% | 44.1% / 45.7% / 43.4% | 14.1% / 25.0% |

세 local regime 평균에서 +0.50 대비 slip 비율 변화는 +0.75에서 +20.8 pp [+16.9, +24.8], +1.10에서 +36.1 pp [+31.1, +41.0]였다. pp는 percentage point이며 두 변화에 대해 동시에 95%를 보정했다.

| 구별 문제 | +0.50 AUROC | +0.75 AUROC | +1.10 AUROC | +0.75 − .50 [95%] | +1.10 − .50 [95%] |
|---|---|---|---|---|---|
| Weak 대 local null | 0.506 | 0.510 | 0.522 | +0.004 [-0.003, 0.011] | +0.016 [0.007, 0.025] |
| Strong 대 weak/null | 0.523 | 0.537 | 0.571 | +0.014 [-0.006, 0.034] | +0.048 [0.021, 0.075] |
| Strong 대 local null | 0.526 | 0.542 | 0.581 | +0.016 [-0.002, 0.035] | +0.056 [0.031, 0.080] |

AUROC는 강도가 높은 표면에 더 높은 posterior mean strength를 부여할 확률로, 같은 점수일 때 0.5를 부여한다. 서로 다른 물체와 orientation이 일상적 관측처럼 섞인 상태에서 계산했다. Weak 대 local null과 strong 대 weak/null의 두 속도 변화에 대해 총 4개 simultaneous interval을 썼다. 나머지 strong 대 local null은 보조 pointwise interval이다. 이 지표의 정답은 표면 강도이고, Brier의 정답은 별도 충돌의 stick 여부다.

## 같은 speed에서 고정 보정보다 나은가

| 표면 분포 | +0.50 Δ [95%] | +0.75 Δ [95%] | +1.10 Δ [95%] |
|---|---|---|---|
| Local 평균 | +0.00067 [+0.00024, +0.00110] | +0.00048 [-0.00012, +0.00107] | -0.00039 [-0.00096, +0.00017] |
| Weak/null 평균 | +0.00055 [+0.00003, +0.00108] | +0.00042 [-0.00026, +0.00109] | -0.00040 [-0.00105, +0.00024] |
| Strong | +0.00090 [+0.00044, +0.00136] | +0.00059 [-0.00004, +0.00122] | -0.00037 [-0.00102, +0.00027] |
| Weak | +0.00073 [+0.00029, +0.00117] | +0.00035 [-0.00035, +0.00105] | -0.00047 [-0.00117, +0.00024] |
| Local null | +0.00037 [-0.00030, +0.00105] | +0.00049 [-0.00020, +0.00118] | -0.00034 [-0.00096, +0.00028] |
| Global null | +0.00023 [-0.00029, +0.00075] | +0.00018 [-0.00026, +0.00062] | -0.00029 [-0.00066, +0.00008] |
| 네 regime 평균 | +0.00056 [+0.00020, +0.00092] | +0.00040 [-0.00009, +0.00089] | -0.00037 [-0.00083, +0.00010] |

각 행은 같은 speed, 같은 signed proposal, 같은 deployment observations를 가진 gate와 고정 32.5% 사이의 비교다. 각 행 안의 세 표 값은 사전에 정한 7개 Brier 대조의 simultaneous interval에 속한다. 표면의 빈도가 다른 집단에는 해당 regime별 loss를 그 집단 비율로 재가중해야 한다.

| 속도 | Strong 0% | 25% | 50% | 75% | 100% | 손익분기 strong 비율 |
|---|---|---|---|---|---|---|
| +0.50 | +0.00055 | +0.00064 | +0.00073 | +0.00081 | +0.00090 | 교차 없음 |
| +0.75 | +0.00042 | +0.00046 | +0.00050 | +0.00055 | +0.00059 | 교차 없음 |
| +1.10 | -0.00040 | -0.00040 | -0.00039 | -0.00038 | -0.00037 | 교차 없음 |

Weak와 local null의 비중을 서로 같게 놓고 strong 비중을 바꾼 재가중 결과다. 표 안의 수치는 Gate−32.5% Brier의 point estimate이다. 손익분기점은 이 혼합 가정 아래의 기술값이며 deployment에서 관측한 표면 비율이 아니다. [재가중 수치](prevalence37.json)

| 고속 − +0.50 | 실제 Gate Brier 차이 [95%] | Gate−32.5% 차이의 변화 [95%] | 공통 +0.50 predictions에서 Gate 차이 [95%] | 배포 action 변경률 |
|---|---|---|---|---|
| +0.75 | -0.00012 [-0.00072, +0.00048] | -0.00019 [-0.00058, +0.00019] | -0.00021 [-0.00052, +0.00009] | 8.4% |
| +1.10 | -0.00114 [-0.00182, -0.00047] | -0.00106 [-0.00151, -0.00061] | -0.00106 [-0.00139, -0.00073] | 8.6% |

속도를 올리면 acquisition likelihood도 바뀌어 13개 deployment probe를 받을 물체가 달라질 수 있다. 네 번째 열은 실제 gate 선택만 각 속도에서 가져와 동일한 +0.50 deployment probabilities 위에서 재채점한 대조다. 이 열의 95% interval은 pointwise이고 실제 온라인 방법의 latency를 뜻하지 않는다.

| 속도 | Physics | 항상 보정 | 세 관측 gate | 네 관측 gate | Soft 보정 | 과거에 고른 고정값 | Gate − 과거 고정값 [95%] | 분할 oracle¹ |
|---|---|---|---|---|---|---|---|---|
| +0.50 | 0.11341 | 0.11470 | 0.11344 | 0.11262 | 0.11203 | 0.11189 | +0.00074 [+0.00035, +0.00112] | 0.11045 |
| +0.75 | 0.11345 | 0.11481 | 0.11351 | 0.11250 | 0.11208 | 0.11196 | +0.00053 [-0.00000, +0.00107] | 0.11051 |
| +1.10 | 0.11335 | 0.11457 | 0.11340 | 0.11148 | 0.11178 | 0.11179 | -0.00032 [-0.00085, +0.00021] | 0.11032 |

과거 자료로 고른 고정 보정량은 44.1%다. v36의 192개 archive에서 +0.50의 Brier로 한 번 고른 뒤 새 세 속도에 그대로 적용했다. 이 방법은 target과 같은 simulator family의 과거 labels를 이용한 보조 비교다. Soft 보정은 같은 βz에 posterior mean near-field strength를 곱했다. ¹분할 oracle은 각 환경에서 64개 deployment 물체의 label로 항상 보정 또는 건너뛰기를 고른 뒤 나머지 64개에서 점수화하고 양쪽 역할을 바꿔 평균했다. 이 값은 실제 정책이 쓸 수 있는 정보보다 많은 label을 요구한다. [과거 자료 및 계수](fixed_validation37.json)

## 속도가 만들어 낸 센서 신호

| 후보 표면 | 속도 | Privileged accuracy | 같은 clean response 비율 |
|---|---|---|---|
| Weak 대 local null | +0.50 | 56.0% | 38.5% |
| Weak 대 local null | +0.75 | 60.1% | 18.4% |
| Weak 대 local null | +1.10 | 63.3% | 3.9% |
| Strong 대 local null | +0.50 | 67.7% | 35.7% |
| Strong 대 local null | +0.75 | 77.2% | 16.8% |
| Strong 대 local null | +1.10 | 84.4% | 2.9% |

이 계산에서는 latent 물체와 orientation, 두 후보 contact law를 모두 알려준 상태에서 하나의 noisy velocity로 어느 표면인지 추측한다. 두 표면의 prior가 같고 공통 Gaussian noise의 표준편차가 σ라면 최적 정확도는 Φ(‖μ₁−μ₂‖/(2σ))다. 동일 물체의 두 반응이 모두 stick이면 impulse가 계수에 무관해 clean velocity가 같을 수 있다. 이 표는 oracle accuracy, 위 표는 실제 posterior의 AUROC이므로 두 수치의 크기를 직접 빼지 않는다.

두 후보 표면과 같은 latent 물체를 주었을 때 weak/null 정확도 상한은 56.0%, 60.1%, 63.3%였다. 두 표면의 clean velocity가 같은 비율은 38.5%, 18.4%, 3.9%였다. 동일한 충돌 상태가 어느 속도에서 바뀌었는지 묶어서 보면 상한 개선이 어디에서 생겼는지 더 구체화할 수 있다.

| +0.50에서 올린 속도 | 처음부터 양쪽 모두 slip | 어느 한 표면이라도 stick → slip | 나머지 |
|---|---|---|---|
| +0.75 | 58.8% / +0.40 pp | 22.9% / +3.69 pp | 18.4% / +0.00 pp |
| +1.10 | 58.8% / +0.49 pp | 37.3% / +6.82 pp | 3.9% / +0.00 pp |

각 칸은 전체 물체 중 해당 비율과, 그 집단이 전체 privileged accuracy 변화에 기여한 percentage point다. 세 열의 기여 합은 해당 속도와 +0.50의 상한 차이와 같다. 속도를 올리면 slip 상태 외에도 tangential velocity, normal impulse, velocity-dependent friction modulation이 함께 바뀐다. 아래 부록에서 kinematics와 modulation을 따로 고정해 이 두 경로를 비교한다. [속도 요인 분해](PHYSICS_FACTOR_KO.md)

![Speed, slip and decision loss](v37.png)

## 보정까지 포함한 GPU p90

| 속도 | 32.5% GPU p90 (ms) | Gate GPU p90 (ms) | Gate 비용 변화 [동시 95%] | +2% 기준 |
|---|---|---|---|---|
| +0.50 | 1,444.8 | 1,446.4 | +0.12% [-0.28%, +0.51%] | 상한 이내 |
| +0.75 | 1,442.1 | 1,443.5 | +0.10% [-0.10%, +0.29%] | 상한 이내 |
| +1.10 | 1,445.6 | 1,443.3 | -0.16% [-0.79%, +0.48%] | 상한 이내 |

RTX PRO 6000 Blackwell Server Edition 한 장에서 환경 8개와 총 1,024개 deployment 물체를 처리하는 한 호출을 측정했다. 각 방법·속도당 8개 timing block × 3회, 총 24회이며 block 안에서 여섯 조합의 실행 순서를 무작위화했다. 속도별 likelihood, 취득할 관측 선택, 선택 관측의 decoding, CPU에서 계산한 correction과 host 반환을 포함한 synchronized wall time이다. Bank 초기 생성, simulated data 생성, 실제 접촉 및 robot reset 대기는 제외했다. [GPU 코드](run37.py) · [완료 기록](results/completion.json)

## 속도만 바꾼 실험

환경마다 서로 다른 물체 네 개를 한 번씩 접촉한다. 첫 세 probe는 높이 0, tangential speed ±1.10에서 교대로 수행하고 첫 부호를 block 안에서 균형화했다. 네 번째는 동일한 물체에서 **높이 23 mm, 방향 +**를 고정하고 tangential speed를 +0.50, +0.75, +1.10으로 바꿨다. 예측할 별도 충돌은 **25 mm, +0.425**로 고정했다. 여기서 속도 수치는 기준 좌표의 incoming x velocity를 1로 둔 simulator의 단위다. 접촉점의 local normal component는 contour에 따라 달라진다. 실물 장비의 m/s로 환산하지 않았다.

세 속도에서 네 번째 물체의 latent geometry, 처음 세 접촉, factual observations와 두 성분의 Gaussian velocity noise(각 표준편차 0.027)를 공유한다. Gate가 관찰하는 값은 noisy outgoing velocity다. Slip 상태와 깨끗한 물리 응답은 평가할 때만 열었다. 새로운 deployment 물체는 환경당 128개다. 각 속도의 acquisition이 그중 13개에 probe, 나머지에 query view를 배정한다. 높이와 속도가 다른 관측을 잘못된 action으로 해석하지 않도록 해당 속도의 Coulomb likelihood를 사용한다. 같은 속도의 두 방법은 동일한 선택과 같은 physics probability p를 받는다.

- **고정 32.5% shrinkage**는 첫 세 관측에서 추정한 signed friction cue z의 32.5%를 항상 적용한다. 최종 예측은 sigmoid(logit(p)+0.325βz)이다. 0.325는 v36 source에서 가정한 네 near-field multiplier 1, 0.3, 0, 0의 균등 평균이다.
- **Hard gate**는 마지막 측정으로 각 후보 surface model의 posterior를 갱신한다. Source에서 기록한 expected Brier gain이 양수면 sigmoid(logit(p)+βz)를, 아니면 p를 반환한다. 두 방법에 같은 β=0.8289760825845313과 같은 z를 사용한다. +0.50에서 v36 모델을 그대로 불러오고, 다른 속도에서는 v36과 동일한 source draws로 네 번째 측정 likelihood만 다시 계산했다. Gate threshold, prior와 gain lookup은 속도별로 다시 맞추지 않았다. [고정된 모델의 기록](freeze37.json) · [Inference](gate37.py)
- **Primary target**은 strong, weak, local null이 동일 비율로 존재하는 조건이다. 세 regime은 높이 0에서 같은 방향성을 보이지만, query 부근에서 원래 방향 효과의 100%, 30%, 0%를 갖는다. 별도 global-null 표면은 모든 높이에서 방향성이 없다. 이 조건을 primary 평균에 넣지 않고 자체 결과와 네 regime을 균등하게 혼합한 결과를 공개한다.
- **새 표본**은 v36과 동일한 두 contour와 두 target friction law에서 독립된 seed로 생성했다. 네 cell에 8개 독립 block씩, 각 block에 8개 환경과 환경당 128개 deployment 물체가 있어 unique latent deployment objects는 32,768개다. 각 물체의 32개 observation noise draws, orientation mirror, regime와 속도는 같은 block 안에 둬 paired interval을 계산했다. 각 cell 안에서 첫 probe 부호별 독립 block은 네 개씩이다. 구간은 이 네 고정 cell 내부의 변동에 대한 bootstrap 근사다. [설계와 seed](design37.json) · [결과 전 분석 계획](ANALYSIS_PLAN_KO.md)

Source에서는 방향성이 없는 query의 gain을 근사할 때 v34의 기존 directional deployment observation을 유지하고 outcome만 isotropic으로 바꾼 부분이 있다. v37 global-null target은 calibration과 deployment 모두 처음부터 isotropic으로 생성했다. 두 방법의 공통 base physics decoder는 population proxy coupling을 안다는 기존 가정을 유지한다. 이 결과를 실제 새 표면 또는 다른 sensor noise 범위로 옮길 때에는 이 두 조건을 별도로 검증해야 한다.

## 논문 기여 가능성

이 실험이 직접 평가한 양은 네 번이라는 고정 budget에서 **속도를 바꾸어 더 식별 가능한 physical response를 만들었을 때, 그 정보가 downstream decision loss로 얼마나 전달되는가**이다. Sticking 상태에서 일부 friction coefficient가 동일한 impulse를 만든다는 관측성 한계는 선행연구에 이미 제시돼 있다. 더 빠른 속도에서 slip이 늘었다는 결과만으로 새 방법의 기여를 주장하기는 어렵다. [Fazeli et al., §3](https://arxiv.org/pdf/1710.04979)

**+0.75에서 속도 변경 자체의 효과와 baseline 우위를 구분해야 한다.** Gate의 Brier가 +0.50에서 +0.75로 바뀐 양은 -0.00012 [-0.00072, +0.00048]였다. 같은 속도에서 32.5%를 대조한 이점이 +0.50보다 커졌는지 보는 차이의 차이는 -0.00019 [-0.00058, +0.00019]다. +1.10과 +0.50의 차이의 차이는 -0.00106 [-0.00151, -0.00061]였다. 이 값은 재질 순위의 AUROC와 독립적으로 확인할 수 있는 decision utility다. 원래 v36의 +0.50에서 Gate−32.5%는 약 +0.00037이었으며, 이번 독립 panel의 +0.50에서는 +0.00067였다. [v36 결과](../query_matched_probe_v36/REPORT_KO.md)

**어떤 결정이 어려운지도 분리할 수 있다.** Gate는 원래 확률과 전체 보정 확률 중 하나를 고르고, shrinkage는 물체마다 그 사이의 별도 확률을 낸다. 표면을 정확히 분류하더라도 두 endpoint 중 하나가 squared error를 가장 작게 만든다는 보장은 없다. 이 차이를 확인하기 위해 환경별 64개 deployment 물체의 label로 binary action을 고른 후 다른 64개로 채점했다. 이때 고정 32.5% 대비 Brier 차이는 속도별 -0.00150, -0.00151, -0.00155였다. 이 진단과 실제 gate의 차이에는 관측 가능한 정보, source–target 차이, 유한한 deployment label 수가 함께 들어간다. 다른 물체의 정답을 사용한 진단값을 배포 가능한 성능으로 쓸 수 없다.

고속에서의 보조 실험은 다음 가설을 구체화한다. 같은 frozen gain table이 0%, 32.5%, 100% 중 하나를 고르게 하면 +1.10의 Brier가 0.11107이고, 고정 32.5%보다 −0.00080 [−0.00122, −0.00037] 낮았다. 기존 binary gate와 비교한 차이도 −0.00040 [−0.00068, −0.00012]였다. +0.75에서는 고정값보다 낮다는 증거가 없었다. +1.10에서의 이득은 0.003보다 작고, 이 3-action 대조는 같은 archive에서 추가한 offline 분석이며 p90을 측정하지 않았다. [세 행동의 선택률과 paired interval](TERNARY_KO.md)

+1.10에서 강한 방향과 weak/null의 구별은 AUROC 0.571까지 좋아졌으나 weak 대 local null은 0.522였다. 같은 실험에서 약한 두 표면만 후보로 제한한 별도 likelihood ratio도 AUROC 0.530이었다. 세 regime 사이에서 gate 결정을 무작위로 다시 배정했을 때보다 실제 배정의 Brier가 약 0.00031 낮았고, 같은 세 표면에서 77.0%는 같은 행동을 골랐다. 높은 속도에서 얻은 추가 정보의 일부가 유용했으나 그 효과의 절대 크기는 작았다. [선택 효과 분해](ROUTING_KO.md) · [두 후보 판별](BINARY_KO.md)

+0.50에서 +1.10으로 갈 때 새로 slip한 36.1%가 전체 Gate−shrink 차이의 변화에 −0.00021을, 이미 slip이었던 60.1%가 −0.00081을 기여했다. 반면 weak와 local null의 물리적 구별 상한 증가는 주로 적어도 한 후보가 stick에서 slip으로 바뀐 물체에서 나왔다. 두 계산은 각각 실제 policy loss와 privileged sensor separation을 다루며 집단을 묶는 기준도 다르다. 정보 상한 증가율을 Brier 개선으로 그대로 읽을 수 없는 이유다. [집단별 손실 변화](SLIP_KO.md) · [속도 요인 분해](PHYSICS_FACTOR_KO.md)

**주장의 범위.** 네 조합에서 독립 seed를 사용했고 source contour·friction 함수와 target 함수가 다르지만, 세 local regime과 global-null이라는 후보 자체는 source와 target이 공유한다. Tangential velocity는 고정 부호의 세 수준만 시험했다. Late query의 방향과 sensor noise는 고정했으며, reference decoder는 population proxy coupling을 알고 있다. 속도 변경에 따른 slip 변화만의 causal effect를 실험으로 분리하지 않았다. GPU p90은 한 장치와 batch size에서 실측했다. 실제 표면에서의 최적 속도나 장비에서의 cycle time에 대한 증거는 아직 없다.

**다음 검증.** 동일 표면·물체에서 명령 속도를 무작위화하고 noisy velocity와 독립적인 slip 판정, 새 물체 충돌의 stick/slip을 함께 기록할 수 있는 실물 실험이 필요하다. 같은 네 번의 비용에서 현재 binary gate와 32.5%를 고정하고, 이번 보조 분석에서 얻은 3-action 정책을 미리 주방법으로 고정해 별도 물체·표면에서 재현한다. 직접 추정한 연속 friction posterior와 비교하고 3-action의 전체 p90도 측정한다. 속도가 달라지면 실제 sensor noise도 달라지는지 측정하고, 같은 재료 내 반복과 새 material session을 구분해 Brier interval을 낸다. 실물 자료 전까지는 이 실험을 속도와 정보, 예측 손실을 분리한 simulation 연구로 제시하는 것이 타당하다.

## 문헌 근거와 후속 연구 질문

## 마지막 probe의 속도를 바꾸면 얻는 정보

속도 변화에 대한 실제 마찰 측정, 충돌에서의 parameter identifiability, 관측 선택의 목적함수, 확률 보정을 다룬 원문 7편을 검토했다. 결과의 단위는 각 연구가 직접 측정한 값으로 유지했다.

| 원문과 연구 질문 | 방법 및 표본 | 결과와 결론 | v37 적용 한계 |
|---|---|---|---|
| Yu et al. (IROS 2016). Pushing의 friction을 속도·방향에 무관한 상수로 놓을 수 있는가? | 로봇으로 steel object의 pose와 force를 기록했다. ABS, Delrin, plywood, polyurethane(PU) 네 지지면에서 speed scan을 10~500 mm/s의 10개 속도로 각 10회씩 진행했다. | Dynamic coefficient of friction은 ABS, Delrin, plywood에서는 속도에 따라 조금 변했고 PU에서는 고속에서 약 1.0까지 증가했다. 연구진은 넓은 속도 범위에서 재료별 검증이 필요하다고 결론지었다. | 지지면 위 평면 이동과 v37의 짧은 impact는 서로 다른 실험이다. 같은 물체의 반복 scan은 독립 재료 수에 포함되지 않는다. [원문, Table II·§V·Fig. 5][Y] |
| Tramsen et al. (J. R. Soc. Interface 2018). Sawtooth의 방향으로 높은 friction 방향을 예측할 수 있는가? | Aspect ratio, stiffness와 substrate roughness를 바꾼 96개 sample–substrate 조합에서 양방향 pulling force를 비교했다. | Geometry가 같아도 조합에 따라 높은 friction 방향이 바뀌었다. 저자들은 실제 접촉 면적과 mechanical interlocking의 상대적 역할로 설명했다. | 특정 방향의 quasi-static pulling coefficient로 다른 속도의 impact에서 stick할 확률을 정하기에는 추가 측정이 필요하다. [원문, §§2–4][T] |
| Sakuma et al. (PRL 2026). 같은 물질에서 sliding velocity에 대한 friction의 민감도가 유지되는가? | Muscovite mica를 22~200 °C, normal stress 100 MPa에서 실물 shear experiment로 측정했다. 네 sliding velocity는 0.01, 0.1, 1.0, 3.0 μm/s였다. | 22 °C에서는 coefficient가 log sliding velocity에 의존했고, 온도가 오르면서 의존성이 약해져 200 °C에서는 측정 범위에서 거의 일정했다. | 고압·고온 mica에서의 기전을 일상 접촉으로 옮기려면 별도 측정이 필요하다. Rate sensitivity와 방향에 따른 차이는 독립 축이다. [원문][M] |
| Fazeli et al. (공개 원고 2017). Planar rigid-body impact로 friction parameter를 식별하고 실제 impulse를 재현할 수 있는가? | 실제 drop 2,000회 중 tracking 등 품질 검사를 통과한 1,718건을 사용했다. 6개 모델에 friction과 restitution을 맞추고 허용되는 impulse의 범위를 비교했다. | 고정 restitution에서 coefficient가 sticking threshold를 넘으면 같은 impulse를 내 해당 사건에서 coefficient를 구별하기 어렵다. 실제 impulse가 각 모델의 예측 범위 안에 든 비율은 35.5~45.8%였다. | 이 비율은 모델이 표현할 수 있는 impulse의 범위를 나타낸다. 실제 slip 중에도 rigid-body family가 설명하지 못하는 충돌이 생길 수 있다. [원문, §3 식 (9)·Table 2][F] |
| Arslan Waltersson & Karayiannidis (IJRR 2025). 물체를 집어 올릴 때 slip을 일으켜 후속 조작에 쓸 물성을 추정할 수 있는가? | UR10에 parallel gripper를 장착하고 두 손가락에서 각각 force와 sliding velocity를 측정했다. 접촉 추정 실험은 wood, 안경집, plastic, cardboard 네 물체에 각각 10회, 5 N으로 수행했다. | Linear exploration으로 static·Coulomb·viscous friction을, 이어지는 rotational exploration으로 contact radius를 추정했다. 두 단계는 2초였고 cardboard와 plastic이 각각 가장 높고 낮은 friction을 보였다. | 동일 접촉의 짧은 time series에서 여러 값을 사용한다. v37처럼 네 번의 noisy velocity만으로 다른 물체의 충돌 확률을 맞히는 문제로 옮기려면 sensor와 budget을 다시 맞춰야 한다. [원문, Algorithm 1·Table 9][W] |
| Zhong et al. (공개 원고 v2 2025). 가장 많은 parameter 정보를 주는 sensor가 미래 quantity of interest에도 최적인가? | Nested Monte Carlo, MCMC, density estimation, Bayesian optimization으로 predictive information gain을 최적화했다. 합성 test problem과 convection–diffusion source inversion에서 1~3개 sensor 배치를 비교했다. | 미래 농도를 예측할 위치와 예측량이 달라지면 높은 utility의 sensor 위치가 달라졌다. Parameter와 그 nonlinear function에 대한 information gain의 관계도 도출했다. | 목적함수는 predictive distribution의 information gain이다. 제한된 선택지만 쓰는 hard gate의 Brier 개선은 별도로 측정해야 한다. [원문, §3·Figs. 14–19][Z] |
| Ranjan & Gneiting (JRSS B 2010). 확률 예측을 고정 비율로 결합하면 calibration이 유지되는가? | Linear opinion pool의 성질을 증명했다. Beta-transformed linear pool을 가상 예와 미국 본토 29개 도시의 강수 확률 예측으로 검토했다. | 서로 다른 calibrated forecasts의 양의 가중 평균은 결합 뒤 calibration을 잃을 수 있었다. 결합 후 재보정 방법을 제시하고 Brier와 구성항을 평가했다. | v37의 0.325는 log-odds correction 크기에 곱한다. Probability를 선형 결합한 이 논문과 수식이 다르다. 같은 holdout과 scoring rule에서 보정들을 대조하는 방법론이 연결된다. [원문, §§2–4][R] |

### 연구 흐름과 달라 보이는 결과

| 연구 사이의 연결 | 확립된 근거와 불확실성 |
|---|---|
| Commanded speed → contact state → 관측되는 velocity | 실제 impact와 rigid model에서 stick 중 일부 friction 값은 동일 impulse를 만든다. 이 구간을 떠나는 것이 식별에 필요한 경우가 있다. 서로 다른 두 slip response가 sensor noise에 비해 충분히 멀어지는지는 별도 문제다. [Fazeli et al.][F] |
| Sliding speed × 표면 조건 → friction response | PU에서는 큰 속도 의존성, 세 다른 지지면에서는 작은 의존성이 보고됐다. Mica는 같은 재료에서도 온도에 따라 의존성이 달라졌다. 재료·normal stress·온도·동작의 차이를 고려하면 보편 rate law를 정할 수 없다. [Yu et al.][Y] · [Sakuma et al.][M] |
| 표면 방향 → query 결과 → correction 선택 | 양방향 마찰 차이의 부호는 물체·표면 조합에 따라 바뀐다. 어떤 물성이 관측에 드러나도 query에 맞춘 utility는 따로 평가해야 한다. Weak와 null에서 같은 결정을 내리는 편이 손실상 최적일 가능성은 v37에서 시험하는 가설이다. [Tramsen et al.][T] · [Zhong et al.][Z] |
| Benefit score → binary gate 또는 shrinkage → held-out probability | Brier score는 확률 오차를 평가한다. 높은 material AUROC만으로 correction 크기의 적절성을 판단할 수 없다. 0.325의 우위 여부를 직접 다룬 증거는 이 7편에 없다. [Ranjan & Gneiting][R] · [Zhong et al.][Z] |

### 결과에서 확인할 주장과 공백

1. **관측성.** 같은 calibration 물체의 latent state와 sensor noise를 대응시켜 speed별 slip rate, weak–local-null AUROC, 두 regime의 velocity separation을 비교한다. Trial과 sensor draw를 독립 surface로 세면 interval이 지나치게 좁아진다. [Fazeli et al.][F] · [Yu et al.][Y]
2. **기전.** Speed를 올리면 slip 외에도 tangential demand, normal impulse와 velocity-dependent coefficient가 함께 달라진다. 같은 speed contrast로 식별할 수 있는 양은 속도의 총효과다. Slip만의 매개효과를 주장하려면 law를 고정한 반사실이나 독립적인 조작이 필요하다. [Yu et al.][Y] · [Sakuma et al.][M] · [Fazeli et al.][F]
3. **예측 이득.** 같은 speed에서 같은 base probability와 direction proposal을 사용하는 gate와 shrinkage를 비교한다. Speed 변경이 gate의 상대 이득까지 높였다는 주장은 두 손실 차이가 속도에 따라 얼마나 바뀌었는지로 평가한다. 결과에서 유리한 속도를 고른 경우 새로운 test panel이 필요하다. [Zhong et al.][Z] · [Ranjan & Gneiting][R]
4. **논문 범위.** Sticking 중 비식별성에는 선례가 있다. 어느 속도까지 weak/null 식별이 증가하고 그중 어느 부분이 제한된 budget에서 후속 충돌의 Brier를 낮추는지를 보여주는 실험이 기여 후보가 된다. 실물에서 시험하려면 probe의 outgoing motion과 다른 물체의 독립 query outcome을 짝지어야 한다. [Fazeli et al.][F] · [Tramsen et al.][T] · [Zhong et al.][Z]

### 문헌의 공백에서 나온 연구 질문 10개

| 질문 | 공백과 중요성 | 변수와 검증 설계 |
|---|---|---|
| 1. 같은 slip rate에서도 weak/null 식별이 달라지는가? | Stick 구간 밖이라는 사실만으로 impulse 차이의 크기는 정해지지 않는다. [Fazeli et al.][F] | 두 속도에서 모두 미끄러지는 동일 물체를 짝지어 velocity separation과 AUROC를 평가하고 하위집단 추정의 범위를 명시한다. |
| 2. Velocity-dependent law와 contact state 변화 중 어느 쪽이 정보를 증가시키는가? | 재료별 rate sensitivity가 다르다. [Yu et al.][Y] | Coefficient의 속도 항을 기준 속도 값으로 고정하고 incoming speed만 올린 조건을 원래 law와 짝짓는다. |
| 3. 높은 속도에서 sensor clipping이 증가하면 순위가 뒤집히는가? | 실물 impact와 측정·모델의 예측 범위에는 제한이 있다. [Fazeli et al.][F] | 장비의 속도별 noise covariance·saturation을 측정하고 동일 예산에서 AUROC와 Brier를 다시 계산한다. |
| 4. 새로운 표면 패치에서도 네 번의 읽기를 재사용할 수 있는가? | 실제 지지면에서 friction의 공간적 변동이 관찰됐다. [Yu et al.][Y] | Patch 거리와 probe–query 지연을 교차하고 object·surface session을 함께 hold out한다. |
| 5. 방향과 함께 재료별 속도 의존성도 학습하면 query 예측이 좋아지는가? | 양방향 마찰 부호와 속도 민감성은 서로 다른 조건에 의존한다. [Tramsen et al.][T] · [Sakuma et al.][M] | 방향 부호·속도 기울기를 독립 생성하고 joint posterior와 frozen cue를 비교한다. |
| 6. 네 번째 probe의 고속 방향에 반대되는 query에도 효과가 있는가? | 방향을 바꾸면 anisotropy 부호의 의미가 달라진다. [Tramsen et al.][T] | Calibration 방향과 deployment query 방향을 독립 배정하고 signed 및 orientation-invariant decoder를 비교한다. |
| 7. Material AUROC 최적 speed와 후속 Brier 최적 speed는 언제 다른가? | Parameter 정보와 미래 quantity of interest 정보는 서로 다를 수 있다. [Zhong et al.][Z] | Training environments에서 두 기준으로 각자 하나의 speed를 선택·고정하고 새 panel에서 paired regret을 측정한다. |
| 8. Hard gate와 continuous shrinkage의 차이는 어느 예측에서 발생하는가? | 확률 결합은 값의 척도와 calibration에 의존한다. [Ranjan & Gneiting][R] | 같은 source labels로 multiplier를 학습하고 calibration, resolution 및 held-out Brier를 boundary별로 측정한다. |
| 9. 어떤 물체를 골라 접촉할지까지 허용하면 추가 이득이 남는가? | 관측 위치의 가치가 예측 대상에 따라 달라진다. [Zhong et al.][Z] | Factual geometry만으로 informative fourth object를 사전 선택하고 무작위 물체에 같은 속도로 접촉한 경우와 비교한다. |
| 10. 실물에서 높은 속도의 이득이 운동·sensor reset 비용을 감수할 만큼 큰가? | 실물 benchmark에서는 force·pose 기록과 반복 reset이 필요하다. [Yu et al.][Y] · [Arslan Waltersson & Karayiannidis][W] | 이전 gripper 실험의 2초 exploration을 참고해 physical cycle time, 에너지, GPU p90과 Brier를 각각 측정하고 동일 장비의 Pareto frontier를 추정한다. |

[Y]: https://arxiv.org/pdf/1604.04038
[T]: https://findresearcher.sdu.dk/ws/files/151776983/rsif.2017.0629.pdf
[M]: https://journals.aps.org/prl/pdf/10.1103/y3jy-nqcd
[F]: https://arxiv.org/pdf/1710.04979
[Z]: https://arxiv.org/pdf/2403.18072
[R]: https://academic.oup.com/jrsssb/article/72/1/71/7076442

[W]: https://research.chalmers.se/publication/549631/file/549631_Fulltext.pdf

## 부록 및 재현

- **Source와 target의 차이.** 이전 source simulator에서 세 속도의 density를 확인하고 +0.50 예측값을 v36과 원소별로 검산했다. [Source 검증표](SOURCE_KO.md) · [Source 결과](source37.json)
- **같은 물체에서의 gate 선택.** Local regime의 결정만 서로 바꾸어 배정한 대조와 항상 선택·건너뛰기의 비용, 방향 판독이 맞았을 때와 틀렸을 때의 결과를 비교했다. [선택 효과 및 Brier 분해](ROUTING_KO.md)
- **후보 표면과 보정량의 수.** Weak와 local null만 후보라고 알려준 likelihood 판별기와, 기존 utility table에 세 가지 보정량을 허용한 offline 정책을 분석했다. [두 후보 판별](BINARY_KO.md) · [세 행동 대조](TERNARY_KO.md)
- **실제 slip 상태와 예측 손실.** 같은 물체가 두 속도에서 보인 slip 전이를 기준으로 Brier 변화량을 분해했다. [Slip 집단별 결과](SLIP_KO.md)
- **물리 요인.** Incoming kinematics에 들어가는 속도와 friction modulation에 들어가는 속도를 simulator 안에서 각각 바꿔 clean velocity의 구별 가능성을 평가했다. [물리 요인 분해](PHYSICS_FACTOR_KO.md)

서버 GPU raw archive 224개를 CPU에서 다시 열어 Brier와 query labels, 세 속도의 slip/velocity, seed와 archive hash를 검산했다. 실행에 사용된 파일 34개의 hash와 서로 다른 seed ID 576개를 확인했다. [재현 방법](REPRODUCE_KO.md) · [독립 계산](stats37.py) · [source 고정 및 확인](SOURCE_KO.md). 서버 정본은 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/probe_speed_v37/REPORT_KO.md`에 있다.
