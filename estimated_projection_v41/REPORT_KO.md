# v41 · 새로운 물체와 접촉 높이에서 contour 변형을 제거한 뒤에도 보정 선택이 도움이 되는가

**두 v38 보정 정책 중에서는 환경별 선택 이득이 남았다. 그러나 같은 decoder에서 보정을 완전히 꺼 두는 편이 평균적으로 더 정확했다.** 처음 보는 flare·roll 물체와 19·31 mm contact height에서, 접촉 결과를 알려주지 않은 별도 물체 32쌍으로 contour 변형 방향을 추정했다. 12개 독립 catalog panel의 576개 block을 끝까지 평가했다. 방향 추정과 이후 보정의 가치를 따로 보면 결론이 달라진다.

| 질문 | 결과 | 판정 |
|---|---|---|
| Target contact label 없이 유효한 방향을 찾았는가? | 형상·높이별 평균 principal angle 10.63–17.01°. Fixed 보정의 held-out Brier가 original 0.125320 → estimated 0.121716, oracle 0.121816. | Estimated−original은 −0.003604 [−0.003875, −0.003332]. Estimated−oracle의 동시 95% 상한 +0.000080은 사전 허용값 +0.000300 이내다. |
| 이 방향을 없앤 뒤에도 보정 정책을 고를 이유가 있는가? | Estimated에서 selector 0.120552, fixed 0.121716, 항상 optimizer 0.121731. | Selector−fixed −0.001164, selector−optimizer −0.001179. 두 동시 95% 구간 모두 0 아래다. 선택 자체의 이득은 사전 실용 기준 0.003에 미달했다. |
| 보정 자체를 써야 하는가? | 같은 estimated decoder에서 보정을 끄면 0.120129. Selector는 그보다 +0.000423 [개별 95% CI +0.000219, +0.000628] 높았고 12개 panel 중 1개에서만 이겼다. | 이 catalog 배합에서 우선 사용할 방법은 **estimated projection + 보정 없음**이다. 두 보정 사이의 선택 성공을 보정 자체의 필요성으로 확대할 수 없다. |
| Latency 허용 기준을 만족하는가? | 1,024개 물체의 warm GPU p90은 estimated + selector 1,445.4 ms, estimated + fixed 1,442.4 ms. Ratio 1.002 [0.995, 1.233]. | 점추정 차이는 +0.21%지만 상한이 사전 허용한 +2%를 넘었다. 같은 latency라는 판정은 유보한다. 64회 passive readout의 실물 획득 시간은 별도다. |

보정 없는 방법까지 같은 GPU에서 직접 맞대결한 추가 측정에서, estimated + 보정 없음의 warm p90은 **1,423.7 ms**, estimated + selector는 **1,440.6 ms**였다. 같은 관측을 처리할 때 보정을 끄면 loss와 latency가 함께 줄었다. [동일 GPU crossover](latency_estimated_zero41.json)

가장 큰 변화는 **같은 물체의 외곽만 바꿨을 때 예측 확률이 흔들리는 정도**였다. Query-view 가지의 평균 절대 변화가 0.05679에서 0.00857로 줄었다. 반대로 변형이 없는 base contour만 평가하면 projection이 Brier를 0.001420 높였다. 실물에 전달할 주장은 주변 contour를 항상 지우자는 것이 아니라, **접촉 결과를 보존하는 변형이 얼마나 자주 일어나며, 그 방향이 물리 정보와 얼마나 겹치는지 함께 측정해야 한다**는 것이다. [주 결과](stats41.json) · [Matched contour 분해](contour_breakdown41.json) · [보정 없는 대조](no_correction41.json)

![방향 복원, 형상·높이별 projection의 효과, 같은 decoder 내 정책 선택 효과](v41.png)

그림 A는 12개 panel의 평균이다. B의 구간은 네 shape·height 대조를 함께 보정했다. C의 원·사각형은 각 decoder 안에서 두 사전 대조를 함께 보정한 95% 구간이고, 삼각형은 보정 없음과 사후 비교한 개별 95% 구간이다. B와 C에서 왼쪽이 첫 번째 방법의 Brier가 낮은 방향이다.

## 무엇을 새 물체에서 검증했는가

가령 로봇이 모서리를 밀 때, 실제로 닿는 곳은 그대로 두고 그 위쪽 외곽만 볼록하게 바꿀 수 있다. 물리적 결과는 같은데 영상만 보고 계산한 미끄럼 확률이 달라지면, decoder가 주변 모양을 물리적으로 너무 강한 증거로 사용한 셈이다. 이때 주변 모양의 변동 방향을 결과 정답 없이 찾아 지울 수 있는지, 그 뒤에도 네 번의 접촉에 따라 보정 세기를 바꿀 이유가 있는지를 분리했다.

| 고정하거나 바꾼 부분 | 실험 절차 |
|---|---|
| 새 물체와 높이 | 과거 데이터와 독립적인 12개 catalog panel에서 flare·roll 형상을 만들고 query contact height를 19·31 mm로 바꿨다. 기존 기준 높이는 25 mm다. 2개 shape × 2개 height × 3개 contour × 2개 contact law로 24개 cell을 평가했다. Saturating과 crossover law는 v40에서 이미 다뤘다. |
| Contour intervention | 각 물체의 base·hollow·bulge는 같은 접촉 위치와 local tangent를 갖도록 만들었다. Factual profile, calibration 충돌, 실제 stick/slip label과 선택된 observation을 대응시켰다. Query view는 접촉 높이의 양옆 6 mm까지 2 mm 간격으로 측정한 7개 좌표다. 좌표별 noise σ는 0.30 mm다. |
| Target label 없는 추정 | Panel·shape·height마다 별도 물체 32개에서 두 번씩, 총 64번 passive readout을 얻었다. 같은 물체 내 signed deformation amplitude는 두 번 독립적으로 달라지지만 그 값·부호를 추정기에 알려주지 않았다. Whitened difference에서 평균을 빼고 첫 principal component를 구했다. 이 과정에 필요한 것은 같은 물체라는 pairing, 좌표 정렬, 같은 deformation을 공유하는 group, 알려진 sensor noise다. |
| Calibration과 배포 | 환경마다 물체를 바꿔 네 번 calibration했다. 세 번은 기존 높이에서 방향을 번갈아 측정하고, 마지막은 23 mm, 속도 +1.10이다. Query 속도는 +0.425다. 한 번의 평가에서 환경당 128개 deployment 물체에 frozen K256 acquisition이 115개 query view와 13개 probe를 배정했다. 모든 방법이 같은 관측을 받았다. |
| Decoder | Original은 7개 좌표 전체를 쓴다. Estimated는 학습한 단위 방향 û를, oracle은 simulator가 아는 u를 써서 whitened query likelihood 공간에서 I−uuᵀ를 적용한다. Probe likelihood는 동일하다. 16,384개 source 가설은 실제 query 높이에 맞춰 관측 template과 가설의 stick event를 새로 계산했다. Acquisition이 사용하는 source 기준 높이 25 mm는 유지했다. 모든 비교에 같은 환경별 두 proxy alignment parameter(geometry와 material의 prior correlation)를 제공해 bank를 생성했다. 이 parameter까지 추정해야 하는 운영 조건은 별도 문제다. |
| Correction | Base stick 확률 p의 log-odds에 βzw를 더한다. β≈0.829와 signed cue z의 추정식은 동결했다. Fixed는 w=0.325, optimizer는 네 번의 calibration을 v38 source gain table에 넣어 w를 얻는다. Selector는 두 값 중 하나를 환경별로 고른다. 보정을 전혀 하지 않으면 w=0이다. |
| Selector의 정보 | 새 contact label·law 이름·shape 이름을 받지 않는다. Source label로 학습한 frozen scorer가 해당 환경에서 이미 획득한 128개 deployment 관측과 calibration을 사용한다. 한 환경의 모든 물체에 한 가지 방법을 적용하고, 각 noise 반복 때 독립적으로 선택한다. 실제 배치가 전부 도착한 뒤 내려야 하는 결정이다. |

**왜 paired difference인가.** 같은 대상의 profile을 xᵢ₁=bᵢ+aᵢ₁d+εᵢ₁, xᵢ₂=bᵢ+aᵢ₂d+εᵢ₂로 쓰면 차분에서 물체마다 다른 bᵢ가 정확히 사라진다. 좌표당 noise를 표준화하면 차분의 covariance는 isotropic noise와 ddᵀ에 비례하는 항으로 구성된다. 독립 Uniform[−1,1] amplitude와 표준화한 방향 v에서는 covariance가 2I+(2/3)vvᵀ다. 네 group의 변형 축 분산은 noise-only 분산의 3.76~6.90배다. 이 조건에서 첫 eigenvector는 변형의 축을 식별한다. 부호는 projection에서 사라진다. 여기서 passive는 충돌 측정 횟수에 들지 않는다는 의미다. 실물에서는 두 외곽 상태를 만들고 물체를 정렬하는 비용이 든다. 같은 강체의 동일 형상만 여러 번 촬영하면 aᵢ₂−aᵢ₁에 신호가 없어 이 방식은 성립하지 않는다. [생성식과 pair 정의](WORLD_KO.md) · [구현](world41.py)

**왜 항상 optimizer와도 비교하는가.** Selector가 거의 모든 환경에서 optimizer 하나만 골라도 평균적으로 fixed를 이길 수 있다. 환경별로 고르는 이득을 인정하려면 같은 projected decoder에서 fixed와 항상 optimizer를 모두 이겨야 한다. 두 정책의 확률 qₒ, q𝒻에서 ΔBrier=(qₒ−q𝒻)(qₒ+q𝒻−2Y)다. 과거 v39/v40 자료에서 이 차이와 연결되는 calibration residual을 예측하도록 ridge regression을 학습했다. v38 보정식을 그대로 사용하면서, 두 정책 중 하나를 고르는 상위 규칙은 v41용으로 학습했다. Source validation에서 선택된 모델의 projected selector−fixed는 leave-law-out에서 −0.000079, leave-shape-out에서 +0.000579였다. 두 결과 모두 hyperparameter 선택에 사용했다. [선택식과 학습 분할](SELECTOR_KO.md) · [Source 검증](selector_source41.json)

**평가 단위.** Panel마다 시작 방향이 다른 두 block, 여덟 환경, 두 표면 방향, strong·weak·null인 세 near-field regime, 32회 observation noise를 대응시켰다. Held-out Brier loss는 stick 사건 Y에 대한 (p−Y)²의 평균이고 작을수록 좋다. 24개 cell을 동일하게 가중했다. 반복 관측이나 재사용된 prototype 수가 confidence interval을 인위적으로 좁히지 않도록 독립 panel 12개를 추론 단위로 삼았다. Estimated 안의 두 selector 대조와 fixed에서의 두 projection 대조를 각각 Bonferroni–t로 보정했다. 각 family 내 양측 동시 95% 구간이다. 생성식·selector·판정 기준의 SHA-256을 target outcome 생성 전에 기록했다. [사전 분석 계획](PLAN_KO.md) · [동결 기록](freeze41.sha256) · [UTC 기록](frozen_utc.txt)

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

## v41 방향 추정 검산

모든 값은 독립 passive calibration 물체에서 얻었다. Angle은 추정 방향과 oracle 방향의 부호를 무시한 각도이며 0°면 완전히 일치한다. 남는 변형 길이 비율은 sin(angle)이다. 같은 물체 pairing을 깨뜨린 control은 각 prefix에서 두 번째 profile만 한 행씩 옮겨 계산했다.

| Shape / height | 4 pairs | 8 pairs | 16 pairs | 32 pairs [panel 평균 95%] | 32 pairs 제거 후 남는 길이 | Unpaired 32 평균 각도 |
|---|---:|---:|---:|---:|---:|---:|
| flare / 19 mm | 34.79° | 22.14° | 19.67° | 12.43° [10.35, 14.51] | 21.5% | 79.82° |
| flare / 31 mm | 41.17° | 23.97° | 18.28° | 13.74° [10.72, 16.76] | 23.7% | 30.50° |
| roll / 19 mm | 45.96° | 34.88° | 28.38° | 17.01° [13.80, 20.23] | 29.1% | 75.45° |
| roll / 31 mm | 32.15° | 23.76° | 15.85° | 10.63° [7.84, 13.42] | 18.4% | 88.47° |

매 group의 첫 32개 pair를 사용하는 기준에서 48개 독립 group calibration에 1,536개의 추가 물체, 3,072회 profile readout이 필요했다. Calibrated group은 두 law 및 모든 variant·시작 block에 재사용했다. Angle을 서로 다른 eigensolver로 재검산했고 oracle 방향도 별도의 compact support 식으로 검산했다. [원자료](direction_audit41.json)

![두 접촉 높이에서 12개 panel의 estimated direction과 oracle 비교](geometry_directions41.png)

그림에서는 방향의 부호를 oracle에 맞춰 표시했다. 실제 projection은 부호에 무관하다.

## 방향 제거가 무엇을 바꿨는가

같은 물체에서 base·hollow·bulge의 실제 stick event가 같기 때문에, 세 contour에서의 평균 Brier가 두 항으로 정확하게 나뉜다. 첫 항은 세 예측 확률의 분산이고, 둘째 항은 세 예측의 평균과 실제 event 사이의 제곱 오차다. 이는 contour에 따른 흔들림을 얼마나 줄였는지와, 흔들림을 없앤 확률의 중심이 얼마나 맞았는지를 구분한다. Fixed 32.5%에서 계산했다.

| 항목 | Original | Oracle | Estimated | Estimated−original, 개별 95% CI |
|---|---:|---:|---:|---:|
| Contour 간 예측 분산 | 0.005295 | 0.000000 | 0.000217 | -0.005078 [-0.005214, -0.004941] |
| Contour 평균 예측의 Brier | 0.120025 | 0.121816 | 0.121499 | +0.001474 [+0.001202, +0.001746] |
| Base만 사용한 Brier | 0.120149 | 0.121816 | 0.121569 | +0.001420 [+0.001170, +0.001671] |
| Hollow·bulge 평균 Brier | 0.127905 | 0.121816 | 0.121790 | -0.006115 [-0.006386, -0.005844] |

세 contour의 비율은 1:1:1이다. 같은 query view 가지에서 base와 두 변형 사이의 보정 전 절대 확률 변화는 original 0.056790, oracle 5.10e-16, estimated 0.008569였다. 같은 물체의 probe 가지는 세 decoder에서 동일했다. [재계산](contour_breakdown41.json)
Base 확률을 1−q로, 나머지 두 변형 확률을 각각 q/2로 놓고 이 실험의 strata를 그대로 재가중하면 estimated와 original의 평균 Brier 순위가 바뀌는 q는 18.8%다. 실세계 prevalence와 그 값의 신뢰구간은 측정하지 않았다.

### 물체 대응 관계와 저렴한 대안

이 보조 비교는 각 panel마다 홀짝으로 지정한 한 block, 32회 중 고정된 noise 0·16만 사용했다. 24개 cell과 12개 panel, 구매한 관측, source bank와 contact label은 모두 같고 추론은 서버 CPU에서 계산했다. GPU p90 비교에는 위 주 실험의 9개 방법과 별도 crossover만 포함한다.

| 같은 물체와 noise에서의 방법 | Group당 추가 profile | Fixed 32.5% Brier |
|---|---:|---:|
| Original | 0 | 0.125342 |
| Oracle projection | oracle 정보 | 0.121875 |
| Estimated projection (matched) | 64 | 0.121743 |
| Unpaired projection (sham) | 64 | 0.199663 |
| Local window (중앙 ±2 mm 세 좌표) | 0 | 0.129383 |
| Finite-variance marginalization (matched) | 64 | 0.121621 |

Estimated−unpaired는 **-0.077920 [-0.084147, -0.071693]**였다. 같은 profile을 유지하고 두 번째 관측의 물체 대응만 한 칸 순환 이동했다. Estimated−local window는 **-0.007641 [-0.008132, -0.007149]**였다. Local window는 contact height와 가까운 세 좌표만 likelihood에 넣는다. [Sham 결과](sham41.json) · [Local window 결과](local_window41.json)

Finite-variance 방법은 pair difference의 첫 eigenvalue에서 sensor noise를 뺀 뒤, 해당 방향의 residual variance를 유한한 값으로 적분했다. 같은 û를 사용해 정보를 부분적으로 남기는 방법이다. Fixed에서 finite-variance−hard projection은 **-0.000122 [-0.000296, +0.000052]**였다. Finite-variance 안에서 selector−fixed는 -0.001124 [-0.001432, -0.000817]였다. Local window와 finite-variance는 주 실험의 평가 도중 별도로 명세한 보조 분석이며 실용 후보 간 순위를 넓혀 보기 위한 것이다. [방법과 결과](marginal41.json)

이 보조 분석의 구간들은 12개 panel의 개별 95% 구간이다. 보조 표본의 평균 Brier를 576개 block의 주 표와 직접 빼지 않았다.

## V41 보정 선택의 환경별 재검증

평가할 물체 128개의 관측으로 한 번 내린 선택이 어느 environment에서 이득을 내는지 검산했다. 아래 수치는 12개 독립 catalog panel에서 계산한 paired estimate다. 괄호는 개별 대조의 95% t interval이며, 여러 대조를 한꺼번에 보정한 주 분석은 본 보고서에서 다룬다. 모든 Brier 차이는 음수일 때 앞의 방법이 우세하다.

| Decoder | Optimizer를 택한 환경·noise 비중, % | 선택한 곳의 optimizer−fixed | 선택하지 않은 곳의 optimizer−fixed | Selector−fixed | Selector−동일 비율 blind mixing |
|---|---:|---:|---:|---:|---:|
| original | 52.1 [50.5, 53.7] | -0.001697 | +0.001863 | -0.000883 | -0.000884 [-0.001034, -0.000733] |
| oracle | 54.8 [52.7, 57.0] | -0.002167 | +0.002604 | -0.001188 | -0.001093 [-0.001241, -0.000946] |
| estimated | 54.8 [52.7, 56.9] | -0.002123 | +0.002609 | -0.001164 | -0.001083 [-0.001232, -0.000935] |

각 panel·cell 내에서 optimizer 선택 비율만 맞추면, 관측을 보지 않고도 `fixed + 선택 비율 × 평균(optimizer−fixed)`의 Brier를 기대할 수 있다. 마지막 열은 실제 selector가 그 기대값보다 낮은 loss의 환경에 optimizer를 배정했는지를 측정한다. 선택한 곳과 선택하지 않은 곳의 두 열에는 각각 그 subset 전체의 실제 평균을 썼다. 이 두 subset은 무작위로 나눈 집단이 아니다.

| Decoder | 원래 decoder와 선택이 다른 비율, % | Selector−원래 decoder에서 내린 선택 재사용 | Selector−더 높은 stick probability 선택 | Selector−같은 배치에서 사후 선택 |
|---|---:|---:|---:|---:|
| original | 0.0 | 0.000000 [0.000000, 0.000000] | -0.000280 [-0.000412, -0.000147] | 0.001141 [0.001034, 0.001248] |
| oracle | 12.8 | -0.000048 [-0.000072, -0.000025] | -0.000583 [-0.000714, -0.000453] | 0.000758 [0.000667, 0.000850] |
| estimated | 12.2 | -0.000054 [-0.000074, -0.000033] | -0.000576 [-0.000699, -0.000453] | 0.000758 [0.000665, 0.000851] |

더 높은 stick probability를 택하는 규칙은 calibration cue의 부호와 두 보정량만으로 실행할 수 있다. 사후 선택은 128개 정답을 확인한 후 그 배치에서 Brier가 작은 방법을 고른 값이다. 이는 같은 배치의 경험적 하한이다.

| Decoder | Cross-fitted−fixed | Cross-fitted−항상 optimizer | Selector−cross-fitted | 같은 배치의 사후 선택−cross-fitted | 두 half에서 우세한 정책이 일치, % |
|---|---:|---:|---:|---:|---:|
| original | -0.001522 [-0.001719, -0.001324] | -0.001532 [-0.001697, -0.001367] | 0.000638 [0.000518, 0.000759] | -0.000502 [-0.000542, -0.000463] | 73.7 [72.5, 74.9] |
| oracle | -0.001414 [-0.001642, -0.001186] | -0.001402 [-0.001559, -0.001245] | 0.000226 [0.000110, 0.000341] | -0.000533 [-0.000576, -0.000489] | 73.1 [71.3, 74.9] |
| estimated | -0.001388 [-0.001617, -0.001160] | -0.001403 [-0.001559, -0.001248] | 0.000224 [0.000105, 0.000344] | -0.000534 [-0.000578, -0.000489] | 72.9 [71.2, 74.7] |

Cross-fitting은 128개의 절반가량에서 실제 label로 방법을 택한 뒤 나머지 물체에 적용하고 역할을 맞바꾼 결과다. 같은 prototype ID의 물체는 항상 같은 half에 둔다. 서로 다른 8개 분할의 평균을 내지만 독립 panel 수는 12개다. 한 방법이 거의 모든 환경에서 우세해도 두 half의 일치율은 높을 수 있다. 실제로 label 정보가 선택에 주는 이득은 cross-fitted 방법과 두 고정 방법의 Brier 대조로 해석한다. 실제 운영에 쓸 수 있는 label-free 선택 방법의 이론적 한계를 추정하지는 않는다.

사전 봉인한 소스와 계수는 [선택식](SELECTOR_KO.md), 보조 대조 정의는 [선택 진단](SELECTOR_DIAGNOSTICS_PLAN_KO.md)과 [cross-fitting 계획](CROSS_FIT_KO.md), 수치는 [진단 JSON](selector_audit41.json) 및 [cross-fitting JSON](crossfit41.json)에 있다.

## 논문에서 어느 주장까지 방어할 수 있는가

**가장 강한 결과는 같은 접촉을 다르게 보이게 만드는 관측 변화의 손익을 분리한 것이다.** 세 contour는 실제 label과 구매한 probe 결과가 같다. Fixed 보정에서 projection은 contour 사이 예측 분산을 0.005078 줄이지만, 세 예측의 평균과 실제 label 사이 오차를 0.001474 키운다. 두 값의 합이 주 효과 −0.003604와 정확히 일치한다. 깨끗한 base에서는 +0.001420 손해, hollow·bulge 평균에서는 −0.006115 이득이다. 같은 catalog를 재가중할 때 변형 비율이 약 18.8% 아래이면 original이 앞선다. 따라서 관측 불변성을 확보하는 것만으로 더 정확한 물리 예측이 보장되지는 않는다. [원 확률 분해](contour_breakdown41.json)

**방향을 추정한 정확도와 Brier의 최적성은 다르다.** Oracle은 실제 변형에 대한 확률 흔들림을 수치 오차까지 없앴다. Estimated는 흔들림이 일부 남았지만, pooled fixed Brier에서 oracle과의 차이는 −0.000100 [−0.000280, +0.000080]이었다. 실용 허용 손실 이내라는 증거가 있으며 oracle을 능가한다는 증거는 부족하다. 형상별로는 roll / 31 mm에서 original 대비 감소가 0.006556으로 가장 컸다. 반면 flare / 19 mm에서 estimated−oracle 구간의 상한은 0.000379였다. Pooled 판정을 모든 새 family에 적용할 수 없다. 독립 계산에서 정확한 oracle 방향조차 깨끗한 7점 profile이 주는 local tangent 추정 분산을 1.54–4.54배 높였다. [형상별 결과](stats41.json) · [Local tangent 정보 분석](PROFILE_INFORMATION_KO.md)

**같은 정보로 똑같이 비율만 조절한 것보다 실제 환경별 선택이 나았다.** Estimated selector는 54.8%에서 optimizer를 택했다. 그곳에서는 optimizer−fixed가 평균 −0.002123이고, 나머지에서는 +0.002609였다. 같은 cell·panel에서 선택 비율만 맞춰 무작위 배정하는 정책보다 Brier가 0.001083 낮았다. 반면 original decoder에서 내린 선택을 projected decoder에 그대로 옮겨도 전체 선택 이득의 약 95%가 남았다. 새 decoder의 확률로 다시 선택해서 더 얻은 이득은 0.000054였다. V40은 law·contour 이름을 아는 조건별 사전 예측이었고 v41은 관측만 쓰는 batch selector다. 양쪽의 절대 Brier는 평가 물체와 높이도 달라 직접 전후 차이로 해석할 수 없다. [환경별 보정 선택 검산](SELECTOR_AUDIT_KO.md)

**선택의 실용 한계도 수치로 드러난다.** 약 64개 물체의 실제 label을 보고 같은 환경의 다른 prototype에서 채점한 cross-fitted 대조는 fixed보다 0.001388 낮았다. Label-free selector가 그 이득의 약 84%를 회수했다. 더 유리하게 같은 배치의 전체 정답으로 두 방법 중 하나를 사후 선택해도 fixed 대비 최대 평균 감소는 약 0.001922다. 이 두 action을 유지하는 한 해당 empirical batch에서 0.003 감소는 불가능하다. 반면 보정 없음이라는 세 번째 action을 비교하면 항상 w=0이 selector를 0.000423 앞선다. 더 복잡한 두 정책 selector를 학습하기에 앞서, 물리 모델의 기본 확률에 보정을 가할 가치가 있는지부터 검증해야 한다. [Cross-fitting](crossfit41.json) · [보정 없음](no_correction41.json)

**왜 보정을 끄면 평균 loss가 더 낮은가.** 보정 전 확률을 p, selector가 가한 변화를 δ라 놓으면 Brier 변화는 2(p−Y)δ+δ²로 정확히 나뉜다. Estimated selector의 첫 항은 −0.000372로 평균 잔차를 줄이는 방향이었지만, 제곱 크기 비용 +0.000796이 더 커서 합은 +0.000423이었다. Law별 같은 대조의 점추정은 saturating에서 −0.000795, crossover에서 +0.001641이었다. Crossover에서는 첫 항도 +0.000801로 방향까지 불리했다. 이 분석은 결과를 확인한 뒤 원인을 분해한 것으로, 새 law를 미리 구별하는 selector의 성능을 뜻하지 않는다. [Brier 항별 원자료](zero_terms41.json)

**연구 기여의 무게.** 같은 대상의 paired difference로 nuisance subspace를 찾아 projection하는 방법은 문헌에 직접적인 선례가 있다. V41에서 논문화할 만한 부분은 물체·높이를 바꾼 matched physics benchmark, 실제 event가 유지되는지에 대한 독립 검산, 그리고 **변형 방향 추정·유용한 geometry 손실·보정 선택의 순이득**을 하나의 held-out loss에서 나눠 보인 점이다. 같은 pair로 변형 분산을 유한하게 적분한 baseline과 hard projection 사이에는 이번 보조 비교에서 우위를 확정할 수 없었다. 두 보정 사이의 selection을 주 알고리즘으로 내세우기에는 보정을 끈 baseline과 실제 운영 비용이라는 반대 증거가 있다. 현재 원고에서는 controlled mechanism study와 재현 가능한 evaluation protocol을 중심 기여로 삼는 편이 타당하다. [가까운 projection 선례](https://arxiv.org/html/2607.22923v1) · [Finite-variance 대조](marginal41.json)

### 어떤 일반화가 아직 필요한가

1. **방향을 찾는 것과 지워도 되는지 판단하는 것은 다른 문제다.** 두 profile에서 자주 달라지는 방향은 관측만으로 찾을 수 있다. 그 방향을 따라 변형해도 실제 stick event가 같다는 사실은 v41의 물리 실험 설계에서 얻었다. 실제 사용자가 건넨 미지의 물체에서 방향의 운동학적 무관성까지 label 없이 식별한 결과는 아니다. Oracle도 실제 변형 축만 아는 방법으로, Brier에 대한 최적 해법은 아니다.
2. **접촉 높이의 정확도가 중요하다.** 물체를 미는 점이 설계 높이에서 조금만 벗어나도 주변 contour가 실제 접촉에 관여한다. 생성식을 미분하면 ±0.25 mm 오차에서 19 mm의 bulge와 base 사이 local slope 차이는 flare에서 최대 약 0.314, roll에서 약 0.203이다. 명목 높이에서는 0이다. 이것은 미분식의 민감도이며, 높이 오차가 있는 상태의 Brier는 측정하지 않았다. 또한 decoder가 받는 입력은 등록된 7점 profile이다. Raw image, 유한한 contact patch, occlusion은 별도 검증이 필요하다. [세계 생성식](WORLD_KO.md) · [독립 물리 계산](physics_audit41.py)
3. **어느 target에서든 바로 쓸 수 있다는 주장은 범위를 넘는다.** 측정에 쓰는 group과 두 상태의 대응이 주어졌고, 각 group에는 추가 물체 32개가 있다. 세 decoder가 공유하는 source bank에는 환경별 proxy alignment도 주어졌다. 평가한 shape family와 height는 각각 두 개이며 contact law 두 개는 historical selector source에도 있었다. 12개 panel의 confidence interval은 이 조건에서 새 catalog를 뽑을 때의 불확실성을 나타낸다. 실물 material 모집단의 불확실성까지 포함하지 않는다.
4. **GPU latency와 데이터 획득 비용의 단위가 다르다.** Fixed와 optimizer에 같은 네 번의 calibration을 제공했으나 fixed의 보정량은 네 번째 응답에 의존하지 않는다. 64개 passive profile은 새 group에 처음 진입할 때 발생하며, selector는 128개 배치의 관측이 모일 때까지 기다린다. 한 번에 한 물체를 다루는 로봇의 전체 반응 시간 또는 작업 성공률은 GPU p90·Brier만으로 계산할 수 없다.

### 다음 한 실험

다음 버전에는 **어떤 방향을 지우면 실제로 위험해지는지**를 먼저 배치하는 편이 좋다. 같은 물체에서 (a) 접촉부는 고정하고 위·아래 외곽만 교체한 조건, (b) 같은 크기의 변형이 접촉부까지 이동한 조건을 만든다. 관측한 profile 차이의 크기와 sensor noise를 일치시키고, 숨긴 contact height 오차를 0·±0.25·±0.5·±1 mm로 사전 지정한다. 보정 action에 w=0을 포함한 세 가지 선택 규칙을 source에서만 동결한다. 물체별 true event flip rate, uncorrected/fixed/selected Brier, projection과 finite-variance·local-window baseline의 paired 차이를 함께 측정한다. 원래 event가 바뀌는 positive control을 포함해야 너무 많은 정보를 지운 경우를 검출할 수 있다.

실물로 옮길 때는 한 강체의 접촉 strip은 고정한 채 비접촉 insert만 교체한 물체와, 접촉 strip 자체도 바꾸는 물체를 함께 사용한다. Camera 및 contact height 오차, 재정렬 시간, slip 비용을 기록하고, 물체·재질 단위 holdout에서 같은 사전 기준으로 평가한다. 이 실물 실험은 v41의 결과에 포함되지 않는다.

## 선행연구 비교와 논문 기여의 위치

문헌 질문은 두 가지다. 한 물체의 여러 관측으로 예측에 방해되는 변형 방향을 찾은 선례가 있는가? 그런 방향을 지운 뒤에도 물리 보정의 세기를 환경마다 바꿀 근거가 있는가? 아래 여덟 연구는 이 질문에 가까운 방법과 실제 접촉 실험을 연결한 문헌 표본이다. 표의 표본 수는 원문의 실험 단위를 따른다. 여러 번 측정한 같은 물체를 독립 물체로 환산하지 않았다.

| 연구 | 연구 질문 | 방법과 자료의 규모 | 주요 관찰과 저자의 결론 | v41과의 차이·한계 |
|---|---|---|---|---|
| Solomonoff et al., ICASSP 2005 | 녹음 채널이 달라도 같은 화자를 알아볼 수 있는가? | Switchboard 2의 일부에서 같은 화자의 여러 session을 묶어 eigenvalue problem을 풀고 SVM 공간에서 nuisance dimension을 제거했다. | Cross-channel recognition이 개선됐다. 같은 대상의 반복 측정에서 nuisance subspace를 추정할 수 있는 직접적인 선례다. | 화자 ID와 session 대응이 있고 목표는 recognition이다. 실제 접촉에서 어떤 shape 변화가 무해한지는 결정해 주지 않는다. [원문 정보와 저자 요약](https://www.ll.mit.edu/r-d/publications/advances-channel-compensation-svm-speaker-recognition) |
| Hosseini-Kivanani, TidyVoice 2026 challenge preprint | 같은 화자가 언어를 바꿀 때 생기는 embedding 변화를 제거할 수 있는가? | 3,666명 training, 808명 development, 2,200명 evaluation 화자. 같은 화자의 언어별 embedding 차이에 PCA를 적용하고 orthogonal complement에서 점수를 냈다. | Development EER가 AS-Norm의 2.70%에서 projection을 더해 2.18%로 감소했다. 같은 대조에서 minDCF는 0.64에서 0.73으로 커져 지표에 따라 손익이 갈렸다. | v41과 가장 가까운 수학적 절차다. 화자 확인과 접촉 예측의 outcome·metric가 다르다. [원문 §2.3·3.1](https://arxiv.org/html/2607.22923v1) |
| Locatello et al., ICML 2020 | 무엇이 달라졌는지 알려주지 않고 관측 쌍으로 요인을 분리할 수 있는가? | 일부 latent factor를 공유하는 image pair를 이용했다. 다섯 dataset, 15,000개 이상의 학습 모델, 100만 개 이상의 downstream classifier를 비교했다. | 관측 간 관계를 주면 여러 benchmark에서 representation이 안정적으로 학습됐다. 이론은 latent factor와 pair 생성 과정에 관한 가정에 따른다. | v41은 7차원으로 정렬된 profile의 한 방향을 제거한다. Pair ID와 좌표 대응도 정보이므로 완전한 unsupervised discovery로 볼 수 없다. [원문 §2–4](https://arxiv.org/pdf/2002.02886) |
| Belrose et al., NeurIPS 2023 | 표현을 최소한으로 바꾸면서 특정 concept를 linear classifier가 읽지 못하게 할 수 있는가? | LEACE의 closed-form linear map을 유도했다. Part-of-speech 정보와 BERT embedding의 gender bias라는 두 응용을 다뤘다. | 주어진 선형 조건에서 concept erasure와 최소 변형을 함께 달성했다. | 지정한 concept와 representation이 필요하다. v41의 paired PCA나 downstream Brier 최적성은 이 정리에서 나오지 않는다. [원문](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d066d21c619d0a78c5b557fa3291a8f4-Abstract-Conference.html) |
| Fazeli et al., 2017 | 실제 impact 관측으로 물리 parameter와 충돌 결과를 얼마나 정확히 식별하는가? | 평면에서 로봇이 물체를 떨어뜨렸다. 2,000회 중 품질 검사를 통과한 1,718회와 여섯 rigid-body contact model을 비교했다. | 관측 impulse의 표현 범위와 model별 식별 한계가 드러났다. 개별 impact에 맞춘 parameter의 물리적 해석에는 주의가 필요했다. | 실험은 접촉 모델 사이의 오차를 측정한다. 이 자료만으로 paired contour projection의 효과나 32.5%라는 계수를 정할 수 없다. [원문 §3–4](https://arxiv.org/pdf/1710.04979) |
| Jolaei et al., Journal of Robotics and Mechatronics 2026 | 보지 못한 접촉 영역을 tactile imprint에서 외삽할 수 있는가? | 세 형상×다섯 크기의 실물, 약 6,900개 tactile sample을 사용했다. Shape classifier와 형상별 외삽, grid scan을 비교했다. | Held-out shape classification accuracy는 94.3%였다. 제한된 sensor resolution에서는 일부 큰 원통과 평면을 혼동했다. | Material·stick Brier를 평가하지 않았다. Geometry를 보존할 필요가 있는 실제 사례라서 무엇을 지울지 결정하는 대조 연구가 된다. [원문 §3–5](https://www.fujipress.jp/jrm/rb/robot003800030740/?full=1) |
| Zhong et al., ActivePusher, ICRA 2026 | Residual physics와 선택적 측정으로 실제 pushing을 개선할 수 있는가? | Simulation 물체 4종과 UR10에서 실물 2종을 시험했다. Skill learning은 물체마다 독립 test action 1,000개, 실제 planning은 20개 문제를 사용했다. | 적은 자료에서 residual model의 예측이 개선됐고 active sampling은 planning에도 이득을 보였다. | 측정 선택·학습·행동이 모두 달라질 수 있는 설계다. v41은 동일 관측을 받아 계산하는 decoder와 correction의 효과를 측정한다. [원문 §V](https://arxiv.org/html/2506.04646v4) |
| Yao et al., Bayesian Analysis | 조건마다 모델 순위가 다를 때 pooling이 특정 모델의 상시 사용을 개선하는가? | Input-dependent stacking weight를 partial pooling했다. 응용 중 하나는 미국 50개 주×40주 polling이며 다음 주 predictive density로 평가했다. | 예측 우위가 입력에 따라 바뀌고 자료가 제한적일 때 계층적으로 weight를 공유하는 이득을 보였다. | 확률 혼합과 hard policy selection은 다르다. 새로운 형상에서 순위를 구별하는 정보가 실제로 있는지는 별도 외부 검증이 필요하다. [원문 §3, §5.3](https://arxiv.org/html/2101.08954) |

### 근거의 연결과 상충하는 결과

* **같은 대상 안에서 변한 정보 → nuisance 추정 → projection.** 음성의 session과 영상의 pair 모두 어떤 관측이 같은 대상에 속하는지 알려 준다. 이 관계를 이용하는 방법 자체는 확립돼 있다. LEACE는 supervision과 제거 목표를 명시하면 어떤 선형 정보가 사라졌는지 증명할 수 있다는 별개의 접근을 제공한다. 따라서 v41이 기여할 수 있는 지점은 paired PCA의 발명이 아니라, 같은 물리적 query를 보존하는 변형에서 그 추정 오차가 물리 확률 예측과 보정 선택에 미치는 영향을 따로 측정한 데 있다. [Solomonoff et al.](https://www.ll.mit.edu/r-d/publications/advances-channel-compensation-svm-speaker-recognition) · [Locatello et al.](https://arxiv.org/pdf/2002.02886) · [Belrose et al.](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d066d21c619d0a78c5b557fa3291a8f4-Abstract-Conference.html)
* **접촉 형상을 보존하면 좋아지는 결과와 형상 성분을 지우면 좋아지는 결과.** Jolaei 등의 target은 손끝이 어디에 닿는지이고 geometry 자체가 답의 일부다. v41에서는 simulator가 contact tangent와 event를 보존하도록 일부 contour만 바꾼다. 실물 tactile patch와 synthetic seven-point profile, 공간적 centroid와 binary Brier의 차이가 두 관찰을 함께 설명한다. 실제 contact height가 흔들리거나 접촉 면적이 커지면 주변 contour가 다시 유효해질 수 있다는 것은 이 비교에서 도출한 가설이다. [Jolaei et al.](https://www.jstage.jst.go.jp/article/jrobomech/38/3/38_740/_article/-char/en) · [Fazeli et al.](https://arxiv.org/pdf/1710.04979)
* **Calibration → correction → policy selection → 실제 행동.** Impact 하나로 parameter를 식별하기 어렵다는 실물 증거와, residual physics로 행동이 좋아졌다는 실물 증거는 양립한다. 전자는 개별 충돌·impulse를, 후자는 여러 학습 action을 거친 planning을 평가했다. 환경별 모델 결합도 historical outcome과 재현 가능한 조건 차이에 의존한다. 같은 실험에서 projection의 평균 이득, 개별 환경의 예측 가능성, 실제 행동 개선을 각각 입증해야 한다. [Fazeli et al.](https://arxiv.org/pdf/1710.04979) · [Zhong et al.](https://arxiv.org/html/2506.04646v4) · [Yao et al.](https://arxiv.org/html/2101.08954)

**증거의 수준.** 검토한 논문들은 paired observation이 정보를 제공하고 접촉 측정의 물리 조건이 예측 가능성을 바꾼다는 점을 서로 다른 자료에서 지지한다. 실물에서 shape와 learned residual이 유용하다는 근거는 물체와 작업에 한정돼 있다. 이 여덟 연구 중 새 실물의 같은 tangent에서 contour를 바꾸고 동일한 friction correction의 환경별 Brier 우위를 외부 검증한 연구는 없었다. 이는 이 문헌 표본에서 확인한 gap이다. 최초성, 임의 형태의 변형 식별, 로봇 작업 성공률은 이 근거로 판정할 수 없다. [Locatello et al.](https://arxiv.org/pdf/2002.02886) · [Jolaei et al.](https://www.jstage.jst.go.jp/article/jrobomech/38/3/38_740/_article/-char/en) · [Zhong et al.](https://arxiv.org/html/2506.04646v4)

### 근거에서 도출한 후속 연구 질문 10개

| 질문 | 근거가 비어 있는 부분과 중요성 | 변수와 검증 방법 |
|---|---|---|
| 1. 실물에서 어떤 contour 변화가 event를 유지하는가? | 실제 충돌 모델의 적용 범위와 tactile geometry의 정보량이 다르다. [Fazeli et al.](https://arxiv.org/pdf/1710.04979) · [Jolaei et al.](https://www.jstage.jst.go.jp/article/jrobomech/38/3/38_740/_article/-char/en) | 같은 접촉부에 교체식 측면을 부착한다. 자세·속도를 고정하고 contour별 stick rate와 Brier를 측정한다. |
| 2. Pair 대응이 얼마나 틀리면 제거가 해로운가? | Pair는 weak supervision이므로 correspondence에 비용과 오차가 있다. [Locatello et al.](https://arxiv.org/pdf/2002.02886) | 대응 오류율 0–100%에서 독립 물체별 angle과 Brier의 break-even point를 측정한다. |
| 3. 방향 하나가 충분한지 target label 없이 판단할 수 있는가? | 과거 subspace 제거는 자료로 nuisance dimension을 정한다. [Solomonoff et al.](https://www.ll.mit.edu/r-d/publications/advances-channel-compensation-svm-speaker-recognition) | 0–3개 deformation axis를 독립 생성해 eigengap·split-half angle로 rank를 정하고 blind test한다. |
| 4. Subspace를 완전히 지우는 것과 일부만 낮추는 것 중 언제 유리한가? | 선형 제거의 최적성은 표현에 가하는 제약에 따라 달라진다. [Belrose et al.](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d066d21c619d0a78c5b557fa3291a8f4-Abstract-Conference.html) | Local tangent와 변형의 정렬도를 바꾸고 soft attenuation 및 nuisance marginalization을 같은 bank에서 평가한다. |
| 5. Contact height 오차는 불변성의 유효 범위를 얼마나 줄이는가? | 실물 tactile geometry의 정보는 위치와 공간 분해능에 좌우된다. [Jolaei et al.](https://www.jstage.jst.go.jp/article/jrobomech/38/3/38_740/_article/-char/en) | 위치 jitter를 0, 0.5, 1, 2 mm로 사전 지정하고 label flip과 risk를 함께 기록한다. |
| 6. Passive calibration의 이득이 획득 비용을 상쇄하는 최소 배치 수는 얼마인가? | 측정 비용까지 고려하면 residual model의 이득이 달라진다. [Zhong et al.](https://arxiv.org/html/2506.04646v4) | Readout·교체·정렬 시간을 재서 pair 수, 배치 재사용 횟수와 함께 cost–risk curve로 보고한다. |
| 7. 실제 마찰 재료가 바뀔 때도 source 보정 순위가 유지되는가? | 실제 충돌에서는 parameter 해석과 전이가 제한된다. [Fazeli et al.](https://arxiv.org/pdf/1710.04979) | Shape와 material을 독립 배정하고 완전히 새로운 material family에서 selector와 fixed의 paired risk를 비교한다. |
| 8. 물체 한 개에서 결정해야 할 때도 선택 이득이 남는가? | 조건별 모델 우위와 그 우위를 추정할 표본 수는 별개다. [Yao et al.](https://arxiv.org/html/2101.08954) | 환경당 1·4·16·128개 순차 도착을 고정하고 미래 관측 없이 hard selector와 hierarchical pooling을 평가한다. |
| 9. Geometry-only 방향을 새로운 sensor로 전달할 수 있는가? | 실제 tactile 해상도와 영상 pair의 정렬 가정이 다르다. [Jolaei et al.](https://www.jstage.jst.go.jp/article/jrobomech/38/3/38_740/_article/-char/en) · [Locatello et al.](https://arxiv.org/pdf/2002.02886) | Camera silhouette와 tactile profile의 calibration 오차를 따로 기록한 뒤 sensor-held-out Brier를 비교한다. |
| 10. 낮아진 Brier가 실패 비용을 실제로 낮추는가? | Planning 실험은 예측 오차와 실제 성공을 구분해 측정했다. [Zhong et al.](https://arxiv.org/html/2506.04646v4) | Stick 확률에 따라 실행·중단하는 threshold와 slip 비용을 먼저 정하고, 독립 실물에서 cycle time·실패율·utility를 기록한다. |

## 실행과 독립 검산

서버 `datacenter3`의 Slurm allocation 2744468에서 시작 직전 compute process 0개, memory 0 MiB인 NVIDIA RTX PRO 6000 Blackwell Server Edition 한 장을 사용했다. 주 실험 576개 block의 dependency fingerprint와 파일별 SHA-256을 검산했다. GPU 환경은 PyTorch 2.7.1+cu128, CUDA 12.8였다. 세 agent가 세계 생성·물리 검산, selector 학습·선택 검산, GPU 실행·결과 검산을 분담했다. [서버 기록](results/completion.json) · [재현 절차](REPRODUCE_KO.md)

Calibration과 배포에 독립 catalog를 사용했다. 12×2×8×128=24,576개 deployment draw에 실제로 서로 다른 prototype은 23,153개였다. 실제 calibration 접촉에 쓰인 서로 다른 prototype은 765개다. 12개 독립 panel 안에서 contact law·형상·높이·시작 방향·noise의 짝 구조를 유지했다. [물리 검산](physics_audit41.json)

별도의 NumPy collision 식으로 3,538,944개 저장 label을 재구성했다. 불일치는 0이었다. 248,832개 환경별 loss 재계산의 최대 절대 오차는 1.67e-16였고, 동일 물체의 세 contour에서 oracle 확률의 최대 차이는 1.66e-13였다. 16,384개 가설의 단일 bank를 점검했을 때, 기존 25 mm event를 그대로 사용하면 실제 19 mm·31 mm에서 각각 295개·180개 가설이 다른 event를 예측했다. 새 높이의 likelihood는 직접 계산한 squared distance와 2.89e-15 이내로 일치했다. [Brier 검산](audit41.json) · [Decoder 검산](decoder_audit41.json)

이미 메모리에 있는 64개 profile에서 estimated direction을 다시 구하는 host 구간은 median 0.153 ms, p90 0.187 ms였다. Estimated template과 event를 새로 만드는 전체 setup의 p90은 7.00 ms였다. 두 값은 12개 panel × 3회 × 3개 정책에 걸친 108회 setup에서 계산했다. 실제 64개 profile의 획득·대응·형상 교체를 포함한 총시간은 별도로 측정해야 한다.

서버 작업 폴더는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/estimated_projection_v41`다. 주 결과는 [통계 JSON](stats41.json), [GPU runner](runner41.py), [label-free 추정](world41.py), [frozen selector](selector41.py)에서 다시 확인할 수 있다.
