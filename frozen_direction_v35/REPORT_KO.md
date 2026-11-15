# v35 · 네 번의 probe로 읽은 friction 방향은 새 물체의 충돌 예측에도 유효한가?

**새 synthetic 물체 두 종류와 directional surface law 두 종류를 교차한 결과, fixed 4의 held-out Brier는 원래 physics decoder 0.11860에서 v34의 frozen correction 0.11277로 0.00583 낮아졌다.** 두 schedule을 같은 비중으로 둔 paired Δ의 95% interval은 [-0.00637, -0.00532]다. 양쪽 schedule에서 사전에 정한 실용적 개선도 재현됐다.

비교한 두 모델은 probe도 observation도 같은 것을 사용했다. 달라진 것은 다음 충돌에서 stick할 확률뿐이다. 방향성이 없는 별도의 표면에서는 오히려 pooled Brier가 0.00607 높아졌다. 따라서 보고할 수 있는 주장은 probe로 읽은 방향이 query 충돌에도 이어지는 조건에서의 **terminal prediction** 개선이다. 아래에 일반적인 score 보정만으로 설명되는 부분과, 같은 evidence가 있어도 보정이 실패하는 조건을 분리했다.

| Calibration motion | 원래 physics | Frozen correction | Δ Brier ↓ | Simultaneous 95% interval | 판정 |
|---|---|---|---|---|---|
| 한 방향 | 0.11885 | 0.11284 | -0.00601 | [-0.00688, -0.00514] | 실용 기준 통과 |
| 양방향 교대 | 0.11835 | 0.11270 | -0.00565 | [-0.00647, -0.00484] | 실용 기준 통과 |

Δ는 같은 world·같은 관측에서 **보정 − 원래 physics decoder**다. 음수가 개선이다. 사전에 정한 실용 기준은 interval 상한이 −0.003보다 작은 것이다. 두 schedule의 구간은 동시에 보정했다. [전체 통계와 원자료 검산](audit35.json)

## GPU p90: 방향을 읽고 보정하는 시간까지 포함한 비용

| Calibration motion | 원래 p90 (ms) | 보정 포함 p90 (ms) | 변화 | Simultaneous 95% interval | +2% 상한 |
|---|---|---|---|---|---|
| 한 방향 | 1,418.1 | 1,420.9 | +0.20% | [-0.28%, +0.67%] | 충족 |
| 양방향 교대 | 1,421.2 | 1,420.9 | -0.02% | [-0.23%, +0.19%] | 충족 |

장치는 **NVIDIA RTX PRO 6000 Blackwell Server Edition** 한 장이다. 한 호출은 물체 1,024개를 함께 처리한다. 각 방법·schedule당 48회, 독립 timing block은 8개다. 반복별 실행 순서를 바꿨다. +2%는 시간 증가에 대해 미리 정한 허용 상한이다. 이 한 세션에서 얻은 interval이며 평균 속도나 단일 물체의 latency로 읽으면 안 된다.

보정 계산만 따로 실행한 server CPU microbenchmark의 p90은 한 방향 0.146 ms, 교대 0.145 ms였다. 원래 확률이 이미 host에 도착한 시점부터 paid rows 추출·reader·1,024개 odds update를 포함한다. 이 값을 전체 GPU p90끼리 뺀 값으로 대체할 수 없다. 서로 다른 실행의 p90은 더하고 뺄 수 있는 단계별 시간이 아니다. [측정 원자료](results/run.json) · [보정 단계만의 측정](head_cost35.json) · [실행 범위](REPRODUCE_KO.md)

## 무엇을 고정하고 무엇을 바꿨나

정답은 물체를 다른 높이와 속도로 충돌시켰을 때 **stick**하면 1, **slip**하면 0이다. 예를 들어 같은 네 번의 probe를 본 두 예측기가 한 물체에 대해 stick 확률 0.6과 0.7을 내면 실제 stick일 때 Brier loss는 0.16과 0.09, 실제 slip일 때는 0.36과 0.49다. 양쪽 모두 같은 물체·같은 noise의 정답으로 채점했다. 방향을 정확히 분류하는 것과 이 손실을 줄이는 것은 다른 과제다.

v34에서는 결과를 본 뒤 고른 보정식이 fixed 4에서 Brier를 단방향 0.00596, 교대 0.00535 낮췄다. 당시 네 family에는 각각 학습된 reader/β 쌍이 대응했다. v35에서는 새 물체의 family를 보고 쌍을 고를 수 없게 했다. 저장 순서상 첫 held-out fold인 `rounded_fine`의 k=4 reader와 β=0.8289760825845313을 모든 조합에 적용했다. 단방향인지 교대인지는 실행자가 아는 action schedule이므로 각 schedule의 기존 reader를 그대로 사용했다. 선택한 reader와 β를 학습한 v34 물체·표면 조합은 `ridged_coarse`다. 같은 frozen 쌍이 v34의 `rounded_fine`에서 냈던 Δ는 각각 −0.00575/−0.00418이었다. 나머지 세 쌍은 sensitivity용이고 primary 선택에 쓰지 않았다. Target outcome으로 fitting이나 threshold 선택을 하지 않았다. [v34의 출발점](../bidirectional_probe_v34/ANCHORED_RESULTS_KO.md) · [검증 전에 기록한 계획](AUDIT_PLAN_KO.md)

1. **서로 같은 관측:** 환경마다 contour와 factual 충돌의 velocity를 측정한 물체 1,024개 중 미리 뽑은 서로 다른 4개에 calibration probe를 한다. Tangential speed의 절댓값은 1.10이며, `same`은 같은 부호 네 번, `alternating`은 양쪽을 번갈아 두 번씩 쓴다. 처음 움직이는 부호는 각 조합 안에서 균형을 맞췄다. 별도의 deployment 물체 128개 중 13개에는 기존 K256 acquisition이 고른 probe, 나머지에는 query view를 적용했다. 한 환경의 두 decoder 사이에 달라지는 것은 마지막 확률 계산뿐이다.
2. **같은 원래 physics decoder:** 공통 Coulomb decoder는 선택한 observation을 받아 각 물체의 stick probability p를 낸다. 두 arm 모두 실제 population proxy coupling α를 알고 있다. 유리한 공통 가정이다. Target contact law, orientation, θ, stick label은 추론기에 제공하지 않는다.
3. **고정 보정식:** paid measurements와 수행한 motion directions로 이 환경에서 양의 friction orientation일 확률 q를 읽는다. z=2q−1, p′=p exp(βz)/(1−p+p exp(βz))이다. q=0.5면 p′=p이고, q>0.5면 p′가 커진다. 확률 0과 1은 유지된다. 한 환경 안에서는 같은 odds multiplier를 쓰므로 그 환경 안 물체 간 확률 **순위**는 바뀌지 않는다. 전체 Brier의 변화는 probability prediction에 관한 것이며 stopping이나 observation selection의 개선을 뜻하지 않는다.
4. **새 world:** v34의 cubic contour·tanh friction response 대신 sinusoid slope와 quartic contour 두 물체 모델, arctangent와 algebraic saturation 두 contact laws를 완전 교차했다. 두 표면의 최대 directional amplitude는 각각 0.21, 0.16이다. 각 block에서 orientation만 뒤집은 두 world가 물체·bank·sensor noise를 공유한다. 같은 두 물체에 isotropic surface도 넣었다. 이 경우 비교할 실제 방향이 없다. 네 주조합 모두에서 calibration과 query는 같은 방향의 anisotropy를 공유한다. 새로운 함수 형태를 사용했지만 이 가정까지 새롭게 만든 것은 아니다. [정확한 함수](target35.py) · [표본과 seed](design35.json)
5. **표본과 구간:** 주조합 4개 × 독립 block 8개 × 환경 8개 × deployment 물체 128개로, orientation과 schedule을 세기 전 latent deployment objects는 32,768개다. 각 물체에 32번의 sensor noise를 사용했다. 네 조합과 두 orientation, 두 schedule에 같은 가중치를 준다. 32개 block을 cell·첫 motion 부호 안에서 resample해 paired interval을 만들었다. Schedule별 주결과는 두 비교에 대한 simultaneous 95% interval을 쓴다. 반복 noise를 독립 표본으로 세지 않았다. 구간은 이 네 설계 조건에 대한 불확실성이며 실제 물체 모집단의 구간이 아니다.
6. **GPU p90:** 한 호출은 환경 8개에서 새 물체 1,024개를 처리한다. 원래 arm과 보정 arm을 섞은 순서로 각 호출을 재실행했다. CUDA synchronize를 양 끝에 두고, CPU 입력부터 acquisition, observation decoding, host probability 반환까지의 wall time을 쟀다. 보정 arm의 paid feature 추출, reader 및 odds update도 안에 들어간다. Bank 최초 생성, simulator의 자료 생성, 장비 reset과 실제 probe 대기는 양쪽 모두 제외했다. Brier는 32개 noise를 평균하지만 latency에는 온라인에서 실제 도착한 observation 하나를 쓴다.

## 어느 물체에서 얼마나 좋아졌나?

| 새 물체 × 표면 | 한 방향 Δ | 양방향 교대 Δ |
|---|---|---|
| sinusoid × atan | -0.00359 | -0.00419 |
| sinusoid × algebraic | -0.00737 | -0.00663 |
| quartic × atan | -0.00806 | -0.00653 |
| quartic × algebraic | -0.00502 | -0.00527 |

이 표는 조합별 point estimate다. 네 조합은 전체 물체·표면에서 무작위 추출한 표본이 아니다. 같은 실험의 방향 판독 성능과 잘못된 부호를 넣었을 때의 결과를 함께 보면 cue가 성공한 이유와 실패 경계를 더 좁힐 수 있다.

| Calibration motion | 방향 accuracy | 방향 Brier | 부호를 잘못 뒤집으면 Δ | 실제 부호를 주면 Δ | 다른 세 frozen 쌍 Δ |
|---|---|---|---|---|---|
| 한 방향 | 89.6% | 0.07984 | +0.02371 | -0.00905 | -0.00677, -0.00623, -0.00717 |
| 양방향 교대 | 89.5% | 0.07497 | +0.02470 | -0.00902 | -0.00675, -0.00619, -0.00689 |

부호 반전은 추정 cue에 −1을 곱한 대조, 실제 부호는 q 추정을 정답으로 바꾼 진단이다. 둘 다 같은 β를 쓴다. 실제 부호를 제공한 진단은 배포 가능한 방법이나 모든 모델에 대한 하한이 아니다. 다른 세 쌍은 v34 모델을 재학습하지 않고 교체한 sensitivity이고, 그중 가장 좋은 값을 보고 모델을 고르지 않았다.

## 무엇이 개선을 설명하는가?

| Calibration motion | 확률을 옮긴 비용 E[d²] | residual과 맞은 정도 2E[d(y−p)] | 차이 Δ |
|---|---|---|---|
| 한 방향 | 0.00959 | 0.01560 | -0.00601 |
| 양방향 교대 | 0.01032 | 0.01598 | -0.00565 |

d=p′−p라 하면 ΔBrier=E[d²]−2E[d(y−p)]가 정확히 성립한다. 첫 항은 예측을 움직인 것 자체의 비용이다. 둘째 항은 움직인 방향과 크기가 원래 예측의 residual에 맞았을 때 생기는 이득이다. 방향 분류 accuracy가 높더라도 두 번째 항이 첫 번째 항보다 작으면 Brier가 나빠진다. 아래에서는 paid evidence와 예측은 그대로 두고 query 정답만 바꾸어 이를 확인했다.

| query law만 바꾼 조건 | 기존 정답이 바뀐 비율 | 한 방향 Δ [95%] | 양방향 교대 Δ [95%] |
|---|---|---|---|
| 같은 directional law | 0.0% | -0.00601 [-0.00674, -0.00525] | -0.00565 [-0.00639, -0.00495] |
| query amplitude 절반 | 3.4% | +0.00162 [+0.00092, +0.00232] | +0.00210 [+0.00132, +0.00287] |
| query만 isotropic | 6.8% | +0.00969 [+0.00904, +0.01034] | +0.01055 [+0.00971, +0.01141] |
| query에서 부호 역전 | 13.5% | +0.02348 [+0.02260, +0.02437] | +0.02463 [+0.02378, +0.02556] |
| 20개 측정 magnitude를 query에 대입 | 4.9% | +0.00538 [+0.00474, +0.00601] | +0.00604 [+0.00526, +0.00681] |

원래 query의 정답은 각 archive의 θ로 독립적으로 다시 계산했고 전체 원래 label과 일치해야 표를 생성하게 했다. 나머지 행은 동일 물체의 query에서만 directional friction을 약화·제거·반전한 반사실이다. 마지막 행은 Tramsen et al.의 공개 S1에 있는 20개 normalized amplitude의 절댓값을 같은 가중치로 사용한 sensitivity다. query의 부호는 원래 환경과 일치시킨 채 magnitude만 바꿨다. reader에는 정답 부호를 주지 않았다. 실물에서 얻은 stick label이나 실물 Brier가 아니다. 이 표의 interval은 조건별 pointwise 95%로 primary 판정에 쓰지 않았다. [사전 기준](TRANSFER_STRESS_PLAN_KO.md) · [계산값](transfer_stress35.json) · [S1 원자료](https://doi.org/10.6084/m9.figshare.c.3950773.v2)

### 이 대조에서 알 수 있는 식별 한계

위 reverse 조건에는 모델을 바꿔도 넘을 수 없는 한계가 하나 있다. 두 실험실이 같은 물체에 완전히 같은 네 번의 probe를 했고, 같은 factual data와 같은 deployment observation을 전달했다고 하자. 한쪽 실험실에서만 **질문한 충돌에서의** directional friction 부호가 바뀐다. 이 실험의 데이터 생성 방식에서는 예측 직전까지 두 실험실을 구분하는 값이 하나도 없다. 어떤 reader를 쓰더라도 그 입력만으로 ‘어느 실험실인가’를 맞힐 수 없다. 실제 재료에서 반드시 이런 상황이 발생한다는 주장이 아니라, 이 설계에서 같은 evidence가 서로 다른 query 결과와 양립한다는 뜻이다.

두 regime을 같은 확률로 고른다고 명시하면 한계의 크기도 계산할 수 있다. 두 정답이 다른 물체에서 공통 evidence에 하나의 확률 u를 내야 할 때 평균 Brier는 [u²+(1−u)²]/2 = (u−0.5)²+0.25다. 따라서 이런 물체는 어떤 u라도 0.25 이상을 지불한다. Latent θ를 추가로 알려주더라도 regime을 알려주지 않으면 같은 하한이 남는다. 전체 하한은 **부호 역전으로 정답이 바뀐 비율 × 0.25**다. 이는 이 인위적 two-regime mixture의 절대 loss 하한이며, primary 평균에서 관측한 오차 전체가 식별 불가능하다는 뜻도, 모든 물리 조건의 minimax bound라는 뜻도 아니다. 이 한계는 우리 보정식에만 생기는 현상이 아니다.

따라서 실패 후 모델의 parameter 수만 늘리는 것보다 정보가 어디에서 들어올 수 있는지 먼저 묻는 편이 좋다. 다음 비교에서는 같은 네 번의 budget 중 하나만 query에 가까운 높이·속도로 옮긴 **query-matched probe**를 배정하고, 그 probe의 response가 두 regime에서 실제로 다른지 실측해야 한다. 차이가 없다면 prediction을 과감하게 이동시키는 대신 base model로 복귀하거나 두 regime에 대한 predictive mixture를 내는 전략이 더 타당하다.

이 panel에서는 역전 시 정답 변화율이 13.5%여서, 위의 균등 혼합 예에서 하한은 0.03384 Brier다.

![Direction transfer and full GPU latency](v35.png)

## 방향이 실제로 없는 surface에서의 검증

앞의 query-only stress와 달리 이 control은 calibration과 deployment 모두에 isotropic law를 사용했고, 별도 16개 seed block에서 처음부터 다시 실행했다. 물리적으로 참인 표면 방향이 없으므로 reader의 방향 accuracy는 정의하지 않았다.

| motion | 원래 Brier | 보정 Brier | Δ | Simultaneous 95% interval | 평균 ∣z∣ |
|---|---|---|---|---|---|
| 한 방향 | 0.10608 | 0.11217 | +0.00609 | [+0.00448, +0.00770] | 0.59886 |
| 양방향 교대 | 0.10568 | 0.11174 | +0.00605 | [+0.00342, +0.00868] | 0.61036 |

두 schedule 평균은 +0.00607 [+0.00449, +0.00786]였다. |z|는 reader가 중립값에서 얼마나 멀어졌는지 나타낸다. 두 방향 중 어느 것인지 구분하도록 학습한 reader에는 “표면에 방향이 없음”이라는 별도 상태가 없다. 가능한 두 부호를 잘 구분하는 것과, 방향이 없는 표면을 알아보는 것은 다르다.

탐색적으로 같은 query-only stress의 amplitude 비율을 0부터 150%까지 2.5%p씩 늘려 평균 Δ의 부호가 바뀐 구간도 확인했다. 한 방향 60~62.5%, 교대 62.5~65%였다. 기준은 각 표면 원래 amplitude의 백분율이다. 이 값은 네 synthetic 조합을 평균한 격자상의 교차이며 deployment gate나 실제 마찰 계수 threshold로 학습·검증한 것이 아니다. [수치와 전체 곡선](threshold35.json)

## v35에서 방향 정보 없이도 같은 개선을 얻을 수 있는가?

## Source에서 고정한 두 가지 대조군

V34의 fold 0에서는 estimated direction의 β를 **ridged_coarse**의 8개 block × 2개 orientation × 2개 schedule에서 공동으로 학습했다. 같은 32개 metrics archive에서 fixed 4의 같은 action을 선택했다. 각 stratum에서 8개 환경 × 128개 deployment worlds × 32개 sensor draws의 예측을 사용했다. 이 수는 1,048,576개 score이며, 같은 label과 prefix를 공유하는 반복을 포함한다. 두 source-only 보정은 target 자료 없이 CPU에서 학습·고정했다. [원래 fit](../bidirectional_probe_v34/anchored34.py) · [source-only fit 및 32개 입력 해시](source_cal35_frozen.json)

원 physics probability를 p, logit을 η라고 할 때 constant offset은 sigmoid(η + 0.01965140895), temperature scaling은 sigmoid(0.55849826698 η)다. T는 1.79051585140이다. 두 방법 모두 p를 [10⁻¹², 1−10⁻¹²]로 제한해 logit을 정의하고, binary log loss에 v34와 같은 0.005의 quadratic ridge를 더했다. Ridge의 기준은 identity에 해당하는 offset 0 및 inverse temperature 1이다. 원본 p가 경계일 때도 이 작은 clipping을 적용했다. Target family, schedule, sign에 관계없이 같은 계수를 사용했다. [학습 코드](fit_source_cal35.py) · [고정 계수와 입력 해시](source_cal35_frozen.json)

## 독립 target에서의 paired Brier

원래 fixed 4 action 및 관측은 다섯 방법에서 같다. T + signed는 먼저 temperature를 적용한 logit에 같은 v34 βz를 더해 sigmoid를 취한다. 음수는 각 contrast에서 왼쪽 방법의 Brier가 낮음을 뜻한다. [결과 확인 전 고정한 비교](SOURCE_CAL_PLAN_KO.md)

CI는 object × surface 내에서 block을 첫 probe 부호로 층화해 12,000회 재표집한 paired max-standardized interval이다. Mirrored orientation·두 schedule·동일 환경의 32개 sensor draws는 block 안에 함께 남겼다. Directional 표의 pooled 및 cell별 모든 30개 contrast에 대해 동시에 95%를 적용했다. Isotropic 표에서는 독립적으로 18개 contrast를 보정했다. 각 CI는 고정된 family의 block 변동에 관한 추정이다. [평가 코드](score_source_cal35.py) · [수치 원본](source_cal35.json)

| Object × surface | Schedule | Original | Offset | Temperature | Signed | T + signed | Signed − offset [95%] | Signed − temperature [95%] | (T + signed) − T [95%] |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pooled | same | 0.11885 | 0.11885 | 0.11808 | 0.11284 | 0.11393 | -0.00601 [-0.00711, -0.00491] | -0.00524 [-0.00631, -0.00416] | -0.00414 [-0.00547, -0.00281] |
| pooled | alternating | 0.11835 | 0.11835 | 0.11766 | 0.11270 | 0.11410 | -0.00565 [-0.00669, -0.00462] | -0.00497 [-0.00593, -0.00401] | -0.00357 [-0.00486, -0.00228] |
| sinusoid_atan | same | 0.12123 | 0.12103 | 0.12164 | 0.11764 | 0.12063 | -0.00339 [-0.00597, -0.00081] | -0.00400 [-0.00642, -0.00159] | -0.00102 [-0.00415, +0.00212] |
| sinusoid_atan | alternating | 0.12176 | 0.12156 | 0.12198 | 0.11757 | 0.12020 | -0.00398 [-0.00499, -0.00298] | -0.00441 [-0.00559, -0.00322] | -0.00178 [-0.00321, -0.00034] |
| sinusoid_algebraic | same | 0.12195 | 0.12194 | 0.12052 | 0.11458 | 0.11452 | -0.00735 [-0.00976, -0.00494] | -0.00594 [-0.00881, -0.00307] | -0.00600 [-0.00893, -0.00307] |
| sinusoid_algebraic | alternating | 0.12213 | 0.12211 | 0.12059 | 0.11551 | 0.11578 | -0.00661 [-0.00821, -0.00501] | -0.00508 [-0.00551, -0.00465] | -0.00481 [-0.00671, -0.00291] |
| quartic_atan | same | 0.11858 | 0.11864 | 0.11687 | 0.11052 | 0.11040 | -0.00813 [-0.01001, -0.00624] | -0.00636 [-0.00766, -0.00505] | -0.00647 [-0.00832, -0.00462] |
| quartic_atan | alternating | 0.11694 | 0.11702 | 0.11572 | 0.11041 | 0.11137 | -0.00661 [-0.00947, -0.00375] | -0.00531 [-0.00834, -0.00227] | -0.00435 [-0.00795, -0.00076] |
| quartic_algebraic | same | 0.11363 | 0.11378 | 0.11326 | 0.10861 | 0.11018 | -0.00517 [-0.00688, -0.00346] | -0.00465 [-0.00623, -0.00308] | -0.00308 [-0.00553, -0.00063] |
| quartic_algebraic | alternating | 0.11256 | 0.11270 | 0.11237 | 0.10729 | 0.10904 | -0.00541 [-0.00771, -0.00311] | -0.00507 [-0.00703, -0.00311] | -0.00333 [-0.00611, -0.00054] |

### Isotropic control

| Object × surface | Schedule | Original | Offset | Temperature | Signed | T + signed | Signed − offset [95%] | Signed − temperature [95%] | (T + signed) − T [95%] |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pooled | same | 0.10608 | 0.10613 | 0.10919 | 0.11217 | 0.11725 | +0.00604 [+0.00413, +0.00796] | +0.00298 [+0.00109, +0.00487] | +0.00806 [+0.00579, +0.01033] |
| pooled | alternating | 0.10568 | 0.10573 | 0.10892 | 0.11174 | 0.11695 | +0.00601 [+0.00287, +0.00914] | +0.00282 [-0.00022, +0.00585] | +0.00803 [+0.00387, +0.01220] |
| sinusoid_isotropic | same | 0.10662 | 0.10635 | 0.11010 | 0.11258 | 0.11829 | +0.00623 [+0.00402, +0.00844] | +0.00248 [+0.00015, +0.00481] | +0.00819 [+0.00548, +0.01091] |
| sinusoid_isotropic | alternating | 0.10633 | 0.10606 | 0.10992 | 0.11255 | 0.11839 | +0.00649 [+0.00153, +0.01145] | +0.00263 [-0.00219, +0.00746] | +0.00848 [+0.00159, +0.01536] |
| quartic_isotropic | same | 0.10554 | 0.10590 | 0.10827 | 0.11176 | 0.11621 | +0.00586 [+0.00271, +0.00900] | +0.00348 [+0.00048, +0.00649] | +0.00793 [+0.00428, +0.01159] |
| quartic_isotropic | alternating | 0.10504 | 0.10539 | 0.10792 | 0.11092 | 0.11551 | +0.00553 [+0.00161, +0.00944] | +0.00300 [-0.00075, +0.00675] | +0.00759 [+0.00278, +0.01240] |

## 해석

- same에서 signed correction과 constant offset의 pooled Brier 차이는 -0.00601이고 simultaneous 95% CI는 [-0.00711, -0.00491]다. signed correction의 이득을 지지한다.
- same에서 signed correction과 temperature scaling의 pooled Brier 차이는 -0.00524이고 simultaneous 95% CI는 [-0.00631, -0.00416]다. signed correction의 이득을 지지한다.
- alternating에서 signed correction과 constant offset의 pooled Brier 차이는 -0.00565이고 simultaneous 95% CI는 [-0.00669, -0.00462]다. signed correction의 이득을 지지한다.
- alternating에서 signed correction과 temperature scaling의 pooled Brier 차이는 -0.00497이고 simultaneous 95% CI는 [-0.00593, -0.00401]다. signed correction의 이득을 지지한다.
- Temperature를 먼저 적용한 뒤 같은 βz를 더하면 same에서 Brier 차이가 -0.00414 [-0.00547, -0.00281]다. 추가 이득을 지지한다.
- Temperature를 먼저 적용한 뒤 같은 βz를 더하면 alternating에서 Brier 차이가 -0.00357 [-0.00486, -0.00228]다. 추가 이득을 지지한다.
- 개별 directional family × schedule 8개 중 두 score-only 기준을 모두 앞선 것은 8개다. Temperature 후 βz의 추가 이득은 7개에서 simultaneous CI 상한이 0보다 작았다.
- Isotropic control에서 temperature 후 βz를 더한 pooled 차이는 same에서 +0.00806 [+0.00579, +0.01033]였다. 방향이 없는 target은 correction이 손해를 일으킬 수 있는 실험적 반례다.
- Isotropic control에서 temperature 후 βz를 더한 pooled 차이는 alternating에서 +0.00803 [+0.00387, +0.01220]였다. 방향이 없는 target은 correction이 손해를 일으킬 수 있는 실험적 반례다.

다섯 방법은 같은 sensor action을 쓰므로 이 결과가 비교하는 양은 관측 뒤 확률의 품질이다. Score-only baseline에 이겨도 추정한 sign이 단일 물리적 원인임을 식별하지는 못한다. Surface의 다른 속성과 reader 출력이 함께 달라질 수 있다. 원 score의 전반적인 보정과 환경 간 판별력이 각각 얼마나 변했는지는 독립 reliability·resolution 분석과 within-environment sign intervention으로 평가할 수 있다. 이 부록의 두 대조군에는 추가 GPU timing을 측정하지 않았다. [Brier decomposition](https://proceedings.neurips.cc/paper_files/paper/2019/file/8558cb408c1d76621371888657d2eb1d-Paper.pdf) · [physics residual의 선행 검증](https://journals.sagepub.com/doi/full/10.1177/0278364920954896)

## 논문 기여 가능성: 무엇을 주장할 수 있나

가장 설득력 있는 결과는 단순한 보정식 자체가 아니다. 이미 지불한 네 번의 접촉에서 읽은 정보를, 나중에 다른 물체가 stick할 확률에 어느 조건에서 옮길 수 있는지 prospective transfer와 실패 대조로 함께 검증했다는 점이다. Shape와 surface response를 바꾸고도 같은 reader와 계수를 썼고, 원래 decoder와 동일한 actions 및 실제 service boundary를 사용했다. Query에서 정보의 의미를 바꾸면 이득이 어떻게 사라지는지도 같은 예측으로 검증할 수 있다.

일반적인 확률 보정과의 구분이 기여의 핵심이다. 한 방향: frozen signed 0.11284, temperature 0.11808, temperature 뒤 같은 signed correction 0.11393. 마지막 두 방법의 차이는 -0.00414 [-0.00547, -0.00281]. 양방향 교대: frozen signed 0.11270, temperature 0.11766, temperature 뒤 같은 signed correction 0.11410. 마지막 두 방법의 차이는 -0.00357 [-0.00486, -0.00228]. 여기서 temperature는 target 결과를 보기 전에 v34에서만 fit했다. 원래 physics에 이긴 결과와 이미 보정된 강한 baseline 위에 남은 결과를 구분해서 보고해야 한다. 합성 모델의 결과는 secondary이고 그 모델의 전체 GPU p90을 별도로 측정하지 않았다.

**현재 증거의 한계.** 낮은 Brier는 held-out event probability가 더 정확했다는 증거다. 그것만으로 reliability 자체의 개선, 실제 friction mechanism의 식별, 더 나은 observation selection을 보인 것은 아니다. Acquisition과 budget은 고정했고 base decoder에는 실제 population coupling을 주었다. 하나의 simulator, 두 contour, 두 directional response와 고정된 query 방향을 사용했다. 마찰의 부호가 query의 접선 운동과 결합하는 방식은 그대로다. 실물에서의 성능 및 로봇 전체 cycle time도 측정하지 않았다.

**제가 논문을 계속 쓴다면:** 지금 결과를 재현·반증이 가능한 핵심 simulation figure로 쓰겠다. 독립적인 robotics 방법 논문의 중심 주장으로 삼기에는 실물 transfer와 강한 물리 baseline이 아직 비어 있다. 다음 실험에서는 (1) 같은 네 번의 force·velocity로 directional friction의 연속크기와 isotropic 가능성을 함께 추정하는 physics decoder를 만든다. (2) probe와 다른 높이·속도에서 새 물체의 stick/slip을 관측하고 object ID와 surface session을 동시에 hold out한다. (3) 원래 네 번의 budget 안에서 query-matched probe 한 번을 쓰는 방법, frozen correction, temperature, 연속 physics model을 비교한다. 실험 전 event 판정과 Brier margin, paired session 수, sensor/reset 포함 latency를 고정한다. 물리 baseline 위에 남는 이득 또는 query-matched probe만이 해소하는 실패가 독립 자료에서도 재현될 때 알고리즘 기여 주장을 강화할 수 있다. 기존 공개 pulling 표에는 이 평가에 필요한 짝지어진 impact outcome이 없다.

## 관련 문헌과 후속 연구 질문

v35의 질문은 부호를 맞히는 데서 한 걸음 더 나간다. 이미 지불한 네 번의 probe에서 구한 friction direction의 확률을 원래 physics decoder에 더하면, 새로운 물체와 표면에서 *다음 접촉의 결과*를 더 정확히 예측할 수 있는가? 아래 일곱 원문은 각각 물리적 부호, physics model의 오차 보정, probability calibration, distribution shift를 다룬다. v34의 보정은 기존 결과를 본 뒤 선택됐다. 따라서 v35에서 증거를 추가하는 부분은 새 target의 결과를 보기 전에 reader·계수·판정 기준을 고정했다는 데 있다. [v34에서 이 가설이 나온 과정](../bidirectional_probe_v34/ANCHORED_RESULTS_KO.md) · [v35 사전 검증 계획](AUDIT_PLAN_KO.md)

## 1. 선행연구에서 실제로 측정한 것

| 원문·연구 질문 | 방법·표본 | 주요 결과와 저자의 결론 | v35 해석에 적용할 때의 한계 |
|---|---|---|---|
| Yu, Bauza, Fazeli & Rodriguez (IROS 2016). 같은 표면에서 friction을 상수로 취급할 수 있는가? | 실제 산업용 로봇으로 stainless-steel 물체 11개, 지지면 4종의 planar pushing을 기록했다. 위치, 방향, 속도, 가속도 등을 바꾸고 force와 pose를 측정했다. 표면별로 100회 반복한 별도 scan도 있다. | 같은 재료에서도 측정 위치에 따라 friction coefficient가 달랐다. 반복 scan에서 감소한 정도도 재료마다 달랐다. 접촉 모델의 단일 상수 가정이 성립하는 범위를 직접 시험할 수 있다. | 실제 데이터의 장점은 force와 pose의 동시 측정이다. 논문에서 재료의 정답을 v35의 이산적인 plus/minus variable로 제공하지는 않는다. [원문 §§IV–V](https://arxiv.org/pdf/1604.04038) |
| Tramsen et al. (J. R. Soc. Interface, 2018). 표면 돌기의 기울기만으로 어느 방향의 friction이 큰지 정할 수 있는가? | 96개 sample–substrate 조합에서 sawtooth를 따라 또는 반대로 당겼다. 상대 stiffness, aspect ratio, roughness를 바꿨다. | 같은 기하학적 방향을 두고도 조합에 따라 friction anisotropy의 부호가 반전됐다. 저자들은 접촉 면적과 mechanical interlocking의 상대적 역할로 해석했다. | 직접 측정한 양은 양방향 pulling force로 구한 coefficient다. v35의 impact 후 binary event와의 관계, 새 물체에서 공통으로 쓸 logit 계수는 별도로 시험해야 한다. [원문 Figs. 2–5](https://findresearcher.sdu.dk/ws/files/151776983/rsif.2017.0629.pdf) |
| Kloss, Schaal & Bohg (IJRR, 2022). 물체 운동을 예측할 때 analytical model의 어느 부분을 학습으로 보완해야 하는가? | Yu et al.의 실제 push에 depth·RGB와 접촉 정보를 추가했다. Pure neural, analytical, hybrid, hybrid + learned error correction을 비교했다. Unseen push configuration 실험에는 100,000개 초과 training examples와 2,500개 조건이 있다. | 충분한 in-domain data에서는 pure neural model이 강했다. Unseen push configuration에서는 hybrid가 pure neural보다 정확했다. Table 3에서 error correction은 rotation error를 35.3%에서 34.5%로 줄였고 translation error는 두 방법 모두 15.6%였다. Ground-truth state를 쓰는 physics 기준은 각각 32.8%, 14.6%였다. | Physics에 learned residual을 붙이는 아이디어에는 직접적인 선례가 있다. 예측 대상은 pose 변화이고, ground-truth state와 inferred state의 입력 조건도 다르다. [원문 §§5.3–5.5, Table 3](https://journals.sagepub.com/doi/full/10.1177/0278364920954896) |
| Guo, Pleiss, Sun & Weinberger (ICML 2017). 이미 학습된 classifier의 confidence를 몇 개 계수로 보정할 수 있는가? | Vision·text datasets에서 held-out validation labels로 temperature 등을 추정했다. Table 1은 CIFAR-10/100, ImageNet, SVHN, 20 News, Reuters, SST의 두 task를 포함한다. | 다수 조합에서 temperature scaling이 Expected Calibration Error(ECE)를 줄였다. 양의 공통 temperature는 한 입력 안의 class 순서를 보존한다. | ECE는 confidence와 accuracy의 차이를 bin별로 측정한다. 환경별 signed cue를 사용한 binary event probability와 측정량이 다르다. [원문 §4, Table 1](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf) |
| Kull, Silva Filho & Flach (AISTATS 2017). 비대칭적인 score 분포에서는 어떤 calibration map이 유리한가? | UCI 41개 dataset을 binary task로 만들고 10회 반복한 5-fold cross-validation과 내부 calibration split에서 Brier score와 log loss를 비교했다. | Identity를 포함하는 beta calibration은 Naive Bayes와 한 AdaBoost variant에서 logistic calibration보다 유리했다. 다른 AdaBoost variant에서는 두 방법 차이가 작았다. Score 분포와 원 model에 따라 효과가 달라졌다. | Input은 원 score다. v35는 원 score에 별도의 환경 관측을 결합하므로 전체 Brier 개선에는 조건부 예측 정보의 추가도 섞인다. [원문 §5, Figs. 4–5](https://proceedings.mlr.press/v54/kull17a/kull17a.pdf) |
| Ovadia et al. (NeurIPS 2019). Source에서 잘 보정한 확률이 dataset shift 후에도 유효한가? | CIFAR-10·ImageNet에 16개 corruption을 각각 5개 강도로 적용했다. Text와 ad-click 실험도 수행하고 Brier score, log loss, ECE 등을 비교했다. | Image shift가 커지면서 accuracy와 calibration이 악화했다. Temperature scaling이 source에서 보인 이점은 강한 shift에서 약해졌고, ensemble은 여러 조건에서 상대적으로 견고했다. | Corruption과 contact law 교체는 서로 다른 shift다. 정해진 조합 전반에서 확인한 현상과 특정 물체에서의 손실 방향을 구분해야 한다. [원문 §§3–5, Fig. 2](https://proceedings.neurips.cc/paper_files/paper/2019/file/8558cb408c1d76621371888657d2eb1d-Paper.pdf) |
| Park, Bastani, Weimer & Lee (AISTATS 2020). Target labels 없이 어떤 shift를 보정할 수 있는가? | Labeled source와 unlabeled target으로 feature alignment·importance weighting·temperature를 결합했다. MNIST·USPS·SVHN과 Office31의 Amazon·DSLR·Webcam 간 transfer를 10회 반복 평가했다. | 일부 transfer에서 source-only 보정보다 ECE를 줄였다. 예컨대 Amazon → Webcam에서 temperature의 26.69%가 결합법에서 13.49%였다. 순위는 transfer별로 달랐다. | 이론은 covariate shift에서 조건부 label law가 유지되고 충분한 support가 겹친다는 전제를 둔다. 새 contact law가 같은 관측에서도 event 분포를 바꾸면 별도의 검증이 필요하다. [원문 §§2–6, Table 1](https://proceedings.mlr.press/v108/park20b/park20b.pdf) |

## 2. 연구 흐름과 knowledge map

| 출발점 | 다음 단계로 이어지는 근거 | 증거의 강도와 남은 질문 |
|---|---|---|
| Surface mechanics → directional response | Tramsen et al.은 반대 방향의 force를 같은 조합에서 측정했다. Yu et al.은 같은 재료라도 사용 이력과 위치가 force에 영향을 줄 수 있음을 보였다. | **직접적인 물리적 근거**가 있다. 하나의 고정 sign이 얼마나 오래 유지되는지는 표면과 측정 방식에 따라 따로 확인해야 한다. [Tramsen et al.](https://findresearcher.sdu.dk/ws/files/151776983/rsif.2017.0629.pdf) · [Yu et al.](https://arxiv.org/pdf/1604.04038) |
| Physics model → residual prediction | Kloss et al.은 구조를 넣은 뒤 learned correction으로 운동 예측의 일부 오차를 줄였다. Yu et al.의 dataset을 재사용했으므로 두 논문은 실물 표본을 공유한다. | **조건부 실험 근거**가 있다. 도움이 된 운동 성분과 input 조건이 특정돼 있어, 방향을 읽을 수 있다는 사실만으로 임의의 query 개선을 예측하기 어렵다. [Kloss et al.](https://journals.sagepub.com/doi/full/10.1177/0278364920954896) · [Yu et al.](https://arxiv.org/pdf/1604.04038) |
| Held-out score → probability calibration → transfer | Guo et al.과 Kull et al.은 단순한 map도 유용할 수 있는 조건을 제시했다. Ovadia et al.과 Park et al.은 평가 분포와 사용 가능한 target 정보가 바뀌면 성능이 달라짐을 보였다. | **설계 원칙을 뒷받침하는 근거**다. 같은 frozen reader와 계수가 새 contact law에서 통하는지는 v35의 고유한 empirical question이다. [Guo et al.](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf) · [Kull et al.](https://proceedings.mlr.press/v54/kull17a/kull17a.pdf) · [Ovadia et al.](https://proceedings.neurips.cc/paper_files/paper/2019/file/8558cb408c1d76621371888657d2eb1d-Paper.pdf) · [Park et al.](https://proceedings.mlr.press/v108/park20b/park20b.pdf) |

**왜 연구 사이에 차이가 생기는가?** Guo et al.은 같은 benchmark 안의 held-out examples로 보정과 시험을 분리했고, Ovadia et al.은 test corruption의 강도를 바꿨다. Park et al.은 deployment inputs로 target distribution을 추정할 수 있었다. V35에서는 새 환경의 독립 calibration probe 네 번이 유일한 추가 정보다. Target 전체를 미리 보는 calibration baseline과 비교한다면 target input 수와 처리 시간부터 맞춰야 한다. [Guo et al., §4](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf) · [Ovadia et al., §4](https://proceedings.neurips.cc/paper_files/paper/2019/file/8558cb408c1d76621371888657d2eb1d-Paper.pdf) · [Park et al., §§2·6](https://proceedings.mlr.press/v108/park20b/park20b.pdf)

## 3. 무엇을 논문 기여로 삼을 수 있는가?

**입증할 수 있는 가설은 좁고 측정 가능하다.** 네 번의 관측과 같은 sensor action을 고정하고, v34의 signed correction을 그대로 적용했을 때 새 synthetic object–surface 조합에서 다음 outcome의 probability error가 줄어드는지를 시험한다. v34에서 여러 후보를 보고 고른 사실을 공개하고, v35에서는 어느 fold와 계수를 배포할지 결과를 보기 전에 정한 것이 중요하다. Physics residual과 scalar calibration에는 선례가 있으므로 기여 후보는 *어느 조건에서 이미 지불한 환경 정보가 frozen physics decoder의 예측을 개선하는지* 보여 주는 paired transfer experiment다. [v34 보조 실험](../bidirectional_probe_v34/ANCHORED_RESULTS_KO.md) · [v35 계획](AUDIT_PLAN_KO.md) · [Kloss et al.](https://journals.sagepub.com/doi/full/10.1177/0278364920954896) · [Guo et al.](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf)

**왜 부호를 잘 읽어도 Brier가 악화할 수 있을까?** 원 확률을 p, corrected probability를 q, 실제 event를 y라고 하면, 한 관측의 Brier gain은 정확히 2(q−p)(y−p)−(q−p)²이다. 첫 항은 offset이 실제 원 예측 오차와 얼마나 맞물리는지, 둘째 항은 확률을 움직인 크기의 비용이다. 원 decoder가 새 환경에서 이미 정확하거나, query와 마찰 방향의 관계가 달라지거나, β가 너무 크면 손해가 날 수 있다. 반대로 pooled Brier가 줄어도 그 원인이 calibration 개선인지, 서로 다른 환경을 더 잘 구분한 결과인지 가르려면 reliability와 resolution을 따로 살펴야 한다. 이 식은 paired loss의 대수적 전개이며, 접촉의 인과 기전을 식별하는 식은 아니다. [Ovadia et al., Brier decomposition](https://proceedings.neurips.cc/paper_files/paper/2019/file/8558cb408c1d76621371888657d2eb1d-Paper.pdf)

**심사 시 가장 큰 공백은 외적 타당성이다.** 새로운 기하·마찰 함수에서의 실험은 해당 simulator 계열 안의 transfer를 시험한다. 실제 표면의 wear, 연속적인 orientation, heterogeneous roughness는 source와 target의 관계를 바꿀 수 있다. 공개 push 자료로 확장하려면 같은 물체·표면의 calibration과 deployment를 서로 다른 실제 push로 나누고, 일치하는 binary endpoint와 부호 측정법을 먼저 정의해야 한다. 원문의 coefficient difference나 pose error 자체를 v35의 Brier 결과에 합칠 수 없다. GPU p90은 해당 장치의 서비스 비용을 나타내며 네 번의 실제 접촉과 reset이 걸리는 시간은 별도 measurement다. [Tramsen et al.](https://findresearcher.sdu.dk/ws/files/151776983/rsif.2017.0629.pdf) · [Yu et al.](https://arxiv.org/pdf/1604.04038) · [실제 push 자료의 데이터 명세](https://mcube.mit.edu/push-dataset/index.html)

## 4. 이 근거에서 이어지는 연구 질문 10개

각 행은 앞으로 반증할 수 있는 가설이다. 앞의 일곱 원문에서 해당 결과까지 확인했다는 뜻은 아니다.

| 연구 질문 | 공백과 중요성 | 변수·검증 설계 |
|---|---|---|
| 1. 실제 새 물체와 새 재료를 동시에 바꿔도 이득이 남는가? | 구조적 방향은 실물에서 반전되고, hybrid model의 외삽 성능은 조건에 따라 달랐다. [Tramsen et al.](https://findresearcher.sdu.dk/ws/files/151776983/rsif.2017.0629.pdf) · [Kloss et al.](https://journals.sagepub.com/doi/full/10.1177/0278364920954896) | 별도 object ID와 surface session을 함께 hold out한다. Four-probe prefix·decoder·event 정의를 미리 고정하고 paired Brier를 비교한다. |
| 2. Directional difference가 0에 가까우면 잘못된 offset의 비용은 얼마인가? | 실제 재료에서는 anisotropy의 크기도 달라진다. 약한 효과를 강한 부호로 판정하면 baseline이 정확한 경우 손해가 날 수 있다. [Tramsen et al.](https://findresearcher.sdu.dk/ws/files/151776983/rsif.2017.0629.pdf) | Independently measured coefficient difference를 연속 축으로 쓰고 isotropic control을 포함한다. Cue magnitude별 gain과 penalty를 계산한다. |
| 3. 평균 보정으로도 같은 이득을 얻는가? | Source-only calibration으로도 확률 오차가 줄어드는 사례가 있어 signed cue의 고유 기여를 분리해야 한다. [Guo et al.](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf) · [Kull et al.](https://proceedings.mlr.press/v54/kull17a/kull17a.pdf) | 같은 source labels에서 constant intercept와 temperature를 학습해 고정한다. 같은 target에서 signed cue·within-stratum shuffled cue와 비교한다. |
| 4. 성능이 흔들릴 때 reader와 β 중 무엇이 원인인가? | Physics model 보정의 효과가 예측하는 성분별로 달랐다. 한 개의 전체 loss로는 실패 지점을 찾기 어렵다. [Kloss et al.](https://journals.sagepub.com/doi/full/10.1177/0278364920954896) | 실측 양방향 force로 얻은 sign을 사용한 diagnostic과 deployable estimated sign의 gain을 비교한다. Sign readout Brier와 event Brier를 구분한다. |
| 5. 보정은 calibration을 개선하는가, resolution을 개선하는가? | Brier score에는 두 효과가 함께 들어간다. [Ovadia et al.](https://proceedings.neurips.cc/paper_files/paper/2019/file/8558cb408c1d76621371888657d2eb1d-Paper.pdf) | 독립 calibration split에서 bias를 점검한 reliability curve와 decomposition을 추정하고 cluster uncertainty를 제시한다. |
| 6. 같은 surface에서도 wear가 진행되면 네 번의 probe를 얼마나 오래 재사용할 수 있는가? | 실제 반복 scan의 friction 변화량이 재료마다 달랐다. [Yu et al.](https://arxiv.org/pdf/1604.04038) | Calibration–deployment 시간 간격과 누적 push 수를 바꾸고 surface session을 단위로 resample한다. |
| 7. Friction 방향과 query의 운동 방향이 회전하면 signed cue가 여전히 도움이 되는가? | 선행 실물 연구는 특정 방향의 pulling으로 상대 friction을 정의한다. [Tramsen et al.](https://findresearcher.sdu.dk/ws/files/151776983/rsif.2017.0629.pdf) | 표면과 query의 상대 각도를 독립적으로 회전하고, 기존 scalar와 source에서 미리 학습한 angular projection을 비교한다. |
| 8. Source에서 고른 correction을 어느 contact regime까지 신뢰할 수 있는가? | Sticking과 sliding 비중이 다른 test set은 물체 운동 예측 난이도가 달랐다. [Kloss et al., §5.4](https://journals.sagepub.com/doi/full/10.1177/0278364920954896) | Slip/stick boundary까지의 거리를 사전에 정의하고 동일한 event prevalence에서 regime별 Brier gain을 비교한다. |
| 9. Target covariates만으로 correction 신뢰 구간을 좁힐 수 있는가? | Covariate shift에 대한 importance weighting은 target support 및 conditional law에 의존한다. [Park et al.](https://proceedings.mlr.press/v108/park20b/park20b.pdf) | Observation marginal만 바꾸는 조건과 같은 observation에서 event law가 바뀌는 조건을 교차한다. Target label 없이 학습한 방법과 frozen 방법을 동일 예산에서 시험한다. |
| 10. 같은 Brier gain이 실제 로봇의 deadline 안에서 유지되는가? | 강한 uncertainty model은 계산비가 클 수 있고, 실제 push는 force·pose 획득이 필요하다. [Ovadia et al.](https://proceedings.neurips.cc/paper_files/paper/2019/file/8558cb408c1d76621371888657d2eb1d-Paper.pdf) · [Yu et al.](https://arxiv.org/pdf/1604.04038) | Batch size와 환경당 deployment 수를 고정해 reader 포함 GPU p90을 재고 센서·reset 포함 robot wall time을 별도로 기록한다. |

이 검토는 prototype과 직접 연결되는 원문을 목적 표집했다. 분야 전체의 최초성, 실물 generalization, adaptive observation selection의 개선은 각각 더 넓은 선행연구 검색, 독립 실물 실험, action을 변경하는 별도 intervention을 필요로 한다.

## 재현 및 검증 기록

Primary와 control을 합친 raw archive 160개와 source 파일 20개의 hash를 별도로 검산했다. Evidence·평가 seed 분리, paid IDs, deployment quota, 충돌 식과 label도 확인했다. [실험 코드](run35.py) · [데이터 생성](target35.py) · [실행 방법](REPRODUCE_KO.md) · [독립 통계 재계산](audit35.py) · [physics 및 hash 검산](physics_audit35.json) · [같은 v34 fold의 역사적 점수](source_replay35.json) · [run 완료 기록](results/completion.json). 서버 경로: `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/frozen_direction_v35`.
