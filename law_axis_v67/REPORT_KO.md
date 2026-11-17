# v67 · 접촉 법칙을 바꿔도 남는 형상 정보와, 조명을 바꿨을 때 달라지는 구매 결정

**새 두 contact law에서 v66의 손익 반전과 좌표 복원 이득이 모두 재현됐다.** 같은 관측으로 예측할 때 축을 부분적으로 약화하면 Wendland에서는 Brier loss가 낮아졌고 Dispersive에서는 높아졌다. 별도의 학습 비교에서는 삭제한 축의 부호 있는 좌표를 되돌려 주면 두 law 모두에서 예측이 개선됐다. 조명 실험에서는 photodiode delay를 보정하고도 네 번째 구매 추천 전환율이 두 조명 **0.30660%**, 같은 조명 **0.16624%**였다. 두 실험은 서로 다른 모집단과 endpoint를 사용한다. [형상 상세 결과](EXPERIMENT_KO.md) · [조명 상세 결과](../paired_light_sham_v67b/REPORT_KO.md) · [문헌·연구 설계 검토](LITERATURE_KO.md)

![새 두 contact law의 paired Brier 차이와 같은 조명 및 두 조명의 최종 구매 전환율](v67_summary.png)

왼쪽은 여섯 primary contrast에 대한 동시 95% interval이다. 오른쪽은 각 arm의 pointwise 95% interval이다. 두 조명 차이의 판정에는 표 아래의 **paired** interval을 썼다. 그림의 음수는 왼쪽에 적힌 첫 번째 방법의 Brier loss가 낮다는 뜻이다. [벡터 그림](v67_summary.pdf)

## 1. 같은 형상과 관측에서 마지막 접촉의 법칙만 바꾸기

v66의 12개 held-out catalog에서 같은 object, 변형, 표면, 영상, 측정 위치, paid query/probe와 noise를 재사용했다. 마지막 impact에서 접선 속도가 커질수록 저항이 증가하는 rate-strengthening law와 감소하는 rate-weakening law를 대칭으로 만들었다. 기준 속도, 변화 폭, 방향 비대칭 항, 압력 분포 및 solver는 같았다. 따라서 이 비교는 같은 이력 뒤에 가한 마지막 impact의 개입 효과를 추정한다. 처음부터 다른 law에서 calibration을 다시 받는 시스템에는 별도 검증이 필요하다. [사전 설계와 식](PLAN_KO.md)

독립 source seed 4개마다 fold와 ripple에서 형상 쌍을 다시 수집하고 두 높이의 projection axis를 재추정했다. 축 사이의 변동을 비교하기 위해 head initialization은 고정했다. 두 개의 실험을 구별했다. 첫 실험은 고정한 위치 규칙 아래 likelihood에서 축 가중치를 줄인다. 두 번째는 같은 구조의 학습된 readout에 해당 좌표를 빼거나 되돌려 준다. 새 law의 label 생성 전 12개 forecast shard와 8개 readout을 봉인했다. 표의 차이는 loss가 낮을수록 유리하며, 독립 단위는 12개 catalog다. [예측 봉인](forecast_seal.json) · [전체 절대 Brier 및 seed별 표](EXPERIMENT_KO.md)

| 마지막 impact의 law | Wendland, 부분 억제 − 원래 likelihood | Dispersive, 부분 억제 − 원래 likelihood | 두 형상 평균, 좌표 복원 − 삭제 |
|---|---:|---:|---:|
| Rate-strengthening | −0.003496 [−0.003887, −0.003105] | +0.006720 [+0.006003, +0.007437] | −0.012047 [−0.013587, −0.010506] |
| Rate-weakening | −0.003764 [−0.004151, −0.003377] | +0.007372 [+0.006752, +0.007993] | −0.021369 [−0.022960, −0.019778] |

**축을 다시 추정해도 효과의 방향은 유지됐고 크기는 달라졌다.** 네 seed 모두, 두 law 모두에서 두 형상의 손익 부호가 반대였다. 각 law에서 seed를 평균한 손익의 부호도 12개 catalog 모두 일치했다. 원래 axis를 고정하면 Dispersive의 부분 억제 손해는 +0.018467/+0.020059였고, 새 axis 평균에서는 +0.006720/+0.007372였다. 새 source 수집은 hollow와 bulge를 별도 noise로 읽었으므로 이 차이에는 수집 프로토콜의 영향도 있다. 같은 noise로 재수집하는 대조에서는 네 seed의 축이 모두 과거 축과 수치 오차 내 일치했다. [과거 axis 대조](reference_axis.json) · [source axis](axes.json)

**왜 label이 바뀐 횟수만으로 손실 방향을 알 수 없는가?** 마지막 law만 바꾸자 stick/slip label은 Wendland에서 9.27%, Dispersive에서 10.94% 바뀌었다. 이미 같은 확률을 냈던 두 방법은 그 물체의 label이 바뀌어도 서로의 Brier 차이가 변하지 않는다. 실제로 두 law 사이의 Brier 차이 변화는 각 label 변화에 두 예측 확률의 차이를 곱해 합친 값으로 계산됐고, 직접 scoring과의 최대 오차는 9.35×10⁻¹⁷이었다. 별도로 hollow와 bulge를 바꾸었을 때 label이 뒤집힌 비율은 두 law 모두 Wendland에서 더 높았다. 그럼에도 억제 손해는 Dispersive에서 더 컸다. [법칙별 사고 실험과 label 표](EXPERIMENT_KO.md)

두 가지 검산을 더 했다. 기존 rational/gaussian-rate 자료에서도 관측 law와 label law를 교차해 v66의 Brier를 최대 5.56×10⁻¹⁷ 이내로 재현했다. 새 law에서는 12개 catalog 전체의 공간·시간 quadrature를 각각 두 배로 높여 1,769,472개 label을 다시 계산했다. 가장 큰 family·law 평균 Brier contrast 변화는 0.00000358 이하였다. [기존 law 교차 검산](legacy_cross67.json) · [전체 fine quadrature 검산](fine_audit67.json)

## 2. 같은 조명에서도 구매가 바뀌는 경로

Calibration bank마다 같은 **8회 impact**를 배정했다. 첫 세 reference가 같은 후속 물체에서 네 번째 측정으로 reference(R)와 impact(I) 중 다른 것을 추천했을 때 한 건의 전환으로 셌다. 상위 router가 달라져도 두 정책이 모두 I를 고르면 전환은 0건이다. 총 24개 catalog, 8개 board, 12,288개 bank, 98,304개 impact slot을 사용했다. 각 slot의 장면은 공유하고 두 판독에는 독립 추가 pixel noise를 넣었다. A/A와 B/B를 sham으로, A/B와 B/A를 두 조명으로 균등 비교했다. 실제 actuator receipt는 없다. [계측 및 endpoint 사전 설계](../paired_light_sham_v67b/PLAN_KO.md)

| 둘째 판독에 가한 개입 | 같은 조명 sham | 두 조명 |
|---|---:|---:|
| 추가 오류 없음, 독립 read noise | 0.02511% | 0.20567% |
| 일관된 marker decoy만 추가 | 0.16491% | 0.30527% |
| Photodiode delay만 추가 | 0.23927% | 0.33181% |
| 두 오류 동시, 보정 전 | 0.31941% | 0.39052% |
| Photodiode delay만 추가, 보정 후 | 0.02644% | 0.20700% |
| 두 오류 동시, 보정 후 | **0.16624%** | **0.30660%** |

별도 commissioning pulse로 board별 차이를 추정했다. 두 오류가 함께 있을 때 보정 후 두 조명−sham은 **+0.14036 percentage points** [동시 95% CI +0.02267, +0.31278]였다. Delay만 있을 때 sham의 보정 전−후 차이는 **+0.21283 percentage points** [+0.08263, +0.38043]였다. 동일 픽셀·wave를 그대로 복사한 sham은 0건이었다. 기존 v64 재생에서는 두 조명 0.186299%를 회수했다. 표의 독립 read noise가 추가된 cell과는 acquisition 조건이 다르다. [전체 단계별 수치](../paired_light_sham_v67b/results67.json)

**어느 오류가 구매까지 남았는가?** 같은 bank에서 delay와 marker를 하나씩 껐다. 보정 후 sham에서 두 오류를 켰을 때 발생한 전환 0.16624% 중 **0.02385 percentage points**는 추가 오류 없이도 일어났다. 새로 생긴 전환 가운데 **0.14107 percentage points**는 marker 단독으로, **0.00133 percentage points**는 잔여 delay 단독으로 재현됐다. 두 오류를 모두 켜야 발생한 새 전환은 이 표본에서 0건이었다. 보정 전에는 둘을 모두 켜야 발생한 sham 전환도 0.06332 percentage points 있었다. 보정 후 두 조명에서 추가 오류 전부터 있었고 끝까지 남은 전환은 0.18430 percentage points였다. 이 분해는 같은 난수와 후속 prefix에서의 simulated 개입을 짝지은 것이다. 현장 오류의 유병률을 추정하지 않는다. [개별 bank 교차 검산](../paired_light_sham_v67b/attribution67.json) · [판독→구매 경로](../paired_light_sham_v67b/paths67.json)

희귀한 전환이므로 개별 prefix를 독립 표본으로 세지 않았다. Catalog와 board를 다시 뽑아 paired interval을 계산했다. Catalog를 하나씩 제외한 보정 후 두 조명의 초과분은 +0.12994~+0.14780 percentage points, board를 하나씩 제외하면 +0.10107~+0.16338이었다. 별도로 v64에 원본 픽셀이 보관된 작은 표본(48 bank)에 한쪽 계측기의 보정 잔차만 주었을 때, +0.25 ms와 +0.50 ms에서 sham과 두 조명이 각각 똑같이 0.24963%, 1.00001% 전환했다. 잔여 지연에 민감한 bank가 있었고, 이 작은 표본의 조명 간 차이는 0이었다. 12,288개 bank 전체의 조명 차이는 더 큰 표본의 다른 read-noise 조건에서 얻었다. [잔여 지연 검산](../paired_light_sham_v67b/residual_sensitivity67.json)

## 3. 논문에서 지지할 수 있는 주장

**주 결과는 contact representation에 두는 편이 적절하다.** 같은 두 형상에 두 반대 속도 의존 법칙을 적용하고 source axis를 네 번 재추정해도, likelihood 억제의 손익 방향이 반대로 나타나고 부호 있는 좌표를 복원하는 이득이 남았다. 두 새 law는 결합된 수식의 대칭 변형이고 기존 solver와 catalog를 공유한다. 서로 다른 물리 계열이나 실물 표면에서 일반화하는 주장은 아직 검증해야 한다. 다음 확인 실험의 판정 단위는 새로운 독립 contact solver 또는 실물 surface와, 처음부터 해당 law에서 얻은 calibration 이력이다. 실제 평면 충돌에서 post-impact velocity를 비교한 Fazeli 등의 실험과도 직접 비교 가능한 endpoint를 추가할 수 있다. [Fazeli 등의 평면 충돌 실험](https://proceedings.mlr.press/v78/fazeli17a.html) · [문헌 비교와 10개 연구 질문](LITERATURE_KO.md)

**조명 결과는 실제 pilot의 대조군과 계측 우선순위를 정한다.** 첫 단계는 같은 표면·같은 impact에서 photodiode 지연을 별도 계측하고 동일 lamp sham과 독립 marker identity를 함께 남기는 것이다. 이번 simulation에서는 timing 보정만으로 생기는 개선과 그 뒤에도 남는 marker 경로를 분해할 수 있었다. 두 조명에서 추천을 더 자주 바꾼다는 사실만으로 더 좋은 측정을 골랐거나 held-out Brier가 낮아졌다고 판단할 수 없다. 실물에서는 두 추천을 먼저 봉인한 뒤 실제 구매·후속 결과까지 수집해야 한다. [실물 pilot의 수집 규약](../paired_light_pilot_v65/REPORT_KO.md) · [LED·photodiode 파형을 직접 비교한 연구](https://www.mdpi.com/1424-8220/24/13/4103)
