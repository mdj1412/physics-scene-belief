# v62 · 낮은 신뢰도의 영상 부호를 보류하면 어떤 후속 행동이 달라지는가?

**선택적 보류를 적용한 reset은 기존 v61 reset보다 평균 held-out Brier가 0.0164 × 10⁻⁶ 낮았다.** 기존 sham과의 차이는 불확실했고 항상 v54 gate를 사용한 정책보다 0.0336 × 10⁻⁶ 높았다. 새 표면에서 물리 원인과 장치 오류를 판별하는 정확도도 낮아졌다. 이번 실험에서 보류한 42,253개 추가 판독은 모두 정확했다. 기존 readout을 덜 쓰자 유료 pilot 이후 행동이 상시 gate로 돌아간 것이 주된 경로였다. [보류 기여 분해](attribution62.json) · [정책 수치](score62.json)

## 같은 움직임, 같은 비용에서의 주 평가

LED pulse 0.20/0.80 ms와 shutter exposure 0.25/0.80 ms의 네 조합을 균등 가중했다. 같은 표면의 8회 impact마다 0/2 ms LED delay와 0/2 ms trigger delay의 네 조합을 각각 두 번 배치했다. 8회 reference도 같이 읽었다. C에서는 sensor fault 발생률이 실제 deployment A/B와 독립이도록 표준화했다. P에서는 실제 두 물체 변형 사이에 물리적 association이 있다. V59의 source likelihood와 odds threshold 2.016을 고정해 후속 행동인 fixed 3R+I와 v54 gate 중 하나를 선택했다. Target label은 성능 평가에서만 사용했다. [사전 판정 기준](SCORE_PLAN_KO.md) · [Likelihood](../sensor_reset_v59/route59.py)

위험의 추론 단위는 24개 catalog와 8개 board다. 48개 표면 각각에 모든 board를 교차하고 board·조건별 64개 pilot을 평가했다. 12,288개 held-out 물체의 expected binary stick Brier를 계산하고, 다른 seed로 구성한 realized binary session(각 128개 물체) 분포의 weighted empirical p90을 구했다. 3개 대조의 각 endpoint별 동시 구간에는 board와 catalog를 독립 복원추출하는 6,000회 paired product bootstrap과 Bonferroni 보정을 적용했다. Cause를 출력하는 대조는 두 개다. [채점 코드](score62.py) · [v61 통계 재현 검산](score_validation62.log)

| 정책 | Active 쌍 판독률 | 원인 판별률 | 실제 저손실 정책 판별률 | 평균 held-out Brier ↓ | Session loss p90 ↓ |
|---|---:|---:|---:|---:|---:|
| Selective reset + camera | 2.372% | 50.283% | 50.029% | 0.157078333 | 0.188423350 |
| V61 reset + camera | 5.047% | 50.582% | 50.028% | 0.157078349 | 0.188423350 |
| V61 sham + camera | 5.047% | 50.490% | 50.026% | 0.157078345 | 0.188423350 |
| 항상 v54 gate | — | — | 50.000% | **0.157078299** | 0.188423350 |
| Selective sham + camera | 2.372% | 50.232% | 50.027% | 0.157078328 | 0.188423350 |
| Nominal reset + camera | 53.318% | 50.650% | 50.414% | 0.157080315 | 0.188426465 |

원인과 저손실 정책의 판별률은 balanced accuracy다. 48개 표면 모두에서 A에서는 fixed, B에서는 gate가 실제 더 낮은 평균 loss를 냈다. 상시 gate는 pilot 해석을 생략하므로 판독률과 원인 판별률이 정의되지 않는다. 전체 유료 쌍 대비 판독률은 selective 1.190%, 기존 2.534%였다. Active 쌍 판독률이 더 높은 nominal은 판독한 쌍 중 38.05%의 mismatch를 틀렸고 downstream 평균 Brier가 더 높았다. [판독률의 분모와 원자료](vision62.json) · [Board별 손실](score62.json)

| Selective reset − 대조 | 원인 판별률 차이 (pp) | 정책 판별률 차이 (pp) | 평균 Brier 차이 (×10⁻⁶) | Session p90 차이 (×10⁻⁶) |
|---|---:|---:|---:|---:|
| 기존 reset | −0.299 [−0.514, −0.110] | +0.001 [−0.006, +0.006] | **−0.0164 [−0.0430, −0.0023]** | 0.000 [−0.805, 0.000] |
| 기존 sham | −0.207 [−0.356, −0.070] | +0.003 [−0.016, +0.023] | −0.0124 [−0.0445, +0.0088] | 0.000 [−0.492, +0.447] |
| 항상 v54 gate | — | +0.029 [−0.003, +0.063] | **+0.0336 [+0.0125, +0.0596]** | 0.000 [−0.104, +1.118] |

Session p90은 연속 근사가 아닌 실제 replay loss의 discrete quantile이므로 같은 값에 걸릴 수 있다. 두 번째 seed에서도 네 primary 정책의 p90은 모두 0.188042715였다. 평균 Brier의 selective − v61 reset 차이는 ultra solver와 두 binary replay 모두 음수였고 동시 95% CI가 0을 넘지 않았다. Gate 대비 손해의 점추정치는 세 검산 모두 양수였으나 두 번째 binary replay에서는 CI에 0이 포함됐다. 8개 board 각각에서 selective는 v61 reset보다 낮고 상시 gate보다 높은 평균을 냈다. 기존 sham과의 순위는 board 5개와 3개로 나뉘었다. [민감도와 leave-one-board-out](score62.json)

## 어느 조합에서 보류했는가

| LED pulse / exposure (ms) | Active 쌍 판독률, v61 → selective | 원인 판별률, v61 → selective | Selective − v61 reset Brier (×10⁻⁶) | Selective − gate Brier (×10⁻⁶) |
|---|---:|---:|---:|---:|
| 0.20 / 0.25 | 4.614% → 2.532% | 50.539% → 50.313% | −0.0129 | +0.0321 |
| 0.20 / 0.80 | 0.423% → 0.001% | 50.044% → 50.000% | −0.0015 | +0.0208 |
| 0.80 / 0.25 | 7.518% → 4.268% | 50.881% → 50.507% | −0.0196 | +0.0475 |
| 0.80 / 0.80 | 7.635% → 2.685% | 50.863% → 50.310% | −0.0317 | +0.0339 |

0.20 ms pulse와 0.80 ms exposure에서는 reader가 이미 희소한 flash와 변형 신호를 더 거절해, 거의 전자센서만으로 정책을 골랐다. 0.80/0.80 ms에서는 기존 reset 대비 차이가 가장 컸고, 3 fault 관계 × 4 optics cell × 3 대조를 동시 보정한 95% CI도 [−0.107, −0.001] × 10⁻⁶였다. 나머지 세 cell에서 같은 comparator에 대한 36-way CI는 0을 포함했다. 어느 cell에서도 gate 대비 selective의 평균 점추정치가 더 낮지 않았다. [모든 조합과 세 대조의 구간](score62.json)

## 정확한 evidence를 제거해서 생긴 변화

A에서 fixed가 gate보다 평균 42.50 × 10⁻⁶ 유리했고, B에서는 잘못된 fixed가 평균 80.51 × 10⁻⁶ 손해였다. 기존 reset이 fixed를 택한 비율은 A 0.378%, B 0.322%였고 selective는 각각 0.289%, 0.232%였다. 같은 pilot에서 비교하면 **0.0964%가 기존 fixed에서 gate로**, **0.0071%가 기존 gate에서 fixed로** 바뀌었다. 각 전환의 전체 평균 Brier 기여는 −0.01698와 +0.00055 × 10⁻⁶였다. 나머지 약 99.90%의 pilot에서는 두 방법의 행동이 같았다. [방향별 행동 변화와 손익](score62.json)

사전 지정한 privileged audit에서 selective가 제거한 기존 optical mismatch 쌍을 simulator truth로 정판독과 오판독으로 나눴다. 오판독만 제거, 정판독만 제거, 둘 다 제거한 세 counterfactual을 original과 같은 원래 전자센서와 likelihood에서 다시 평가했다. 두 집합을 어느 순서로 제거하는지 평균해 기여를 구했다. 원래 판독 79,701개에서 오판독이 0건이었고 selective가 남긴 37,448개 역시 오판독이 없었다. **정판독 42,253개를 제거한 결과가 관측된 평균 Brier 변화 전부**였다. 오판독 제거의 현재 효과는 0이며, 향후 잘못된 부호를 받을 때의 선택적 보류 효과는 이 결과만으로 알 수 없다. [검증 계획](ATTRIBUTION_PLAN_KO.md) · [Attribution 구현](attribution62.py) · [Identity 검산과 수치](attribution62.json)

| 장치 fault와 최적 행동의 연결 | Selective 원인 판별률 | Selective 정책 판별률 | Selective − v61 reset (×10⁻⁶) | Selective − gate (×10⁻⁶) |
|---|---:|---:|---:|---:|
| Source와 일치 | 50.283% | 50.107% | +0.0181 [−0.0015, +0.0386] | −0.0165 [−0.0593, +0.0298] |
| 독립 | 50.283% | 50.029% | −0.0164 [−0.0430, −0.0023] | +0.0336 [+0.0125, +0.0596] |
| Source와 반대 | 50.283% | 49.950% | −0.0509 [−0.1014, −0.0148] | +0.0837 [+0.0452, +0.1319] |

동일한 계측과 표면 물리를 둔 채 장치 빈도가 어느 deployment와 연결되는지만 바꿨다. 독립과 반전에서는 정확하지만 정책 선택에 불리하게 작용하는 source 단서의 사용을 줄이면서 기존 reset보다 loss가 낮아졌다. Source와 같을 때의 점추정치 방향은 반대였고 구간에는 0이 포함됐다. 손익의 절대 규모가 작고 실제 hardware board 또는 다른 simulator에서 재현되었다는 증거는 없다. 다음 판별 실험에는 충분한 빈도로 **기존 dual이 자신 있게 틀리는 관측**이 있어야 confidence가 오류를 선별하는 성능을 검증할 수 있다. [모집단별 추정값](score62.json) · [현상별 계측](vision62.json)
