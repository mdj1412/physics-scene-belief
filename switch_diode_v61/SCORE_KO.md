# v61 · 두 지연을 분리한 뒤 어떤 판단이 달라졌는가

## 무엇을 예측하고 어떻게 평가했는가

표면마다 4 sessions × 2개 물체에서 8회 impact·8회 reference를 관측한다. 두 전자센서의 mismatch가 반복되는 원인은 session 안에서 실제 변형이 지속되는 **physical persistence(P)** 또는 공용 upstream의 **common-mode error(C)**다. 원인과 다음 deployment의 A/B 환경은 별도로 생성한다. A에서는 대체로 네 번째 관측을 항상 impact로 사는 fixed 3R+I가, B에서는 세 reference로 네 번째 센서를 고르는 v54 gate가 유리하다. 각 pilot에서 초기 관측으로 이 둘 중 하나를 고른 뒤 동일한 frozen terminal decoder로 별도 물체의 stick 확률을 예측한다. [계획](SCORE_PLAN_KO.md) · [Frozen policy](../sensor_reset_v59/route59.py)

원인 판별률은 P/C posterior MAP의 balanced accuracy다. 정책 판별률은 각 표면·환경에서 실제로 Brier가 낮은 행동을 정답으로 삼아 양쪽 class에 균등 가중한 balanced accuracy다. A/B의 사전 지정 행동과의 agreement 및 비용을 무시한 A/B MAP도 구했다. Loss-weighted policy가 fixed를 살 odds cutoff 2.016은 source에서 동결했다. Target label을 쓰는 곳은 손익·판별률을 **평가하는 단계**다. Surface와 board 이름, 두 지연의 정답, 아직 사지 않은 센서의 값은 router에 제공하지 않는다. [채점 코드](score61.py)

각 catalog의 두 표면마다 8개 새 board를 모두 사용한다. Board마다 64개 Monte Carlo pilot을 배정해 같은 두 표면·board·물체·noise에서 reader와 reset을 대응시켰다. Board와 surface를 단순히 곱해 독립 표본 수를 늘리면 동일한 board에 의존하는 observation을 반복으로 센다. 6,000회 paired product bootstrap에서 24개 catalog와 8개 board를 각각 독립 복원추출한 뒤 두 가중치의 곱으로 각 method를 재평가했다. 평균 Brier와 session loss p90, 정책 판별은 각 endpoint에서 두 primary contrast에 대해 동시 95% CI를 사용했다. 원인 판별은 진단하지 않는 상시 gate를 빼고 sham과만 비교했다. P/C와 A/B에는 1/2, (LED, trigger) 0/2 ms의 4개 조합에는 1/4을 부여했다. C에서 high/low fault 빈도는 A/B 양쪽에 균등 배정한다. [사전 명세](SCORE_PLAN_KO.md) · [Raw CDF 및 이전 결과 재현 검산](score_validation61.log)

평균 held-out Brier에는 두 deformation 방향에서 별도로 시뮬레이션한 stick/slip과 그 발생 확률을 사용했다. Session loss p90은 별도 binary replay에서 먼저 128개 물체의 squared error를 session별로 평균한 뒤 policy가 만든 pooled 분포에서 계산했다. Bootstrap마다 discrete weighted quantile을 다시 계산하고, 주 추정치에서 바로 앞 outcome까지의 CDF와 해당 값까지의 CDF가 0.9를 사이에 두는지 확인했다. 이 수치는 8회 pilot 자체의 loss나 물체 한 개의 loss percentile과 다르다. [계산 코드](score61.py)

## 주 결과 · 새 surface에서 fault 빈도와 유리한 정책이 독립인 경우

| Reader와 행동 | P/C 원인 판별률 | 저손실 정책 판별률 | 평균 held-out Brier ↓ | Session loss p90 ↓ |
|---|---:|---:|---:|---:|
| Switch + photodiode, reset + camera | 54.722% | 50.016% | 0.158664300 | 0.189778805 |
| Switch + photodiode, sham + camera | 54.097% | 50.024% | 0.158664302 | 0.189778909 |
| 항상 v54 gate | — | 50.000% | **0.158664110** | 0.189778909 |
| LED only, reset + camera | 52.184% | 50.194% | 0.158665066 | 0.189778909 |
| LED only, sham + camera | 51.866% | 50.180% | 0.158664854 | 0.189778909 |
| Nominal window, reset + camera | 52.298% | 50.401% | 0.158666485 | 0.189779237 |
| Switch + photodiode, reset, 비용 비가중 MAP | 54.722% | 51.532% | 0.158668576 | 0.189779237 |
| 항상 fixed 3R+I | — | 50.000% | 0.158679726 | 0.189782768 |

어떤 물체에서도 gate가 원인을 추정할 필요는 없다. 원인 판별의 빈칸은 이에 해당한다. A에서 fixed, B에서 gate인 순위가 각각 48개 모든 held-out surface에서 유지되어 사전 A/B 대응과 실제 저손실 행동 BA는 이번 표본에서 같았다. 전체 결과는 reader에 따라 실제 구매한 impact가 얼마나 달랐는지도 함께 제공한다. [정책·원인·조건별 원자료](score61.json)

| Dual reset에서 뺀 대조 | 원인 판별률 차이, pp | 정책 판별률 차이, pp | 평균 Brier 차이, ×10⁻⁶ | Session loss p90 차이, ×10⁻⁶ |
|---|---:|---:|---:|---:|
| Dual sham | +0.625 [+0.412, +0.839] | −0.008 [−0.037, +0.018] | −0.002 [−0.066, +0.057] | −0.104 [−1.073, +0.298] |
| 항상 v54 gate | — | +0.016 [−0.001, +0.037] | **+0.190 [+0.105, +0.287]** | −0.104 [−1.267, +1.281] |

두 장치로 보정한 reset은 sham보다 원인 판별률이 높았다. 평균 Brier와 session loss p90에서 sham보다 유리하다는 증거는 없었고, 평균 Brier에서는 gate가 유리했다. Ultra solver와 두 independent binary replay에서도 reset − gate의 평균은 각각 +0.193, +0.241, +0.214 ×10⁻⁶였다. 세 검산 모두에서 해당 차이의 CI 하한은 0보다 컸고 reset − sham의 CI는 0을 포함했다. 두 번째 replay에서 세 정책의 session loss p90은 reset 0.190187037, sham 0.190187037, gate 0.190186933이었다. [Bootstrap와 민감도](score61.json)

A에서 fixed의 이득은 평균 46.626 ×10⁻⁶, B에서 fixed를 잘못 쓰면 손해가 77.858 ×10⁻⁶였다. Dual reset이 fixed를 산 비율은 A 1.293%, B 1.261%로 거의 같았다. Source에서 고정한 odds threshold를 비용 비가중 MAP로 바꾸면 저손실 정책을 더 자주 맞혔지만, 비싼 오판을 더 많이 허용해 Brier가 0.158668576으로 올랐다. 각 board를 개별로 볼 때 reset − gate의 평균 Brier 차이는 8개 모두 양수(+0.091~+0.264 ×10⁻⁶)였다. 하나씩 제외한 추정치도 +0.180~+0.204 ×10⁻⁶였다. 반면 reset − sham은 board별로 4개에서 음수, 4개에서 양수였다. [Board별·leave-one-board-out 집계](score61.json)

## LED 지연과 camera trigger 지연을 따로 바꾼 효과

| LED / trigger 지연 | Dual reset 원인 BA | LED-only reset 원인 BA | Nominal reset 원인 BA | Dual − LED-only reset 평균 Brier, ×10⁻⁶ |
|---|---:|---:|---:|---:|
| 0 / 0 ms | 54.603% | 55.005% | 56.149% | −0.019 |
| 0 / 2 ms | 54.840% | 55.253% | 48.447% | −0.020 |
| 2 / 0 ms | 54.603% | 50.000% | 56.149% | **+0.176** |
| 2 / 2 ms | 54.840% | 48.477% | 48.447% | **−3.198** |

네 cell을 모두 합친 dual − LED-only의 평균 Brier 차이는 reset에서 −0.765 ×10⁻⁶ [동시 95% CI −1.197, −0.400], sham에서 −0.552 ×10⁻⁶ [−0.873, −0.276]였다. 원인 BA는 각각 +2.538 pp [+2.046, +2.968], +2.231 pp [+1.789, +2.651] 높아졌다. 각 cell의 점추정치는 위 표에, 24개 대조를 보정한 cell별 reset − sham/gate 구간은 전체 결과에 있다. [전체 조합](score61.json)

**정확한 판독이 정책에 불리해진 cell도 있었다.** LED만 2 ms 지연된 조건에서 LED-only는 여섯 frame을 확보하지 못해 모든 camera mismatch 판독을 포기했다. 이때 LED-only reset은 50%의 원인 판별과 gate에 가까운 평균 Brier 0.158664119를 보였다. Dual reset은 원인 판별을 54.603%로 높이면서 평균 loss는 0.158664295로 올랐다. 양쪽이 함께 늦으면 LED-only가 판독한 133,011 active mismatch 중 132,966개를 잘못 읽었고, dual은 자기 reader가 판독한 190,229개에서 오류가 없어 큰 손해를 줄였다. 낮은 fault-independent Brier를 확보하는 데 필요한 조건에는 원인 파악 이후 그 정보로 어떤 행동을 선택하는지도 포함된다. [판독 자료](vision61.json) · [정책 결과](score61.json)

추가 factorial analysis에서 LED-only reset의 LED main effect는 +1.492 ×10⁻⁶, trigger main effect는 +1.699 ×10⁻⁶, 두 요인의 interaction은 **+3.372 ×10⁻⁶** [pointwise 95% CI +1.980, +4.947]였다. Main effect는 반대쪽 요인을 평균하고, interaction은 (2,2)−(2,0)−(0,2)+(0,0)으로 계산했다. Dual reset에서는 LED main effect와 interaction이 수치 오차 내 0이었고, trigger main effect는 +0.011 ×10⁻⁶ [+0.001, +0.027]였다. 같은 표면에서 반사실적 timing만 바꾼 현재의 두 시점·합성 광학 구간에 대한 결과다. [Factorial code](factorial61.py) · [Factorial 결과](factorial61.json)

## 원인 정보가 deployment에 유용해지는 조건

| C의 fault 빈도와 A/B의 관계 | Dual reset 원인 판별률 | Dual reset 정책 판별률 | Reset − sham 평균 Brier, ×10⁻⁶ | Reset − gate 평균 Brier, ×10⁻⁶ |
|---|---:|---:|---:|---:|
| Source 대응 유지 | 54.722% | 50.859% | −0.147 [−0.220, −0.084] | −0.329 [−0.433, −0.221] |
| 독립 | 54.722% | 50.016% | −0.002 [−0.066, +0.057] | +0.190 [+0.105, +0.287] |
| Source 대응 반전 | 54.722% | 49.173% | +0.144 [+0.063, +0.223] | +0.710 [+0.534, +0.883] |

각 행은 해당 population에서 timing 네 cell을 균등 평균했으며, 구간은 행별 두 대조에 보정했다. 원인 판별률이 같은 이유는 A/B를 균등 가중해 판독한 데이터의 marginal distribution이 같기 때문이다. 상대적으로 높은 장치 fault가 어느 deployment의 이득을 시사하는지만 바뀌었다. 이 설계는 기록 장치를 개선해도 환경에 따라 그 신호가 잘못된 정책 단서가 되는 반례를 제공한다. 실제 site의 연관 강도, board clock drift, 물리적 sensing 비용이나 임상·산업적 효과 크기는 이 합성 성능만으로 정할 수 없다. [표면·board별 재현 bundle](score61_cells.npz)
