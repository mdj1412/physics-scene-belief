# v60 · 동기화 보정의 효과와 유리한 정책을 선택할 수 있는 조건

**장치의 fault 빈도와 deployment 환경을 독립시킨 주 검증에서는 표식 보정 후에도 reset + camera의 평균 held-out Brier가 두 대조보다 낮지 않았다.** Corrected reset에서 corrected sham을 뺀 차이는 +0.014×10⁻⁶ [−0.026, +0.051]이었고, 상시 v54 gate를 뺀 차이는 +0.366×10⁻⁶ [+0.278, +0.458]이었다. 양수는 손해다. Trigger 지연과 fault–surface 관계를 독립적으로 조작했고 marker-corrected policy를 공통으로 고정한 결과다. [사전 판정 기준](SCORE_PLAN_KO.md) · [전체 수치](score60.json)

## 어떤 결정을 채점했는가

A에선 fixed 3R+I, B에선 frozen v54 gate를 선택한다는 v56의 사전 지정에 따라 A/B를 선택하는 policy와 원인 P/C를 분리했다. P에서는 실제 physical mismatch 상태가 session 내 두 물체에서 지속된다. C에서는 물체별 실제 mismatch가 독립이나 두 local sensor가 공용 upstream의 잘못된 신호를 받을 수 있다. Cause와 deployment 환경은 별도다. Source에서 A에 높은 C fault 빈도 29.58%, B에는 낮은 4.08%가 연결되어 있었다. V60에서는 이 배정을 유지하거나 독립시키거나 반대로 바꾼다. 독립 배정은 각 실제 환경에 두 빈도를 반반 갖는 동일 예산 pilot을 표면별로 가중했다. 물체 순서와 sensor reset은 무작위이며 모든 electronic reset에는 같은 ID의 sham이 대응한다. 세 trigger 조건은 impact clip의 0%, 20%, 100%가 2 ms 늦는 설계다. Primary에는 세 지연을 1/3씩, P/C와 A/B를 각각 1/2씩 넣었다. 현장 빈도의 추정값은 없다. [설계](SCORE_PLAN_KO.md)

V59에서 고정한 source likelihood는 같은 surface에서 얻은 4개 pair의 paid electronic·camera mismatch와 valid 여부를 입력으로 쓴다. Timestamp 정답·실제 fault assignment와 deployment label은 입력에 없다. 원인 추정은 A/B를 marginalize한 P/C posterior의 MAP다. 정책 판별에는 두 가지 수치를 구했다. 첫째, loss-weighted policy가 실제로 선택한 행동과 사전 지정 A/B 행동의 balanced accuracy를 평가한다. 둘째로 A/B posterior에서 비용을 무시한 MAP의 accuracy를 비교한다. 전자의 odds cutoff는 v56 손익에서 정한 2.016이다. [Frozen likelihood](../sensor_reset_v59/route59.py) · [채점 코드](score60.py)

새로운 독립 단위는 24개 catalog다. Material 48개와 두 rate law에서 12,288개 held-out 물체를 평가했다. Fine solver에서 fixed로 바뀔 때 평균 Brier 이득은 A 39.287×10⁻⁶, B는 오히려 89.005×10⁻⁶ 손해였다. B의 사전 대응은 48개 표면 모두에서 맞았고 A는 47개에서만 맞았다. Catalog 5006의 material 0에서 A여도 gate가 5.193×10⁻⁶ 유리했다. 이 예외를 제외하지 않았다. 따라서 표에는 출판 전 검증용으로 실제 surface별 저손실 행동을 정답으로 다시 삼아, 그 정답 class 내에서 계산한 balanced accuracy도 넣었다. 해당 진단에서만 target label을 썼다. 모든 deployable 행동은 동일하다. 알려진 A/B만 쓰는 privileged 대조는 target regime을 볼 수 있으므로 hazard를 재배정해도 고른 행동을 유지하도록 검산했다. [표면별 채점](score60_cells.npz) · [결과와 감사](score60.json)

## Primary 결과 · deployment 환경과 C fault 발생률이 독립

| 정책 | 원인 P/C BA | 실제 선택과 A/B 대응 BA | 실제 저손실 행동 BA | 평균 Brier ↓ | Session p90 ↓ |
|---|---:|---:|---:|---:|---:|
| 보정 reset + camera, loss-weighted | 55.832% | 50.008% | 50.019% | 0.159348796 | 0.188999951 |
| 보정 sham + camera, loss-weighted | 55.103% | 50.021% | 50.035% | 0.159348781 | 0.188999951 |
| 미보정 reset + camera, loss-weighted | 52.748% | 50.365% | 50.373% | 0.159351587 | 0.189000517 |
| 미보정 sham + camera, loss-weighted | 52.374% | 50.369% | 50.382% | 0.159350858 | 0.189000398 |
| 보정 reset + camera, MAP | 55.832% | 51.379% | 51.400% | 0.159355682 | 0.189002678 |
| 보정 sham + camera, MAP | 55.103% | 51.430% | 51.428% | 0.159356582 | 0.189002678 |
| 항상 v54 gate | — | 50.000% | 50.000% | **0.159348430** | 0.188999400 |
| 항상 fixed 3R+I | — | 50.000% | 50.000% | 0.159373289 | 0.189012408 |

Mean held-out Brier는 선택된 frozen decoder의 binary stick loss를 임의 deformation 방향에 대해 적분했고, session p90은 128개 물체에 대한 realized session 평균 분포의 90번째 백분위수다. Delay별 p90을 평균하지 않고 pooled empirical CDF를 반전했다. 10,000회 paired catalog bootstrap을 사용했다. Primary에서 각 endpoint별 사전 지정한 두 대조에 대해 동시 95% CI를 계산했다. Cause는 진단을 내지 않는 gate와 비교하지 않았다. [코드와 수치](score60.py) · [원자료](score60_cells.npz)

| 보정 reset에서 뺀 대조 | 원인 판별 변화 (pp) | 선택한 A/B 행동 변화 (pp) | 평균 Brier 변화 (×10⁻⁶) | Session p90 변화 (×10⁻⁶) |
|---|---:|---:|---:|---:|
| 보정 sham | +0.729 [+0.652, +0.804] | −0.013 [−0.027, +0.001] | +0.014 [−0.026, +0.051] | 0.000 [−1.043, 0.000] |
| 항상 v54 gate | — | +0.008 [−0.002, +0.018] | +0.366 [+0.278, +0.458] | +0.551 [−0.358, +1.848] |

Corrected reset은 A에서 1.531%, B에서 1.515%를 fixed로 바꿨다. Sham에서는 각각 1.499%, 1.457%였다. 두 정책이 서로 다른 행동을 산 pilot의 가중 비중은 0.901%였다. Reset으로 gate에서 fixed로 바뀐 비율은 0.473%, 반대 전환은 0.428%였다. Reset은 24개 catalog 중 8개에서 sham보다 평균이 낮았고, 상시 gate보다 낮은 catalog는 0개였다. 잘못 B에서 fixed로 움직이는 비용이 A에서 올바르게 전환하는 이득보다 높은 조건에서, 원인을 더 잘 맞히는 것만으로 reset 비용을 회수하지 못했다. Ultra solver와 두 independent deployment seed의 실제 binary loss 모두에서 gate 대비 평균 손해였고, reset−sham CI는 매번 0을 포함했다. [Paired 전환 비용](switch60.json) · [Solver·seed 검산](score60.json)

## Trigger 지연과 surface fault의 독립 조작

다음 표는 새 물체들의 결과에서 **보정된 reset loss-weighted policy**를 비교한 것이다. Marker와 readout buffer가 지원하는 0/2 ms에서 보정 성공·실패를 포함한 최종 paid code가 세 지연 조건 모두에서 같았다. 이 세 cell을 독립된 세 차례 재현으로 세지 않는다. Mean 차이 CI는 3개 fault relation × 3개 delay × 2개 대조의 18개 family 전체에 Bonferroni 보정했다. [모든 cell 수치](score60.json)

| C의 surface fault 배정 · 각 delay 0/20/100% | 원인 BA | A/B 행동 BA | MAP A/B BA | 평균 Brier | Session p90 | Reset − sham (×10⁻⁶) | Reset − gate (×10⁻⁶) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Source와 일치 | 55.832% | 51.069% | 55.145% | 0.159348118 | 0.188997075 | −0.145 [−0.226, −0.085] | −0.312 [−0.395, −0.223] |
| 서로 독립 | 55.832% | 50.008% | 51.379% | 0.159348796 | 0.188999951 | +0.014 [−0.039, +0.062] | +0.366 [+0.251, +0.485] |
| Source와 반대 | 55.832% | 48.947% | 47.614% | 0.159349474 | 0.189000398 | +0.174 [+0.122, +0.234] | +1.044 [+0.859, +1.241] |

Matched와 reversed가 같은 P/C 판별률을 보이는 이유는 두 A/B target의 비중이 같고 C branch의 같은 acquisition을 재배정했기 때문이다. 반대로 전자·camera discrepancy가 A일 확률을 나타내느냐 B일 확률을 나타내느냐는 달라진다. High와 low hazard를 모든 target에 각각 동일하게 지정한 추가 대조에서는 원인 BA는 59.64%와 52.02%, gate 대비 평균 손해는 각각 +0.624, +0.107×10⁻⁶이었다. 따라서 fault가 자주 일어나 원인 자체를 더 잘 찾더라도 그 원인과 실제 손익의 연결이 없으면 정책 효용은 낮다. [추가 대조](score60.json)

| 지연된 impact 비중 · 독립 hazard | 미보정 reset 원인 BA | 보정 reset 원인 BA | 미보정 reset 평균 Brier | 보정 reset 평균 Brier | 보정 − 미보정 (×10⁻⁶) |
|---|---:|---:|---:|---:|---:|
| 0% | 56.219% | 55.832% | 0.159348815 | 0.159348796 | −0.020 |
| 20% | 53.534% | 55.832% | 0.159349538 | 0.159348796 | −0.742 |
| 100% | 48.490% | 55.832% | 0.159356407 | 0.159348796 | −7.611 |

0%에서 검출되지 않은 marker는 유료 clip을 결측으로 바꾸므로 미보정과 보정이 동일할 필요가 없다. 지연이 있을 때 표식으로 window를 옮기면 같은 reset 정책 자체의 loss가 낮아진다. 이 개선과 상시 v54 gate를 이기는 것은 다른 비교다. V59의 matched hazard를 전제로 동결한 posterior를 바꿀 때의 결과는 별도의 sensitivity에 해당한다. Primary에서는 새 threshold를 맞추거나 표면별로 가장 좋은 조합을 고르지 않았다. [검증 기준](SCORE_PLAN_KO.md) · [입력 hash와 채점 확인](score60_cells.npz)
