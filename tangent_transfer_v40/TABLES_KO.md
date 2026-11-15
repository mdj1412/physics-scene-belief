## 사전 예측과 독립 target 결과

Δ는 **optimizer Brier − fixed Brier**다. 음수이면 optimizer의 확률이 실제 stick 여부에 더 가까웠다. 사전 Δ는 v39 자료의 replay로 계산했다. 새 law나 v40 물체의 정답을 사용하지 않았다. 조건별 구간은 12개 독립 catalog panel, 12개 조건에 대해 보정한 Bonferroni–t 동시 95% 구간이다.

| Tangent / contour / law | 사전 Δ | Fixed Brier | Optimizer Brier | 실측 Δ [동시 95%] | 판정 | Optimizer 우세 panel |
|---|---:|---:|---:|---:|---|---:|
| shelf / base / saturating | +0.000394 | 0.108100 | 0.108141 | +0.000041 [−0.000731, +0.000813] | 미확정 | 6/12 |
| shelf / base / crossover | +0.001200 | 0.118167 | 0.120063 | +0.001896 [+0.001248, +0.002543] | 예측 확인 | 0/12 |
| shelf / hollow / saturating | −0.000667 | 0.112403 | 0.111265 | −0.001138 [−0.002130, −0.000147] | 예측 확인 | 11/12 |
| shelf / hollow / crossover | +0.000207 | 0.120954 | 0.121824 | +0.000870 [+0.000085, +0.001654] | 예측 확인 | 1/12 |
| shelf / bulge / saturating | +0.001214 | 0.108352 | 0.109281 | +0.000929 [+0.000276, +0.001582] | 예측 확인 | 1/12 |
| shelf / bulge / crossover | +0.001939 | 0.119088 | 0.121691 | +0.002603 [+0.001942, +0.003263] | 예측 확인 | 0/12 |
| ridge / base / saturating | −0.000271 | 0.107173 | 0.106325 | −0.000848 [−0.001421, −0.000276] | 예측 확인 | 12/12 |
| ridge / base / crossover | +0.000468 | 0.116559 | 0.116652 | +0.000093 [−0.000449, +0.000635] | 미확정 | 6/12 |
| ridge / hollow / saturating | +0.000302 | 0.106589 | 0.106353 | −0.000236 [−0.000651, +0.000179] | 미확정 | 10/12 |
| ridge / hollow / crossover | +0.000963 | 0.117108 | 0.117697 | +0.000589 [+0.000089, +0.001089] | 예측 확인 | 0/12 |
| ridge / bulge / saturating | −0.001160 | 0.111419 | 0.109743 | −0.001676 [−0.002526, −0.000825] | 예측 확인 | 12/12 |
| ridge / bulge / crossover | −0.000362 | 0.119019 | 0.118398 | −0.000621 [−0.001352, +0.000110] | 미확정 | 9/12 |

음수·양수 어느 쪽도 동시 구간으로 확인할 수 없으면 미확정으로 기록했다. 각 panel 안에서 두 시작 방향과 모든 paired 환경을 먼저 평균했다. 개별 물체의 승패 적중률과 다른 단위다. [사전 예측](PREDICTIONS_KO.md) · [Panel별 Brier](stats40.json)

### 부호를 맞히는 것과 이득 크기를 맞히는 것

| Forecast에 쓰는 정보 | 12개 cell의 MAE | 부호 적중 | Spearman ρ | Full − 해당 forecast의 MAE 차이 [paired bootstrap 95%] |
|---|---:|---:|---:|---:|
| v39 전체 평균 | 0.001051 | 5/12 | 정의되지 않음 | −0.000570 [−0.000761, −0.000384] |
| Tangent family만 사용 | 0.000960 | 5/12 | 0.531 | −0.000479 [−0.000712, −0.000306] |
| Tangent와 law를 사용하고 contour 변화를 무시 | 0.000704 | 10/12 | 0.756 | −0.000223 [−0.000340, −0.000136] |
| Tangent, law, 관측 contour를 모두 replay | 0.000481 | 11/12 | 0.916 | 기준 |

사전에 Δ<0인 조건에서만 optimizer를 고르면 Brier는 0.113387이다. 항상 fixed와의 차이는 −0.000357 [−0.000490, −0.000224], 항상 optimizer와의 차이는 −0.000565 [−0.000717, −0.000414]였다. 두 구간은 각각 pointwise 95%다. 이 선택은 설계자가 아는 shape·law를 사용하므로 미지 표면의 online router 성능과 구분해야 한다. [Forecast 채점](forecast_eval40.json)

## 같은 tangent와 물체에서 off-contact contour만 바꾼 효과

표에서 Δ 변화는 (변형 contour의 optimizer−fixed) − (base의 optimizer−fixed)다. 음수일수록 해당 변형이 optimizer에 상대적으로 유리하다. 아래의 절대 Brier 변화는 각 정책 안에서 base와 비교한 값이다.

| 조건 | Δ 변화 [8개 동시 95%] | Fixed Brier 변화 | Optimizer Brier 변화 | Query를 받은 물체의 평균 절대 base probability 변화 |
|---|---:|---:|---:|---:|
| shelf / hollow / saturating | −0.001180 [−0.001520, −0.000839] | +0.004303 | +0.003124 | 0.03334 |
| shelf / bulge / saturating | +0.000888 [+0.000584, +0.001191] | +0.000252 | +0.001140 | 0.03410 |
| shelf / hollow / crossover | −0.001026 [−0.001373, −0.000679] | +0.002787 | +0.001761 | 0.03316 |
| shelf / bulge / crossover | +0.000707 [+0.000402, +0.001012] | +0.000921 | +0.001628 | 0.03403 |
| ridge / hollow / saturating | +0.000613 [+0.000385, +0.000840] | −0.000584 | +0.000028 | 0.03772 |
| ridge / bulge / saturating | −0.000827 [−0.001125, −0.000530] | +0.004246 | +0.003419 | 0.03420 |
| ridge / hollow / crossover | +0.000496 [+0.000270, +0.000721] | +0.000549 | +0.001045 | 0.03764 |
| ridge / bulge / crossover | −0.000714 [−0.001022, −0.000406] | +0.002460 | +0.001746 | 0.03401 |

형상을 바꾼 모든 짝에서 stick label·sensor 선택·signed cue·보정량이 같았다. Probe branch에서 base probability도 같았다. 두 경로를 교환해 loss를 다시 계산하면 contour 효과 전부가 query view를 통한 base probability 변화로 복원된다. 이 결과는 접촉 위치를 고정한 본 simulator의 설계와 실측을 함께 반영한다. [분해와 negative control](stats40.py) · [검산 수치](stats40.json)

## GPU p90

| 방법 | 1,024개 물체의 GPU p90 | Matched-panel bootstrap 95% 구간 |
|---|---:|---:|
| Fixed 32.5% | 1,432.1 ms | [1,427.0, 1,437.2] ms |
| Frozen v38 optimizer | 1,434.4 ms | [1,427.5, 1,437.9] ms |

비율은 1.00157 [0.99911, 1.00384]로, 상한 ≤1.02인 사전 조건을 충족했다. 서버 GPU 한 장에서 24개 timing cluster, 각 방법당 72회 측정했다. 한 번의 warm online 호출에 acquisition, 선택한 센서의 decoder, 보정 선택과 host 반환이 들어간다. 실제 probe·로봇 reset·bank 구축은 제외했다. [GPU 실행](run40.py) · [Timing 원자료와 산출 방식](stats40.json)
