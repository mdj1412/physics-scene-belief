# v69 실측값에 대한 solver 비교

1,023개 실제 golf bounce의 inbound state를 입력해 outbound state를 예측했다. Campaign A는 artificial turf, B는 natural turf다. A에서 학습하면 B의 693회 모두, B에서 학습하면 A의 330회 모두를 평가했다. **동일한 planar contact와 Coulomb friction에서 PyBullet과 v68 방식의 normal-compliant solver는 동일한 tangential velocity를 예측했다.** 두 방식은 실측값과 상당한 차이가 있었다. Tangential spring을 추가했을 때의 개선은 평가 표면과 event 정의에 따라 달랐다. [원 논문](https://link.springer.com/article/10.1007/s12283-023-00442-4) · [실측 CSV 및 변수 정의](https://static-content.springer.com/esm/art%3A10.1007%2Fs12283-023-00442-4/MediaObjects/12283_2023_442_MOESM2_ESM.zip)

## 실측 endpoint와 단위

CSV의 x/y velocity는 ball radii/s, 양의 spin은 anticlockwise다. Nominal ball radius 0.02135 m로 변환했다. Signed contact-point velocity는 u = vx + Rω다. Primary near-stick event는 lift-off에서 |u|≤0.50 m/s로 정하고 0.25, 1.00 m/s도 확인했다. 이는 구의 속도와 회전에서 계산한 운동학적 tolerance다. 바닥과 구가 눌린 동안 local stick/slip을 직접 측정한 값은 없다. Secondary sign-reversal event는 u_in × u_out < 0이다. 속도가 연속이면 이 두 시각 사이에 적어도 한 번 0을 지나간다. 측정된 표면들의 추가 rebound 중 holdout 693회에서 56회, 330회에서 7회가 primary event에 해당했다. 같은 두 holdout에서 sign reversal은 각각 539회와 215회였다. 각 row가 가리키는 video, 동일 ball serial, turf site별 contact patch를 연결하는 식별자는 원본 CSV에 없다. [측정 규약 및 실험 해석](https://arxiv.org/abs/2302.02758)

## 구현 및 parameter fitting

- 저자들이 제공한 6개 수치 열만 사용했다. Source에서 CSV 순서의 연속 24행을 한 block으로 만들고 0-based block index modulo 4가 1인 block을 uncertainty calibration에, 나머지를 fitting에 사용했다. A의 경우 240/90, B의 경우 525/168이다. 두 source subgroup을 나누는 경계는 특히 B에서 실험 장치의 기록된 session 경계로 확인되지 않았다. 전체 target campaign은 한 번에 holdout했다.
- 세 모델에 공통으로 source의 정상 속도에 대한 least squares로 restitution e를 정했다. Tangential fit에는 vx와 Rω의 squared error를 같은 가중치로 사용했다. Solid-sphere inertia I/(mR²)=0.4를 가정한다. 실제 golf ball의 층별 mass distribution을 측정하지 않았다. R, m을 engine unit으로 각각 1로 놓은 similarity transform 때문에 engine의 angular velocity는 원래 rim speed와 수치가 같다.
- PyBullet DIRECT는 회전하는 sphere와 고정 plane으로 실제 실행했다. 두 body damping, rolling friction, spinning friction, contact ERP와 contact slop을 0으로 두었다. Sliding 때의 friction impulse를 min(μ(1+e)|vn|, |u|/3.5)로 제한하는 독립 rigid impulse 계산과 비교해 전체 trial에서 차이는 7.2×10⁻¹⁵ m/s 이하였다.
- v68의 asymmetric normal spring과 Coulomb friction을 회전하는 구에 이식했다. Loading 때 spring stiffness k, unloading 때 e²k를 사용했다. 두 단계의 normal impulse는 각 80개 interval에서 oscillator의 force integral로 구했다. Tangential compliance가 없으면 flat-plane impact 동안 |u|는 impulse 누적량에 따라 단조 감소한다. 최종 normal impulse가 같아 두 solver가 일치한다. 이 결과를 curved patch 또는 v68의 multi-site contour 전체로 일반화할 수 없다.
- 세 번째 모델에는 Coulomb bound를 적용한 tangential spring과 return mapping을 추가했다. μ와 k_t/k_n을 source에서 함께 선택했다. Selected e, μ_rigid, μ_memory, k_t/k_n은 A에서 0.5488/0.25/0.425/0.0625이고 B에서 0.2693/0.5/0.5/0.25였다. 독립적으로 측정한 material constant는 없다. 관찰한 양은 pre/post velocity이므로 contact time, patch radius, force history는 이 fitting으로 식별되지 않는다.
- 확률 계산에는 parameter fitting에 쓰지 않은 source calibration rows의 signed error r=u_obs−u_pred를 사용했다. 각 test trial에 대해 u_pred+r 전체 중 각 event의 발생 비율을 Jeffreys smoothing (k+0.5)/(n+1)로 계산했다. 이는 campaign 간 residual law가 이동하면 영향을 받는다. Source prevalence baseline은 같은 calibration rows의 event 비율만 사용했다.

## Target 결과

| Source → target | 모델 | vx MAE / RMSE (m/s) | u MAE / RMSE (m/s) | Brier 0.25 / 0.50 / 1.00 | Sign reversal Brier |
|---|---|---:|---:|---:|---:|
| A → B | PyBullet | 6.803 / 8.184 | 5.617 / 8.137 | .03300 / .08276 / .23636 | .18217 |
| A → B | Normal-compliant | 6.803 / 8.184 | 5.617 / 8.137 | .03300 / .08276 / .23636 | .18217 |
| A → B | Tangential spring | 6.648 / 8.110 | 5.392 / 7.491 | .03236 / .08231 / .22182 | .25414 |
| A → B | Source prevalence | — | — | .03078 / .07469 / .21042 | .18323 |
| B → A | PyBullet | 4.100 / 4.518 | 3.704 / 4.495 | .01183 / .02832 / .05884 | .09556 |
| B → A | Normal-compliant | 4.100 / 4.518 | 3.704 / 4.495 | .01183 / .02832 / .05884 | .09556 |
| B → A | Tangential spring | 4.150 / 4.574 | 3.949 / 4.771 | .01549 / .04163 / .10387 | .12672 |
| B → A | Source prevalence | — | — | .01071 / .02493 / .05217 | .23764 |

양방향 모두 source prevalence가 세 solver의 primary Brier loss를 밑돌았다. Sign-reversal에서는 PyBullet과 normal-compliant가 source prevalence보다 낮았으며 B→A에서 차이가 컸다. A→B에서 tangential spring은 COM MAE를 0.155 m/s 낮췄지만 reversal Brier는 0.072 증가했다. B→A에서는 두 지표가 모두 악화됐다. 서로 다른 지표의 결론이 갈리므로 near-stick, reversal, in-contact stick의 정의를 구분해야 한다.

## Numerical audit 및 실행

Compliant solver와 scoring은 Slurm job 2744471의 node56, NVIDIA RTX PRO 6000 Blackwell Server Edition에서 float64 PyTorch 2.7.1+cu128로 실행했다. 시작 시 GPU memory 0 MiB, utilization 0%, compute PID 없음이 기록됐다. PyBullet DIRECT는 같은 allocation의 CPU에서 구동했다. CPU wall time과 GPU time의 속도 비교는 수행하지 않았다. PyBullet time step을 반으로 줄였을 때 모든 rebound 값이 같았다. Compliant normal-only solver와 독립 rigid solution은 3.6×10⁻¹⁴ m/s 내에서 같았다. Tangential spring을 각 phase 160-step으로 재실행했을 때 component별 최대 변화는 0.065 m/s, 평균 변화는 0.020 m/s 이하였다. Primary Brier는 A→B .08251, B→A .04173으로 주요 비교 방향이 유지됐다. 첫 Bullet run은 half-step check에서 contact slop의 영향을 발견해 결과 산출 전에 중지했다. 별도 one-impact unit check에서 이를 확인하고 기본값 대신 contactSlop=0을 지정했다. [수정 기록](numerical_note.md)

- [실험 설계](plan.md) · [재현 코드](solver.py) · [GPU 실행 스크립트](run.sh)
- [전체 실측·예측 결과](predictions.csv) · [수치 검증과 parameter](metrics.json) · [희귀 event 빈도와 baseline](marginal_baselines.json) · [원시 실행 로그](rerun.log)

## 같은 campaign에서의 재검증

Whole-campaign transfer 결과를 본 뒤, 같은 surface와 장치 안에서도 결과가 유지되는지 별도 비교했다. 각 campaign에서 연속 24행 block을 4개 fold로 돌렸다. 각 회차에서 한 fold를 test, 바로 다음 fold를 residual calibration, 나머지 둘을 parameter fitting에 사용했다. 모든 1,023개 trial이 방법별로 정확히 한 번 test가 됐다. A에서는 block 하나가 6회씩 반복한 launch setting 네 개를 담는다. B에서는 반복 수가 일정하지 않아 block 경계를 넘는 동일 setting이 일부 있을 수 있다. 이 검증은 same-campaign interpolation에 해당한다. [실행 전 기록](within_plan.md) · [전체 결과](within_metrics.json) · [trial별 값](within_predictions.csv)

| Holdout campaign | 모델 | vx MAE (m/s) | u MAE (m/s) | Near-stick Brier | Reversal Brier |
|---|---|---:|---:|---:|---:|
| A | PyBullet / normal-compliant | 4.829 | 4.636 | .02083 | .11132 |
| A | Tangential spring | 4.593 | 3.347 | .02614 | .09773 |
| A | Source prevalence | — | — | .02142 | — |
| B | PyBullet / normal-compliant | 5.820 | 2.494 | .07467 | .08349 |
| B | Tangential spring | 5.841 | 2.626 | .08453 | .09484 |
| B | Source prevalence | — | — | .07435 | — |

A에서 spring은 contact-point velocity MAE와 reversal Brier를 줄였지만 near-stick Brier를 높였다. B에서는 세 지표가 모두 악화됐다. Near-stick을 확률적으로 가려내기 어렵다는 결과는 서로 다른 campaign 사이의 transfer에만 국한되지 않았다. 이 표의 평가 단위는 두 campaign으로 한정되며 반복 impact를 독립 surface로 셀 수 없다.
