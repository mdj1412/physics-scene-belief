# v54 · Paired held-out 평가

## 같은 decoder에서 네 번째 관측을 바꾼 결과

Primary gate가 12개 catalog의 5개 post-retention 조건, 4 environments, 256 prefix에서 전부 reference를 골랐다. 따라서 평균과 session p90이 fixed 4R과 정확히 일치했다. Host/GPU가 gate를 계산하는 시간에는 여전히 차이가 날 수 있다. 첫 세 쌍의 sign count만 쓰거나 terminal이 보유한 교체 전 네 쌍의 분포를 무시한 두 acquisition rule도 모두 reference를 선택했다.

| 정책 | Impact 선택률 | 평균 held-out Brier | Session loss p90 | Ultra-resolution 평균 / p90 |
|---|---:|---:|---:|---:|
| Nominal adaptive | 0.0% | 0.158526317 | 0.182367131 | 0.164145365 / 0.188434839 |
| Fixed 4R | 0% | 0.158526317 | 0.182367131 | 0.164145365 / 0.188434839 |
| Fixed 3R+I | 100% | 0.158518914 | 0.182310373 | 0.164139877 / 0.188393042 |
| 독립 random gate (목표 50%) | 50.22% | 0.158523162 | 0.182339519 | 0.164143295 / 0.188406914 |
| Source robust gate, 같은 nominal terminal | 35.29% | 0.158501188 | 0.182322383 | 0.164120697 / 0.188394010 |
| 10개 sign pattern의 leave-one-catalog-out lookup | 62.89% | 0.158502608 | 0.182309672 | 0.164122591 / 0.188384235 |

Frozen v53 continuous joint로 각 후보의 네 번째 측정을 받은 예측을 복원한 뒤 동일한 128개 deployment 물체의 potential outcome으로 채점했다. Mean은 deployment randomness에 대해 적분한 조건부 Brier를 prefix와 catalog 안에서 평균했다. P90은 실제로 하나의 binary outcome를 뽑은 물체 128개의 session 평균 loss 분포에서 계산했다. 모든 e∈{0,.25,.5}, g∈{0,.25,.5,1}, κ∈{0,.25,.5,.75,1}를 균등 가중하며, g는 surface 교체가 session 전체의 방향을 반전시킬 확률이다. 12개 catalog가 uncertainty의 단위다. 세 쌍의 prefix와 이후 관측·outcome은 여러 조건에서 공유되므로 이를 독립 표본으로 세지 않았다. [고정한 평가 계획](SCORE_PLAN_KO.md) · [채점 코드](score54.py)

| Primary에서 앞의 gate에서 대조를 뺀 값 | Mean 차이와 simultaneous 95% CI | Session p90 차이와 simultaneous 95% CI |
|---|---|---|
| Adaptive − fixed 4R | 0 [0, 0] | 0 [0, 0] |
| Adaptive − fixed 3R+I | +0.00000740 [+0.00000049, +0.00001432] | +0.00005676 [−0.00000907, +0.00009420] |

Mean은 paired t interval, p90은 같은 catalog를 함께 재추출한 5,000회 paired bootstrap이다. 두 대조와 두 endpoint를 한 family로 보정했다. 동일한 예측에서 얻은 [0, 0]은 물리적 모집단에서 두 정책이 항상 같다는 주장을 뒷받침하지 않는다. 두 번째 deployment noise seed의 p90 차이도 +0.00001897이었다. Ultra-resolution에서는 adaptive − fixed 3R+I의 평균 차이 +0.00000549 [−0.00000134, +0.00001232], p90 차이 +0.00004180 [+0.00000120, +0.00008207]이었다. 48-session 규모에서 더 큰 mean loss를 냈던 비율은 256회 중 49.2%, 더 큰 표본 p90은 39.5%였다. 이 빈도는 독립된 표면 256개를 시험한 결과가 아니다. [전체 endpoint](score54.json) · [Catalog와 fleet](catalog54.npz) · [Session replay](session54.npz)

## 어떤 순위는 맞고 어떤 결정이 틀렸는가

Nominal model에서 계산한 impact의 예상 Brier gain과 reference의 예상 gain의 차이를 기준으로 사분위수를 나눴다. 양수는 impact에 유리하다. 아래 실제 gain은 같은 prefix에서 두 경로의 expected Brier를 각각 구한 뒤 reference loss에서 impact loss를 뺀 값이다. 실제 gain은 분석자만 볼 수 있었다.

| 사분위 · impact를 가장 낮게 평가한 순서 | 예측한 gain 차이 (×10⁻⁶) | 실제 gain 차이 (×10⁻⁶) |
|---:|---:|---:|
| 1 | −86.1 | −80.4 |
| 2 | −55.8 | +11.0 |
| 3 | −33.9 | +32.9 |
| 4 | −23.0 | +66.0 |

네 집단의 평균 순서는 보존됐지만 예상 이득이 0을 넘는 prefix가 없었다. 이 결과는 모든 개별 prefix의 순위가 맞았다는 보장을 주지 않는다. True reference–impact mismatch e가 0이면 fixed impact의 평균 손실이 fixed reference보다 g별로 0.000010–0.000086 높았다. e=.25에서도 0.000087–0.000209 높았고, e=.5에서는 0.000169–0.000234 낮았다. 첫 세 weak/reference 쌍은 e를 직접 관측하지 않는다. [구간 및 조건별 결과](score54.json)

## 다른 prior, 같은 구매 비율, transfer 오차

- Source robust selection은 nominal continuous joint를 공통 terminal로 유지하면서 mean에서 fixed 4R 대비 −0.00002513 [−0.00002817,−0.00002209], fixed 3R+I 대비 −0.00001773 [−0.00002269,−0.00001277]을 기록했다. p90은 각각 −0.00004475 [−0.00007397,−0.00001231] 및 +0.00001201 [−0.00003885,+0.00004318]이었다. 이 규칙에서 평균 이득을 보인 사실이 두 fixed보다 작은 p90까지 입증하지 않는다.
- Uniform e prior를 acquisition과 terminal에 함께 쓰면 7.54%에서 impact를 골랐고 mean 0.158645767, p90 0.182170451이었다. 같은 uniform terminal의 fixed 4R은 0.158650070 / 0.182179466, fixed 3R+I는 0.158671046 / 0.182187393이었다. 탐색적 mean 차이는 두 fixed 모두보다 작았지만 p90 차이의 interval은 둘 다 0을 포함했다. Uniform gate에 nominal terminal을 고정하면 0.158519058 / 0.182358339였다.
- Primary와 같은 구매율의 독립 random gate는 0%여서 fixed 4R과 같다. 50% random의 결과는 위 표와 같다. 10개 sign-pattern lookup은 판독 가능한 쌍의 수와 agreement 수로 bin을 나눠, 각 test catalog 이외의 11개 catalog에서 평균 Brier가 작은 관측을 고른다. 이 탐색 대조는 다른 catalog의 **deployment 정답**을 학습에 이용한다. Fixed impact 대비 mean −0.00001631 [−0.00002068, −0.00001193], p90 −0.00000070 [−0.00004428, +0.00002812]였다. Fold 간 training set이 겹치는 이 interval은 추정한 lookup을 고정한 불확실성이다.
- 활성 impact 방향의 10%를 잘못 읽고 known error rate를 terminal에 반영했다. Nominal adaptive는 그런 측정을 한 번도 구입하지 않아 결과가 같았다. Robust gate의 mean/p90은 0.158520207 / 0.182336167, 같은 센서의 fixed 3R+I는 0.158578840 / 0.182345927이었다. 이 값은 특정 오독 모델을 적용한 sensitivity다. Secondary interval은 각 규칙 안의 네 비교에 대해 동시 보정했고 규칙 탐색 전체에 대한 familywise 확증은 아니다. [Secondary CI 및 readout sensitivity](secondary54.json)

## 네 번째를 미리 선택할 수 있는 범위

| 진단 | 평가 prefix에서 네 번째를 고르기 전에 본 정보 | Impact 선택률 | Mean | Session p90 |
|---|---|---:|---:|---:|
| Hidden-cell lookup | e, drift event, κ와 다른 11개 catalog의 해당 조건 평균 | 51.35% | 0.158456141 | 0.182255074 |
| Privileged prospective, 8 futures | 교체 전후 prefix, 실제 환경·e·κ와 독립 네 번째 후보 sample들 | 52.65% | 0.158456129 | 0.182261258 |
| Privileged prospective, 32 futures | 동일 | 54.51% | 0.158446761 | 0.182252377 |
| Privileged prospective, 128 futures | 동일 | 54.89% | 0.158443162 | 0.182245284 |
| Both-future hindsight | 현재 물체의 실제 네 번째 reference와 impact를 **둘 다** 본 뒤 latent expected risk 비교 | 25.56% | 0.158403538 | 0.182184145 |

Prospective에서는 현재 세 물체를 고정하고, 같은 catalog·환경에서 다른 calibration prefix의 네 번째 물체만 donor로 썼다. 이미 지불한 세 object ID와 같은 물체 및 현재 prefix의 네 번째 trial은 donor에서 제외했다. 후보 선택 후 원래 희생 물체의 관측으로 돌아와 최종 예측을 평가했다. 128개 donor에서 fixed 4R 대비 mean이 −0.00008316, fixed 3R+I 대비 −0.00007575 낮았다. Latent deployment risk와 true e는 실배포에서 주어지지 않는다. 이 수치는 도움이 되는 regime이 존재하는지 보는 Monte Carlo 진단이며 post3로 달성할 수 있는 보장치가 아니다.

Hindsight는 두 센서의 실현값을 동시에 지불했다는 특권과 실제 환경 risk를 이용한다. 여기서 0.158403538은 **현재 두 potential forecast 중 하나를 택할 때의 평균 loss 하한**이다. Session p90을 최소화하도록 고른 정책이 아니므로 표의 p90을 수학적 하한으로 부르지 않는다. [Prospective 계산](prospective54.py) · [비교 결과](prospective54.json)
