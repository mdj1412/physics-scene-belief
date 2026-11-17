# v41 물체, 접촉 높이와 label-free 방향 추정

두 test shape는 기존 shelf·ridge에서 계수만 바꾼 재표집이 아니라 서로 다른 해석식으로 정의했다. 물체의 기본 offset·slope·curvature 위에 형상별 profile을 더한다. u=y/0.025일 때 flare의 추가 slope는 0.068·[1−exp(−u²/0.90)]이고, roll은 −0.072u/√(1+u²)이다. Profile은 이 식의 적분이며 y=0에서 0이다. 접촉 law는 v40의 saturating·crossover다. 새 환경에 옮기는 축은 latent object, base geometry, deformation basis 및 query height다.

| 측정 | 설정 |
|---|---|
| Factual profile과 probe | Factual profile 높이는 −10~+10 mm다. 기존 네 번의 calibration 중 마지막은 23 mm·+1.10이고, deployment probe는 0 mm다. |
| Held-out query | Query 충돌 속도는 +0.425다. Query height는 이전 25 mm에서 19 또는 31 mm로 바뀌고, query view는 각각의 새 높이를 중심으로 −6, −4, −2, 0, +2, +4, +6 mm에서 측정한다. |
| Contour variant | 같은 물체에 base, hollow, bulge를 적용한다. 네 quartic lobe의 경계가 11, 19, 23, 31, 39 mm에 있다. 각 lobe는 양 끝에서 위치·미분값이 0인 16x²(1−x)²이고, signed 최대 진폭은 0.90 mm다. Flare의 상대 진폭은 (0.80, −0.85, 0.30, 0.95), roll은 (−0.45, 0.55, 0.95, −0.55)다. |
| Contact control | 이 때문에 0·19·23·31 mm에서 세 variant는 접선과 위치가 일치하고, factual 및 네 번의 물리 측정은 같아야 한다. 하나의 global contour가 두 query height를 모두 정의한다. |
| 표본 분리 | 매 독립 panel·α에서 calibration과 deployment 용도의 prototype support를 서로 다른 seed로 생성한다. 동일 panel 내의 shape·height·law는 같은 latent support에서 물체를 뽑을 수 있다. 방향을 학습하는 passive calibration 물체는 다시 별도 seed로 직접 생성한다. 이들은 네 번의 충돌 물체 및 deployment 물체와 독립이다. |

단위 mm로 나타낸 bulge − base의 순수한 7점 측정 변위는 다음과 같다. Hollow는 반대 방향이다. 두 높이가 같은 물리 위치를 측정하는 경우 변위도 일치한다.

| Shape / height | 7개 좌표의 변위 | 길이 / sensor σ |
|---|---|---:|
| flare / 19 mm | +0.405, +0.720, +0.405, 0, −0.765, 0, +0.152 | 4.020 |
| flare / 31 mm | +0.152, +0.270, +0.152, 0, +0.481, +0.855, +0.481 | 3.819 |
| roll / 19 mm | −0.228, −0.405, −0.228, 0, +0.495, 0, +0.481 | 2.875 |
| roll / 31 mm | +0.481, +0.855, +0.481, 0, −0.278, −0.495, −0.278 | 4.208 |

## PCA에 들어가는 정보

각 panel·shape·height에서 독립된 calibration 물체 32개를 준비해 각각 두 profile을 같은 좌표에서 측정한다. 두 측정의 signed deformation amplitude는 독립 Uniform[−1,1]이고 주어진 순서는 방향과 무관하다. 두 측정에는 독립 Gaussian sensor noise(좌표당 σ=0.30 mm)를 더한다. 알고 있는 정보는 물체 내 pairing과 좌표 대응이다. Estimator 입력은 (32,2,7)의 관측 tensor뿐이다. Pair에 대응하는 amplitude, 실제 subspace, contact response, query label을 넘기지 않는다. 같은 shape·height의 양쪽 law와 세 variant, 여러 deployment object는 추정한 한 방향을 공유한다.

Whitened difference matrix의 각 행은 (profile₂−profile₁)/σ이다. 이를 행 표본에 대해 centering한 뒤 첫 번째 right singular vector를 단위 방향으로 사용한다. Projection은 단위 방향 u의 부호에 무관한 I−uuᵀ이다. 같은 물체의 base profile가 pair 차분에서 정확히 상쇄되므로 서로 다른 물체의 구분과 무관한 rank-one 신호의 방향을 측정할 수 있다. 이 가정은 repeatable contour 변화 또는 registered profile pairing에 의존한다. 방향 학습에는 한 group당 64회 passive readout이 필요하다. 같은 group에서 여러 배치에 재사용할 때와 처음 관측해야 할 때의 비용을 구분해야 한다. Oracle 방향은 생성식에서 계산한 참 변위를 sensor σ로 나눠 정규화한다.

보조 진단은 같은 32개 pair의 4, 8, 16, 32개 prefix로 angle arccos(|ûᵀu|)과 원래 변위 길이 대비 제거 뒤 남는 비율 sin(angle)을 기록한다. 각 prefix의 두 번째 관측을 한 행씩 순환 이동해 같은 물체 pairing을 깨뜨린 뒤 동일한 PCA를 적용한 negative control도 계산한다. 같은 물체를 여러 catalog처럼 세지 않고 각 group의 독립 panel 평균을 요약한다. [생성 코드](world41.py) · [방향 검산](direction_audit41.py)
