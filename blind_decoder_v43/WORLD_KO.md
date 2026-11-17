# v43 물리 생성기와 blind OOD의 범위

V43은 기존 6개 latent coordinates를 쓰는 rigid point-impact 문제와 센서를 유지하면서, 표면의 위치 특성과 속도에 따른 마찰 함수를 독립적으로 바꾼다. Source archives의 flare·roll 및 saturating·crossover가 target의 화면 이름만 바뀌어 나타나는 shortcut을 줄이기 위해 2×2×2의 생성식을 새로 작성했다. 식과 계수는 target 점수를 열기 전에 정했다. Target의 두 family씩이 전체 가능한 형상·재질·마찰 현상을 대표한다는 가정은 두지 않는다.

| 축 | 새 family | 달라지는 물리량 |
|---|---|---|
| Object shape | fold, ripple | Fold는 중심을 비켜난 위치의 logistic hinge를 적분한 외곽이다. Ripple은 기본주파수와 배주파수의 두 harmonic으로 기울기를 만든다. 각 외곽에는 다른 변형을 적용한다. Fold는 좌우 비대칭의 compact sine-square lobe, ripple은 compact quartic에 harmonic modulation을 곱한 lobe다. |
| Surface field | inlay, woven | Inlay는 22 mm 근방의 국소 material insert가 범위 내 friction scale과 anisotropy를 바꾼다. Woven은 24 mm 주기의 공간적 rib가 두 양을 바꾼다. 한 표면에서 서로 다른 두 speed law를 적용할 수 있다. |
| Contact law | rational, gaussian_rate | Rational의 stick threshold는 속도 제곱의 rational 함수, 방향성 항은 부호가 있는 Hill 함수다. Gaussian-rate는 0.64 부근의 speed band에서 static/kinetic 차와 directional gain이 변한다. 모두 동일한 Coulomb impulse solver 안에서 사용하는 effective law다. |

Nominal 19/31 mm와 네 번째 probe의 23 mm는 lobe의 knot다. 세 contour의 local position 및 slope가 그 지점에서 정확히 일치한다. Factual 센서 범위와 기존 calibration contact 위치에도 deformation은 없다. 따라서 같은 seed의 base·hollow·bulge 사이에서 실제 물체, 관측 noise, 네 번의 calibration response 및 nominal label을 고정할 수 있다. 실제 query만 ±0.125·0.25·0.5·1 mm로 이동하면 물체마다 다른 local tangent를 읽으므로 label을 다시 계산한다. 사전 가중치로 후보를 고를 때 offset은 관측되지 않는다. 이 설계에서 offset 변경은 셀별 systematic placement error이며, 하나의 물체가 연속해 아홉 번 이동하다가 변형·마모되는 실험과 다르다.

Oracle row에는 7점 profile에서의 analytic deformation difference를 sensor noise로 표준화한 단위벡터만 전달한다. 방향의 부호는 projection에 영향을 주지 않는다. Family 이름과 latent는 생성·평가 측에 있고 blind row에는 선택된 noisy view 또는 접촉 response만 전달한다. Catalog support seed는 panel·latent population·calibration/deployment lane·source/target에 따라 분리하고, target family 간에는 같은 base prototype을 고의로 공유한다. 따라서 confidence interval에 쓰는 독립 단위는 개별 cell 수가 아닌 catalog panel이다.

두 cloud 재질은 표현 가능한 표면 field의 합성 예제다. 새로운 law는 접촉시간 동안 friction history를 누적하는 동적 solver나 실측한 마찰 모델을 뜻하지 않는다. 기존 장비·관측 단위·지지면과 접촉하는 수식의 공통 부분이 있기 때문에 이번 비교가 평가하는 것은 family holdout에서의 transfer다.

보조 기하 예측은 v42와 마찬가지로 apparent slope s와 실제 tangent 변화 t를 써서 2ts−s²가 양수면 contour 방향을 지우는 것이 손해라고 예측한다. 이 식은 slope squared error를 Brier의 surrogate로 쓴 방향 예측이다. 새 target 결과 없이 수치 미분으로 계산하고, 이후 동일 shape·height·offset에서 hollow/bulge만 고른 Brier 차이의 부호와 대조한다. Simulator의 slope와 실제 offset을 아는 진단이며 배포 정책의 feature로 쓰지 않는다. [물리 생성식](world43.py) · [독립 수치 미분 및 event 식](physics_check43.py)
