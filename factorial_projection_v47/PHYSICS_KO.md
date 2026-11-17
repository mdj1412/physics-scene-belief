# 접촉 위치와 normal variance를 맞춘 v47의 두 개입

접촉 전에 같은 형상으로 보이던 물체도 하중을 받으면 다르게 휠 수 있다. v47에서는 평상시 외곽을 같게 한 접촉 부위에서, 7점 optical query에 남는 변형의 방향과 실제 load가 표면의 기울기를 바꾸는 정도를 따로 조작했다. 하나의 물체·카메라·측정 noise를 네 조건에 배치하므로, 두 요인에 따른 Brier 변화에 서로 다른 물체가 섞이지 않는다.

| 조작 | 동일하게 묶은 변수 | 바꾼 변수와 검산 |
|---|---|---|
| Optical alignment | 물체, profile deformation의 크기, 접촉 전 영상, solver와 target label | 명목 높이별로 source unit axis를 u, family별로 u에 수직인 unit vector를 v라 하면 query에 ±1.20 mm × (√a u + √(1−a) v)를 더했다. 7개 noise 표준편차가 같아 whitening 후에도 squared cosine a는 0.10 또는 0.90이다. 양 방향 모두 Euclidean norm 1.20 mm로, channel당 변위를 뜻하지 않는다. 이 조작은 calibrated sensor readout에 주입한 변환이다. |
| Mechanical coupling | 카메라, factual evidence, paid profile/probe, 최초 contact coordinate, nominal pressure footprint | Null에서는 unloaded curve를 그대로 썼다. Strong에서는 nominal tangent에 ±0.14의 slope 변화를 가하고 위치에 따라 다르게 휘는 loaded curve를 적분했다. 물성을 관측한 paired material sample이 아니라 prescribed load response다. Label은 두 조건 모두 finite pressure impulse 후 residual tangential speed ≤ 0.020 m/s로 계산했다. |

동일 좌표 주변의 local curvature는 공통인 두 smooth compact lobe로 만들었다. Keel·crown·channel·bead는 접촉부 바깥에서 dip, peak, 비대칭 wing, 짧은 복합 bump의 조합이 다르다. 각 family 쌍의 7점 query 위치를 두 높이에서 비교했을 때, 가장 비슷한 두 family도 적어도 한 곳에서 0.68 mm의 unloaded profile 차이가 있었다. 19·31 mm의 ±2.5 mm 구간을 촘촘히 샘플링한 profile과 slope의 family 간 최대 차이는 0이다. 세 offset(−0.5, 0, +0.5 mm)에서 ±2 mm normal 측정 범위와 broad patch 전체가 그 구간 안에 들어간다. 같은 height와 offset으로 고정한 것은 **최초 접촉 중심**이다. 변형된 중심 slope가 바뀌면 강도에 따라 총 normal impulse도 달라질 수 있다. [Generator](world47.py) · [Geometry audit](geometry_checks47.json)

Strong load에서 nominal height를 기준으로 d=(x−h)/2.5 mm를 두고 네 family의 slope 변화를 각각 ±0.14 × 1, ±0.14 × (1+0.40d²), ±0.14 × (1+0.70d), ±0.14 × (1+0.40 sin(πd))로 정했다. Family별로 해당 slope의 원시함수를 명시하고 finite difference 오차가 최대 9.8×10⁻¹¹임을 확인했다. 따라서 같은 unloaded normal variance를 지녀도 넓은 patch에서는 force가 전달되는 방향이 다르다. 실제 재질에서 이 식을 추정한 것은 아니다. [Geometry audit](geometry_checks47.json)

12개 catalog × 4 환경 × 128개 물체를 공통으로 사용했다. 각 catalog에서 2 surface × 2 contact law를 네 환경에 하나씩 배치해 catalog에 따라 순환했다. 각 물체는 4 family × 2 variant × 2 coupling × 2 height × 2 pressure radius(0.25·2 mm) × 3 offset으로 평가했다. 1,179,648개 design slot 가운데 null coupling의 variant·family 반복은 기계적으로 같아 한 번 계산한 후 일치 여부를 검증하고 공유했다. 이 구조와 primary case COUNT를 구별하면, 정확히 같은 null 조건을 접은 물리 계산은 663,552개이다. Optical alignment와 noise 반복은 이 숫자에 들어가지 않는다. 12개의 catalog가 불확실성 추정 단위다. [Finite-pressure evaluator](mechanics47.py) · [Mechanical checks](outcomes_checks47.json)

| 같은 물체에서 hollow↔bulge로 바꿔 stick/slip이 달라진 비율 | Null coupling | Strong coupling, 95% CI |
|---|---:|---:|
| Keel | 0% | 33.41% [32.50, 34.32] |
| Crown | 0% | 34.07% [33.15, 34.99] |
| Channel | 0% | 33.61% [32.61, 34.62] |
| Bead | 0% | 33.77% [32.78, 34.75] |
| 네 family 동일 가중 | 0% | **33.72% [33.01, 34.42]** |

Family별 strong 구간은 네 비교에 대해 보정한 paired catalog 평균의 simultaneous t interval이다. 마지막 줄은 별도 pooled 95% interval이다. Strong끼리 서로 다른 두 family의 label을 대조하면 결과가 갈리는 비율은 0.33%~1.73%였다. 각 family에서 개입을 켜지 않은 상태 대비 같은 signed variant의 label이 바뀐 비율은 16.71%~17.04%였다. Hollow와 bulge끼리 바꾼 33.72%와 구별해야 한다. 미리 정한 catalog 0·6에서 12×6 quadrature를 24×12로 높여 확인한 196,608개 slot에서 label 차이는 0건, 저장한 residual speed의 차이는 최대 5.96×10⁻⁸ m/s였다. [결과와 수치 검산](outcomes_checks47.json)

서버 CPU에서 32개 target 물체에 4 family × 2 height × 3 offset × 2 variant × 2 alignment × 2 coupling을 별도 렌더링해 6,144 clip을 대조했다. 같은 family·height·offset·물체에서는 optical alignment, mechanical coupling, signed variant를 바꿔도 pixel bytes가 모두 같았다. True secant-normal variance의 family 간 최대 차이도 0이었다. 실제 영상에서 계산한 estimator는 창 밖의 물체도 참조하므로, local analytic geometry의 일치와 measured x·variance의 family 간 일치는 서로 다른 검산이다. [동일 이미지 검산](pixels_optics47.json)

이 설계에서 coupling 자체를 판별할 수 없는 까닭은 같은 관측에 서로 다른 loaded solver를 직접 연결했기 때문이다. 이 한계는 실험 장치에서 성립한다. 실제 접촉 전 영상에 미세한 deformation이나 compliance 단서가 있는 물체, 또는 하중을 가한 추가 관측이 있는 상황에까지 적용하려면 별도의 물리 데이터를 수집해야 한다.

전체 12개 catalog의 실제 GPU forecast archive에서 대응하는 물체의 추정 x를 family 사이에 대조하니 차이가 있는 경우는 0건이었다. True x에 대한 pooled MAE는 0.093916 mm, 오차 p90은 0.194714 mm였다. 네 family의 측정 variance 범위(max−min)는 median과 p90이 0, 평균 7.62×10⁻⁷ rad², p99 7.28×10⁻⁶, 최대 0.003423 rad²였다. 같은 family의 signed variant 간에는 x·variance가 모든 입력에서 정확히 일치했다. 공통 noise를 사용한 실제 paid query 쌍으로 squared cosine을 다시 계산하면 설계값 0.10·0.90에서 벗어난 최대치가 3.68×10⁻⁷이었다. [전체 GPU 입력 검산](observed_optics47.json)
