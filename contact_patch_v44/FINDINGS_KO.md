## 5. 수치 설정을 바꾼 전체 재검증

Primary에서는 221,184회 모두 접촉이 검출됐다. 사전에 정한 소표본으로 time step과 mesh를 점검한 뒤, 결과의 부호가 설정에 의존하는지 알아보기 위해 같은 두 검사를 전체 물체로 확장했다. 아래 두 재실행은 사후 민감도 분석이다. Target label만 새로 생성했고 사전 예측과 두 decoder 확률은 유지했다. [전체 수치 분석](NUMERICAL_FORECAST_KO.md) · [원속도와 contact manifold 검산](NUMERICAL_KO.md)

| Solver 설정 | Primary 대비 개별 event 변화 | 예측 적중 수 / 48 (tool-aware · carryover · always-harm) | Fold D nominal / shifted | Ripple D nominal / shifted |
|---|---:|---:|---:|---:|
| 20 µs · 0.10 mm mesh | 기준 | 43 · 38 · 34 | +0.002496 / +0.005250 | +0.004756 / +0.005324 |
| 10 µs · 0.10 mm mesh | 12,320회 (5.57%) | 47 · 42 · 38 | +0.003327 / +0.005975 | +0.006565 / +0.007136 |
| 20 µs · 0.05 mm mesh | 4,149회 (1.88%) | 43 · 38 · 34 | +0.002455 / +0.005411 | +0.004062 / +0.005087 |

Primary에서 동시 95% 구간이 0을 벗어났던 31개 cell의 부호는 두 재실행에서도 모두 같은 방향으로 구별됐다. 전체 재실행에서 time step 때문에 바뀐 네 cell은 전부 작은 반지름의 ripple였고, primary의 구간은 0을 포함했다. 세 수치 설정 전체에서 v43 대비 바꾼 다섯 예측은 모두 점추정 부호가 맞았다. 평균의 부호 안정성과 개별 접촉 event의 안정성에는 이처럼 차이가 있었다. 접선 속도 threshold를 0.01 또는 0.04 m/s로 바꿔도 네 primary 평균 D는 모두 양수였다. Primary에서 0.4 ms에도 하중이 남은 충돌은 1.68%였다. 사전 선택한 소표본을 0.8 ms까지 연장한 후속 검사에서는 13,824회 중 label 변화가 없었다. [해상도별 48개 cell](numerical_forecast44.json) · [종료 시점 확인](horizon44.json) · [속도 threshold 분석](stats44.json)

## 6. 논문으로 어느 주장까지 가능한가

**직접 뒷받침되는 기여**는 이미 훈련된 physics decoder에서 contour 방향을 지우는 선택의 손익이 *tip이 도달하는 실제 위치*에 따라 바뀔 수 있고, 그중 일부를 새 solver의 결과를 보기 전에 찾아냈다는 것이다. V43의 변형 방향 자체는 정확히 아는 privileged 조건에서도 이런 손실이 생겼다. Fold 19 mm의 세 조건은 새로운 물체, 세 수치 설정, 강한 historical carryover를 모두 견뎠다. 독립 solver에서 달라질 결과를 미리 명시했다는 점은 반복적인 simulator tuning에 대한 대안 설명을 줄인다. 위치 불확실성이 충돌 법선을 바꾼다는 기초 원리 자체는 기존 연구에도 있다. [사전 예측](FORECAST_KO.md) · [독립 solver 검증](FORECAST_RESULT_KO.md) · [Meißenhelter et al.](https://cgvr.cs.uni-bremen.de/papers/icra40_24/ICRA40_2024_UncertainPhysics.pdf)

**실제로 decoder 선택을 바꾸면 얼마인가.** 모든 높이·signed offset·도구를 균등 가중하고 예상 D가 음수인 11/48개 cell에서만 projection을 적용해 보았다. 같은 물체에서 original을 항상 쓴 경우보다 평균 Brier가 0.001320 낮았고 [paired 차이 −0.001839, −0.000802], carryover로 decoder를 고른 경우보다 0.000993 낮았다 [−0.001263, −0.000724]. 두 구간은 두 비교에 대한 동시 95% 구간이다. 이 분석은 사후에 계산했다. 선택 규칙은 사전 예측 부호를 그대로 사용했지만, 실제 signed offset과 shape를 알고 있다는 조건이 붙는다. 실제 접촉점을 모르는 배포 환경에서 기대할 이득의 크기는 아직 열려 있다. [동일 cell 재채점](diagnostic_routing44.json) · [계산 코드](diagnostic_routing44.py)

**논문에서 제한해야 할 주장**도 세 가지다. 첫째, 높이와 signed offset을 묶으면 단순한 always-harm의 15/16이 tool-aware의 14/16보다 높다. 따라서 장비별 평균 손익만 필요한 사용자에게 이 forecast가 우월하다는 근거는 없다. 둘째, 새 solver의 충돌 결과를 맞혔지만 같은 family·material field·기존 factual sensor를 재사용했다. 독립적인 실물·family·sensor에 대한 외부 타당성은 미검증이다. 셋째, 2 mm sole에서도 양의 하중점은 매우 짧은 구간에 모였다. 실물의 접촉 압력이나 stick/slip을 측정했다는 주장은 과도하다. 공개된 실물 impact 연구에서도 solver·parameter에 따라 slide와 bounce 재현성이 달라졌다. [조건별 검산](forecast_eval44.json) · [Acosta et al.](https://arxiv.org/abs/2110.00541) · [Yuan et al.](https://people.csail.mit.edu/yuan_wz/GelSight1/ICRA15_2740_FI.pdf)

**판단과 다음 확인 실험.** 현 단계의 설득력은 geometry representation에 관한 기전 검증에서 가장 높다. 범용 active acquisition이나 실물에 적용할 decoder 선택법의 독립 논문으로는 증거가 부족하다. 다음 한 번의 실험에서는 초기 contact 위치와 pressure area의 영향을 분리하는 것이 우선이다. 같은 최초 접선·위치를 갖되 compliance가 다른 두 pad를 만들고, 별도로 동일 경도에서 최초 위치만 이동시킨다. 새 contour family의 양·음 손익 cell을 각각 사전에 지정하고 query와 factual sensor를 모두 바꿔 측정한다. 실제 offset을 아는 진단, 영상에서 offset을 추정하는 선택, always-original을 동일 관측 비용으로 비교한다. 접촉점 이동만으로 사전 예측과 Brier 이득이 유지되면 support 위치 설명이 강해진다. 위치를 맞춘 뒤에도 pressure area에 따라 손익이 움직이면 분포를 표현하는 contact model이 필요하다는 근거가 된다. 실물 장비가 없는 단계에서는 pressure를 적분하는 solver로 이 대조를 먼저 시험할 수 있다. [Elandt et al.](https://arxiv.org/pdf/1904.11433)
