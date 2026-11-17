## 문헌 비교와 현재 증거의 범위

이번 검토는 접촉 위치의 불확실성, 면접촉의 계산 방식, 실제 slip 측정이라는 세 쟁점을 다룬 원저를 골랐다. 아래 논문은 서로 다른 outcome을 평가하므로 오차를 합산하거나 논문 간 성능 순위를 매기지 않았다.

| 연구 질문·원저 | 방법과 표본 단위 | 결과와 결론 | v44 해석의 한계 |
|---|---|---|---|
| 같은 실물 impact를 여러 simulator가 얼마나 설명하는가? Acosta et al. | 실제 cube toss와 Cassie robot의 22회 jump 자료에서 Drake·MuJoCo·Bullet의 contact parameter를 식별하고 궤적을 비교했다. | Cube의 궤적을 대체로 재현했으나 bounce와 slide가 바뀌어 최종 자세 오차가 커지는 사례가 있었다. | 한 engine에서 다른 engine으로 옮겨도 실물에서 어떤 endpoint가 맞는지 알 수 없다. 저자들은 실측 궤적을 사용했다. [원문 §§III–V](https://arxiv.org/pdf/2110.00541) |
| 단일 contact point가 넓은 접촉에서 필요한 wrench를 표현할 수 있는가? Elandt et al. | 내부 pressure field로 contact surface와 net force·moment를 근사했다. Gripper로 pencil을 잡고 회전시키는 예제 등을 시연했다. | 접촉 분포에 따라 달라지는 회전 구속을 설명할 수 있었다. | Bullet의 소수 활성 contact point 사이 거리와 연속 압력장의 면적은 다른 측정량이다. [원문, Fig. 4](https://arxiv.org/pdf/1904.11433) |
| Slip–stick transition을 포함한 compliant dynamics를 안정적으로 적분할 수 있는가? Castro et al. | Regularized friction용 TAMSI를 개발하고, 흔들리는 parallel-jaw gripper가 mug를 잡는 경우 등에서 안정성과 비용을 평가했다. | Transition 부근의 stiff dynamics를 고려하는 적분이 수치적 어려움을 줄였다. | 적분기·friction regularization이 다르다. v44의 half-timestep 대조는 개별 label과 평균 Brier에 대한 자체 검산이 필요하다. [원문 §VI](https://arxiv.org/pdf/1909.05700) |
| 실물에서 contact patch 전체가 한 번에 미끄러지는가? Yuan et al. | GelSight elastomer의 영상 sequence에서 접촉 영역 내부의 shear·torsional deformation과 slip 전파를 측정했다. | 주변부에서 slip이 시작돼도 중앙부에 static contact가 남는 incipient slip을 관찰했다. | Rigid tip의 충돌 후 tangential speed로 정의한 binary event는 국소 incipient slip 자체를 측정하지 않는다. [원문](https://people.csail.mit.edu/yuan_wz/GelSight1/ICRA15_2740_FI.pdf) |
| 위치 오차로 변하는 normal이 충돌 후 상태에 어떻게 전파되는가? Meißenhelter et al. | Armadillo·Bunny·Cup mesh의 충돌에서 sphere hierarchy 및 분포 근사를 sampling·linearization과 비교했다. | 이 시험에서 제안한 근사가 대량 sampling보다 빨랐고 단순 선형화보다 결과 분포에 가까웠다. | 위치 → normal의 중요성에는 선례가 있다. Frozen decoder에서 어떤 contour 정보를 지울지의 Brier 예측은 해당 실험의 outcome이 아니다. [원문 §III](https://cgvr.cs.uni-bremen.de/papers/icra40_24/ICRA40_2024_UncertainPhysics.pdf) |
| Rigid-body impact parameter가 실물 물성으로 해석되는가? Fazeli et al. | 2,000회 planar drop 중 1,718회 유효 관측을 사용해 여섯 contact model의 impulse와 파라미터를 분석했다. | 실제 impulse를 표현할 수 있는 범위가 제한되고 맞춘 parameter의 물리적 해석이 어려운 사례가 있었다. | 독립 solver와 같은 결과를 얻어도 실제 friction coefficient나 실제 contact area를 식별했다는 증거가 되지는 않는다. [원문](https://arxiv.org/pdf/1710.04979) |

**연구 지식 지도.** 위치오차와 tip 외곽 → 실제 접촉 위치와 법선 → contact impulse·접선 운동 → stick 가능성이라는 물리 경로가 있고, 7점 profile → geometry likelihood → stick 확률이라는 추정 경로가 있다. 위치 불확실성이 법선에 전파되고 접촉 분포가 wrench에 영향을 준다는 근거는 독립적으로 축적되었다. 두 경로를 동일 물체에 짝지어 **어느 방향의 정보를 지우면 어떤 조건에서 Brier가 오르내리는가**를 사전 예측하는 부분이 이 연구의 좁은 질문이다. [Meißenhelter et al.](https://cgvr.cs.uni-bremen.de/papers/icra40_24/ICRA40_2024_UncertainPhysics.pdf) · [Elandt et al.](https://arxiv.org/pdf/1904.11433) · [v43 결과](../blind_decoder_v43/REPORT_KO.md)

**엇갈리는 근거의 해석.** 단순한 접촉 모형이 cube의 전체 궤적은 잘 맞히면서도 접촉 직후 bounce와 slide를 다르게 낼 수 있다. 실물 궤적의 평균 오차, 공간적인 partial slip, 미래 물체의 binary Brier는 측정 단위가 다르다. Acosta et al.의 궤적 재현과 Fazeli et al.의 impulse 한계, Yuan et al.의 부분 미끄러짐은 이 차이를 감안하면 양립한다. v44에서 실험적으로 개입한 radius·sole span의 효과는 테스트한 engine과 outcome 정의에 한해 해석한다. [Acosta et al.](https://arxiv.org/pdf/2110.00541) · [Fazeli et al.](https://arxiv.org/pdf/1710.04979) · [Yuan et al.](https://people.csail.mit.edu/yuan_wz/GelSight1/ICRA15_2740_FI.pdf)

### 남은 연구 질문 열 가지

| 질문 | 근거와 중요성 | 가능한 검증 |
|---|---|---|
| 1. 실제 압력 중심이 geometric support보다 나은 predictor인가? | 분포를 적분한 wrench와 한 접촉점은 다른 정보를 보존한다. [Elandt et al.](https://arxiv.org/pdf/1904.11433) | Pressure-weighted normal, 첫 normal, 마지막 normal로 별도 forecast를 고정해 새 손익 부호를 비교한다. |
| 2. Radius와 compliance는 언제 같은 효과를 내는가? | 넓은 rigid sole와 compliant pad의 접촉 형성이 다르다. [Elandt et al.](https://arxiv.org/pdf/1904.11433) | 같은 외곽의 경도만 바꾸어 contact area·stick 확률·Brier를 교차 측정한다. |
| 3. Micro-slip과 gross slip 중 어느 label을 예측해 왔는가? | 실제 patch에서는 가장자리부터 미끄러질 수 있다. [Yuan et al.](https://people.csail.mit.edu/yuan_wz/GelSight1/ICRA15_2740_FI.pdf) | 촉각 영상에서 두 event를 각각 표기하고 속도 기반 label과 비교한다. |
| 4. 회전 자유도를 주면 fold/ripple 패턴이 달라지는가? | 접촉 moment는 회전 제어 과제에서 중요했다. [Elandt et al.](https://arxiv.org/pdf/1904.11433) | 같은 질량·자세에서 angular inertia만 바꾸고 net moment와 부호를 예측한다. |
| 5. 어느 solver 해상도에서 부호가 안정되는가? | 마찰 전이의 수치 강성이 적분을 어렵게 한다. [Castro et al.](https://arxiv.org/pdf/1909.05700) | Time step·mesh spacing·solver iterations를 독립적으로 변화시키고 같은 물체의 event flip 및 paired Brier를 비교한다. |
| 6. 실제 drift 분포를 써도 평균 predictor가 바뀌는가? | 위치 분포에 따라 normal과 충돌 후 분포가 달라진다. [Meißenhelter et al.](https://cgvr.cs.uni-bremen.de/papers/icra40_24/ICRA40_2024_UncertainPhysics.pdf) | 장비의 signed placement error를 센서로 먼저 추정한 뒤 해당 빈도로 cell을 가중한다. |
| 7. 관측과 query 충돌을 모두 다른 engine으로 옮겨도 되는가? | 실제로 solver 간 이동을 제한하는 원인은 과제별로 다르다. [Acosta et al.](https://arxiv.org/pdf/2110.00541) | Factual/probe engine × query engine을 교차해 representation shift와 label shift를 분리한다. |
| 8. 실제 마찰 계수와 모형의 유효 계수가 달라도 예측하는가? | 실측 impulse를 맞춘 coefficient는 실제 물성으로 단정하기 어렵다. [Fazeli et al.](https://arxiv.org/pdf/1710.04979) | 독립 마찰 측정과 trajectory fit을 분리하고 새 표면 session에 대해 검증한다. |
| 9. 같은 도구를 가정한 shape prior만으로도 예측이 충분한가? | V43에서는 단순히 항상 projection이 손해라고 해도 여러 조건에서 방향이 맞았다. [v43 검산](../blind_decoder_v43/PHYSICS_RESULTS_KO.md) | Carryover·always-harm·tool-aware forecast 간 불일치 cell을 다른 형상에서 미리 지정해 평가한다. |
| 10. 고밀도 geometry 관측 한 번을 구매할 가치가 있는가? | 실제 접촉점의 불확실성은 기하 입력의 가치를 제한한다. [Meißenhelter et al.](https://cgvr.cs.uni-bremen.de/papers/icra40_24/ICRA40_2024_UncertainPhysics.pdf) | 같은 배포 시간에 고밀도 profile과 접촉점 camera를 대조하고 acquisition·reset을 포함한 Brier와 작업 실패 비용을 보고한다. |
