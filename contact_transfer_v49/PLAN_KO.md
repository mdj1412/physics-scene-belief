# v49 · 독립 contact solver와 두 종류의 transfer

## 고정할 설계
v48에서 개발·동결한 직접 보정과 projection policy를 그대로 적용한다. Source fitting은 v47, 기하·카메라·센서·query는 v48 target 12개 catalog를 짝지어 재사용한다. v49에서 바뀌는 것은 full-load outcome을 계산하는 독립적인 contact dynamics와 two-station, weak-to-full state의 결합법이다. 본 시험은 같은 geometry에서 solver transfer를 분리한다. 새 object holdout 재수집은 아니다. Policy는 target outcome, solver identity, 조건 ID나 transfer 확률을 보지 않는다.

- 공간 계수 ρ ∈ {0, .5, 1}. Query weak state를 draw한 다음 다른 station이 확률 ρ로 그대로 복사하고 그렇지 않으면 (0,−,+) prior (.5,.25,.25)에서 독립 draw한다. 같은 station에서는 같은 state, 같은 sensor realization을 쓴다.
- 하중 간 방향 유지확률 κ ∈ {0,.25,.5,.75,1}. Active query weak state의 방향이 full load에서 κ의 확률로 유지되고 나머지는 반전한다. Inert는 inert로 유지된다. κ=.5에서는 active 방향만 무정보다. 공간 draw와 부호 유지 draw는 조건부 독립이다.
- Conditional state별 policy 예측과 full-load label을 먼저 따로 계산하고 joint distribution에 따라 Brier를 정확히 적분한다. 이로써 같은 trial에서 두 intervention의 contrast를 평가하고 Bernoulli draw의 우연한 차이를 줄인다. 각 정책의 추가 관측 예산은 0.4 N preload 1회, scalar signed displacement 1개, reset 1회다. Always-original과 direct-none은 추가 접촉이 없다. Fixed는 source 선택 31 mm station을 쓴다.
- Primary 효과는 query-local direct에서 (i) fixed-site direct, (ii) 같은 query-site projection gate의 Brier를 각각 뺀 값이다. 두 효과가 모두 음수이고 15개 cell × 2개 contrast의 simultaneous 95% upper bound가 0 미만이면 double win으로 센다. 12개 독립 catalog에 paired Student t + Bonferroni, catalog 안에서 family/height마다 같은 가중치와 state prior를 적용한다. 의미 있는 최소 효과에 대한 실물 사양이 없어 작은 값을 유의성만으로 해석하지 않는다.
- Controls는 original, direct-none, fixed probe에 local mapping을 적용한 fixed-shared, fixed projection gate다. 같은 위치인 31 mm에서는 local/fixed 두 direct 예측의 차이가 정확히 0이어야 한다. Algebraic endpoint interpolation과 개별 binary-outcome squared error로 scoring을 교차 검산한다. 동일 데이터에서 얻은 경계값은 exploratory estimate로 제시한다.
- Label을 보기 전의 예상은 κ가 높고 ρ가 낮을수록 local direct가 fixed direct를 이기기 쉽다는 것이다. 거의 항상 반전하면 강한 signed signal을 잘못 해석하는 local direct가 손해를 볼 수 있다. Local direct와 local gate의 차이는 ρ와 무관해야 한다. 새 solver에서 signed state에 따른 stick 확률의 순서가 반전하거나 거의 같으면 기존 효과가 유지된다는 보장은 없다.
- GPU latency는 inference 전체에 대해 동일한 idle allocation에서 method order를 섞고 synchronize한다. 512 objects/batch, warmup 제외. GPU p90과 paired CI를 비교하고 로봇의 sensing/reset은 물체당 횟수와 물리 시간 가정으로 따로 계산한다. Simulation label 생성 시간은 deployment latency에서 제외한다.
