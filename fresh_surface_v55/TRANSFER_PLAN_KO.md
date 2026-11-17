# v55 · 평가 분포에 대한 추가 분석 계획

Fresh catalog 결과를 보기 전에 다음 sensitivity를 정한다. Primary는 v54와 동일하게 세 reference–impact mismatch 수준 e=0, 0.25, 0.5를 균등 가중한다. Sensitivity에서 e=0.5의 비율 w를 0에서 1까지 움직이고 나머지 두 수준에 (1−w)/2씩 준다. 다른 요인의 가중치와 정책은 고정한다. 각 catalog에서 gate loss에서 fixed 3R+I의 loss를 뺀 값, 같은 구매율 random policy의 loss를 뺀 값을 집계한다. 각 곡선은 양 끝점 간 직선이다. 부호가 바뀌면 교차점, 바뀌지 않으면 조사한 전 범위의 부호를 기록한다. e는 gate가 볼 수 없고 이 분석에서 사후에 공개하는 평가 조건이다. 전개한 결과로 gate threshold나 prior를 다시 정하지 않는다.

Matched-rate random이 같은 I 구매량에서도 차이를 보이는 이유는 identity E[L_gate−L_random]=Cov(A,L_I−L_R)로 확인한다. A는 prefix로 정해진 0/1 impact 결정이다. Random action의 확률은 전체 prefix의 평균 A이고, 각 action의 예측을 따로 loss로 변환후 가중한다. Session마다 forecast를 섞으면 이 identity와 평가하려는 정책이 달라진다. Goal은 mean에서 나타나는 차이를 설명하는 것이고, 이 식으로 tail quantile까지 설명하지 않는다. 환경에 적용할 w의 실제 분포는 측정되지 않았다.
