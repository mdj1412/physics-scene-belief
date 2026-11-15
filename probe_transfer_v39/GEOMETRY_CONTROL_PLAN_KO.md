# 형상만 바꾸는 보조 대조

주 실험이 실행되는 동안, 주 비교의 target loss를 확인하기 전에 추가한 진단이다. Stage와 optimizer 및 네 사전 방향은 그대로 둔다. 48개 cell-specific catalog 각각에서 catalog ID의 홀짝으로 이미 정해진 두 block 중 하나를 택한다. 같은 v39 prototype, 초기 기울기와 곡률, 마찰·noise를 유지하고, shelf·ridge의 contour perturbation만 각각 v38의 shoulder·bead 식으로 교체한다. 공통 evidence·deployment acquisition·두 보정·held-out label을 그 물리 조건에서 다시 생성한다. 두 형상에서 각각 구한 Δ(optimizer−fixed)의 차이를 catalog 안에서 대응시켜 계산한다. 이 비교는 위치별 contour의 총효과를 추정한다. 형상을 바꾸면 contact normal·sensor 값·선택·실제 outcome이 함께 변할 수 있다. 과거 v38 catalog나 v38의 shape별 curvature 배율까지 복원하지 않는다.
