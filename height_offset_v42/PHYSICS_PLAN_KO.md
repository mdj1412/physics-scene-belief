# v42 물리 검산 계획

두 nominal query height 19·31 mm에서 실제 높이만 −1, −0.5, −0.25, −0.125, 0, +0.125, +0.25, +0.5, +1 mm 옮긴다. +는 좌표 y 증가 방향이다. Factual evidence, contact calibration 네 번, 7-point query view, 관측 noise, decoder의 template과 source 가설의 event는 nominal frame을 유지한다. 같은 frame에서 추정한 deformation axis 역시 같다. 23 mm에서 실행하는 네 번째 calibration은 자체의 높이 오차가 없다. Offset은 기록·채점에만 쓴다.

독립 12개 panel은 예전 데이터에서 사용하지 않은 v41 world generator의 catalog index 256–267을 사용한다. Calibration과 deployment support는 각각 별도 stream의 2,048개 prototype이고 passive profile 32쌍도 다른 stream에서 얻는다. 이전 v41 fitting에 사용된 0–11 panel과 seed 및 물체 hash 중복을 조사한다. 관측을 반복하거나 같은 물체를 여러 offset에서 시험해도 독립 panel 수는 12다.

**두 가지 label flip.** 같은 물체·law·표면 방향·regime에서 각 offset마다 (1) hollow 또는 bulge와 같은 높이의 base를 비교해 contour가 바꾼 event 비율을 기록한다. (2) 같은 contour의 nominal event와 비교해 높이를 옮긴 영향 전체를 구한다. 물체별 slip→stick 및 stick→slip을 따로 계수한다. Base만의 nominal 대비 flip은 contour lobe 없이 나타나는 ordinary height effect다. 형상·nominal height·signed offset 각각의 analytic slope increment를 검산한다.

**숨긴 접촉 위치의 한계.** 특정 물체·형상·law·regime과 offset 크기까지 알려도 부호를 동전 던지기로 정하고 관측에 offset을 담지 않으면 같은 입력에서 두 실제 event가 다를 수 있다. 두 부호의 event가 서로 다를 때 최적 constant probability는 0.5이고 평균 Brier는 0.25다. 따라서 이 이상적인 정보 조건에서의 irreducible Brier floor는 0.25 × P[Y(+d) ≠ Y(−d)]다. 9개 signed offset을 균등하게 섞는 경우의 floor는 물체마다 아홉 label의 평균 m을 내고 m(1−m)을 평균한다. 두 수치는 arbitrary real-world height noise의 하한으로 옮기지 않는다.

**물리·점수 검산.** 노이즈와 물체 index를 sidecar seed에서 다시 얻고, 별도로 쓴 NumPy normal/tangent collision 식으로 모든 actual-height event 및 paid calibration의 velocity를 비교한다. 0% Brier 대조의 정확한 항등식도 확인한다. 같은 물체의 예측 차이가 d=p_est−p_orig일 때, 실제 variant label로 채점한 두 decoder의 loss 차이에서 같은 높이의 base label로 채점한 차이를 빼면 −2·mean[d·(Y_variant−Y_base)]다. 이 항은 ordinary height effect로 설명할 수 없는 contour-dependent event 변화의 기여다.
