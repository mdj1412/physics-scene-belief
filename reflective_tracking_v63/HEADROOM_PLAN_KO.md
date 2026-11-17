# v63 · 같은 판독 수에서 실제로 바꿀 수 있는 행동

각 8-impact pilot에서 기존 reader가 읽은 n쌍과 confidence가 남긴 k쌍을 고정한다. 가능한 모든 k-subset을 동일한 frozen likelihood에 넣었을 때, gate와 fixed 3R+I가 모두 가능한 pilot의 비율을 측정한다. 0<k<n과 달리, 이 지표는 어떤 쌍을 남길지에 따라 실제 deployment 정책까지 바뀌어야 양수다. Read quality가 바뀌어도 action이 거의 고정돼 있다면 downstream effect가 작을 것으로 예측한다.

24 catalog와 8 board를 교차한 동일 panel에서 (1) 어느 subset도 action을 바꿀 수 없는 빈도, (2) 두 action이 모두 가능할 때 평균 손실이 낮은 쪽을 항상 고르는 privileged bound, (3) confidence 및 matched random 대비 그 차이를 계산한다. 권한이 있는 진단에서만 catalog·material·regime별 held-out expected risk를 사용한다. 두 정책의 risk를 새로 보아 선택한 in-sample bound이므로 실제 정책의 예상 성능 또는 out-of-sample 측정치로 해석하지 않는다. 미래 개별 이진 label을 본 뒤 예측 확률을 고르지 않는다. Primary처럼 C의 fault 빈도와 저손실 policy의 관계를 독립시킨다. Gate와 fixed가 실제로 어느 쪽이 낮은지는 출력값으로 다시 확인한다.
