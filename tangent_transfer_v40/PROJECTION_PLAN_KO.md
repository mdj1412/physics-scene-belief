# 사전 보조 대조 · 한 개의 관측 방향을 제거한 decoder

실행 전에 curve 설계식에서 얻는 7차원 query view 변형 방향 d를 알고 있다고 가정한다. 부호, 진폭, tangent family, contact law, 정답은 입력으로 쓰지 않는다. Query observation과 K256 template을 같은 sensor 표준편차로 나눈 뒤 d/σ와 평행한 성분을 양쪽에서 제거하고, 나머지 여섯 차원으로 posterior를 계산한다. Probe branch, acquisition, v38 optimizer와 fixed는 고정한다. 같은 물체에서 세 contour의 observation 차이는 d와 평행하므로 사전 예측은 새 decoder에서 contour별 Brier 차이가 수치 오차까지 0이라는 것이다. d 방향에 원래 유용한 정보가 있으면 base 성능은 나빠질 수 있다. 전체 평균 Brier의 방향은 사전 지정하지 않는다.

12개 matched catalog에서 catalog ID 홀짝으로 block 하나를 정하고, 12개 조건 모두 같은 배치에서 original decoder와 projected decoder의 두 보정을 paired 비교한다. Nuisance 방향 하나를 정확히 알고 있다는 강한 가정, 주 실험의 절반을 사용하는 보조 실험, 별도 p90 없음도 보고한다. 목표는 주 효과의 재추정과 별개로 알려진 contour 변형에 대한 민감도와 정보를 제거하는 비용을 수치화하는 것이다.
