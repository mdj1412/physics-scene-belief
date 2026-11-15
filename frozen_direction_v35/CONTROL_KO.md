## 방향이 실제로 없는 surface에서의 검증

앞의 query-only stress와 달리 이 control은 calibration과 deployment 모두에 isotropic law를 사용했고, 별도 16개 seed block에서 처음부터 다시 실행했다. 물리적으로 참인 표면 방향이 없으므로 reader의 방향 accuracy는 정의하지 않았다.

| motion | 원래 Brier | 보정 Brier | Δ | Simultaneous 95% interval | 평균 ∣z∣ |
|---|---|---|---|---|---|
| 한 방향 | 0.10608 | 0.11217 | +0.00609 | [+0.00448, +0.00770] | 0.59886 |
| 양방향 교대 | 0.10568 | 0.11174 | +0.00605 | [+0.00342, +0.00868] | 0.61036 |

두 schedule 평균은 +0.00607 [+0.00449, +0.00786]였다. |z|는 reader가 중립값에서 얼마나 멀어졌는지 나타낸다. 두 방향 중 어느 것인지 구분하도록 학습한 reader에는 “표면에 방향이 없음”이라는 별도 상태가 없다. 가능한 두 부호를 잘 구분하는 것과, 방향이 없는 표면을 알아보는 것은 다르다.

탐색적으로 같은 query-only stress의 amplitude 비율을 0부터 150%까지 2.5%p씩 늘려 평균 Δ의 부호가 바뀐 구간도 확인했다. 한 방향 60~62.5%, 교대 62.5~65%였다. 기준은 각 표면 원래 amplitude의 백분율이다. 이 값은 네 synthetic 조합을 평균한 격자상의 교차이며 deployment gate나 실제 마찰 계수 threshold로 학습·검증한 것이 아니다. [수치와 전체 곡선](threshold35.json)