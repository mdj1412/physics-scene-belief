**같은 calibration에서 world에 따라 최적 observation이 바뀌는 조건을 만들었다.** 새로 추출한 1,280개 worlds에서 동일 stick query를 사용했다. 32회 observation noise를 둘로 나눴을 때 두 half 모두에서 query view가 0.02 Brier 넘게 유리한 world는 773개(60.4%), probe가 유리한 world는 373개(29.1%)였다. 좁은 curvature·넓은 friction에서는 probe의 Brier가 query view보다 평균 0.23371 낮았고, 넓은 curvature·좁은 friction에서는 query view의 Brier가 평균 0.21908 낮았다.

**Factual gate는 always-query를 이겼지만, 고정 K256까지 이긴다는 사전 superiority 기준을 충족하지 못했다.** Gate, K256, always-query의 held-out Brier는 각각 0.076195, 0.076308, 0.106122였다. Gate−always-query 차이는 −0.029927 [95% CI −0.035475, −0.024266], gate−K256 차이는 −0.000113 [−0.002066, +0.002067]이었다. K256 자체가 전체 선택의 25.3%에서 probe를 고른 강한 adaptive observation selector였다.

**P90에서는 K256이 더 빨랐다.** Gate의 GPU p50은 0.926 ms로 K256의 3.421 ms보다 짧았지만 p90은 4.023 ms 대 3.503 ms였다. 81.0%의 입력에서 gate가 action을 직접 선택하고, 나머지 19.0%에서 K1024 acquisition을 실행한 결과와 일관된다. Always-query의 p90은 0.757 ms였다. 이 비용은 GPU에서 선택 후 posterior를 갱신하기까지의 시간이며 실제 probe 동작 시간은 포함하지 않는다.
