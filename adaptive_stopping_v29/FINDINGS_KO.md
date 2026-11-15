**평가 때 사용할 probe 수까지 동일하게 고정하면, 추가 calibration probe를 환경별로 재배분해 held-out Brier loss를 조금 낮출 수 있었다.** 24개 독립 block에서 환경 8곳당 calibration probe 32회와 test probe 104회를 두 정책 모두에 배정했다. Adaptive stopping의 common-decoder Brier는 0.10717, 환경마다 4회씩 배정한 fixed 4는 0.10806이었다. 차이는 **−0.00089**, 두 사전 지정 비교에 대한 simultaneous 95% interval은 **[−0.00172, −0.00006]**이었다. 사전에 정한 실질적 개선 폭 0.003에는 미치지 않았다.

관측을 자유롭게 선택하는 streaming에서는 **0.10590 대 0.10629**였고, 차이의 구간은 **[−0.00124, +0.00045]**였다. 24개 block 전체에서 두 정책은 각각 768회의 calibration을 사용했고, adaptive가 실제 고른 test probes는 2,341회로 fixed 4의 2,397회보다 56회 적었다. 같은 단가를 probe당 c라고 하면 접촉 지출 차이는 streaming에서는 −56c, batch quota에서는 0이다. Query view 등 다른 센서 비용은 포함하지 않았다.

첫 2회 이후 점수는 어느 환경에서 **관측 선택이 더 자주 바뀔지**를 잘 구분했다. 환경 두 곳을 비교한 순서 일치율은 78.9%였다. 반면 어느 환경에서 **관측을 바꾸는 것이 Brier를 더 낮출지**에 대한 일치율은 50.3%였다. 이 차이가 이번 연구에서 가장 분명하게 드러난 한계다.

Warm GPU p90은 streaming에서 adaptive 3.497 ms와 fixed 4 3.494 ms, batch 128개 전체에서 424.33 ms와 424.37 ms였다. 해당 환경의 posterior fit·score·bank 생성까지 포함하면 batch p90은 506.30 ms와 481.15 ms로 adaptive가 5.2% 더 걸렸다. 센서와 물체 reset에 걸리는 실제 시간은 측정하지 않았다.
