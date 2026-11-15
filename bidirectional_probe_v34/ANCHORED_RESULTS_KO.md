### 원 physics decoder를 기준으로 해도 이득이 남는가?

처음 설계한 terminal logistic recalibration은 방향 추정치를 쓰면 neutral recalibration보다 낮은 loss를 냈지만, 둘 다 원래 physics decoder보다 나빴다. 전체 계수·온도까지 다시 fit한 비교만으로는 sign을 실제로 쓸 가치가 있다고 말하기 어렵다. 이 결과를 본 **후**, fixed 4와 기존 sensor 선택을 그대로 두고 원래 확률의 log-odds에 β·(2P(plus|prefix)−1)만 더하는 anchored control을 추가했다. β 한 개만 뺀 object·surface의 자료에서 학습했다. Cue가 0이면 원 확률과 완전히 같은 예측이다.

| 평가 자료 | 원 fixed 4 Brier | Estimated sign 보정 | True sign 보정(진단) | Estimated − 원 Brier [simultaneous 95%] |
|---|---:|---:|---:|---:|
| v34 단방향 | 0.12570 | 0.11974 | 0.11780 | −0.00596 [−0.00664, −0.00528] |
| v34 양방향 | 0.12560 | 0.12025 | 0.11779 | −0.00535 [−0.00605, −0.00464] |
| v33 별도 물체, 음의 방향만 | 0.12570 | 0.11882 | 0.11684 | −0.00688 [−0.00765, −0.00611] |

추가 probing이나 sensor 변경 없이 원래 decoder를 기준으로도 0.003보다 큰 개선이 나왔다. 실험자가 알려 준 true sign 없이 방향 추정치만으로도 이득이 있었다. 이 결과가 지지하는 후속 가설은 **다음 물체에 어떤 관측을 살지 바꾸기보다, 이미 산 관측을 해석하는 데 환경의 마찰 방향을 쓰자**는 것이다. 다만 primary 결과를 본 뒤 고른 모델이며 v33 archive의 기존 결과도 연구자가 본 적이 있다. 위 구간은 사후 분석 선택의 불확실성을 반영하지 않는다. 이 보정의 추가 GPU p90은 따로 실측하지 않았다. [보조 대조 계획과 식](ANCHORED_PLAN_KO.md) · [전체 수치와 β](anchored34.json) · [코드](anchored34.py)
