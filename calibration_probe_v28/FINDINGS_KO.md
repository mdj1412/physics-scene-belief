**서로 다른 과거 world를 한 번씩 probe하면, factual evidence만으로 보이지 않던 velocity–friction shift의 방향을 식별할 수 있었다.** 64개 target cohort에서 0회는 방향을 정할 수 없었다. 1·4·16회의 정답은 각각 48/64, 55/64, 61/64였다. 16회에서 50/64개 cohort는 한 방향에 95% 넘는 posterior probability를 부여했고 이 중 방향 오류는 0건이었다. 이는 시험한 shift 크기와 sensor에서의 관측값이다.

**최종 예측의 Brier와 관측 선택의 Brier는 다른 결론을 주었다.** 0회 대비 native Brier 개선을 처음 검출한 budget은 4회였다. Neutral K256 대비 개선을 처음 검출한 budget은 16회였고, 그 차이는 −0.00542 [−0.01050, −0.00035]였다. 그러나 16회와 neutral이 고른 관측을 같은 known-prior decoder로 채점하면 차이는 −0.00118 [−0.00538, +0.00303]이었다. 16회에서 선택한 action은 0회와 8.1%의 worlds에서 달랐으나, 사전 판정 기준에서 관측 선택 자체가 더 유용해졌다고 확인한 budget은 없었다.

**접촉이 비싸다면 4회와 16회의 차이가 중요하다.** 4회의 native Brier는 0.10937, 16회는 0.10855였다. 0회→16회의 평균 개선 중 약 86%가 첫 4회에서 나왔다. 동일한 환경에서 128개 test worlds를 처리할 때 두 방법의 총 probe 횟수는 test world당 0.126과 0.214였고, warm GPU p90은 모두 약 3.26 ms였다. 16회가 모든 배포 조건에서 경제적인지는 실제 접촉·reset 비용과 환경이 유지되는 기간에 달려 있다.
