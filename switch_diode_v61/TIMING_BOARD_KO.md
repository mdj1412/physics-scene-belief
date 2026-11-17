# Timing의 board cluster sensitivity

첫 catalog의 raw replay가 통과한 후 24개 catalog의 pooled p90을 열기 전에 보조 분석을 정했다. Timing은 catalog당 1개 board만 쓰므로 loss에서 사용한 board × catalog 완전 교차 설계를 따르지 않는다. 8개 board cluster를 복원추출하고 선택된 board마다 3개 catalog를 다시 뽑아 8,000회로 p90의 reset−sham/gate 동시 95% 구간을 구한다. Board 하나를 제외할 때의 pooled 비교 범위와 같은 block의 paired median도 따로 보고한다. 기존 primary catalog bootstrap과 표본 및 결론을 합치지 않는다.
