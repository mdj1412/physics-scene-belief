# Resource scheduling addendum

2026-11-11 약12:50 KST, actual evaluation 전에 추가했다. Inference method, sample count, estimator, primary contrasts, seed 또는 outcome-based stopping rule은 바꾸지 않았다.

처음에는 짧은 allocations를 preparation에만 사용할 계획이었다. 첫 9 cases의 acquisition 소요 시간이 약35–40s이고 짧은 allocations에 약7분이 남은 것을 확인했다. 이에 allocation2676324에는 cases56–59, 2676325에는 cases60–63을 auxiliary.py로 배정했다. 새 case 시작은 240s local deadline의 90s 전까지만 허용한다. 긴 worker는 low indices부터 순서대로 수행하므로 short worker가 완료된 long-worker 도착 훨씬 이전에 이 disjoint cases를 끝내고, main은 complete marker가 있는 cases를 건너뛴다. 만약 unfinished auxiliary case가 발생하면 owner 종료 확인 후 main이 재개한다.

각 world의 Bridge와 모든 SNIS 비교는 해당 world를 실행한 동일 GPU에서 측정한다. Main의 actual evaluation은 64 policies가 모두 고정된 이후 수행한다. 이 변경은 resource scheduling에만 해당한다.
