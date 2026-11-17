# v61 · 새 물체와 surface에 대한 held-out deployment

24개 catalog key 5100~5123에서 각각 두 새 material instance와 직전 실험의 두 rate law를 교차한다. Material × law마다 128개의 새 deployment object를 추출한다. 교체 전·후 support를 포함한 parameter vector가 v50/55/56/57/58/59/60과 겹치는지 검증한다. Frozen GPU forecaster, v54 gate와 continuous joint terminal은 v60에서 그대로 사용한다. 같은 물체의 PyBullet fine·ultra 결과를 각각 구하고 independent binary replay seed 61041729, 61084591를 정한다. Ground truth는 evaluation에서만 사용한다.

현재 node55/2744468과 node56/2744471의 GPU는 0 MiB·0%·compute PID 없음이다. Node55 한 장에서 forecast 및 latency를 측정한다. 단계마다 resource를 다시 확인하고 외부 process가 출현하면 guard가 이를 감지한다. 다른 두 allocation의 기존 Python 작업은 계속 실행 중이다. World 준비 및 solver는 Slurm compute node에서 CUDA를 끄고 실행한다.
