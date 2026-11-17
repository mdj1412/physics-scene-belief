# v57 · 비용 및 GPU p90 사전 계획

기존 사용자 Slurm allocation에서 2744468/node55와 2744471/node56이 각각 0 MiB, 0% utilization, 다른 compute PID 0개임을 확인했다. 다른 두 allocation은 각 35,171 MiB 및 40,025 MiB를 사용해 건드리지 않았다. 새 forecast 생성은 node55, 아래 측정은 node56에서 수행한다. 시작 직전 재확인하고 전 과정에서 foreign process를 감시한다.

정책은 MAP 4×2, MAP 8×1, source-risk weighted 4×2, 항상 v54 gate, 항상 fixed 3R+I의 다섯 개다. 8×1에서는 두 모집단의 marginal likelihood가 같으므로 tie를 v54 gate로 정한다. 같은 결과를 주는 고정 대조는 pilot readout을 처리하지 않아도 된다. 각 catalog의 두 surface 중 하나는 A, 하나는 B이며 균형 배정한다. 각 surface에서 두 contact law에 속한 두 개 session이 해당 pilot을 공유한다. 실제 두 signed pilot readout의 8쌍과 session grouping에서 outer route를 온라인으로 계산하고, v54 gate 유지시에만 세 weak/reference 쌍으로 4번째 센서를 선택한다. 구매된 판독과 고정 continuous joint로 네 session의 512개 deployment object를 예측한다. 두 potential action의 숨긴 판독을 교대로 오염시켜 사용하지 않았는지 확인하고, freeze된 forecast와 매 실행의 결과를 대조한다.

각 catalog마다 모든 정책을 한 번씩 warmup한 뒤 balanced 5-period crossover를 다섯 번 측정한다. 두 catalog를 한 쌍으로 하여 같은 배치 순서를 정·역방향으로 실행한다. 이로써 각 정책의 period 위치와 pooled carryover 빈도를 맞춘다. 모든 24개 catalog, 정책당 120개 synchronized timing을 사용한다. CUDA를 호출 전후 동기화하고 sensor 선택과 terminal의 CPU/GPU 비용, object별 GPU inference와 반환을 포함한다. Sensor 장비, 파일 읽기, 생성/렌더링, model initialization은 제외하고, 이미 획득하여 메모리에 들어온 raw pilot readout을 실제로 전송·판독하는 online path를 포함한다. 사전 latency 대조는 MAP 4×2에서 다른 네 정책을 뺀 p90이며, 16,000회 paired catalog bootstrap 및 네 대조 Bonferroni simultaneous 95% interval을 적용한다.

각 design은 서로 다른 8개 물체에 reference와 impact를 각각 한 번 지불한다. 동일한 session 안의 둘째 물체도 독립 specimen이므로 contact/reset을 생략하지 않는다. 8×1은 8개, 4×2는 4개 pilot session의 setup이 필요하다. Contact/reset count, 동일한 cap을 주는 padding과 실제 예상 slot 시간, 추가 setup 및 물체 교체 비용을 구분한다. 실제 setup 시간이 없으므로 별도 S parameter로 계상한다.
