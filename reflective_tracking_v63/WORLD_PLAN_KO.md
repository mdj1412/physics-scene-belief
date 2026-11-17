# v63 신규 deployment, 물리 budget과 GPU 측정

8300~8323 catalog에 각각 새 surface 2개와 기존 두 contact law, 각 surface × law당 128개 물체를 생성한다. 배포 물체 12,288개 및 두 calibration support에 대해 과거 archive와 6차원 parameter 일치가 없음을 검사한다. 독립 CPU PyBullet fine·ultra 해석, frozen common forecast와 v54 acquisition cutoff를 유지한다. Binary replay seeds는 63041729와 63084591이다. 광학 부호의 independent validation sensor는 정책에 제공하지 않는다.

각 정책에 동일한 8 impacts를 주고 selective·같은 수 무작위 유지·상시 v54 gate의 GPU p90을 비교한다. 세 정책당 24 catalog × 네 optical fault cell, memory-resident raw 영상·파형의 판독부터 각 정책이 구매한 terminal observation을 사용한 common GPU forecast가 끝나는 순간까지 잰다. 독립 validation sensor와 물리 촬영 및 reset 시간은 latency에 포함되지 않는다. 무작위 대조는 동일한 실제 pilot에서 confidence 기준으로 k를 산출한 뒤 최초 v61 유효 쌍 중 k개를 uniform sampling한다. 모든 정책은 downstream에 동일한 frozen decoder를 사용한다. 세 정책 순서는 전체에서 균형 있게 교차하고 각 catalog를 통째로 재추출한 paired bootstrap 16,000회로 두 비교를 동시 보정한다. 실제 hardware GPU는 2744468 node55에 할당된 한 장으로, 각 실행 직전 process 유무, 사용 memory, utilization을 확인하고 실행 도중 guard를 둔다.
