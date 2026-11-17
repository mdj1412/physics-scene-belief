# 추가 compute 검산

같은 512개 물체로 계산하는 common decoder가 총 wall time에서 차지하는 비중을 해석하기 위해, 24 catalog × 네 오류 조건에서 같은 raw input을 CPU에서 판독해 최종 two-surface 정책 고르기까지만 따로 잰다. 각 block에서 1회 warmup, 7회 paired 반복, 중복되지 않는 random seed와 무작위 method order를 사용한다. 전체 forecast가 포함된 주 GPU p90을 대체하지 않는다. CPU 비교는 별도 Slurm step에서 단일 thread와 숨긴 CUDA로 수행한다. Timing용 대표 fixture는 surface와 교차한 board 전수 중 24 catalog당 각 한 board를 사용하므로, board 8개 및 board 안 catalog 3개 단위의 계층 bootstrap을 보조 보고한다.
