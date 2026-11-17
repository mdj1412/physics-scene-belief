# v62 보조 검증 · 실제 다음 관측이 바뀔 때의 GPU p90

Primary의 무작위 timing sample에서는 96개 block 모두 selective reset, v61 reset·sham, 상시 v54 gate가 같은 후속 R/I를 구입했다. 신뢰도 보류가 행동을 실제로 바꾸는 경우의 비용을 별도로 확인했다. 세 번째 관측까지의 유료 deployment reference로 v54를 실행하고, 같은 표면의 같은 8회 impact에 대한 selective reset 및 v61 reset의 기록된 action code를 비교했다. 정확히 한 표면에서 outer 정책이 달라지고 후속 네 session 중 한 개 이상에서 실제 네 번째 R/I 구매가 바뀐 경우만 eligible로 두었다. Brier·stick label·solver 결과·GPU 시간은 표본 선정에 쓰지 않았다. [사전 기준](RARE_TIMING_PLAN_KO.md) · [선정 코드](select_rare62.py)

Primary의 장비 조건 및 숨은 원인·환경의 사전 배정을 그대로 사용해 24 catalogs × 4 cell × 512 paired pilot을 검색했다. 49,152개 (catalog, cell, episode) 조합 중 96개가 이 조건을 충족했고, 서로 다른 catalog 16개에서 episode 한 개씩을 선택했다. 네 pulse/exposure cell에는 각각 3·3·5·5개 catalog가 배정됐고 8개 board 중 7개가 포함됐다. 실제 selection에서는 selective가 v61의 fixed 선택을 gate로 바꾼 경우 13개, gate에서 fixed로 바꾼 경우 3개였다. Gate로 바뀌는 후보 92개에 비해 fixed로 바뀌는 후보는 3개 catalog의 4개뿐이었다. 선택한 3개가 서로 다른 catalog를 유지하는 조건에서 가능한 최대치다. [조건별 후보](rare_eligible62.json) · [선택한 catalog와 episode](rare_selection62.json)

선정한 16개 episode의 원래 seed로 같은 물체, 전자센서, switch·photodiode와 영상 pixel을 다시 만들었다. GPU 실행은 이 원본 raw 자료부터 시작해 모든 policy의 표면별 outer action, 실제 후속 관측, terminal mixture 및 512개 물체의 GPU forecast를 독립 저장 code·forecast와 검산했다. 각 catalog에서 세 정책을 한번 warmup하고 두 번 측정했으며 두 번째 실행에서는 첫 실행의 정책 순서를 뒤집었다. 실행 직전 node55/2744468의 GPU가 0 MiB·0%·compute PID 없음임을 다시 확인했다. [Benchmark](rare_bench62.py) · [실행 기록](rare_guard62/results/worker_exit.json)

| 정책 | Rare-switch cohort GPU p90 (ms/512 objects) |
|---|---:|
| Selective reset | 555.024 |
| V61 reset | 553.494 |
| 항상 v54 gate | 555.187 |

16개 catalog의 네 deployment session씩 64개 중 실제 다음 관측이 바뀐 session은 23개(35.94%)였다. 측정한 32개 paired block에서 selective reset의 p90은 v61 reset보다 **1.529 ms 높았고** gate보다 **0.163 ms 낮았다**. Block별 paired time difference의 median은 각각 +1.392 ms, −1.328 ms였다. 세 정책의 최대 forecast replay 오차는 0이었다. 이 별도 실행에서 세 정책의 절대 p90은 모두 553~555 ms였다. 실행 시점과 선택된 물체·표본이 모두 바뀌었으므로 주 실험의 약 480 ms와의 차이를 switch가 초래한 비용으로 계산할 수 없다. [전체 timing·실제 구매](rare_timing62.json) · [Frozen benchmark checksum](rare_freeze62.sha256)

이 p90은 행동이 바뀌는 사례를 의도적으로 모은 표본에서의 관측값이다. 16개 catalog의 2회 측정을 실제 장비 운영에서의 32개 독립 환경으로 세지 않는다. 원래 무작위 timing sample의 p90에 합산하지 않았으며, 세부 방향별 latency 우위·실물 sensing 시간·모집단 tail 개선을 판정할 근거로 사용하지 않는다.
