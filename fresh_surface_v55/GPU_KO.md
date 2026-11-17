# v55 · 새 surface·session의 GPU latency와 sensing·reset 비용

## 측정 결과

Slurm allocation 2744471의 node56에서 NVIDIA RTX PRO 6000 Blackwell Server Edition가 사용률 0%, 메모리 0 MiB, 기존 compute PID 0개일 때 시작했다. Guard가 실행 중 새로 들어온 GPU process를 기록하지 않았다. v54에서 사용한 source-risk gate 파일의 SHA-256은 그대로 `b4fbd2ba1d72c664337b2cd6bd0794af023a3901f87d67fa639211f7a779f3ea`이다. Torch 2.7.1+cu128, CUDA 12.8에서 새 catalog 24개, 96개 session을 사용했다. [사전 계획](TIMING_PLAN_KO.md) · [실험 코드](bench55.py) · [환경과 고정 파일](timing_freeze55.sha256)

| 배포 정책 | GPU p90 (ms/512 objects) | Gate-only host p90 (ms/4 sessions) | Gate−각 대조, GPU p90 차이 [simultaneous 95% CI], ms |
|---|---:|---:|---:|
| Frozen source-risk gate | 435.900 | 0.1289 | 기준 |
| Fixed 3R+I | 435.783 | 0.0017 | +0.117 [-0.828, +2.708] |
| Rate-matched random | 435.730 | 0.0172 | +0.170 [-1.460, +3.441] |

각 policy는 같은 새 session에서 첫 세 쌍의 판독값을 받아 네 번째 센서를 정하고, 지불한 네 번째 readout과 과거 측정에 nominal continuous joint를 적용한다. GPU에서 512개 배포 물체의 factual action을 고르고 purchased sensor로 두 contact height를 각각 256개씩 처리한 뒤 signed forecast를 혼합한다. 시간은 host에서 route 판단을 시작하기 직전부터 마지막 GPU 동기화 직후까지의 wall time이다. Target label 계산이나 원시 센서 측정시간, model load는 포함하지 않았다. 동일 decoder의 거동을 따로 저장한 forecast와 모든 timed batch에서 비교했으며, 최대 절대 오차는 0였다. [실측 집계](timing55.json) · [실행 guard](timing_guard55/results/worker_exit.json)

앞의 네 session을 합쳐 512개 물체를 동시에 처리했다. 각 catalog에서 세 policy의 여섯 순열을 무작위 순서로 배치해 실행 위치와 직전 처리한 policy를 균형화했다. Policy당 별도의 첫 warmup 이후 144번 timed call을 실행했다. 차이의 interval은 catalog의 짝을 보존해 16,000회 bootstrap하고 두 대조에 대해 Bonferroni 보정했다. 두 차이의 interval 모두 0을 포함한다. 통계적으로 검출되지 않은 차이는 사전 지정 margin에 대한 latency 동등성의 입증과는 구분한다. [원시 timing 예시](timing55/panel_00.npz)

## 구매율과 소요 시간

전체 held-out prefix의 gate impact 구매율은 42,589/122,880 = 34.6590%다. Gate가 총 몇 번 impact를 사는지 집계하는 데 label은 필요하지 않았다. Random의 Brier 평가는 이 비율로 두 potential loss를 평균한다. Timing에서 random은 96개 session마다 독립 난수를 부여해 gate와 같은 수의 ticket을 미리 배정했다. 여섯 번의 repeat마다 96개 중 정확히 35개가 gate와 random에서 각각 impact를 받았고, fixed 3R+I는 96개 모두 impact를 받았다. Purchase quota는 맞추되 어떤 session이 ticket을 받는지는 그 session의 prefix나 loss와 독립이다. Timed call에서 난수 생성과 cutoff 비교를 다시 수행했다. [구매율 원본](sensing55.json) · [구매율 집계 코드](sensing55.py)

| 교체 후 정책 | Weak / Reference / Impact (session 평균) | Contacts / resets | T=0.52초 | T=1.04초 | T=2.08초 |
|---|---:|---:|---:|---:|---:|
| Frozen source-risk gate | 4 / 3.65341 / 0.34659 | 8 / 8 | 5.740초 | 5.920초 | 6.280초 |
| Fixed 3R+I | 4 / 3 / 1 | 8 / 8 | 5.400초 | 5.920초 | 6.960초 |
| Rate-matched random | 4 / 3.65341 / 0.34659 | 8 / 8 | 5.740초 | 5.920초 | 6.280초 |

한 weak와 reset에 0.44초, reference와 reset에 1.04초를 두었다. T는 impact 시행, 방향 판독, reset, 다음 trial을 시작할 때까지 걸리는 전체 시간이다. Impact 구매율 r에서 평균 시간은 5.92 + r(T−1.04)초/session이다. T=2.08초면 fixed 3R+I가 gate보다 0.680초/session 더 걸린다. 128개 배포 물체당 5.309 ms/object다. Gate와 random의 기대 sensing 시간은 세 가정 모두 같다. T가 reference보다 짧으면 fixed 3R+I의 시간 절감도 가능하다. 교체 전의 공통 8 contacts·8 resets와 각 deployment의 공통 weak contact·reset을 교체 후 calibration 횟수와 분리했다. 실물 T, 센서의 고장과 물체 마모 비용은 측정되지 않았다.
