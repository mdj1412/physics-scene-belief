# v56 · Calibration impact를 포함한 GPU 및 sensing 비용

## Endpoint

한 surface에서 8개 pilot reference–impact 쌍을 지불한 뒤, 세 정책은 같은 교체 후 네 쌍과 동일한 continuous joint decoder를 사용한다. Slurm node56의 RTX PRO 6000 Blackwell(2744471)이 0 MiB·0%·compute PID 0개인 점을 확인하고 guard에서 실행했다. Fresh 24 catalog에서 48개 material마다 1개의 pilot을 뽑고, material 안의 두 law 각각에 deployment session 한 개를 배정했다. Material의 high-mismatch 비중은 .35 또는 .52로 균등 배정했다. w는 방향이 절반의 확률로 어긋나는 session의 모집단 비중이다. Pilot이 받은 개별 부호 불일치율은 별도로 계산한다. 각 catalog의 네 session이 512개 물체를 구성한다. [측정 계획](TIMING_PLAN_KO.md) · [실행 코드](bench56.py)

Switch timing은 읽힌 reference와 impact 부호에서 유효 pair 및 불일치를 직접 세어 w를 추정하고 fixed I 전환을 결정하는 순간부터 시작한다. V54 gate를 유지하는 session에서만 첫 세 쌍으로 네 번째 센서를 다시 고른다. 두 대조도 동일한 물리 pilot을 받으며 어떤 센서를 살지 정할 때 해당 측정은 무시한다. 그 뒤 구매한 네 번째 readout만 읽어 terminal mixture를 갱신하고 GPU에서 구매한 object별 sensor를 사용해 signed forecast 두 개와 최종 확률을 계산한다. CPU wall clock 양끝에서 CUDA를 동기화한다. Sensor synthesis, 물리 capture·reset, model initialization은 제외한다. 24개 catalog마다 세 정책 실행 순서 여섯 순열을 균형화하고 각각 1회 warmup을 제외했다. 최종 paired interval은 16,000회 catalog bootstrap에 두 대조의 Bonferroni 보정을 적용한다.

## Synchronized GPU p90

| Policy | GPU p90 (ms/512개) | CPU 선택 p90 (ms/4개 session) | Switch−대조 GPU p90 차이 [동시 95% CI], ms |
|---|---:|---:|---:|
| 8-pilot switch | 433.370 | 0.1355 | — |
| Frozen v54 gate | 433.598 | 0.1259 | -0.228 [-0.965, +0.468] |
| Fixed 3R+I | 433.193 | 0.0022 | +0.177 [-1.992, +0.788] |

환경은 NVIDIA RTX PRO 6000 Blackwell Server Edition, PyTorch 2.7.1+cu128, CUDA 12.8였다. 정책당 144개의 batch에서 측정했다. Timing fixture의 96개 deployment session 중 pilot이 fixed로 전환한 수는 38개다. 최종 impact를 구입한 수는 switch·v54 gate·fixed 순서로 57, 32, 96개였다. 별도로 생성한 GPU forecast와 매 timed call을 비교한 최대 절대 오차는 0이었다. Guard가 확인한 다른 GPU process는 0개였다. [Timing 집계](timing56.json) · [Raw batch 예시](timing56/panel_00.npz) · [Guard 결과](timing_guard56/results/worker_exit.json)

두 GPU p90 차이 모두 95% interval이 0을 포함한다. 이 표본으로 latency 우열 또는 사전 margin을 만족하는 동등성은 확정할 수 없다.

## 같은 contact 수와 같은 경과시간

Pilot 한 쌍당 2.0 N reference 한 번과 같은 자리의 impact 한 번, 각 접촉 뒤 reset 한 번이 필요하다. 방향을 읽을 수 없는 경우도 다시 시도하지 않고 비용을 청구한다. 교체 뒤 K개 pilot에 2K contacts와 2K resets, deployment calibration에 session당 8 contacts와 8 resets가 든다. K=8 pilot 한 번 후 session 하나를 운영하면 **24 contacts·24 resets**, 두 law session이 같은 pilot을 쓰면 합계 **32 contacts·32 resets**다. 공통 교체 전 history의 8 contacts·8 resets와 deployment 물체 각각의 공통 weak 접촉은 비교에서 따로 둔다.

Weak+reset을 0.44초, reference+reset을 1.04초, impact+reset을 T초로 두고 H개 session이 surface pilot을 공유한다고 가정한다. K, H와 네 번째 I 구매율 r에서 평균 비용은 다음과 같다.

- Pilot당 비용은 K(1.04+T)초다.
- Pilot을 포함한 deployment session당 평균은 K(1.04+T)/H + 4.88 + (1−r)1.04 + rT초다.
- 동일 wall-clock 예산을 맞추려면 세 정책의 네 번째 slot을 긴 측정에 맞춘다. 이때 공통 비용은 K(1.04+T)/H + 4.88 + max(1.04,T)초다. 짧은 측정을 한 session은 차이만큼 대기하고, 대기 중 추가 관측을 구입하지 않는다. 모델링한 이 대기시간은 판독값이나 loss를 바꾸지 않는다.

K=8, 주 평가 w=.35/.52에서 switch의 외부 fixed I 전환율은 46.596%, 최종 I 구매율은 switch 65.422%, frozen gate 35.251%, fixed 3R+I 100%였다. 같은 surface 내 pilot과 deployment가 조건부 독립이므로 각 material의 전환율과 gate 선택률에서 I 구매율을 계산한 뒤 material 및 w를 균등 평균했다. [Label-free ledger](sensing56.json) · [계산식과 검산](sensing56.py)

| Pilot당 deployment session H | Impact+reset T | Switch 대기 전 | Frozen gate 대기 전 | Fixed 3R+I 대기 전 | 동일 시간 예산 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.52 | 18.060초 | 18.217초 | 17.880초 | 18.400초 |
| 1 | 1.04 | 22.560초 | 22.560초 | 22.560초 | 22.560초 |
| 1 | 2.08 | 31.560초 | 31.247초 | 31.920초 | 31.920초 |
| 2 | 0.52 | 11.820초 | 11.977초 | 11.640초 | 12.160초 |
| 2 | 1.04 | 14.240초 | 14.240초 | 14.240초 | 14.240초 |
| 2 | 2.08 | 19.080초 | 18.767초 | 19.440초 | 19.440초 |

이는 장비에서 stopwatch로 잰 시간이 아니다. Impact 뒤 실제 reset이 더 오래 걸리면 세 센서 경로의 공통 slot이 길어진다. 두 상수 정책이 현장에서 pilot 자체를 생략할 수 있다면 절약되는 시간은 공통 pilot 비용 K(1.04+T)/H초/session이다. v56의 같은-budget 대비 Brier 차이만으로 이 pilot을 새로 구입하는 것이 실제로 가치 있는지 결정할 수 없다.
