# v52 · 표면 교체 뒤 calibration 정책의 GPU latency와 물리 비용

## 측정 경로

같은 RTX PRO 6000 Blackwell에서 policy마다 네 환경의 기존 네 쌍과 새 네 쌍을 받아 session 결정을 처음부터 계산했다. 각 catalog에서 배포 물체 512개(환경당 128개)의 factual acquisition, 구매한 sensor profile 처리, original decoder, 선택된 signed mixture 또는 unsigned predictor를 실행하고 전후 GPU synchronization을 포함한 wall time을 쟀다. 네 정책이 같은 calibration 관측과 같은 물체·관측 noise를 받는다. 교체 전과 후의 물체는 제공된 서로 다른 surface 행에서 뽑으며, test fixture의 실제 retention과 drift indicator는 정책에 들어가지 않는다. 보관한 forecast로부터 session을 하나씩 별도 재생한 값과 온라인 prediction을 대조한다.

정책당 12개 catalog × 8회, 총 96개 512-object timed batch를 사용한다. Catalog마다 warmup 후 두 번의 Williams crossover에서 실행 순서와 직전 정책을 균형화한다. Catalog를 cluster로 16,000회 paired bootstrap한다. Likelihood selector 대비 동일한 세 action의 vote selector와 단순 post-only vote soft 간 두 timing 차이는 각각 97.5% percentile interval을 사용해 simultaneous 95% CI로 보고한다. 모델 계수와 posterior bank는 메모리에 올린 뒤 동일하게 재사용하며, sensor 물리 동작·표면 교체·physics label 생성은 이 시간 밖에 있다.

## 배포 GPU p90

| 정책 | ms/512 objects [pointwise 95% CI] |
|---|---:|
| Likelihood 3-action selector | 468.12 [467.70, 473.56] |
| Simple vote 3-action selector | 467.28 [467.01, 467.75] |
| Post-only vote soft | 467.49 [466.80, 469.65] |
| Always unsigned | 467.50 [466.85, 469.48] |

Likelihood selector에서 sign vote selector를 뺀 p90 차이는 +0.84 ms [동시 95% CI +0.06, +9.55], post-only vote와의 차이는 +0.63 ms [−2.20, +9.87]였다. 새 estimator가 더 빨라지는 효과는 관찰되지 않았다. 실제 hardware 환경 한 장의 측정값이고, 같은 표면에서 특정 action을 선택한 조건별 latency는 별도로 추정하지 않았다.

네 환경의 posterior update와 action 선택만 수행한 K=4 host p90은 likelihood selector 0.10174 ms, vote selector 0.06072 ms, post-only vote 0.01324 ms였다. K=0·1·2·4·8에서 likelihood host p90은 0.09676·0.10246·0.10098·0.10174·0.10617 ms였다. 측정 fixture 48개 session에서 keep/refit/unsigned 횟수는 likelihood 8/16/24, vote 12/15/21이었다. 이는 latency가 세 경로를 모두 지났는지 확인하는 수치이며, 정확도 평가 전체의 action 빈도는 별도로 집계했다.

Node56, allocation 2744471, Torch 2.7.1+cu128 및 CUDA 12.8에서 완료했다. GPU worker는 시작 시 0 MiB·0%와 빈 process 목록을 확인했고 foreign process를 감지하지 않았다. 모든 온라인 예측의 최대 replay error는 4.44×10⁻¹⁶이었다. [측정 결과](timing52.json) · [Batch별 원자료](timing52) · [GPU guard 완료 기록](timing_guard52/results/worker_exit.json)

## Sensing과 reset 예산

환경당 기존 네 쌍은 8 contacts와 8 resets를 이미 지불한 공통 관측이다. 교체 후 K쌍을 더 얻으면 2K contacts와 2K resets를 추가한다. 배포 물체당 query-local weak preload 한 번과 reset 한 번이 따로 들어간다. 교체 후 네 쌍은 환경당 8 contacts와 8 resets이며, 교체 전 예산까지 합치면 환경당 calibration 16 contacts, 16 resets가 된다. Selector는 keep이나 unsigned를 고르더라도 이미 구입한 측정 비용을 지불한다. Always-unsigned는 post probe를 처음부터 생략할 수 있다. Post-only vote는 기존 calibration을 저장하지 않은 환경에서도 같은 예측을 계산할 수 있다.

시간은 접근 100 ms, 측정 50 ms, 후퇴 100 ms, 안정화 40 ms, loading 4 N/s와 unloading 8 N/s로 둔 시나리오다. Weak 0.4 N은 440 ms, reference 2.0 N은 1,040 ms이므로 calibration 한 쌍에 1.48초다. K=1·2·4·8에 따라 환경당 추가 시간은 1.48·2.96·5.92·11.84초다. K=4에서 128개에 공통 적용하면 교체 후 추가 46.25 ms/object로, 개별 deployment touch까지 486.25 ms/object다. 기존 네 쌍까지 새로 취득하는 전체 cycle이라면 532.5 ms/object다. 표면 교체, robot travel, reset failure와 잔류 변형은 미계측이다.
