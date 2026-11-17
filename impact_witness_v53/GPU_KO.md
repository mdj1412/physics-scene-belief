# 교체 후 impact 관측의 sensing·reset 비용과 GPU latency

## 같은 예산에서 어떤 측정이 달라지는가

교체 전 공통 예산은 weak 0.4 N + reference 2.0 N 네 쌍이다. 교체 후 mixed protocol은 처음 세 쌍을 그대로 유지한다. 네 번째 물체에서는 weak 접촉 후 reset하고, 같은 위치에서 impact를 일으켜 변형 방향을 한 번 읽은 뒤 reset한다. 기존 네 번째 reference를 생략한다. 배포 시에는 calibration 물체와 별개인 128개 물체에 query-local weak 접촉을 각각 한 번 쓴다. 교체 후 표의 수량은 한 session 기준이다.

| Post-swap protocol | Weak contacts | Reference contacts | Impact contacts | Contacts / resets | 전체 calibration time |
|---|---:|---:|---:|---:|---:|
| Reference four | 4 | 4 | 0 | 8 / 8 | 5.92 s |
| Three reference + one impact | 4 | 3 | 1 | 8 / 8 | 4.88 s + T |

T는 impact 시행, 방향 판독과 reset까지 마치는 시간이다. 같은 mixed protocol을 받는 continuous joint, 세 행동 selector, sign vote 사이에는 장치 속도와 무관하게 sensing과 reset 차이가 없다. 두 protocol을 비교할 때 **횟수와 소요 시간은 별개의 예산**이다. v52 동작 시간 가정에서 weak 접촉 한 번은 0.44 s, reference는 1.04 s다. 같은 길이 1.04 s의 네 번째 slot을 예약하면 T가 이 안에 들어오는 장치에서만 같은 시간 예산이 성립한다. T가 더 작으면 남는 시간을 대기 시간으로 처리할 수 있다. 이 시뮬레이션은 고속 센서로 부호를 얻는 실제 시간, 에너지, 마모, 표면 교체, specimen handling을 측정하지 않았다. T에 판독 지연과 안정화를 모두 포함해야 한다.

| T 가정 | Mixed session | Reference four와 차이 | Mixed 추가 시간 / 배포 물체 |
|---:|---:|---:|---:|
| 0.52 s | 5.40 s | −0.52 s | 42.19 ms |
| 1.04 s | 5.92 s | 0.00 s | 46.25 ms |
| 2.08 s | 6.96 s | +1.04 s | 54.38 ms |

등시간 가정에서 두 방식은 교체 전후 합계 16 contacts/16 resets, 11.84 s다. 새 calibration의 session 비용과 물체당 0.44 s deployment touch를 합하면 486.25 ms/object다. 교체 전 네 쌍도 이 session에 전부 청구하면 532.50 ms/object다. Impact 방향 판독은 물리 slot에 포함한다. GPU 시간은 물리 slot 종료 후 지불하는 별도 비용이다.

## GPU 측정 경로

같은 서버 GPU에서 12 catalog를 사용한다. 각 timed batch는 calibration 기록이 서로 다른 네 session, 각 session의 배포 물체 128개로 이루어진다. 계수와 GPU decoder는 미리 올리지만, 매 batch마다 old/new calibration의 raw readings부터 posterior, 행동 또는 혼합 계수를 다시 계산한다. GPU에서 factual acquisition과 original physics decoder를 실행하고, 같은 약한 접촉을 해당 정책의 signed mixture 또는 unsigned forecast에 반영한다. 처음과 끝의 CUDA synchronization을 포함한 wall time을 GPU p90으로 보고한다. Host에서 측정하는 posterior 계산 시간도 별도로 기록한다. 물리 slot과 impact camera 또는 센서의 raw image 처리 시간은 GPU timing 밖에 있다. GPU가 받는 것은 이미 판독된 한 개의 ternary gauge sign이다.

각 catalog에서 각 정책의 warmup 한 번 뒤 8-arm Williams crossover를 한 번 완주해 96개 512-object batch를 정책별로 얻는다. 각 위치에 각 정책이 한 번씩, 각 선행–후행 조합도 한 번씩 나타나도록 순서를 균형화한다. Catalog를 단위로 16,000회 paired bootstrap하여 marginal p90 차이의 interval을 계산한다. Mixed continuous joint에서 mixed selector, 같은 세 행동의 vote, post-only vote를 각각 뺀 세 비교는 Bonferroni 동시 95% interval을 쓴다.

정확한 측정 경로를 확인하기 위해, 매 batch의 실제 GPU 출력을 보관된 independent-object forecast와 단일 session씩 재계산한 policy weights로 조합한 예측에 대조한다. Mixed 정책을 호출할 때 숨겨진 네 번째 reference 값을 +999 mm 교란한 후 action과 mixture weight가 그대로인지 검사한다.

별도의 optical diagnostic에서는 실제로 렌더링한 네 session의 uint8 line scan(각각 2시점 × 2마커 × 49픽셀)을 입력했다. Server CPU의 centroid와 differential-displacement 방향 판독은 8,640개 four-session batch에서 median **0.02751 ms**, p90 **0.03549 ms**였다. Trigger, exposure, sensor readout과 hardware reset은 포함하지 않았다. 이 microbenchmark와 GPU 측정의 p90을 더한 값은 end-to-end p90을 보장하지 않는다. [Line-scan 측정 코드](scan_latency53.py) · [수치](scan_latency53.json)

예를 들어 T=2.08 s이고 128개 물체에 비용을 분담하면 네 번째 reference 대신 impact를 쓰는 추가 비용은 0.008125 s/object다. 평균 Brier를 목적함수로 쓸 때, continuous joint의 sensor 교체 이득 0.00000740은 대기 시간 1 s를 Brier 0.00091 이하로 평가할 때에만 이 시간 비용을 상쇄한다. 세 행동 selector의 sensor 교체 이득 0.00006138을 같은 방식으로 나누면 0.00755 Brier/s다. 이는 지정한 T와 본 catalog 평균의 조건부 환산이며, 실제 생산 비용은 별도 측정해야 한다.

## GPU p90 결과

| 관측·정책 | GPU p90 ms/512 [pointwise 95% CI] | Cold posterior host p90 ms/4 sessions |
|---|---:|---:|
| 3R+I · continuous likelihood | 461.54 [460.96, 464.24] | 0.14959 |
| 3R+I · v52-style selector | 461.68 [461.16, 464.75] | 0.16734 |
| 3R+I · same-action sign vote | 461.87 [461.07, 464.74] | 0.12289 |
| 3R+I · post-only simple vote | 461.45 [460.41, 464.73] | 0.02119 |
| 3R+I · impact sign masked | 461.38 [460.50, 463.55] | 0.13690 |
| 4R · continuous joint | 461.34 [460.72, 463.57] | 0.13652 |
| 4R · frozen v52 selector | 461.26 [460.91, 464.13] | 0.10829 |
| 4R · frozen vote selector | 461.41 [461.00, 464.25] | 0.06766 |

같은 3R+I continuous joint에서 selector·same-action vote·post-only vote를 뺀 p90 차이는 차례로 **-0.14 ms** [-1.07, +1.70], **-0.32 ms** [-1.55, +0.42], **+0.09 ms** [-1.95, +0.94]였다(동시 95% CI). 세 차이 모두 zero를 포함한다. 이 batch와 장치에서 빠르다고 단정할 수 없다.

Node56의 Slurm allocation 2744471, NVIDIA RTX PRO 6000 Blackwell Server Edition, Torch 2.7.1+cu128, CUDA 12.8에서 실행했다. 첫 검사에서 메모리 0 MiB, 이용률 0%, 다른 compute process가 없었고 guard는 foreign process 없이 정상 종료했다. 모든 방법의 maximum replay error는 4.44×10⁻¹⁶이었다. [측정 결과](timing53.json) · [각 batch의 raw timing](timing53) · [GPU guard 기록](timing_guard53/results/worker_exit.json)

### 보조 대조 · continuous blending을 유지한 sign count

같은 GPU에서 12 catalog마다 두 정책을 역순으로 교차해 각각 48회 측정했다. 3R+I continuous likelihood 465.70 ms와 continuous sign-count mixture 465.40 ms를 비교한 차이는 +0.30 ms [95% paired catalog bootstrap -1.37, +1.37]였다. Raw witness부터 같은 인코딩·decoder·최종 혼합까지 실행했고 sign-count weights는 평가 캐시와 일치했다. 이 두 수치는 같은 보조 run 안에서 비교하며, 위 8-arm run의 시계값과 직접 차감하지 않는다. [보조 GPU 수치](vote_timing53.json) · [보조 raw timing](vote_timing53.npz)
