# v62 · 새로운 deployment와 GPU·sensing 비용 검증

## 새 물체·표면과 공통 decoder

새 catalog 8000~8023마다 두 새로운 surface instance와 기존 두 rate law, material × law당 128개 물체를 배정했다. 총 12,288개 held-out deployment object와 교체 전·후 support를 포함한 6차원 parameter vector는 36,864개다. GPU forecast 전 이들 모두가 서로 고유하고, v47·48 및 v50~61의 보존된 호환 archive와 v61에서 제외한 5100번대 세트까지 합한 313,744개 고유 vector와 겹치지 않는지 확인했다. Material parameter·board는 새로운 draw이며, contact engine과 기존 두 law family는 재사용했다. [사전 계획](WORLD_PLAN_KO.md) · [검사 대상 archive와 개수](support62.json) · [Generator](generate62.py)

같은 물체에서 변형 방향을 유지하거나 뒤집은 PyBullet trajectory 두 개를 만들었다. 최종 residual tangential speed가 0.020 m/s 이하면 stick이다. 6,144개 active object 중 2,236개(36.39%)에서 변형 방향을 바꾸자 stick/slip이 뒤바뀌었다. Inactive 물체의 변화는 0건이었다. 24,576개 endpoint 중 820개(3.34%)에서 fine/ultra solver resolution에 따라 stick label이 달랐다. Source에서 동결한 decoder의 양쪽 예측, v54 gate·fixed 3R+I의 측정, 두 binary replay(seed 62041729와 62084591)를 같은 물체에 적용했다. [World audit](world62.json) · [168개 archive checksum](deployment62.sha256)

## 비용을 측정한 단위

새 실험 전 Slurm allocation 네 곳의 UUID, 사용 memory, utilization, compute PID를 확인했다. 두 allocation에서는 기존 Python 작업이 약 40 GB씩 쓰고 있었다. 두 idle allocation은 0 MiB·0%·compute PID 없음이었고, node55/2744468의 NVIDIA RTX PRO 6000 Blackwell Server Edition 한 장에 forecast와 inference benchmark를 배정했다. 각 GPU 단계 직전에 사용 상태를 다시 확인하고 guard로 다른 compute process 출현을 감시했다. Solver·영상/파형 생성·통계에는 Slurm compute CPU와 CUDA 비활성화를 사용했다. [Forecast guard 종료](generation_guard62/results/worker_exit.json) · [Timing guard 종료](timing_guard62/results/worker_exit.json)

**End-to-end GPU p90**은 두 surface에서 미리 구입해 memory에 놓인 32개 raw clip과 각 contact의 두 ADC trace부터 시작해 512개 deployment object의 stick probability를 만드는 실제 시간이다. Pixel과 waveform 판독·abstention, 전자센서 encoding, outer selection, 필요한 session에서 inner v54 gate, terminal mixture와 GPU forecast를 포함한다. Host clock 전후에 CUDA를 synchronize했다. 8개 board × 3개 catalog로, catalog마다 4 pulse/exposure cell을 모두 사용했다. 6개 arm에 각각 warmup을 하고 정책당 96회 측정했다. 같은 block에서는 raw evidence, 전자센서 reset 배정, 후속 물체와 terminal을 공통으로 쓰며 실행순서·직전 arm 빈도를 Williams crossover로 균형화했다. Acquisition·기계 동작·파일 읽기·모델 초기화는 이 p90에서 제외했다. [사전 계측 계획](TIMING_PLAN_KO.md) · [Benchmark 코드](bench62.py)

| 정책 | 96개 block GPU p90 (ms/512 objects) |
|---|---:|
| Selective reset | 480.932 |
| V61 reset | 480.983 |
| V61 sham | 480.828 |
| 항상 v54 gate | 480.108 |
| Selective sham | 481.173 |
| Nominal reset | 480.623 |

| Selective reset − 비교 방법 | p90 차이, ms | Paired catalog 동시 95% CI, ms | Board cluster 동시 95% CI, ms |
|---|---:|---:|---:|
| V61 reset | -0.050 | [-13.667, +0.977] | [-14.546, +1.209] |
| V61 sham | +0.105 | [-1.148, +1.037] | [-12.447, +1.521] |
| 항상 v54 gate | +0.824 | [-0.172, +1.639] | [-6.142, +2.060] |

Selective reset의 GPU p90과 세 대조군 간 차이에 대한 동시 95% CI가 모두 0을 포함했다. 주 CI는 catalog 24개 16,000회 paired bootstrap이다. Timing에는 board마다 3개 catalog가 nested되어 있어 추가로 8개 board 및 각 board 내 3개 catalog를 단계적으로 재표집해 8,000회 비교했다. 정확도 실험에서는 8개 board와 표면 48개가 완전 교차한다. 모든 arm에서 저장된 code·결정 및 GPU forecast replay가 일치했고 최대 absolute forecast error는 0이었다. [GPU timing](timing62.json) · [Board 민감도](timing_board62.json)

두 표면의 outer 행동과 뒤이은 네 session의 sensing 선택이 모두 같은 block은 selective reset 대 v61 reset·v61 sham·gate에서 각각 96, 96, 96개(각 96개 중)였다. 각 reader는 실제 raw 자료를 판독했다. 전체 loss 표본에서 드물게 나타난 행동 전환의 latency는 이 96개 block으로 따로 측정할 수 없었다. Paired block별 시간 차이의 median은 각각 -0.009, -0.182, +0.659 ms였다. 같은 행동에 대한 기술통계이며 알고리즘별 GPU kernel 시간이나 인과적 latency 분해로 확대할 수 없다. [경로 감사](route_timing62.json) · [검산 기준](ROUTE_TIMING_PLAN_KO.md)

별도 CPU microbenchmark에서 동일한 32개 clip의 판독과 outer routing만 정책당 576회 실행했다. Node56 p90은 selective reset 1.2619 ms, v61 reset 1.2366 ms, v61 sham 1.2580 ms, 항상 gate 0.00220 ms였다. Selective reset − v61 reset 차이는 +0.0253 ms [pointwise 95% CI −0.1412, +0.2280]였다. 이 측정은 별도 node에서 이루어졌으므로 GPU p90에 더하거나 특정 GPU kernel의 시간으로 해석할 수 없다. [CPU 측정 결과](reader_cost62.json) · [Raw microbenchmark](reader_cost62_raw.npz)

## LED, shutter, sensing·reset 장부

모든 방법이 표면당 4개 session과 서로 다른 물체 8개에서 **8 impacts + 8 references + 16 mechanical resets**를 지불했다. 실제 reset arm에만 4 electronic resets가 추가된다. Selective와 기존 v61 arm은 동일한 16개 raw camera clips(128 frames, 51,840 bytes)와 32개 ADC traces(4,096 bytes)를 사용한다. 보류된 sign은 missing으로 인코딩되고, 이미 구입한 contact의 비용은 그대로이며 재촬영하지 않는다. Gate에도 같은 pilot 예산을 부과하지만 의사결정에 사용하지 않는 영상·파형은 online에서 처리하지 않는다. [조건별 비용 장부](cost62.json)

각 cell 안에서 정책 간 sensing 일정을 맞췄으나 pulse와 exposure 조건 간에는 물리량이 다르다. 표면당 16번 발광하므로 0.20/0.80 ms pulse의 programmed LED on-time은 각각 3.2/12.8 ms다. 동일한 광출력을 가정하면 긴 pulse의 광에너지는 4배다. 0.25/0.80 ms exposure는 128개 frame 전체에서 shutter-open time이 32.0/102.4 ms다. 이것은 이미 할당된 frame 주기 안의 적분시간이며 추가 wall time이나 측정 전력을 의미하지 않는다. 실제 장치의 광출력·전력·열·trigger 반응은 측정하지 않았다. [비용 산출식](cost62.py)

Independent-hazard 모집단의 후속 deployment에서 네 번째 관측으로 impact를 구매한 비율은 selective reset 35.0880%, v61 reset 35.1461%, v61 sham 35.1211%, v54 gate 34.9186%였다. Reference 및 impact 비용을 reset 포함 1.04초로 두는 기존 모형에서는 첫 deployment까지 22.56초다. Impact만 2.08초일 경우 공통 worst-case budget은 31.92초이고 실제 구매율을 적용한 기대시간은 각각 31.2449, 31.2455, 31.2453, 31.2432초다. 전자 reset·clock setup이 다른 작업과 겹치지 않으면 실물에서 해당 시간을 별도로 측정해야 한다. [Sensing·reset 장부](cost62.json)

## 재현

서버의 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/exposure_abstention_v62`에서 동일 project tree의 frozen decoder, v54 gate, v59 router 및 v61 reader를 사용한다. CPU allocation에서 `bash run_world62.sh prepare`와 `bash run_world62.sh preaudit`를 순서대로 실행한다. 통과하면 CPU에서 `bash run_world62.sh solver`, 실제로 비어 있는 GPU에서 `bash run_world62.sh forecast`를 실행한다. 이후 CPU에서 `panel`, `audit` 단계를 실행한다. 24개 영상 pilot이 준비되면 빈 GPU에서 `bash run_timing62.sh`를 실행한다. 각 catalog의 파일에는 실제 경과시간, 실행 순서, 물리 조건, 보드, 선택, 최대 replay 오차가 담겼다. [World runner](run_world62.sh) · [GPU runner](run_timing62.sh) · [Timing 원자료 예시](timing62/panel_00.npz)
