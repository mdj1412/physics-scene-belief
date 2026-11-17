# v61 · 새 물체의 독립성, contact 계측 비용과 GPU p90

## 새 surface와 outcome

새 catalog 24개, surface instance 48개에 기존 rate law 두 개를 교차해 12,288개 held-out 물체를 만들었다. 교체 전·후 calibration support까지 합한 6차원 parameter vector 36,864개는 서로 다르며, 이전 실험에서 보존한 고유 vector 233,472개와 일치한 것은 없었다. 같은 deployment 물체의 변형 방향을 유지하거나 뒤집어 PyBullet fine·ultra로 각각 trajectory를 생성했다. Residual tangential speed 0.020 m/s 이하를 stick으로 정했다. Active 6,144개 중 2,204개(35.87%)에서 두 방향의 stick label이 달랐다. Inactive 물체에서는 달라진 것이 없었다. Fine/ultra resolution 사이에서는 24,576개 endpoint 중 879개(3.58%)가 달랐다. 물체와 material parameter는 새 draw지만 contact solver와 rate law family는 공유한다. [생성 계획](WORLD_PLAN_KO.md) · [최종 audit](world61.json)

**Holdout 누출을 잡은 과정.** 최초에 선택한 서로 다른 catalog ID 5100~5123은 v60과 100이 더 컸다. Generator에서 calibration과 old calibration의 seed에도 100의 차이가 있어 v60의 old support 5,375개와 같은 calibration vector가 만들어졌다. Deployment 물체 자체의 중복은 0이었다. Policy 비교를 수행하기 전에 이 세트 전체를 제외했고, 새 ID 6000~6023에서 모든 물체, label과 GPU forecast를 다시 생성했다. 최종 support는 제외한 5100~5123의 support와도 겹치지 않는다. 서버에 제외한 세트와 당시 audit 실패·원래 계획을 보관했다. [역할별 중복 개수](support61.json) · [처음의 audit](excluded_5100/audit61.log) · [확정된 dataset checksum](deployment61.sha256)

## 같은 8회 impact 예산에서 측정한 시간

Node55/Slurm allocation 2744468의 NVIDIA RTX PRO 6000 Blackwell Server Edition 한 장을 썼다. 조사 시 다른 두 GPU에는 각각 40 GB 이상의 기존 workload가 있었고, node55/2744468과 node56/2744471은 0 MiB, 0%, compute PID 없음이었다. 두 GPU 중 node55 한 장만 forecast와 timing에 사용했다. 준비, solver, waveform/영상 생성, 통계는 별도 Slurm CPU에서 CUDA를 비활성화해 실행했다. GPU 단계마다 사용 상태를 다시 확인하고 외부 process를 감시했다. [Forecast 종료 기록](generation_guard61/results/worker_exit.json) · [Timing 종료 기록](timing_guard61/results/worker_exit.json)

한 호출에서 2개 surface의 32개 raw 8-frame clips, 각 contact의 두 채널 128-sample waveform, 같은 두 전자센서의 readout과 deployment 관측을 입력해 4개 session, 512개 물체의 stick 확률을 예측한다. Dual reader는 독립 contact switch와 photodiode의 edge, camera ROI를 읽는다. LED-only reader는 camera ROI 시점에 맞춘다. Nominal reader는 고정된 창을 쓴다. 공통으로 부호 인코딩, frozen posterior, loss-weighted 정책 선택, 필요한 session의 v54 gate와 terminal mixture, 최종 GPU forecast를 실행한다. Host wall clock의 시작·종료에서 CUDA synchronization을 수행했다. 상시 gate는 같은 pilot 예산을 부과받아도 의사결정에 쓰지 않는 raw 영상과 waveform 처리를 건너뛴다. 장비 setup·촬영·물리 contact·디스크 읽기·모델 초기화 시간은 별도로 취급한다. [계측 코드](bench61.py) · [사전 계획](TIMING_PLAN_KO.md)

각 catalog에서 네 LED × trigger shift 조건을 각각 측정했다. P/C와 target A/B를 균등하게 구성하고 common fault 발생률의 high/low는 해당 조건에서 A/B 양쪽으로 같은 비중이 되게 배정했다. 6개 정책을 warmup하고 24 catalogs × 4 cells = 96개 대응 block에서 순서·직전 정책을 Williams crossover로 균형화했다. 같은 block에서는 두 surface의 동일한 raw readout, deployment와 board를 쓴다. Timing에서 각 catalog는 하나의 board만 쓰고 8개 board가 각각 세 catalog에서 관측된다. Accuracy/loss 실험에서는 8개 board가 48개 surface 모두와 교차된다. 따라서 timing의 board별 p90은 12 block의 기술통계로 해석해야 한다.

| 정책 | GPU p90 (ms/512 objects) |
|---|---:|
| Dual reset | 489.066 |
| Dual sham | 488.846 |
| 항상 v54 gate | 480.114 |
| LED-only reset | 480.668 |
| LED-only sham | 481.778 |
| Nominal reset | 480.844 |

| Dual reset에서 뺀 대조 | GPU p90 차이 (ms) | Catalog bootstrap 동시 95% CI (ms) | Board 재추출 sensitivity 동시 95% CI (ms) |
|---|---:|---:|---:|
| Dual sham | +0.220 | [-11.807, +13.277] | [-13.767, +13.776] |
| 항상 v54 gate | +8.952 | [-0.999, +16.128] | [-3.424, +17.250] |

6개 정책 모두 표면별 raw code와 terminal probability가 독립 forecast와 일치했다(최대 절대오차 0). 모든 latency bin의 실행 위치 및 직전 정책 조합은 각각 16회였다. CI는 24개 catalog 16,000회 paired bootstrap으로 두 비교를 보정했다. Timing에서 board와 catalog가 중첩되는 점을 고려한 부가 분석에서는 8개 board와 해당 board당 3개 catalog를 계층적으로 8,000회 재추출했다. 동일 block 내부 dual reset − sham/gate의 paired median 차이는 각각 +0.198, +1.037 ms였다. [End-to-end 결과](timing61.json) · [Board cluster sensitivity](timing_board61.json) · [사전 비교와 보조 분석의 구분](TIMING_BOARD_KO.md)

별도 CPU microbenchmark에서 두 표면(32 clips)의 pixel·waveform 판독부터 outer routing까지 576회를 측정했다. Node56의 p90은 dual reset **0.5642 ms**, dual sham **0.5629 ms**, LED-only reset **0.4066 ms**, 상시 gate **0.00146 ms**였다. 같은 reset에서 dual로 바꾼 preprocessing의 p90 차이는 **+0.1576 ms** [pointwise 95% CI +0.1523, +0.1639]다. 이 시간은 node55 end-to-end GPU latency와 별도로 측정했으므로 두 수치를 합산할 수 없다. [Reader timing](reader_cost61.json)

## Sensing과 reset 장부

한 surface에서 4개 session, 서로 다른 물체 8개, **8 impacts, 8 references, 16 mechanical resets**를 모든 정책에 부과했다. 실제 reset arm은 둘째 물체 전 무작위로 배정된 센서에 **4 electronic resets**가 추가된다. Dual과 LED-only 정책 모두 같은 16 clips(128 frames, 51,840 bytes)를 읽었다. Dual의 추가 관측은 contact switch와 독립 photodiode 각각의 16개 waveform이다. 16 contacts × 2 channels × 128 uint8 samples = **4,096 bytes**이며 별도의 impact를 지불하지 않는다. Hardware, 배선, ADC, off-camera 기준시계와 trigger의 설치 및 장비 대기 비용은 측정하지 않았다. [원시 관측별 비용](cost61.json)

독립 hazard 모집단에서 후속 네 번째 deployment observation으로 impact를 구매한 비율은 dual reset **35.763%**, dual sham **35.738%**, 상시 gate **34.931%**였다. 이전 실험과 같은 가정으로 첫 deployment의 기계·sensing 시간을 모델링하면, reference와 impact 각각 reset 포함 1.04초에서 세 정책 모두 22.56초다. Impact만 2.08초면 같은 capacity budget이 31.92초이고 실제 구매율로 계산한 padding 전 기대 시간은 각각 약 31.252, 31.252, 31.243초다. 장비에 별도 비중첩 reset 비용 R, surface당 독립 타이밍 회로 준비 S가 있고 같은 pilot을 H개 session에 재사용하면 dual reset에는 추가 (4R+S)/H, dual sham에는 S/H가 적용된다. 상시 gate는 실제 시스템에서 목적에 따라 사용하지 않는 사전 pilot 자체도 생략할 수 있다. [모형 및 조건별 costing](cost61.py)

## 재현 순서

서버의 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/switch_diode_v61`을 기준으로 sibling project의 frozen decoder와 v59 likelihood를 재사용한다. 사용 가능한 Slurm compute allocation 안에서 `bash run_world61.sh prepare` 뒤 CPU로 `bash run_world61.sh solver`, 비어 있는 GPU에서 `bash run_world61.sh forecast`를 수행한다. 양쪽이 끝나면 CPU로 `bash run_world61.sh panel`, `bash run_world61.sh audit`를 실행한다. 24개 paid pilot·fixture가 준비된 상태에서 비어 있는 GPU로 `bash run_timing61.sh`를 실행한다. [GPU guard와 검산](run_timing61.sh) · [World runner](run_world61.sh) · [원본 seed 충돌 검산](verify_support61.py)

`timing61/panel_00.npz` 등 catalog 단위 파일에는 개별 block의 6개 경과시간, 실제 outer 선택, 구입한 deployment impact, crossover 실행 순서, timing cell·board audit index와 독립 replay 오차가 저장된다. [첫 catalog의 timing 기록](timing61/panel_00.npz)
