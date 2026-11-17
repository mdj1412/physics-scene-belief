# v59 · Fresh physics, sensor reset 예산과 GPU p90

## 새 표면과 물체

새 catalog key 4000–4023에서 서로 다른 surface instance 두 개와 rate law 두 종류를 교차하고 각 환경에서 128개 물체를 평가했다. 배포 물체 12,288개, 교체 전·후 support까지 포함한 6차원 physical parameter vector는 36,864개다. 이전 v50, v55–v58의 159,744개와 byte 단위로도 겹치지 않았다. 네 contour 계열과 두 contact law, PyBullet solver는 앞선 실험과 공유한다. 새로운 물체 ID가 새로운 물리 법칙을 대표한다고 해석할 수 없다. [생성 계획](WORLD_PLAN_KO.md) · [검산](world59.json)

각 deployment object에서 두 deformation 방향의 contact intervention을 fine 및 ultra resolution으로 실행했다. Residual tangential speed가 0.020 m/s 이하일 때 stick으로 채점했다. 방향 intervention으로 stick/slip label이 바뀐 물체는 6,144개 active 중 2,286개(37.21%)였다. Inactive에서는 0개였다. 두 방향 전체 24,576개 endpoint 중 855개(3.48%)에서 fine과 ultra label이 달랐다. 측정된 geometry를 frozen model에 입력해 두 후보의 GPU forecast를 얻고, 공통 terminal에서 선택 정책이 허용받은 관측으로 합성했다. Independent binary deployment replay seed는 59041729, 59084591이다. [Generator](generate59.py) · [168개 archive checksum](deployment59.sha256)

## 동시 촬영과 기기 reset 비용

한 표면에서 두 설계 모두 독립 session 네 개, 서로 다른 물체 여덟 개, 2.0 N reference 여덟 번과 실제 impact 여덟 번을 지불한다. 따라서 pilot 비용은 **16 contacts·16 mechanical resets**다. 두 전자 센서가 동시에 관측하므로 16 contacts에서 방향 부호를 총 32번 읽을 기회가 있다. 독립 camera를 켜면 같은 16 contacts에서 16개 clip을 촬영하며 1,000 fps optical 모델의 여섯 frame 기준 96개 frame이다. 두 번째 물체 앞에서 한 센서의 local latch를 해제하는 정책은 **4 electronic sensor resets**를 추가한다. 이때 새 contact를 구입하거나 surface를 교체하지 않는다. Reset이 없는 쪽에도 shadow ID를 배정해 channel 선택 비대칭을 검사할 수 있다. [Pilot 계획](PILOT_PLAN_KO.md) · [Cost ledger](cost59.json)

한 pilot을 이후 H개 deployment session에서 공유하면 두 설계의 physical contacts와 mechanical resets는 각각 16+8H개다. 교체 전 공통 calibration 및 deployment 물체마다 필요한 공통 weak readout은 증분 비용에서 제외했다. H=1일 때 reference+reset과 impact+reset을 1.04초로, weak+reset을 0.44초로 놓으면 모두 22.56초다. Impact+reset을 2.08초로 놓아 네 번째 deployment sensor의 단가가 달라지면 모든 정책에 공통인 padded cap은 31.92초다. 아래는 첫 deployment까지 pad하지 않은 기대값이다. [산출 코드](cost59.py)

| MAP 설계 | 마지막 deployment impact 구매율 | T=2.08초에서 물리 비용 |
|---|---:|---:|
| Reset, camera | 58.44% | 31.488초 |
| Sham, camera | 60.52% | 31.509초 |
| Reset, camera off | 59.74% | 31.501초 |
| Sham, camera off | 59.27% | 31.496초 |

독립 session setup 한 번의 추가 시간을 S, 전자 센서 reset 한 번을 R, camera clip당 겹쳐서 수행할 수 없는 추가 시간을 K라 두면 공통 4S/H에 해당 설계의 4R/H, 16K/H를 더한다. H=1, T=2.08초의 model에서 두 camera-on arm의 물리시간 차이는 reset 쪽이 0.0216초 짧아 R이 약 5.4 ms를 초과하면 총시간 순서가 뒤집힌다. T=1.04초에서는 이 물리시간 차이가 없다. 실물의 reset·동기화·촬영 지연은 계측하지 않았다. 항상 사용하는 gate나 fixed는 실제 배포에서 pilot을 생략해도 같은 예측을 낸다.

## 실행 경계와 동기화

시작 당시 사용자 할당 네 장을 각각 Slurm step에서 조회했다. 두 GPU가 0 MiB·utilization 0%·compute PID 없음이었고 두 대에는 별도 학습 프로세스가 있었다. Node55의 allocation 2744468에 있는 GPU UUID `GPU-acff86bb-b0ee-515f-5139-502744220e28` 한 장으로 forecast와 모든 online benchmark를 순차 진행했다. 단계마다 다시 idle 여부를 확인하고 자기 worker tree 밖 GPU process가 나타나는지 guard가 감시했다. CPU contact solver와 bootstrap은 compute node에서 별도로 실행했다. [Forecast 종료](generation_guard59/results/worker_exit.json) · [Primary timing 계획](TIMING_PLAN_KO.md)

Primary 측정은 raw signed reference·impact와 camera sign이 메모리에 있을 때 시작한다. 각 정책의 관측 encoding, 할당받은 sensor의 reset ID 처리, 원인과 환경에 대한 joint likelihood, 네 번째 sensor 선택, 실제 frozen GPU prediction까지 포함한다. 새로운 surface 두 개에서 각기 pilot을 한 번씩 지불했고, 한 batch는 두 surface × 두 law × 128개 물체로 구성한다. 24개의 catalog마다 6개 arm을 한 번씩 warm up한 뒤 balanced Williams crossover로 3회씩 측정해 정책당 72개 full-call latency를 얻었다. CUDA를 양끝에서 synchronize한 wall clock을 사용했다. 24개 catalog의 paired bootstrap 16,000회로 reset 및 camera에 대한 4개 차이를 계산했다. 95% CI는 각 차이의 pointwise와 네 대조를 함께 보정한 simultaneous interval을 구분한다. 기계 작동, 영상 취득, 파일 읽기, 모델 초기화, PyBullet은 online 측정 밖에 있다. [Primary benchmark](bench59.py)

| 정책 | GPU p90 (ms/512 objects) |
|---|---:|
| Reset, camera | 502.576 |
| Sham, camera | 484.082 |
| Reset, camera off | 491.465 |
| Sham, camera off | 489.684 |
| 항상 v54 gate | 495.334 |
| 항상 fixed 3R+I | 500.849 |

Reset − sham은 camera on에서 +18.494 ms(동시 95% CI −24.718, +512.944), camera off에서 +1.780 ms(−371.506, +465.387)였다. Camera on − off는 reset에서 +11.111 ms(−17.967, +499.040), sham에서 −5.602 ms(−495.913, +42.391)였다. 일부 catalog에서 여러 정책의 wall time이 함께 0.5초대에서 1초 안팎으로 변해 90번째 백분위수 추정이 불안정했다. 외부 GPU process는 없었으나 지연 변동의 원인까지 추정하지는 않았다. 어느 차이도 0에서 구분되지 않았고 네 정책의 지연시간이 동등하다는 증거도 되지 않는다. 48개의 pilot에서 각 MAP arm의 fixed 선택 횟수는 표의 순서대로 19, 17, 14, 14회였다. 6개 arm 모두 독립 forecast replay와의 최대 absolute probability error는 0이었다. [Latency·paired CI](timing59.json) · [Timing worker 종료](timing_guard59/results/worker_exit.json)

## 영상 pixel에서 예측까지

추가 검증은 같은 24개 catalog, 동일한 48개 material pilot의 첫 replicate, 동일한 96개 deployment session에서 수행했다. 각각의 reset counterfactual에 정확히 같은 6-frame image를 제공했다. 두 material × 8개 물체 × reference/impact 2개의 총 32개 clip에서 상대적 fiducial 위치를 계산하고 signed camera observation을 생성하는 CPU pixel tracking부터 512개 물체의 GPU forecast 종료까지 측정했다. 실제 촬영·capture transfer·hardware reset은 제외하며 영상 render와 파일 loading도 시간 밖에서 끝냈다. 기존 sign-level camera 값은 사용하지 않고 선택 불가능한 camera-only false detection은 policy에서 가렸다. [Tracking 코드](frame59.py) · [End-to-end benchmark](frame_bench59.py)

각 catalog에서 reset 및 sham 각각 한 차례 warm up하고, 짝수/홀수 catalog에 ABBA/BAAB crossover를 배정해 두 arm을 2회씩 측정했다. 정책당 48개 sample에서 p90과 16,000회 paired catalog bootstrap 95% CI를 계산했다. 별도 timing run이므로 sign-level p90에서 pixel p90을 빼 tracker 시간으로 해석하지 않는다. [Frame timing 원자료](frame_timing59.json)

같은 영상에서 마지막 GPU 확률이 나오기까지 p90은 **reset 483.758 ms, sham 483.919 ms**였다. Reset − sham은 −0.161 ms [paired 95% CI −17.435, +4.410]였다. 두 arm의 replay 최대 absolute error는 0이었고 guard가 감지한 foreign GPU process는 0개였다. 따라서 이 구성에서 pixel tracking을 포함해도 reset으로 인한 계산 지연 차이를 확인하지 못했다. 실제 장치가 센서를 재초기화하는 데 걸리는 시간은 이 측정에 들어 있지 않다. [Frame timing](frame_timing59.json) · [Guard 종료](frame_timing_guard59/results/worker_exit.json)
