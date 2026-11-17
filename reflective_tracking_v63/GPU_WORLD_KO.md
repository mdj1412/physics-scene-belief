# v63 · 신규 표면의 physics와 정책별 compute 비용

## 새 표면과 공통 예측

Catalog 8300~8323의 48개 surface instance에 기존 두 rate law를 교차하고 surface × law당 128개 물체를 생성했다. 배포 물체 12,288개 및 두 calibration support의 6차원 parameter vector 합계 36,864개는 서로 고유하고, v62까지의 비교 가능한 350,608개 이전 vector와 일치하는 사례가 없었다. Board와 물체·표면의 draw는 새로 생성했고 기존 PyBullet implementation과 rate-law family를 유지했다. Optical fault는 영상에서 바뀌며 동일 물체의 실제 stick/slip target은 네 fault cell에서 공통이다. [중복 검산](support63.json) · [표면 생성](generate63.py)

각 물체를 변형 방향을 유지한 경우와 반전한 경우로 해석했다. 변형이 활성인 물체 6,144개 중 2,199개에서 두 방향의 stick/slip label이 달랐다. Fine과 ultra 두 해상도 사이에는 24,576개 binary endpoint 중 796개에서 차이가 생겼다. 배포 단계의 plus·minus 확률은 하나의 공통 frozen GPU forecast로 계산했고, 별도로 63041729·63084591 seed를 사용해 binary session replay 두 개를 저장했다. [World 요약](world63.json) · [168개 입력과 출력의 checksum](deployment63.sha256)

## GPU p90과 판독 비용

각 호출에서는 두 표면의 paid raw camera clip 32개와 switch·photodiode 파형을 읽고, 선택된 관측을 적용해 네 deployment session의 물체 512개의 최종 stick 확률을 얻었다. Always-v54 gate에는 같은 physical budget을 주되 사용하지 않는 영상·파형의 판독은 생략한다. 무작위 보류는 실제 서비스에서 구현 가능한 비용을 재기 위해 각 pilot의 confidence 판독 수 k를 직접 구하고, 같은 raw frame에서 v61 유효 집합을 계산한 뒤 균등하게 k개를 남겼다. 두 reader는 같은 optical feature 추출을 공유한다. 무작위 baseline의 loss는 모든 subset을 정확하게 주변화하고 timing에서는 한 subset을 구입하는 배포 절차를 쟀다. 평가 전용 독립 센서, 시뮬레이션과 물리 촬영·reset 대기시간, 디스크 읽기·모델 초기화는 세 정책의 GPU p90에서 제외했다. 매 호출 전후에 CUDA를 synchronize해 CPU 판독과 최종 GPU 작업을 포함한 wall time을 기록했다. [동결 계획](WORLD_PLAN_KO.md) · [GPU benchmark](bench63.py)

Node55, Slurm allocation 2744468의 NVIDIA RTX PRO 6000 Blackwell Server Edition 한 장을 사용했다. Common forecast 및 timing을 시작하기 직전 다른 compute PID 없음, 사용 memory 0 MiB, utilization 0%를 확인했으며, 실행 중에는 guard가 외부 process를 감시했다. 별도의 Slurm CPU에서 hidden CUDA로 solver·광학·통계를 실행했다. 24 catalog × 네 fault cell에서 각 방법을 한 번씩 재고 catalog별 warmup은 제외했다. 세 방법이 각 순번에 32회씩 오도록 순서 및 인접 순서를 균형화했다. [Forecast guard](generation_guard63/results/worker_exit.json) · [Timing guard](timing_guard63/results/worker_exit.json)

| 정책 | GPU p90, ms / 512 objects | 상위 fixed 선택 / 192 | 최종 impact 구매 / 384 |
|---|---:|---:|---:|
| Confidence | 483.573 | 2 | 100 |
| Same-count random | 496.661 | 2 | 100 |
| V54 gate | 496.339 | 0 | 96 |

Confidence − random의 GPU p90 차이는 −13.089 ms [95% simultaneous CI −126.278, +32.249], confidence − gate는 −12.766 ms [−131.276, +46.449]였다. 보드를 먼저 재추출하고 그 안의 세 catalog를 재추출한 민감도에서도 두 interval은 0을 포함했다. 첫 두 catalog에서 세 방법 모두의 소요시간이 후속 catalog보다 길어 bootstrap p90 차이의 불확실성이 커졌다. 이들을 포함한 사전 명세의 전체 표본을 유지했다. [주 GPU 결과](timing63.json) · [Board resampling](gpu_board63.json)

대표 96개 block에서 confidence와 random이 실제로 선택한 최종 관측은 384개 모두 일치했다. 두 정책과 gate 사이에는 384개 중 4개의 최종 관측이 달랐다. 저장된 forecast와 세 방법 모두에서 최대 재생 오차는 0이었다. 이 p90은 실제 입력에서 각 정책을 실행한 비용이며, confidence와 random이 서로 다른 관측을 고르는 드문 사례의 비용을 따로 판정할 수 없다.

별도 Slurm CPU(node56, 한 thread, CUDA 비활성)에서 같은 96개 raw block에 7회씩, 정책당 672회 판독과 상위 행동 결정까지만 재었다. Confidence p90은 0.610 ms, same-count random은 0.797 ms, gate는 0.001 ms였다. Confidence − random의 차이는 −0.187 ms [95% simultaneous CI −0.194, −0.180], confidence − gate는 +0.609 ms [+0.603, +0.614]였다. 같은 k를 구한 후 subset을 무작위로 고르는 추가 계산이 작은 차이를 만들며 common forecast가 포함된 GPU p90의 크기와 구분된다. 이 CPU 분해는 독립적인 별도 측정이므로 GPU p90에서 직접 빼서 kernel 시간을 추정하지 않았다. [CPU reader 결과](reader_cost63.json) · [방법](time_audit63.py)

## Sensing 비용과 재현

표면당 네 session, session당 서로 다른 물체 두 개에 8 reference + 8 impact, 총 mechanical reset 16회를 배정했다. 각 contact에서는 raw clip 하나를 얻으므로 표면당 16개 clip, 128 frame이다. Confidence와 random은 둘째 물체 직전 electronic reset 1회를 사용하여 표면당 4회다. Gate는 reset·camera 관측을 정책 선택에 사용하지 않는다. Independent sensor는 기존 접촉에 동시에 기록되며 정책은 읽을 수 없다. 실물 장비에서 필요한 별도 전력·시간·재설정 비용은 미측정이다. 보류 후 추가 impact나 촬영은 허용하지 않았다.

Remote project의 reflective_tracking_v63에서 순서대로 Slurm compute CPU에서 `bash run_world63.sh prepare`, `preaudit`, `solver`를 실행한 뒤 빈 GPU에서 `bash run_world63.sh forecast`를 실행한다. CPU에서 `panel`과 `audit`를 완료한 뒤 24 catalog의 광학 fixture가 준비되면 빈 GPU에서 `bash run_timing63.sh`를 실행한다. 대표 fixture는 catalog마다 한 board를 사용하며 여덟 board가 각각 세 catalog에 배정된다. 24개 catalog의 원자료에는 순서, paid raw input 재생 검산, 상위 행동과 최종 구매, 시간 및 예측 재생 오차가 있다. [GPU 원자료 예제](timing63/panel_00.npz)
