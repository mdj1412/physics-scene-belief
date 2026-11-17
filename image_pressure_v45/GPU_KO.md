## Shape 정보 없이 쓰는 projection axis

Original은 일곱 query profile 좌표를 모두 사용한다. Source-axis decoder는 알려진 query 높이 19·31 mm마다 하나의 축을 고정했다. 과거 v44에서 같은 물체·형상·위치를 hollow와 bulge로 바꿔 관측한 query profile 차이를 구했다. 재질·law 조합의 반복을 피하려고 inlay/rational 조건만 사용하고, 같은 sensor noise를 공유하는 첫 번째 draw를 골랐다. 양쪽에서 실제 query를 구매한 11,040쌍/높이의 7차원 차이를 정규화하고, fold/ripple 전체에서 두 번째 모멘트의 가장 큰 eigenvector를 취했다. 일곱 좌표의 sensor 표준편차가 같으므로 whitening의 공통 scale은 정규화 시 상쇄된다. 학습에서 stick/slip label을 사용하지 않았다. Fold와 ripple는 source에서 노출된 family이고 target catalog의 실제 변형 방향은 입력하지 않았다. Source geometry는 controlled variant를 짝지어 측정할 수 있다는 조건을 쓴다.

19 mm에서 첫 축이 차지하는 정규화된 profile 차이 energy는 67.93%, 31 mm에서 58.28%였다. 각 family별 평균 차이 벡터와의 absolute cosine은 각각 0.824와 0.763이었다. Source-axis arm에서는 각 물체의 nominal height로 이 단일 축을 불러온다. 위치 추정값만이 original 또는 source-axis likelihood 사용 여부를 결정한다. 같은 물체에서 shape별 실제 변형 축을 허용한 privileged 결과는 보조 진단으로 분리했다. [축 학습 코드](source_axis45.py) · [동결된 두 축](source_axis45.json)

## 새 물체와 GPU 비교

12개 새 catalog의 support key는 500–511, seed는 450000000 + 100000 × panel이다. 4개 환경별 128개 물체에 같은 factual evidence를 넣었다. K256은 환경마다 115개 물체에 query view, 13개에 deployment probe를 배정했다. 같은 조건의 sensor noise를 4회 생성하고, 구매하지 않은 값은 NaN으로 제거한 뒤 decoder에 전달했다. Projection은 query view likelihood에 적용된다. Probe를 산 물체의 원래·generic·privileged 예측은 동일하다. 모든 조건의 hypothesis bank와 K256 acquisition은 v43 artifact를 재사용했다. [GPU runner](runner45.py) · [예측 동결 SHA-256](gpu_freeze45.sha256)

서버의 4개 GPU allocation에서 process 및 memory를 먼저 점검했다. 이미 다른 학습이 진행 중인 두 장을 유지하고 node55의 2744468과 node56의 2744471을 사용했다. 각각 짝수·홀수 catalog의 예측을 완료한 뒤 node56의 같은 NVIDIA RTX PRO 6000에서 모든 방법의 timing을 측정했다. 동시 물체 512개 × sensor noise 1회에 대해 매 panel 무작위 순서 1회 warm-up과 3회 본측정을 수행했다. 2개 높이와 2개 shape가 균등하게 나타나도록 12개 대표 조건을 정하고, 각 batch 안에서 세 offset을 거의 같은 수로 섞었다. 한 timing batch의 visual input은 512×3×32×96 = 4,718,592 bytes의 미리 촬영된 uint8 frame이다. 렌더링과 실제 camera exposure, 실물 probe와 reset은 모델 latency에서 제외했다.

Visual 경로는 CPU에 있는 raw frame을 GPU로 전송하고, 영상에서 위치를 추정하고, 고정 표로 route를 정한 뒤 K256과 posterior inference를 수행한다. Oracle은 GPU로 전송한 정확한 좌표로 동일한 표를 조회한다. Always-original은 위치 입력이 없다. 세 방법 모두 실제로 구매한 sensor 한 개를 선택해, 물체별로 지정된 likelihood를 한 번 계산한다. 전체 물체를 두 decoder로 각각 처리하고 나중에 출력을 고르는 비용은 발생하지 않는다. 물체가 서로 다른 branch를 쓰면 query를 나눠 계산하므로 batch 분할의 비용은 포함된다. 고정 source template은 warm measurement 이전에 준비했다. 각 방법의 출력이 따로 계산한 두 frozen decoder의 해당 행과 일치하는지 512개 물체에서 대조했다. 통계 구간은 paired catalog bootstrap 16,000회의 95% percentile interval이다. [전체 latency 코드](timing45.py) · [GPU 독립 검산](audit_gpu45.json)

## 과거 label에서 axis와 짝을 맞춘 보조 gate

Primary gate는 과거 privileged decoder의 sign table을 source-axis decoder에 그대로 적용한다. 이를 점검하기 위해 source-axis decoder를 v44의 과거 물체 첫 8개/환경에서 다시 실행하고, v43 point model이 실제 좌표에서 주는 과거 label로 여섯 위치의 projection−original Brier 차이를 계산했다. 이 절차의 입력에는 v45 target이 없으며 rule과 sample 수를 target 평가 전에 고정했다. 18.5, 19, 19.5, 30.5, 31, 31.5 mm에서 각각 +0.010297, +0.001347, +0.020747, −0.002293, +0.000061, +0.001514였다. Supplementary source-matched gate는 30.5 mm에서만 projection을 고른다. 31 mm의 D는 작은 양수이고 12개 source catalog 중 하나씩 제외하면 여섯 번 부호가 바뀐다. 30.5 mm의 부호는 모든 leave-one-catalog-out fit에서 음수였다. [source-only code](matched_source45.py) · [패널별 D와 lookup](matched_source45.json)

## 실제 latency와 반복 측정

첫 crossover의 panel 0에서는 다섯 방법 모두 median이 약 2.1초였고, panel 1–10은 대체로 약 460–474 ms였다. 모든 record를 포함한 첫 실험의 p90에는 이 공통 지연이 크게 반영됐다. 원인을 특정할 CPU 측정치는 없어 첫 실험을 그대로 보존했다. 이후 같은 allocation에서 팀의 다른 작업을 node55로 분리하고 Slurm이 허용한 CPU core 10으로 worker를 고정해, 같은 raw input·같은 policy 순서·같은 GPU로 전체 crossover를 반복했다. 패널별 cgroup 관측에서 CPU throttling은 0회였다. 기존 프로세스 감시에서도 외부 GPU process는 관찰되지 않았다. 두 측정에서 bias나 family·route rule을 변경하지 않았다. [두 측정과 paired bootstrap](timing_comparison45.json) · [CPU affinity 및 cgroup 기록](timing_repeat45/resources.json)

| 방법 | 첫 측정 p90, ms | 분리 재측정 p90, ms [95% CI] | 분리 재측정 위치 처리만의 p90, ms |
|---|---:|---:|---:|
| Always-original | 769.38 | 465.19 [459.79, 467.10] | 0.01 |
| Oracle 위치, primary | 735.37 | 465.69 [462.43, 465.85] | 0.30 |
| 영상 위치, primary | 785.50 | 468.19 [465.69, 481.65] | 3.16 |
| Oracle 위치, source-matched | 737.89 | 464.41 [462.41, 465.46] | 0.23 |
| 영상 위치, source-matched | 654.04 | 467.75 [466.12, 478.36] | 3.14 |

Visual−original p90 차이는 첫 측정 +16.11 ms [−101.00, +262.44], 분리 재측정 +2.99 ms [−0.30, +17.29]였다. 분리 재측정에서 visual−oracle 차이는 +2.50 ms [+0.88, +16.06]였다. 완전한 camera exposure나 물리 probe 시간을 더한 실시간 robot 지연은 이 수치에서 산출할 수 없다. 분리 재측정의 512개 물체당 4.72 MB frame 전송·추적·위치표 조회의 p90은 3.16 ms였다. 두 측정 모두 576개 probability archive에서 뽑은 해당 방법의 참조 결과와 최대 절대 오차 4.45×10⁻¹⁶ 미만으로 일치했다. [예측 및 순서 검산](audit_gpu45.json) · [반복 계산](timing_comparison45.py) · [GPU 해제](gpu_release45.log)
