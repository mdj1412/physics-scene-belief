# v47 GPU 경로와 재현 자료

## 실행 조건

사용자 allocation 네 장에서 Slurm job별 GPU UUID와 compute process를 확인했다. 2744467과 2744469는 각각 40,125 MiB와 40,127 MiB를 쓰는 별도의 Python 작업이 실행 중이었다. 2744468/node55와 2744471/node56는 0 MiB, utilization 0, compute process 없음이었다. 각 GPU worker는 시작 직전 상태를 다시 확인했다. Guard는 10초마다 다른 process가 나타나는지 검사하고 충돌 시 해당 worker만 멈춘다.

12개 catalog를 두 카드에서 짝수와 홀수로 나눠 RTX PRO 6000 Blackwell, CUDA 12.8, PyTorch 2.7.1에서 계산했다. Catalog support 700–711의 population 생성은 allocated CPU에서 실행했다. 각 catalog는 4 environment × 128 draws를 제공했다. 환경별 material pairing은 4개 surface–law 조합을 한 번씩 포함하며 catalog에 따라 순환한다. 모든 방식에서 factual evidence, K256 acquisition, source hypothesis bank와 4회 paid sensor noise가 같다. Environment마다 115개 query와 13개 deployment probe를 고른다.

## 관측과 decoder 경로

- Camera는 세 장의 96×32 uint8 frame을 제공한다. Family와 variant 사이에 같은 catalog·height·offset이면 camera nuisance seed도 같다. Camera view에는 별도의 range sensor에 적용한 optical axis intervention과 under-load coupling이 입력되지 않는다.
- 7개 query channel에만 prescribed deformation을 더한다. Inference에 넘기기 전 선택되지 않은 sensor를 NaN으로 바꾼다. Original 및 projected decoder는 동일한 사실·유료 관측을 처리한다. Probe arm에서는 두 확률이 일치해야 한다.
- V46 rule은 영상에서 얻은 x와 normal variance로 기존 source lookup을 그대로 조회한다. V47 rule은 추가로 실제 구매한 query profile에서 추출한 두 scalar feature를 사용한다. Probe arm에서는 false로 복귀한다. V46의 camera route 기록은 probe arm에서도 가끔 true이지만 실제 probe posterior에는 영향을 줄 수 없다.
- Archive에서 original·projected의 확률은 float64, diagnostic paid profile은 float32다. 실제 모델의 gate는 float64 원 관측에서 계산한다. Mean risk 계산 때 projected와 original을 offline에서 짝지어 비교한다. 실제 timing에서는 두 개를 미리 계산해 선택하지 않고, 한 물체가 선택한 decoder를 한 번만 실행한다.

## GPU p90 측정

별도 node56 카드 한 장에서 12개 representative batch를 측정했다. Batch마다 512개 물체와 1 paid noise를 사용한다. 서로 다른 세 offset을 무작위로 섞으며 4 family 각각 두 height가 모두 들어간다. 각 batch에서 한 차례 warm-up 뒤 5회씩 method 순서를 섞고 CUDA 동기화 wall-clock을 쟀다. Factual acquisition, 선택된 sensor masking, 필요한 frame 전송과 feature extraction, routing, mixed decoder 및 결과 반환을 포함한다. 실물 camera 노출·physical reset·offline rendering·target label 계산 시간은 포함하지 않는다. Batch p90 차이의 구간은 동일 catalog 12개를 16,000회 복원 추출하여 매번 p90를 다시 계산한다. 이는 단일 물체의 latency가 아니다.

수치 및 독립 검산은 [GPU 감사 결과](gpu_audit47.json), timing raw는 [timing](timing), 실제 Slurm 장치·버전은 [runs](runs), inference 전 동결은 [SHA 목록](freeze47.sha256)에 있다.
