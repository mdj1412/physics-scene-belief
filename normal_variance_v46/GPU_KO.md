# V46의 held-out 예측과 GPU 측정

## 데이터와 접근 가능한 정보

서버에서 support 600–611과 seed 460000000 + 100000 × panel을 사용했다. 12개 catalog 각각에서 4개 환경 × 128개 물체를 추출하고 terrace, dimple에 같은 물체와 observation noise를 대응시켰다. 6,144회 추출에 5,976개의 고유 prototype이 있었다. 각 panel에서 2개 shape × 2개 surface × 2개 law × 3개 variant × 2개 nominal height를 교차했다. 모든 정책에서 factual evidence를 입력한 동일 K256이 환경별 115개 query view와 13개 deployment probe를 선택했다. 구매하지 않은 observation은 NaN 처리한 후 원래 decoder와 v45 source-axis decoder에 같은 evidence를 전달했다. 4개 sensor noise 각각에서 두 확률을 저장한 배열은 [2 decoders, 1 correction, 4 environments, 128 objects, 4 noises]이다. 보관한 576개 파일의 SHA와 2,359,296개 확률을 검산했다. Deployment probe를 사용한 모든 물체에서 두 decoder의 확률 차이는 0이었다. [검산](audit_archive.json) · [실행 코드](runner46.py)

Original은 7점 contour query profile 전체를 사용한다. Source-axis decoder는 과거 fold/ripple에서 쌍으로 관측한 profile 차이에 주성분분석을 적용해 학습했던 v45의 높이별 축을 고정해 쓴다. 두 새 family의 shape 이름과 실제 변형 방향은 encoder·selector 입력에서 제외했다. Source axis가 설명한 source profile energy는 19 mm에서 67.93%, 31 mm에서 58.28%였다. 시뮬레이터만 새 shape 함수를 사용해 관측을 만든다. [V45 동결 axis](../image_pressure_v45/source_axis45.json) · [새 geometry](geometry46.py) · [V46 사전 SHA](gpu_freeze46.sha256)

## 측정 설계

서버에서 유효한 allocation 네 개를 확인했다. Node55의 job 2744468과 node56의 job 2744471이 모두 0 MiB, utilization 0이고 compute process가 없어 두 GPU에 짝·홀 panel을 배분했다. 다른 두 GPU는 35,171 MiB, 40,127 MiB를 점유 중이었다. 동일한 NVIDIA RTX PRO 6000 Blackwell의 같은 node56 GPU에서 네 정책을 교차 측정했다. CPU core를 Slurm으로 bind하고 target solver 작업은 다른 node에 분리했다. 데이터 수집과 독립 label 계산은 별도 compute 단계다.

한 timing batch는 물체 512개의 세 장짜리 96×32 uint8 frame, 총 4,718,592 bytes이다. 카메라를 사용하는 세 정책에는 host→GPU frame 전송과 위치 측정을 포함하고 x+normal variance에는 주변 일곱 surface normal의 각도 분산과 표 조회를 추가했다. 모든 경로에서 선택·posterior·동기화·출력 전송을 함께 잰다. 각 물체의 실제 route에 따라 한 번만 likelihood를 계산한다. Warm template, 촬영 완료된 frame을 시작점으로 사용한다. 실제 robot reset, camera exposure, offline rendering, target solver 시간은 별도다. 12개 대표 batch는 두 높이와 두 shape를 균형 배분하고 세 offset을 매 batch에 섞었다. 각 panel에서 정책 순서를 섞어 1회 warm-up, 4회 본측정했다. p90 및 대비 구간은 paired catalog bootstrap 16,000회를 사용했다. [측정 경로](timing46.py) · [재표집 코드](audit46.py)

**배포 경로 검산.** 실제 target clip 221,184개를 다시 생성해 각 묶음의 SHA-256을 확인하고, 같은 GPU에서 pixel→x→variance→route를 끝까지 실행했다. 점수표에 사용한 CPU 추정 경로 대비 좌표의 최대 절대 차이는 0.0000267 mm, normal variance 최대 차이는 0.0000742 rad²였다. 판정 임계값을 넘어 route가 달라진 clip은 legacy x·refit x·x+normal 세 경로 모두 없었다. [전체 GPU replay](route_replay46.json)

## GPU p90 및 불확실성

| 배포 정책 | 512개 물체의 primary p90 (ms) | Paired panel bootstrap 95% CI | Frame 전송·feature·lookup 경로만의 p90 (ms) |
|---|---:|---:|---:|
| Always-original | 465.64 | [465.23, 470.71] | 0.01 |
| V45 frozen x | 471.35 | [470.51, 476.07] | 3.15 |
| V45 source에서 refit한 x | 471.10 | [470.32, 475.33] | 3.10 |
| X + normal variance | 473.53 | [471.97, 485.91] | 4.60 |

Joint−v45 x의 p90 차이는 **+2.18 ms** [−3.10, +14.49]이며, joint−always-original은 **+7.89 ms** [+1.73, +20.56]였다. 전자에는 0이 포함돼 표본 p90만으로 가치와 비용을 동일하게 판정하기 어렵다. V45 x와 refit x는 source 추정 D는 달라도 같은 여섯 위치에서 같은 결정을 내렸다. 동일 GPU에서도 p90 추정량은 실행 순서와 주변 조건에 따라 변동할 수 있다. 첫 crossover에서 CPU는 node56의 core 10에 고정됐고 시작·종료 cgroup의 throttled time은 0이었다. 모든 직접 mixed-dispatch 확률과 보관한 분기별 결과의 최대 차이는 4.44×10⁻¹⁶이었다. [Primary p90·대비·재표집](audit_latency46.json) · [실행 당시 자원](latency46/resources.json)

두 번째 crossover에서는 동일한 입력과 같은 GPU에서 정책 순서의 seed만 바꿨다. Always-original, v45 x, refit x, joint 순서로 p90은 각각 **470.76, 475.67, 475.42, 478.18 ms**였다. Joint−v45 x는 +2.51 ms [−11.68, +18.66], joint−always-original은 +7.42 ms [+4.84, +23.84]였다. 위치 경로 p90은 3.14 ms, 두 feature 경로는 4.66 ms였다. 두 측정의 전체 처리시간 차이와 feature extraction 자체 비용을 분리해 볼 수 있다. [독립 순서 재측정](audit_latency46_repeat.json)

**Target family가 알려진 진단.** Primary 결과를 확인한 후 두 새 형상의 hollow·bulge가 만드는 7점 profile 차이를 exact geometry로 평가했다. Source sensor noise로 whiten한 뒤 정규화한 각 family·height의 축으로만 기존 template을 교체하고, 고정한 K256·구매한 sensor·hypothesis bank로 모든 물체의 별도 projection 확률을 재계산했다. 이 분석은 target outcome을 학습에 사용하지 않으며, family 이름과 제어된 target variant의 모양을 사전에 아는 비용을 가정한다. [Axis 계산과 검산](target_axis46.py) · [고정된 네 축](diag_axis46/axes.json) · [배치 완료 상태](diag_axis46/complete.json)
