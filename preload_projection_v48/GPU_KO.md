# v48 서버 GPU 실행 및 비용의 범위

v47과 동일한 K256 factual acquisition, 7점 profile 및 probe likelihood, source-axis template와 contact decoder를 사용했다. 네 새 contour family와 12개 새 support(800–811)를 사용하고, 4 environments × 128 objects로 구성한 catalog마다 두 forecast를 저장했다. Held-out label을 읽는 함수는 GPU worker에 없다. 각각의 측정은 mm 단위 scalar 하나이며 정책은 실제 구매한 station의 한 번의 측정과 기존 paid profile만 본다. 한 물체의 다른 signed state나 두 station의 값을 함께 주지 않는다.

## 어떤 계산이 시간에 포함되는가

12개 대표 512-object batch에서 4개 방법을 같은 단일 GPU로 교차 실행한다. 각 batch의 prior state 구성은 inert 256개, negative 128개, positive 128개이며 순서를 섞는다. 방법별 warm-up 1회 후 5회씩 순서를 매번 새로 뽑는다. 각 실행에서 K256 선택을 처음부터 다시 하고 구매하지 않은 공통 sensor는 NaN으로 가린다. Fixed와 preload는 station scalar를 host에서 GPU로 전송하고 source-trained risk 계산을 수행한다. None은 기존 paid profile만 쓴다. Always-original에는 추가 sensor나 risk fit이 없다. 모든 방법은 실제 선택된 decoder 1개로 posterior를 구하고 결과를 host에 되돌린다.

Timing은 이 경로 바깥에서 생성한 synthetic observation을 시작점으로 한 synchronized wall-clock이다. 소프트웨어에서 두 후보를 만들어 채점하는 일과 runtime 단일 경로를 분리했고, 각 timing의 route와 probability를 archive와 대조한다. CPU raw scalar 생성 시간은 로봇 sensing time을 대표하지 않는다. 추가 action은 fixed/preload 모두 물체당 weak contact 1회와 reset 1회이며, 같은 station의 측정 protocol을 쓴다. Always-original과 none의 추가 sensing·reset 횟수는 0이다. 512개 batch의 GPU p90와 한 물체의 물리 cycle은 별개이므로 둘을 바로 더하거나 batch p90를 512로 나누어 단일 robot latency를 만들지 않는다.

## 할당과 재현

서버 경로는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/preload_projection_v48`다. 4개 allocation을 확인하여 이용 중인 2개(당시 각 약 40 GB, compute process 실행 중)는 제외했다. 2744468/node55와 2744471/node56에서 0 MiB, utilization 0 및 process 없음이 확인됐다. 짝수·홀수 catalog를 두 NVIDIA RTX PRO 6000 Blackwell에 나누고 최종 p90는 2744471의 단일 GPU에서 측정한다. 실행 guard는 다른 작업이 나타나면 자신이 시작한 worker만 중지한다.

GPU 원자료와 수치 재현은 [forecasts](forecast), [timing batch](timing), [freeze](freeze48.sha256), [실행 시각과 장치](runs), [audit code](gpu_audit48.py), [검산 결과와 paired bootstrap 구간](gpu_audit48.json)에 있다. Bootstrap은 같은 catalog block을 방법 간에 함께 복원 추출해 p90를 매번 재계산한다(16,000회). 한 GPU에서 12개 batch에 대한 불확실성이다.

## Primary GPU p90 결과

| 방법 | GPU p90 ms / 512 [bootstrap 95% CI] |
|---|---:|
| Always-original | 467.86 [467.02, 468.15] |
| Fixed probe | 471.97 [471.21, 472.74] |
| Query-local preload | 472.11 [471.51, 472.92] |
| No-extra profile gate | 471.47 [470.97, 472.01] |

Local−fixed는 +0.13 ms [−0.45, +1.07], local−original은 +4.24 ms [+3.80, +5.09]였다. Fresh support 내 중복을 제외한 prototype 수의 합은 5,964개였다. 저장한 candidate probability 1,572,864개에서 inherited physical-probe arm의 두 candidate가 같음을 확인했다. Catalog별 deployment policy를 CPU에서 다시 계산해 각각 2,359,296개 decision을 GPU archive와 대조했고 4개 방법의 mismatch가 모두 0이었다. 같은 station에서 fixed와 query-local도 완전히 같은 route를 썼다. Randomized timing 경로의 최대 probability replay 차이는 3.33×10⁻¹⁶이었다. [감사 결과](gpu_audit48.json)

## 사후 standalone direct calibration과 end-to-end crossover

동일 GPU·512 objects에서 별도 무작위 순서로 12개 catalog × 4회씩 측정했다. 각 batch에서 항상 original posterior를 GPU에서 계산한 다음 반환 확률과 동일한 query-local scalar에 대해 7항 calibration을 host에서 적용했다. ① original 466.04 ms [465.69, 466.71], ② local gate 470.40 ms [470.16, 470.76], ③ direct local 466.61 ms [465.81, 468.58]였다. Direct−gate는 **−3.79 ms [−4.54, −1.79]**, direct−original은 +0.56 ms [−0.49, +2.83]였다. Direct 경로는 새로운 posterior를 만들지 않고 동일 original posterior의 반환값을 보정한다. 추가로 계산한 direct probability와 offline 7항 전개식의 최대 차이는 4.44×10⁻¹⁶이었다. 두 timing session의 절대 수치는 따로 비교하고, 효과는 각 session 내부 paired difference로 판단한다. [측정과 비교](direct_timing48.json) · [실행 코드](direct_timing48.py)
