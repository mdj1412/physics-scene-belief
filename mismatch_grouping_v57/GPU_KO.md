# v57 · Session grouping의 GPU p90과 물리 예산

## 측정 대상

서버에 할당된 GPU 중 node55/2744468과 node56/2744471에서 각각 0 MiB, utilization 0%, compute process 0개를 확인했다. 새 physics forecast는 node55, 아래 online prediction은 node56의 NVIDIA RTX PRO 6000 Blackwell Server Edition에서 측정했다. 나머지 두 allocation에서 돌던 workload는 유지했다. [사전 측정 계획](TIMING_PLAN_KO.md)

각 catalog의 두 material에 환경 A/B를 하나씩 교대로 배정하고, material별 독립 pilot 하나의 결정을 두 law의 deployment session에 공유했다. 네 session에 128개씩, 512개의 fresh deployment object를 한 batch에 넣었다. Online call에서는 메모리에 적재된 두 부호 readout의 세션·물체 grouping을 읽고 likelihood로 outer policy를 정한다. V54 gate를 유지한 session에서만 세 weak/reference 쌍으로 네 번째 센서를 고른다. 센서를 선택한 후 구입한 값만으로 frozen terminal mixture를 갱신하고 별도 posterior를 거쳐 GPU forecast를 반환한다. 선택하지 않은 센서의 값을 sentinel로 바꿔 terminal에 영향을 주지 않는지 검산했고 각 call에서 실제 route·예측·예측을 위한 object별 sensor가 독립 replay와 일치하는지 검사했다. 디스크로부터의 archive 로딩이나 pilot를 생성한 simulator, 실제 sensor capture와 기계 reset은 GPU latency에 포함하지 않는다. [Timing 코드](bench57.py)

각 catalog에서 다섯 정책을 한 번씩 warmup한 후 다섯 period의 순서를 균형화한 다섯 차례 crossover를 수행했다. 인접 catalog끼리 서로 반대 순서를 사용해 다섯 정책의 위치와 직전 정책의 영향을 분산시켰다. CUDA를 양끝에서 동기화한 host wall clock의 p90을 보고한다. 24개 catalog 전체를 paired bootstrap으로 16,000회 재추출했고, MAP 4×2에서 다른 네 정책을 빼는 비교 family에 Bonferroni 동시 95% interval을 적용했다. 추가로 online 센서의 선택만 수행한 host p90을 측정했다.

## 측정 결과

| 정책 | GPU p90 (ms/512 objects) | Host routing p90 (ms) | MAP 4×2 − 해당 정책 (ms), 동시 95% CI |
|---|---:|---:|---:|
| MAP 4×2 | 440.573 | 0.1259 | 기준 |
| MAP 8×1 | 440.218 | 0.1304 | +0.355 [-0.392, +0.789] |
| Risk-weighted 4×2 | 440.413 | 0.1359 | +0.161 [-0.747, +0.689] |
| 항상 v54 gate | 440.536 | 0.1227 | +0.038 [-0.409, +0.953] |
| 항상 fixed 3R+I | 440.322 | 0.0037 | +0.251 [-0.526, +0.899] |

24개 catalog에서 정책당 120번씩 측정했다. 512개 물체를 동시에 예측할 때의 총 시간이다. 다섯 정책 모두 다섯 순서에 각각 24회씩 나타났고 서로 다른 두 정책이 앞뒤로 이어지는 경우도 각각 24회였다. 모든 선택에서 cached forecast에 대한 최대 absolute probability error는 0였다. PyTorch 2.7.1+cu128, CUDA 12.8, node56 allocation 2744471을 사용했다. [원시 timing과 장치 상태](timing57.json) · [순서·rare action 검산](timing_audit57.json) 네 latency 차이의 동시 interval이 모두 0을 포함해 이 설계로 compute 우위를 확인하기 어렵다.

실제 timing fixture의 material pilot 48개 중 MAP 4×2와 risk-weighted 4×2가 fixed를 고른 수는 각각 22개와 0개였다. 따라서 rare risk switch의 conditional GPU p90은 이 표에서 추정하기 어렵다. 별도 CPU replay에서 두 환경과 두 material에서 switch가 발생한 실제 paid pilot을 골라 96개 surface case의 route와 terminal posterior를 독립 검산했다. 최대 absolute posterior error는 0였다. Risk-weighted 4×2에서 항상 v54 gate와 항상 fixed를 뺀 exploratory latency는 각각 -0.123 ms [-0.645, +0.792], +0.090 ms [-0.333, +0.753]였다. 두 비교 안에서만 동시 95% interval을 조정했다.

## Sensing·reset ledger

각 설계는 surface당 8개 서로 다른 sacrificial specimen을 사용한다. 물체마다 2.0 N reference와 impact를 각각 한 번 측정하고 reset한다. 유효한 두 방향을 읽지 못하더라도 비용은 지불한다. 같은 session의 다음 물체 역시 새로운 specimen이므로 둘째 판독의 접촉과 reset을 할인하지 않았다. **Pilot은 세 설계 종류(8×1, 4×2, 공통 예산의 고정 대조) 모두 16 contacts·16 resets다.** 이후 post-exchange calibration의 8 contacts·8 resets를 더하면, 배포를 한 번만 할 때 surface당 24 contacts·24 resets다. 한 pilot을 H개 session이 공유하면 총 16+8H contacts와 같은 수의 resets다. 교체 전 공통 8 contacts·8 resets, deployment object별 공통 weak 측정은 이 증분 비교에서 제외했다.

두 protocol은 pilot session을 시작하는 횟수가 다르다. 8×1은 독립 session 8개, 4×2는 4개다. 물체 준비·교체는 양쪽 모두 서로 다른 8개에 필요하다. Surface나 실험 장치를 초기화해 독립 session을 만드는 추가 시간 S, 개별 물체 준비시간 P가 별도라면 setup 항은 각각 8S+8P, 4S+8P다. 이 실험은 simulation이므로 S와 P의 실측값은 없다. S와 P는 각 contact의 reset에 이미 포함한 시간을 제외한 추가 비용이다.

아래의 T는 impact와 그 reset을 합친 시간(초), 1.04초는 reference와 그 reset의 모델값, 0.44초는 weak와 그 reset의 모델값이다. g는 pilot session 수, r은 마지막 deployment 센서로 impact를 구입한 비율이다. 세션당 소유시간은 (8(1.04+T)+gS+8P)/H + 5.92 + r(T−1.04)초다. 정책을 완전히 동일한 elapsed budget에서 비교하려면 네 번째 센서를 더 긴 slot에 맞추고, pilot도 공통 8S까지 padding한다. 이때 상한은 (8(1.04+T)+8S+8P)/H + 4.88 + max(1.04,T)초로 모든 정책에서 같다. 고정 정책도 같은 pilot을 지불하되 routing에서 읽지 않는다. 운영상 고정 정책은 16 contacts·16 resets를 모두 생략해도 예측을 그대로 낼 수 있다. [Sensing 계산](sensing57.py) · [전체 T·H·S 비용표](sensing57.json)

| 정책 | A/B 균등 mixture에서 outer fixed 전환율 | 마지막 I 구매율 | T=1.04초 | T=2.08초 |
|---|---:|---:|---:|---:|
| MAP 4×2 | 40.742% | 61.549% | 22.56초 | 31.52초 |
| MAP 8×1 | 0% | 35.113% | 22.56초 | 31.25초 |
| Risk-weighted 4×2 | 0.446% | 35.402% | 22.56초 | 31.25초 |
| 항상 v54 gate | 0% | 35.113% | 22.56초 | 31.25초 |
| 항상 fixed 3R+I | 100% | 100% | 22.56초 | 31.92초 |

표는 H=1, S=P=0에서 padding 전 값이다. T가 2.08초라면 공통 cap은 31.92초다. H=2면 T=1.04/2.08초의 공통 cap은 각각 14.24/19.44초다. T=2.08인 예에서 MAP 4×2는 네 번째 impact를 더 자주 사서 MAP 8×1보다 deployment당 약 0.275초 더 든다. Session setup을 줄여 회수하는 시간은 4S/H이므로 H=1에서 S가 약 0.069초를 넘으면 4×2의 총 모델 시간이 더 짧아진다. 이는 실제 setup·reset 시간을 측정한 결과가 아니다.
