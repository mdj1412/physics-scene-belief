# v49 · Frozen policy와 GPU 측정

## 어느 입력을 썼는가

v49에서 재학습한 parameter는 없다. v48의 direct artifact는 v47 source catalog 12개에서 original posterior의 잔차를 7개 항의 regularized least squares로 fit했다. 입력은 이미 지불한 sensor arm, query 높이, original stick probability, 0.4 N에서 측정한 signed differential displacement다. 같은 query 위치에 probe를 둔 direct와 31 mm에 고정한 direct는 source에서 학습한 station mode를 그대로 쓴다. 두 위치가 다른 경우 fixed는 source의 spatial transfer 0.60에서 학습한 remote 계수를 쓴다. 측정 위치의 차이와 source에서 정한 계수의 차이를 분리하기 위해 fixed 관측에도 local 계수를 적용한 fixed-shared를 따로 기록했다. 같은 위치 31 mm에서는 두 direct 예측이 모든 state, optical profile 및 noise에서 bitwise 일치했다. [배포 코드](policy49.py) · [Forecast 생성](forecast49.py)

Projection gate는 selected observation이 query profile일 때에만 source-trained conditional risk를 계산하고, 원래 posterior 또는 projection posterior를 한 번 실행한다. 선택되지 않은 query profile은 NaN mask다. Source에서 고정한 높이별 profile feature와 displacement bin을 쓴다. Spatial dependence와 weak-to-full retention은 채점용 intervention이다. 실제 policy에는 station의 숨은 상태나 population-level ρ, κ가 들어오지 않는다.

## 비용을 어떻게 측정했는가

GPU가 비어 있음을 확인한 node56의 RTX PRO 6000 Blackwell에 한정해, 물체 512개와 noise realization 하나의 전체 배포 경로를 쟀다. Factual acquisition, 이미 구매한 관측의 포장, gate 및 선택한 decoder 한 번, direct일 경우 host의 scalar calibration, GPU synchronize까지 wall time에 포함했다. Label solver, 가상 센서 생성과 물리 장비 시간은 분리했다. 12개 catalog 각각에서 (ρ,κ)=(0,0),(1,1) 순서를 무작위로 섞었다. 각 stratum에서 4개 정책의 순서를 다시 무작위로 바꾸고 warmup 1회 뒤 3회를 기록했다. 따라서 정책당 72개 timed batch가 있고, 12개 catalog를 block으로 삼아 paired bootstrap 16,000회로 p90 차이의 pointwise 95% interval을 계산했다. 비교하는 단위는 512개 물체의 elapsed time이다. 두 극단은 latency 측정용이고 Brier 비교는 모든 15개 조건에 대해 계산한다. [Benchmark](bench49.py)

## 측정 결과

| Policy | p90 전체 [95% CI], ms | (ρ,κ)=(0,0) | (ρ,κ)=(1,1) |
|---|---:|---:|---:|
| Always-original | 432.12 [431.79, 432.74] | 431.88 | 432.41 |
| Query-local direct | 431.87 [431.67, 432.07] | 431.86 | 431.86 |
| Fixed-site direct | 432.19 [431.90, 432.85] | 432.18 | 432.14 |
| Query-local projection gate | 436.03 [435.59, 436.40] | 435.85 | 436.55 |

직접 보정 두 방식 사이 p90 차이는 local − fixed **−0.32 ms [−0.88, +0.02]**였다. 같은 local readout을 받아 projection을 선택하면 direct보다 **4.15 ms [3.79, 4.59]** 더 들었다. Actual online 확률과 저장 forecast의 최대 절대 차이는 4.44×10⁻¹⁶이었다. 공통으로 쓰는 GPU 비용이 지배적인 두 direct 정책은 같은 sensing budget에서 측정 위치로만 차이가 난다. 서로 다른 v48·v49 run 사이의 절대 latency 변화로 solver 속도를 추론할 수 없다. [전체와 stratum별 결과](timing49.json) · [Raw timing](timing49/) · [동결 확인](benchmark_freeze49.sha256)

GPU guard는 10초 간격으로 외부 compute process를 감시했다. Forecast와 crossover 모두 종료 코드 0, 관측된 foreign process 0이었다. 별도의 CPU audit에서는 target label을 읽지 않고 source artifact 식으로 각 policy의 확률을 재구성했다. V48에서 배포한 remote-state map으로 되돌린 fixed-site에서도 같은 예측인지 확인했다. 검산한 여섯 경로에서 각각 2,359,296개 확률을 비교했고 최대 절대 차이는 5.55e-16이었다. [검산 코드](audit_policy49.py) · [수치 기록](audit_policy49.json)
