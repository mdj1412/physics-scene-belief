# v64 보조 실험 · 같은 impact를 반대 조명으로 촬영한 simulation

## 질문과 비교 범위

이 절은 실제 반사 표면에서 관측된 발생률을 추정할 수 있는지와 별도로, lighting intervention의 영향을 계산하는 방법을 검증한다. 기존 v63의 새 표면 48개·board 8개와 held-out deployment 물체 12,288개를 고정하고 새로 생성한 12,288개 calibration pilot에서 각 8개 물체를 사용했다. 총 98,304개 impact opportunity다. 네 P/C·환경 가설은 같은 물체를 공유하는 counterfactual이다. 생성한 2방향 raw clip, strain과 전자센서 자료를 저장했다. v63의 deployment label·frozen decoder를 재사용했기 때문에 이것은 동일 test bank의 민감도 분석이다. [사전 계획](SIMULATION_PLAN_KO.md) · [구현](illumination64.py) · [원자료 checksum](sim_raw64.sha256)

Reference는 2.0 N이고, 모든 정책이 동일한 8 references, 8 impacts, 16 mechanical resets를 배정받는다. Confidence와 같은 수 무작위 보류는 electronic reset 4회를 한다. 상시 v54 gate는 calibration 판독에 무관하게 세 deployment reference에서 결정한다. −30°와 +30°는 각각 사전 지정한 별도 정책 조건이며 한 정책은 여덟 접촉 내내 한 방향의 조명만 사용한다. 비교용 counterfactual 생성은 두 방향에 같은 latent deformation, camera noise, ADC, strain, 전자센서 표식을 사용한다. 두 영상 중 target loss가 작은 방향을 고르는 선택은 없다.

두 방향에 공통인 ghost opportunity 확률은 35%이고, 별도의 single-frame tracking error 확률은 15%다. Lens/camera 방향을 0°로 두었다. Surface별 facet 평균은 −18°~+18°이고, 각 접촉에서 표준편차 9°를 더한다. 원래 영상에서 움직임을 반전시킨 광학 ghost의 세기는 facet tilt와 조명 half-angle 간 차이를 사용한 Gaussian angular lobe로 정했다. 강한 ghost를 만드는 표면 방향은 두 조명에 상관되고, 폭 26°이면 폭 8°보다 양쪽에 동시에 보일 수 있다. Transient error와 underlying facet은 lamp 사이에 공유한다. 실제 BRDF 또는 현장 fault 빈도를 측정하여 fit한 상수는 없다. 8회 impact로 두 방향의 영상을 동시에 구매하거나 새 데이터로 reader를 학습하지 않았다. [계측 모형](illumination64.py)

## 어떤 오류를 셌는가

Simulation에서 확인할 수 있는 operational event를 **ghost blend가 0.5를 넘고 transient가 없으며, 기준 v61 reader가 reference·impact를 모두 읽었으나 mismatch를 오판한 경우**로 지정했다. 이 정의에 포함되는 수는 광학 mirror에 끌린 잘못된 방향을 나타낸다. 명시적 marker ID를 출력하는 tracker가 아니므로 실제 물체의 identity switch 빈도와 동일시할 수 없다. 표의 %는 활성 pair가 분모다. 각 행 196,216개에는 P/C·환경 counterfactual이 포함돼 있다. [부호·분모 집계](sim_score64.json)

| Angular width / lamp | Accepted v61 / 활성 pair | 강한 ghost 오판 / 활성 pair | Confidence의 전체 부호 오류율 / accepted |
|---|---:|---:|---:|
| 8° / −30° | 17,370 / 196,216 | 1,362 / 196,216 (0.694%) | 873 / 9,504 (9.19%) |
| 8° / +30° | 17,479 / 196,216 | 1,239 / 196,216 (0.631%) | 722 / 9,507 (7.59%) |
| 26° / −30° | 16,731 / 196,216 | 3,553 / 196,216 (1.811%) | 2,276 / 9,028 (25.21%) |
| 26° / +30° | 16,495 / 196,216 | 3,302 / 196,216 (1.683%) | 2,113 / 8,877 (23.80%) |

독립 noise·dropout·quantization을 가진 differential strain sensor는 196,216개 활성 쌍 중 145,344개(74.07%)에서 두 부호를 읽었고, 그중 1,381개(0.950%)의 mismatch 판정이 틀렸다. 같은 underlying strain에 반응하면서 서로 다른 noise를 쓰는 별도의 simulated sensor다. Camera policy는 strain 값을 입력받지 않는다. [센서와 판독](sim_score64.json)

## 조명으로 실제 후속 구매가 바뀌는가

같은 8개 물체에 조명만 바꾸고 frozen confidence router를 다시 실행했다. 상위 선택은 표면의 후속 물체에서 네 번째 관측을 **고정 impact**로 하거나 **v54 gate에 위임**하는 것이다. 위임해도 v54가 그 session에서 impact를 고르면 실제 구매는 같다. 따라서 동일 deployment session에서 끝까지 수행한 **실제 R/I 전환율**을 함께 보고한다. 아래 CI는 24개 catalog와 8개 board의 paired product bootstrap이다. [전환 계산](sim_score64.py) · [실제 acquisition mask의 독립 검산](sim_pair_check64.py)

| Lighting intervention | 상위 선택 전환율 | 실제 네 번째 R/I 구매 전환율 |
|---|---:|---:|
| 8° lobe, −30° → +30° | 0.254% | 0.165% |
| 26° lobe, −30° → +30° | 0.321% | 0.208% |
| 두 lobe 동일 비중 | **0.288%** [0.076, 0.542] | **0.186%** [0.049, 0.352] |

같은 물체의 두 조명에서 같은 random 우선순위를 쓰는 64개 lottery로 무작위 보류의 전환율도 추정했다. 상위 선택 0.291% [0.067, 0.565], 실제 구매 0.188% [0.043, 0.365]였다. 같은 deployment 입력에서 상시 v54 gate는 조명으로 바뀌는 구매가 0%다. 주 정책인 confidence의 실제 구매 전환율은 deployment의 모든 acquisition mask를 직접 합산한 별도 코드에서도 일치했다. 한 방향 이상에서 강한 ghost 오판으로 집계된 활성 쌍 중 양쪽에서 같은 조건으로 틀린 비율은 좁은 lobe에서 0/2,601, 넓은 lobe에서 1,757/5,098(34.46%)였다. 좁은 lobe는 두 조명의 강한 영역이 모형상 겹치지 않으므로, 이 0은 field의 경험적 발견으로 해석할 수 없다. [짝지은 광학·행동 검산](sim_pair_check64.json)

## 같은 8회에서의 held-out 위험

각 pilot에서 v61이 수용한 n개 중 confidence가 남긴 수 k를 고정하고 모든 k-subset의 실제 행동별 Brier를 적분했다. v54 gate의 고정값은 같은 v63 deployment bank에서 재검산한 값이다. 네 조명 조건을 사전에 균등 가중하고 P/C와 장치·환경의 연결을 v63 primary와 같게 유지했다. 새 capture에서 세 정책을 재실행해 얻은 결과다. [정책 채점](sim_score64.py) · [전체 결과](sim_score64.json)

| 조건 | Confidence | 같은 수 무작위 보류 | v54 gate |
|---|---:|---:|---:|
| 8° / −30° | 0.155701538423 | 0.155701540684 | 0.155701480487 |
| 8° / +30° | 0.155701519906 | 0.155701521896 | 0.155701480487 |
| 26° / −30° | 0.155701559363 | 0.155701559376 | 0.155701480487 |
| 26° / +30° | 0.155701542307 | 0.155701544875 | 0.155701480487 |
| **전체 평균** | **0.155701540000** | **0.155701541708** | **0.155701480487** |

Confidence − random은 **−0.00171 × 10⁻⁶**, 두 비교에 대한 동시 95% CI는 [−0.01241, +0.00821]였다. Confidence − gate는 **+0.05951 × 10⁻⁶** [+0.01021, +0.11934]였다. 사전 가중치에서 유의한 random 대비 Brier 개선은 확인되지 않았고 v54가 confidence보다 낮은 평균 loss를 냈다. Bias가 큰 persistent ghost와 드문 후속 행동 변화로 인해, 계측 실패가 곧바로 downstream loss 개선의 기회를 뜻하지 않는다.

## GPU 측정

Datacenter3 node55의 Slurm 2744468에서 RTX PRO 6000 Blackwell 한 장의 기존 process·메모리·utilization이 모두 비어 있음을 재확인한 후 실행했다. Memory-resident synthetic frames와 두 ADC waveform에서 실제 confidence·같은 수 무작위 추출·상시 gate를 실행하고, 각 후속 선택과 frozen GPU forecast까지 CUDA synchronization을 포함해 측정했다. 24 catalog × 4조건, 정책당 96 paired block, 한 번에 512 물체다. 각 방법이 실행 순서의 매 위치에 32번씩 배정됐다. [GPU 구현](sim_latency64.py) · [GPU 결과](sim_latency64.json)

| 방법 | GPU p90, ms / 512 물체 |
|---|---:|
| Confidence | 483.406 |
| 같은 수 무작위 보류 | 482.531 |
| 상시 v54 gate | 480.780 |

Confidence에서 두 대조의 p90을 뺀 차이는 각각 +0.876 ms [동시 95% CI −6.475, +14.028], +2.627 ms [−0.273, +15.143]였다. 전체 384개 terminal session의 같은 96개에서 세 정책 모두 impact를 구매해 timing 표본은 서로 다른 정책 경로가 발생할 때의 비용을 직접 분리하지 못한다. 저장된 forecast 대비 재생 오차는 세 방법 모두 0이었다. GPU guard는 경쟁 process 없이 종료했다. 장비의 조명 전환·impact·reset·video export·실물 sensor 동기화 시간은 미측정이다. [실행 로그](sim_gpu64.log) · [Guard](sim_guard64/results/worker_exit.json)

실험 자료의 SHA-256과 72개 raw panel의 무결성을 확인했다. 이전 v63의 비교용 potential loss는 `../reflective_tracking_v63/scored63`와 frozen model에 의존한다. [실험 checksum](sim_bundle64.sha256) · [Raw panel checksum](sim_raw64.sha256)

