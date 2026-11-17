# v48 · 약한 preload 한 번을 어디에 가하고, 관측한 변형을 어떻게 사용할 것인가?

**미래에 접촉할 위치에 0.4 N preload를 한 번 가하고, 표면 양쪽의 signed displacement로 projection을 선택하자 12개 새 catalog 모두에서 두 baseline보다 held-out Brier loss가 낮아졌다.** Query-site preload는 **0.154652**, 같은 접촉 예산의 fixed probe는 **0.155636**, always-original은 **0.165778**이었다. 추가 관측 없이 같은 방식으로 훈련한 router는 **0.158575**였다. 따라서 추가 접촉 자체에서 얻은 개선량은 0.003923이고, 이미 한 번 접촉하기로 했을 때 위치까지 맞춘 개선량은 0.000984다. [사전 분석 계획](PLAN_KO.md) · [12개 catalog 결과](score48.json)

**논문에서 더 유망한 방향은 query 위치에서 측정한 변형을 최종 예측에 직접 반영하는 것이다.** 같은 local 관측으로 original decoder의 stick 확률을 직접 보정한 source-only 대조는 **0.145195**였다. 센서 입력을 두 decoder 중 하나를 고르는 데만 쓰면 예측에 유용한 정보 일부가 남는다. 모든 물리 결과와 displacement는 simulation에서 생성했고, 실제 sensing·reset 시간은 아래에서 장비 가정별로 계산했다. [직접 보정 결과](direct48.json)

## 1. 무엇을 독립적으로 바꿨나

V47에서는 optical profile과 pre-impact 영상이 같은 두 물체가 하중 아래에서는 다르게 휠 수 있었다. V48에서는 물체를 약하게 누르고 원상 복귀시킨 뒤 나중의 contact outcome을 예측한다. 접촉 전·약한 하중 상태에서, nominal station 양쪽 ±1.5 mm에 있는 두 fiducial의 높이 변화를 한 snapshot에서 측정한다. 오른쪽 변화에서 왼쪽 변화를 빼면 scalar 한 개가 나온다. 오른쪽이 더 많이 움직였는지 알 수 있어 displacement의 크기와 방향을 함께 보존한다. **Force는 0.4 N 명령값이고 policy에 들어가는 실제 측정 채널은 signed displacement 한 개다.**

- Source는 v47의 keel·crown·channel·bead와 12개 catalog다. Target은 새로 만든 **ogee·cove·reed·flute**, support 800–811의 12개 catalog다. 각 catalog의 4 environment × 128 draws를 합하면 6,144회이고, support 내 중복을 제외한 prototype은 5,964개다. 같은 contact site 전후 ±2.5 mm에서는 네 family의 unloaded 위치와 기울기를 맞췄다. 바깥 영역과 하중을 받았을 때의 bending mode는 다르다. [World](world48.py) · [Geometry 검산](pre_physics48.json)
- Optical variant의 부호와 axis alignment(0.10, 0.90)는 mechanical state와 독립적으로 교차했다. Mechanical state는 inert 50%, 음·양 방향 active 각 25%로 가중했다. 두 높이 19/31 mm, 세 offset, 두 finite pressure radius, 두 surface·두 law 및 네 paid-noise 반복을 사용했다. Primary stick label은 impact 이후 residual tangential speed ≤0.020 m/s다. 양·음 변형을 바꿨을 때 실제 label이 달라진 비율은 34.27% [catalog 95% CI 33.00, 35.55]다. [물리 검산](outcomes_physics48.json)
- Signed displacement는 state 번호를 그대로 읽지 않고 loaded contour에서 두 지점의 차이를 계산해 만들었다. Weak-load 변형은 2.0 N reference에서 지정한 변형의 20%로 놓았다. 무잡음 active 신호의 크기는 0.0684–0.0858 mm이고 station·object gain, bias, 표준편차 0.028 mm의 readout noise를 더했다. Weak force에서 full-load 방향이 전달되는 정도는 이 simulator의 핵심 가정이다. [Sensor generator](sensor48.py)
- Query와 12 mm 떨어진 다른 station은 확률 0.60으로 query state를 복사하고 나머지는 같은 prior에서 새 state를 뽑는다. 우연한 일치까지 포함한 기대 state 일치율은 75%다. 이 값은 실제 물체의 측정치가 아니다. 각 query state로 조건화한 matched Monte Carlo 실험이며, 하나의 영구적인 물체 state map에서 두 query를 연속 관측한 궤적도 아니다. [Sensor 및 비용 설명](PHYSICS_KO.md)

## 2. 어떤 정책을 같은 예산으로 비교했나

모든 방법은 v47의 동일한 factual evidence, K256 acquisition과 original/projected posterior를 사용한다. 환경별 128개 중 115개는 기존 seven-point query view를, 13개는 기존 deployment probe를 산다. 아래 preload는 그 **뒤에 추가되는** 관측이다. 기존 deployment probe를 산 물체에서는 두 decoder의 출력이 같지만, 두 paid policy는 그 물체에서도 preload와 reset을 모두 지불한다. Router가 보는 것은 명령한 query 높이, 이미 구매한 range profile(미구매 시 masking), 해당 정책이 실제로 구입한 displacement다.

| 정책 | 추가 관측 | 어떤 확률을 내는가 |
|---|---|---|
| Always-original | 없음 | 기존 profile을 그대로 사용한 original posterior |
| No-extra router | 없음 | 이미 구매한 profile로 source에서 학습한 상대 Brier에 따라 original/projected 선택 |
| Fixed probe | 모든 물체에서 31 mm, 0.4 N, 한 번 | Query와 같은 station에서는 local model, 다른 station에서는 source에서 같은 구조로 학습한 remote model로 선택 |
| Query-site preload | Query가 19 mm면 19 mm, 31 mm면 31 mm, 0.4 N, 한 번 | 해당 station의 signed displacement와 기존 profile로 선택 |

Source 8개 catalog에서 모델을 fit하고 별도 source 4개에서 고정 위치를 골랐다. Fixed 19와 fixed 31의 validation Brier는 0.156280과 0.156243이었다. 이후 31 mm를 고정하고 12개 source catalog로 계수만 다시 학습했다. 두 위치의 source 차이는 작다. 각 측정 구간에서 7개 profile feature와 그 2차 항을 쓰는 ridge regression이 **D = Brier(projected) − Brier(original)**을 추정한다. D가 음수일 때만 projection한다. 미래 state, family 이름, target label 및 다른 station의 측정값은 배포 경로에서 제외했다. Query가 31 mm일 때 두 paid 정책의 입력과 모델이 모두 같으므로 결과도 정확히 같아야 한다. [Fit 및 source validation](ROUTER_KO.md) · [동결 artifact](policy48.json)

## 3. 한 번 눌러서 coupling과 방향을 실제로 구별했나

다음 값은 source에서 별도로 threshold를 정한 sensor diagnostic이다. Projection 선택의 정확도와 Brier는 다음 절에서 직접 평가한다.

| Held-out 지표 | Query-site preload | Fixed 31 mm |
|---|---:|---:|
| Coupling AUROC, active 대 inert | **0.9391** [0.9318, 0.9465] | 0.8527 [0.8452, 0.8602] |
| Active임을 알고 있을 때 부호 정확도 | **99.47%** [99.37, 99.57] | 89.51% [89.01, 90.01] |
| Inert / negative / positive 세 상태의 balanced accuracy | **85.76%** [84.62, 86.90] | 75.27% [74.02, 76.53] |

99.47%는 세 상태를 모두 맞히는 비율이 아니다. Query-site 측정에서도 실제 negative의 20.0%, positive의 13.6%가 inert로 분류됐다. 변화가 발생했음을 판별하는 것이, 활성임을 아는 상태에서 방향을 구별하는 것보다 어려웠다. 소스에서 고정한 threshold 및 전체 confusion matrix를 함께 보관했다. [Sensor audit](sensor_identification48.json)

## 4. Held-out Brier loss와 GPU p90

| 정책 | Brier ↓ | GPU p90, ms / 512 objects [95% CI] ↓ | 전체 projection 선택률 | 추가 touch / reset |
|---|---:|---:|---:|---:|
| Always-original | 0.165778 | **467.86** [467.02, 468.15] | 0% | 0 / 0 |
| No-extra router | 0.158575 | 471.47 [470.97, 472.01] | 60.52% | 0 / 0 |
| Fixed probe | 0.155636 | 471.97 [471.21, 472.74] | 55.93% | 1 / 1 |
| Query-site preload | **0.154652** | 472.11 [471.51, 472.92] | 54.20% | 1 / 1 |

| Query-site preload에서 비교 정책을 뺀 값 | Paired Brier 차이 [95% CI] | Paired GPU p90 차이, ms [95% CI] |
|---|---:|---:|
| − Always-original | **−0.011126** [−0.011718, −0.010535] | +4.24 [+3.80, +5.09] |
| − Fixed probe | **−0.000984** [−0.001179, −0.000790] | +0.13 [−0.45, +1.07] |
| − No-extra router | **−0.003923** [−0.004152, −0.003693] | +0.64 [−0.24, +1.51] |

처음 두 Brier 비교는 사전 지정한 두 개의 Bonferroni simultaneous interval이다. 마지막은 secondary paired interval이다. Noise, offset, radius, factorial 조합은 catalog 안에서 먼저 평균했다. 독립 통계 단위는 catalog 12개이며, 두 primary 비교 모두 12/12에서 query-site가 이겼다. GPU는 빈 allocation 두 장으로 forecast를 나누고 한 장의 RTX PRO 6000 Blackwell에서 12개 batch × 5회, 순서를 섞은 crossover로 p90을 측정했다. Host→GPU scalar 전송, 공통 acquisition, 선택 및 실제 단일 decoder를 포함한다. 16,000회 paired catalog bootstrap 구간이다. Fixed와 query-site 사이에 작은 compute 차이가 있다고 판정할 근거는 부족하다. [Paired Brier](score48.json) · [GPU audit](gpu_audit48.json) · [GPU 실행 범위](GPU_KO.md)

**같은 위치에서는 차이가 0이었다.** 31 mm에서는 두 정책의 Brier가 모두 0.134367이었다. 19 mm에서는 query-site 0.174937, fixed 0.176906으로 차이가 −0.001969 [두 높이 동시 95% CI −0.002358, −0.001580]였다. 이득은 inert에서 작았고(−0.000040, 구간이 0 포함) negative active에서 −0.001218, positive active에서 −0.002639였다. 네 family 각각의 query-site − fixed 차이도 −0.000879에서 −0.001096 사이였다. [조건별 결과](score48.json)

**방향의 부호가 중요했다.** 같은 source에서 displacement의 절댓값만으로 학습하면 Brier가 0.158135였다. Signed policy의 추가 이득은 −0.003483 [−0.003701, −0.003266]이었다. 같은 물체, optical evidence와 noise를 유지한 채 preload를 평가할 mechanical state와 독립적으로 재배정한 기대 Brier는 0.160285였다. 관측을 하나 더 받아 rule의 선택 빈도가 달라지는 것만으로 이득을 설명하기 어렵다. [대응을 끊은 대조](score48_checked.py)

![Query-site preload의 Brier와 station 간 transfer sensitivity](preload_v48.png)

왼쪽은 query-site에서 세 comparator를 뺀 Brier 차이다. 오른쪽은 두 station이 같은 state를 복사하는 확률을 바꾼 결과다. 음수는 query-site가 유리함을 뜻하며 음영은 각 조건의 pointwise catalog 95% CI다.

## 5. 이득이 사라지는 두 가지 조건

**물체 전체가 같은 방향으로 휘면 위치를 맞출 필요가 줄어든다.** 다른 station의 state 복사 확률이 0 / 0.30 / 0.60 / 0.85 / 1.00일 때 frozen fixed 대비 query-site 이득은 각각 0.002020 / 0.001504 / 0.000984 / 0.000450 / 0.000157이었다. 복사 확률 1에서 fixed에도 완전한 state 전달을 반영한 source calibration을 적용하면 query-site − fixed가 **+0.000029 [−0.000055, +0.000112]**로 바뀐다. Station별 noise와 gain은 여전히 다르다. 이 구간으로 동등성을 단정할 수는 없지만, 이 조건에서 query 위치가 더 정확하다는 증거는 사라졌다. [별도 transfer replay](transfer48.json)

**약하게 누를 때와 실제 충돌에서 휘는 방향이 달라지면 이득이 줄어든다.** 사전 지정한 polarity replay는 inert는 유지하고 active의 ± 측정과 full-load outcome 대응만 뒤집었다. 항상 뒤집히면 query-site가 0.164670, fixed가 0.163071까지 악화됐다. 두 대응을 혼합한 점추정에서 reversal이 약 **38.1%**면 fixed 대비 local의 우위가 사라지고, 약 **39.2%**면 추가 접촉 없는 router 대비 이득도 사라진다. 이 수치는 원자료 두 끝점 사이의 선형 혼합으로 얻었고 실제 재료의 부호 반전율은 측정하지 않았다. [동결한 민감도 계획](PLAN_KO.md) · [Replay 점수](score48.json)

## 6. Sensing·reset 비용과 손익분기점

| 물체당 추가 비용 | Always-original / No-extra | Fixed probe / Query-site |
|---|---:|---:|
| Touch, reset 횟수 | 0, 0 | 각 1, 1 |
| Prescribed sensing cycle | 0 | **250 ms** = 2 mm 접근 100 + 0.4 N ramp 100 + 측정 50 |
| Prescribed reset cycle | 0 | **190 ms** = unload 50 + 2 mm 후퇴 100 + 안정화 40 |
| Station까지 횡이동을 제외한 기본 합계 | 0 | **440 ms/object** |

횟수는 실행한 policy의 예산이고, 시간은 속도·하중·대기를 지정한 **시나리오 추정치**다. 실제 장비로 잰 cycle time, reset 실패율, 잔류 변형 또는 소비전력은 없다. Differential displacement만으로 전체 normal travel이나 mechanical work도 알 수 없다. 시작·종료점을 query 위치로 두고 60 mm/s로 횡이동한다고 가정하면 fixed만 19 mm trial에서 24 mm를 왕복한다. 두 높이 동일 가중 평균은 fixed **640 ms**, local **440 ms**다. 중간 지점에서 출발해 복귀하면 둘 다 **640 ms**다. 실제 배치에 따라 물체당 물리 비용이 달라지며, 512-object GPU p90과 다른 단위다. [비용 가정](PHYSICS_KO.md)

한 번 더 누르는 것을 정당화하려면 **no-extra router와의 차이**가 기준이다. 이 비교의 손익분기 비용은 물체당 **0.003923 Brier-equivalent**(95% CI 0.003693–0.004152)다. 예를 들어 추가 물리 시간을 선형 패널티로 계산하고 기본 0.44초가 걸린다면, 1초당 패널티가 약 **0.00892 Brier** 미만일 때 평균 순이득이 남는다. 안전·처리량·손상에 실제로 얼마를 부여할지는 운용 장비가 결정한다. 같은 횟수의 fixed probe와 비교할 때는 공통 비용이 상쇄되고 위치별 이동 비용만 별도로 남는다.

## 7. 논문의 가장 강한 대안 설명을 시험했다

측정값 자체가 유용하다면, 굳이 projection 여부를 고르지 않고도 이길 수 있다. 이를 위해 **source-only direct calibration**을 추가했다. Original posterior가 예측한 확률과 signed displacement, 둘의 상호작용으로 binary outcome에 대한 residual을 ridge regression으로 학습하고 확률을 [0,1]로 제한했다. Query 높이와 기존에 구매한 sensor 종류에 따라 계수를 나눴다. Source는 같은 v47 archive뿐이며 target scoring 전에 coefficient checksum을 남겼다. Primary 두 비교 밖에서 수행한 보조 실험이다.

| 동일한 측정 예산 | Projection router의 Brier | Original 확률에 직접 반영한 Brier |
|---|---:|---:|
| 추가 관측 없음 | 0.158575 | 0.158014 |
| Fixed 31 mm 한 번 | 0.155636 | 0.150776 |
| Query-site 한 번 | 0.154652 | **0.145195** |

같은 local preload에서 direct − router는 **−0.009457 [−0.011584, −0.007330]**였다(전체 및 두 inherited sensor arm에 대한 세 simultaneous interval). 기존 deployment probe를 산 경우에는 original과 projected가 같아서 routing에 재량이 없지만, direct update는 여기서도 정보를 사용할 수 있다. 이 차이만으로는 결과 전체가 설명되지 않는다. 기존 query view를 산 물체로 한정해도 direct 0.140498 대 router 0.149232였다. Direct끼리의 query-site − fixed도 −0.005581 [−0.006727, −0.004435]였다. 이득은 관측 위치와 측정값을 예측에 쓰는 방식 양쪽에서 나온다. [Direct calibration 구현](direct48.py) · [Source artifact](direct_source48.json) · [별도 채점](direct48.json)

별도의 GPU crossover에서 같은 catalog 12개를 다시 사용하고 세 정책의 실행 순서를 무작위로 바꿨다. 정책마다 48개 timed batch를 측정했다. 512개 물체당 end-to-end p90은 always-original 466.04 ms, projection gate 470.40 ms, direct calibration 466.61 ms였다. 같은 실험 안에서 direct − gate는 **−3.79 ms [−4.54, −1.79]**, direct − always-original은 +0.56 ms [−0.49, +2.83]였다. Direct는 original posterior 하나를 계산한 뒤 같은 scalar로 확률을 갱신한다. 저장 예측과 online 경로 사이의 최대 확률 차이는 4.44×10⁻¹⁶이었다. 이 수치는 §4와 별개로 실행한 crossover에서 얻었다. [Timing 원자료 요약](direct_timing48.json) · [측정 코드](direct_timing48.py)

## 8. 논문에서 어떤 주장을 지지하는가?

가장 방어하기 좋은 주장은 **접촉 전 영상에서 비슷해 보이는 물체라도 국소 변형이 stick/slip을 바꾸는 환경에서는, query 위치에서 한 번 측정한 signed displacement가 같은 비용의 fixed-site 측정보다 예측에 유용할 수 있다**는 것이다. 같은 위치에서 두 정책이 정확히 일치하고, 다른 위치에서는 active 상태에서 차이가 커지고, station 간 상태가 완전히 전달되는 control에서는 차이를 확인하기 어려웠다. 세 관찰이 같은 해석을 지지한다. 단순한 direct calibration이 projection gate보다 정확하고 빨랐기 때문에, 현재 증거로는 projection 선택 알고리즘 자체를 논문의 주기여로 삼기 어렵다. Query 높이로 접촉 위치를 정하는 정책만 평가했으므로 여러 후보의 가치를 추정하는 adaptive acquisition의 우월성은 미검증이다.

**다음 확인 실험은 실물에서 weak-load 신호가 실제 동작까지 전달되는지 측정하는 것이다.** 같은 물체의 두 station에 force-controlled preload를 가하고 독립적인 displacement readout으로 두 위치의 부호 일치율을 잰다. 별도 reset 뒤에는 query 높이에서 full-load stick/slip을 기록한다. 순서와 station을 무작위화하고 weak-to-full polarity reversal, 잔류 변형, cycle time, reset 실패도 보존한다. 정책 비교는 direct-local, direct-fixed, no-extra, projection-local을 사전 고정하고, object family와 surface session 단위로 source와 holdout을 분리한다. 결과를 보기 전에 장비에서 허용할 sensing 시간과 최소 Brier 개선을 정할 수 있다. 이 설계면 신호가 없는 이유가 낮은 sensor resolution인지, 공간적으로 균일한 물체인지, 하중이 바뀔 때 변형까지 뒤집히는지 따로 점검할 수 있다.

현재 simulator는 weak-load와 full-load에서 변형을 공유하는 구조를 명시했고 같은 constitutive assumption을 양쪽에 쓴다. 새로운 contour에서 얻은 결과는 이 구조 안에서의 재현이다. 실제 힘과 변위의 관계, 실제 접촉에서의 국소성, optical tracking 오차와 reset 비용은 독립 실험이 남아 있다.

## 선행연구 검토와 연구 질문

여섯 편을 세 질문으로 읽었다. 약한 접촉에서 무엇을 실제로 측정할 수 있는가, 얻은 측정값을 예측에 어떻게 사용할 수 있는가, simulator에서의 이득을 어디까지 일반화할 수 있는가이다. 아래 표의 표본 단위와 endpoint는 서로 다르다.

| 연구 | 질문과 방법 | 표본·데이터 | 결과, 한계, v48과의 관계 |
|---|---|---|---|
| Su et al., 2012 | BioTac의 force feedback으로 접촉 방향을 맞추고 indentation으로 compliance를 구별할 수 있는가? | Barrett arm, aluminum plate 1개와 서로 다른 rubber 4종, 목표 normal force 10 N | 상대 compliance를 구별했다. 낮은 force 구간에서는 sensor skin과 joint cable의 변형이 측정값에 섞였다. v48이 가정한 0.4 N에서 signed displacement를 읽을 수 있는지는 독립적으로 확인해야 한다. [원문](https://pmc.ncbi.nlm.nih.gov/articles/PMC3405524/) |
| Van der Merwe et al., RSS 2023 | Partial point cloud와 force/torque로 loaded geometry와 contact patch를 동시에 추정할 수 있는가? Simulated supervision으로 Neural Deforming Contact Fields를 학습했다. | 실물에 46 mm sponge, 두 depth camera, force/torque sensor를 사용했다. Material identification에 flat press 16회를 사용했다. | 실물에서 point-cloud baseline보다 나은 재구성을 보고했다. 여러 sensor와 별도 material fitting을 사용했으므로 one-touch future-slip 예측의 근거로 바로 옮길 수 없다. [원문](https://www.roboticsproceedings.org/rss19/p080.pdf) |
| Pfrommer et al., CoRL | 물리적 contact를 반영한 learned dynamics가 impact와 stiction을 예측하는가? Signed distance와 contact-frame Jacobian을 학습했다. | 실제 cube toss 750회 중 정제한 570개 trajectory | 구조화한 모델에서 물리적으로 타당한 trajectory 예측을 보였다. v48의 두 decoder 중 고르기보다 관측을 outcome에 직접 연결하는 대안의 근거다. Soft surface에서 local preload의 순가치는 별도 문제다. [원문](https://proceedings.mlr.press/v155/pfrommer21a/pfrommer21a.pdf) |
| Mozannar & Sontag, ICML 2020 | 입력별로 예측기와 expert 중 하나를 선택하는 학습 목표를 어떻게 구성하는가? Cost-sensitive surrogate와 일관성을 분석했다. | 이론 및 일반 classification 실험. 물리 접촉 표본 없음 | 상대 오류로 두 predictor 중 하나를 고르는 원리는 이미 연구됐다. 물체를 실제로 한 번 더 만지는 비용이나 측정 위치는 별도 문제다. [원문](https://proceedings.mlr.press/v119/mozannar20b.html) |
| Luo et al., 2026 preprint, PhysiCalWorld | Exploratory interaction clip을 physics descriptor로 압축해 물체별 dynamics에 맞출 수 있는가? | 네 환경에서 action-conditioned rollout과 planning을 평가했다. | 탐색 후 예측이 개선됐다. Clip의 정보량과 수집 비용, latent rollout 단위가 0.4 N scalar/Brier 실험과 다르다. 같은 prefix를 동일 budget으로 제공한 비교가 필요하다. [원문](https://arxiv.org/abs/2610.30125) |
| Acosta et al., RA-L 2022 | Simulator가 실제 impact trajectory를 어느 정도 재현하는가? 시스템별 contact parameter를 fitting하고 Drake·MuJoCo·Bullet을 비교했다. | 실제 tossed cube와 Cassie landing, 두 물리 시스템 | 탄성 regime와 운동에 따라 정확도가 달랐다. v48의 quadrature 수렴은 solver 내부 수치 검증이다. 실제 약한 preload와 이후 impact 사이의 부호 관계는 검증하지 않는다. [원문](https://arxiv.org/abs/2110.00541) |

**지식 지도.** Force command → 실제 material/tool compliance → loaded displacement → 관측으로 추정한 state → 미래 stick 확률이라는 경로에서, Su et al.은 앞 두 단계의 계측을, NDCF는 loaded geometry 추정을, ContactNets는 contact dynamics 예측을 다룬다. PhysiCalWorld는 interaction history에서 예측에 필요한 정보를 학습한다. V48은 한 관측을 두 가지 방식, 즉 decoder 선택과 직접적인 probability update에 넣어 같은 endpoint를 채점한다. Deferral의 일반적 원리와 이 물리 실험의 기여를 구분해야 한다. [Su et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC3405524/) · [NDCF](https://www.roboticsproceedings.org/rss19/p080.pdf) · [ContactNets](https://proceedings.mlr.press/v155/pfrommer21a.html) · [PhysiCalWorld](https://arxiv.org/abs/2610.30125) · [Deferral](https://proceedings.mlr.press/v119/mozannar20b.html)

**근거의 강도와 겉보기 충돌.** 촉각 탐색이 재질이나 변형에 대한 정보를 준다는 점은 실물 연구들이 공통으로 뒷받침한다. 한 번의 매우 약한 preload가 이후 큰 하중에서의 부호를 충분히 알려주는지는 재료·센서·위치에 조건부다. V48의 높은 sign accuracy와 Su et al.의 낮은 force에서 겪은 어려움은 모순되지 않는다. 후자는 실제 fingertip과 actuator deformation이 섞인 displacement를 측정했고, 전자는 두 fiducial의 차이와 명시한 noise model을 사용했다. 고정 위치와 query 위치의 비교, sensor 비용 및 직접 predictor까지 동일 조건으로 맞춘 real-world evidence는 이 문헌 집합에서 부족하다. [Su et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC3405524/) · [실물 impact 검증](https://arxiv.org/abs/2110.00541)

### 다음 연구 질문 10개

| 질문 | 메울 gap, 변수와 검증 방법 |
|---|---|
| 1. 0.4 N에서 실제 표면 변형의 부호가 읽히는가? | Force를 단계별로 바꾸고 unloaded/loaded fiducial, 독립 force transducer, 기구 자체의 displacement를 함께 기록한다. Material session별 sign accuracy와 detection limit를 구한다. [Su et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC3405524/) |
| 2. 작은 preload와 실제 impact 사이에서 부호가 몇 번 뒤집히는가? | Backing stiffness와 표면 마감을 독립 변경한 시험편에서 weak/strong load를 짝짓고 부호 반전율·Brier를 측정한다. [NDCF](https://www.roboticsproceedings.org/rss19/p080.pdf) · [실물 impact](https://arxiv.org/abs/2110.00541) |
| 3. Fixed probe를 얼마나 떨어뜨려도 유효한가? | 0–12 mm 이상의 probe 거리별 spatial covariance와 policy gain을 독립 시험편에서 측정한다. 실제 거리에서 v48 transfer sweep의 가정을 검증할 수 있다. [NDCF](https://www.roboticsproceedings.org/rss19/p080.pdf) |
| 4. 더 저렴한 force channel로 같은 정보를 얻는가? | 같은 preload trajectory에서 differential displacement, force/torque, 두 신호 결합을 비교하고 calibration 시간까지 센다. [Su et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC3405524/) · [NDCF](https://www.roboticsproceedings.org/rss19/p080.pdf) |
| 5. Learned contact decoder에 직접 관측을 주면 routing이 더 필요한가? | Original, projected, smooth direct update와 contact-aware learned dynamics에 같은 source budget과 held-out physical object를 주고 Brier 및 calibration을 비교한다. [ContactNets](https://proceedings.mlr.press/v155/pfrommer21a.html) |
| 6. 어떤 물체는 만지지 않아도 되는가? | No-touch, fixed one-touch, expected-value stopping을 같은 물체에 되풀이하고 Brier 감소와 측정·reset 비용의 실측 교환비를 추정한다. [Deferral](https://proceedings.mlr.press/v119/mozannar20b.html) · [PhysiCalWorld](https://arxiv.org/abs/2610.30125) |
| 7. Preload가 다음 query를 바꾸는가? | No-touch와 preload trial 순서를 무작위화하고 독립 reset·휴지시간을 조작해 hysteresis 및 마모가 event 자체에 미치는 영향을 측정한다. [NDCF](https://www.roboticsproceedings.org/rss19/p080.pdf) · [실물 impact](https://arxiv.org/abs/2110.00541) |
| 8. 동일 측정 예산에서 짧은 interaction clip이 scalar보다 더 가치 있는가? | 동일 trajectory에서 단일 수치와 limited-frame encoder를 학습하고 촬영·전송·훈련 비용까지 맞춘다. [PhysiCalWorld](https://arxiv.org/abs/2610.30125) |
| 9. Binary threshold를 바꿔도 효과가 유지되는가? | Slip speed 연속값, threshold별 Brier, 실제 pose tracking의 해상도를 함께 보고한다. [ContactNets](https://proceedings.mlr.press/v155/pfrommer21a.html) · [실물 impact](https://arxiv.org/abs/2110.00541) |
| 10. 완전히 새 재료와 날짜에서 이득이 남는가? | Probe-site 선택과 preprocessing을 source에서 고정하고, material family·tool·session·날짜를 통째로 제외해 순차적으로 재현한다. [Su et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC3405524/) · [실물 impact](https://arxiv.org/abs/2110.00541) |


## 재현과 검산

서버 작업 폴더는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/preload_projection_v48`이다. 빈 allocation 두 곳에서 forecast를 분담했고, 모든 latency 비교는 node56의 같은 GPU에서 각 crossover 안에 짝지었다. 역할을 world·sensor 검산, source-only policy 학습, GPU 실행·timing 검산으로 나눴다. [사전 계획](PLAN_KO.md) · [동결 목록](master48.sha256) · [물리 검산](PHYSICS_KO.md) · [학습 검산](ROUTER_KO.md) · [GPU 검산](GPU_KO.md)

Target 평가 개시 시점은 2026-11-16 11:49:06 UTC다. Primary scoring의 secondary decomposition에서 NumPy boolean 배열에 단항 음수를 먼저 적용한 식이 실행 오류를 냈다. 곱셈 결과에 부호를 적용하도록 괄호 한 쌍을 옮겼다. 동결본과 실행본, 수정 checksum을 함께 보존했다. Policy, 배정과 loss 정의는 그대로이며, case별 squared error를 별도 집계해 대조했다. Direct calibration은 별도로 동결한 secondary 비교다. [개시 기록](TARGET_GO) · [동결 채점 코드](score48.py) · [실행한 코드](score48_checked.py) · [수정 checksum](scoring_repair48.sha256) · [Primary 결과](score48.json) · [GPU 결과](gpu_audit48.json)
