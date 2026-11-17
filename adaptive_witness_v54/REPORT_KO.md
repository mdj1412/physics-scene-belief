# v54 · 표면 교체 후 첫 세 쌍으로 마지막 calibration을 고를 수 있는가?

**같은 continuous joint decoder와 네 쌍의 예산에서, 첫 세 쌍을 사용해 네 번째 측정을 배정하면 두 fixed 정책보다 평균 held-out Brier loss를 줄일 수 있었다.** 다만 사전 지정한 주가설은 실패했다. Nominal expected value of sample information(EVSI)은 모든 경우에 reference를 골랐다. Acquisition에서 더 넓은 mismatch prior를 쓰거나, 다른 catalog에서 학습한 sign-count 규칙을 썼을 때만 평균 이득이 나타났다. 두 방식 모두 fixed 3R+I보다 session loss p90이 낮다는 근거는 얻지 못했다.

## 무엇을 언제 관측했는가

표면을 교체한 뒤 서로 다른 세 물체에 weak preload(0.4 N)와 reference(2.0 N)를 가해 displacement를 읽었다. 네 번째 물체를 만지기 전에, 앞선 세 쌍만 보고 같은 weak preload 뒤에 다시 reference를 측정할지(R), 실제 impact 방향을 관측할지(I) 결정한다. 고정된 v53 continuous joint decoder는 선택을 마친 뒤 교체 전 네 쌍과 교체 후 지불한 네 쌍을 합쳐, 별도 deployment 물체의 slip 확률을 예측한다. Active impact가 없거나 신호가 약한 경우도 평가에 포함했다. 선택 시점에는 네 번째 weak reading, 구입하지 않은 센서, 이전 표면의 측정값, catalog·law·surface ID와 deployment label을 숨겼다. [설계](DESIGN_KO.md) · [정책과 입력 검증](POLICY_KO.md)

기존 12개 held-out simulation catalog에서 네 environment와 다섯 방향 유지율 조건, 256개 calibration realization을 조합해 61,440개 prefix를 만들었다. Reference–impact mismatch 세 수준과 session drift 네 수준에서 같은 prefix를 공유하고 각 수준을 균등 가중했다. 각 session에서 128개 deployment 물체에 두 관측 후보를 짝지었다. 평균 Brier는 binary outcome noise를 적분한 값이다. Session loss p90은 실제로 binary outcome을 하나씩 뽑아 계산한 session 평균의 90번째 백분위수다. Uncertainty의 독립 단위는 12개 catalog다. [채점 계획](SCORE_PLAN_KO.md) · [채점과 solver sensitivity](RESULTS_KO.md)

## 같은 예산에서 얻은 결과

| 네 번째 관측 선택 정책 | I 구매율 | Mean held-out Brier ↓ | Session loss p90 ↓ | GPU p90 A (ms) | GPU p90 B (ms) |
|---|---:|---:|---:|---:|---:|
| Fixed 4R | 0% | 0.158526317 | 0.182367131 | 466.83 | 465.68 |
| Fixed 3R+I | 100% | 0.158518914 | 0.182310373 | 466.78 | 465.91 |
| Nominal EVSI · primary | 0% | 0.158526317 | 0.182367131 | 466.81 | — |
| Source-risk gate · secondary | 35.29% | **0.158501188** | 0.182322383 | — | 466.20 |
| Cross-fitted sign counts · secondary | 62.37% | 0.158502081 | 0.182311486 | — | 465.83 |

GPU p90에는 첫 세 쌍의 판독값에서 시작해 네 번째 센서를 고르고 구입한 측정으로 512개 물체의 예측을 끝내는 시간까지 포함했다. Datacenter3의 빈 RTX PRO 6000 Blackwell에서 A는 정책당 72 batch, B는 정책당 48 batch를 측정했다. 같은 열 안에서 비교해야 한다. B에서 source-risk gate의 p90 차이는 4R 대비 +0.52 ms [−0.33, +2.03], 3R+I 대비 +0.29 ms [−0.87, +2.00]이었다. A의 두 차이도 simultaneous 95% interval에 0이 포함됐다. 물리 장비의 응답 시간은 이 측정에 포함되지 않는다. [실행 환경·원시 timing과 검증](GPU_KO.md)

## 왜 primary는 실패했는가

Nominal prior에서 reference와 impact 사이에 방향이 뒤집힐 확률 e의 평균은 0.10이다. 이 모델은 reference가 앞으로의 impact에 대해서도 대체로 좋은 측정이라고 본다. 실제 평가에서 어느 방향을 선택할지 결정할 때 모델은 e를 알 수 없었다. 네 번째 impact의 예상 이득에서 reference의 예상 이득을 빼면 61,440개 prefix 전부에서 음수였다. Quadrature를 세밀하게 바꾼 3,840개 prefix에서도 선택이 뒤집히지 않았다. 결과적으로 fixed 3R+I 대비 primary의 mean 차이는 **+0.00000740** [동시 95% CI +0.00000049, +0.00001432], p90 차이는 **+0.00005676** [−0.00000907, +0.00009420]이었다. Solver resolution을 높이면 mean 차이의 interval은 0을 포함했다. [수치 검산](POLICY_KO.md) · [민감도](RESULTS_KO.md)

![네 번째 impact의 예상 이득과 실제 held-out 이득](prefix_value54.png)

그림에서 양수는 impact가 reference보다 평균 Brier를 줄인다는 뜻이다. 예측 순위로 나눈 네 집단에서 실제 이득은 낮은 순서대로 **−80.4, +11.0, +32.9, +66.0 × 10⁻⁶**이었다. 모델은 어떤 집단에서 impact가 상대적으로 더 유용한지는 포착했지만, 각 집단의 절대 가치를 너무 낮게 예측했다. 이 그림은 집단 평균에 대한 진단이다. 개별 session의 최적 행동이나 e의 부호를 식별했다고 해석할 수 없다. [수치와 interval](score54.json)

## 두 adaptive 정책은 무엇을 다르게 배웠는가

**Source-risk gate**는 양쪽 센서를 평가할 때 e를 0–0.5의 uniform distribution으로 적분하고, 지불한 측정을 예측으로 바꿀 때는 nominal decoder를 그대로 썼다. Gate의 평균 loss 차이는 4R 대비 −0.00002513 [−0.00002817, −0.00002209], 3R+I 대비 −0.00001773 [−0.00002269, −0.00001277]이었다. Impact를 정확히 같은 35.29%의 확률로 독립적으로 사는 lottery와 비교해도 평균이 **0.00002252** 낮았다. 따라서 단순히 impact를 덜 사서 평균이 좋아졌다는 설명으로는 결과를 설명하기 어렵다. 3R+I 대비 session p90 차이는 +0.00001201 [−0.00003885, +0.00004318]이었다. [구매율을 맞춘 대조](propensity54.json) · [Secondary interval](secondary54.json)

**Cross-fitted sign counts**는 첫 세 쌍에서 읽을 수 있는 부호의 수와 그 부호들이 얼마나 일관되는지를 여섯 가지 경우로 나눴다. 한쪽 여섯 catalog의 평균 downstream Brier로 각 경우의 구매 결정을 정한 뒤 나머지 여섯에서 검증하고 양쪽 역할을 바꿨다. 두 fold 모두 유효 판독이 하나도 없으면 R, 두 개 또는 세 개가 모두 같은 방향 관계를 보이면 I를 골랐다. 나머지 세 경우에서 두 fold의 결정이 달랐다. Mean은 12개 평가 catalog 각각에서 두 fixed보다 낮았다. 학습까지 매번 다시 한 2,000회 paired bootstrap에서 3R+I 대비 mean 차이는 −0.00001683 [−0.00002084, −0.00001407], 같은 구매율의 lottery 대비 −0.00001962 [−0.00002261, −0.00001738]이었다. 3R+I 대비 p90 차이는 +0.00000111 [−0.00002959, +0.00002179]이었다. 이 interval은 탐색한 모든 규칙에 대한 동시 보장이 아니다. [학습 코드](sign_bins54.py) · [Fold별 결과](sign_bins54.json) · [재학습 bootstrap](resample_bins54.json)

**강건성과 비용.** Synthetic optical readout에서 localization jitter를 0.85 pixel로 높였을 때 sign-count gate의 mean은 0.158523324, 4R은 0.158526317, 3R+I는 0.158558290이었다. 4R 대비 우세는 12개 중 7개 catalog로 줄었다. 모든 정책은 교체 후 8 contacts와 8 resets를 사용한다. Impact 측정과 reset에 걸리는 시간을 T초, I 구매율을 r라 두면 모델상의 소요 시간은 5.92 + r(T−1.04)초/session이다. T=1.04면 모두 5.92초이고, T=2.08이면 4R 5.92초, source-risk 6.29초, sign counts 6.57초, 3R+I 6.96초다. 실물 T는 측정하지 않았다. [Optical sensitivity](validate_bins54.json) · [비용 계산](GPU_KO.md)

## 논문에 남길 주장과 다음 판정

v54가 지지하는 주장은 **동일한 decoder와 측정 횟수에서 prefix가 관측의 상대적 가치를 구분할 수 있고, 잘못된 acquisition prior는 이 신호를 모든 경우에 같은 행동으로 바꿀 수 있다**는 것이다. 구매율을 맞춘 lottery는 prefix 자체의 역할을 분리하며, primary의 예측과 실제 gain 사이의 차이는 실패 과정을 관찰 가능하게 한다. 두 adaptive 정책이 3R+I보다 줄인 평균 Brier는 절대값으로 약 0.000017–0.000018(상대 약 0.011%)이다. 이 크기가 실물에서 중요한지, loss가 큰 session을 보호하는지, GPU 시간을 절감하는지는 입증되지 않았다.

우선 확인할 실험은 새로운 held-out surface·session에서 **source-risk gate 하나와 fixed 3R+I**를 동결하여 비교하는 것이다. 실험 전에 e 분포와 optical noise가 바뀌는 조건을 따로 정하고, 얻게 될 Brier 감소에 지불할 최대 추가 시간과 실용적으로 의미 있는 효과를 명시해야 한다. 같은 구매율의 lottery와 4R을 함께 유지하면 sensing budget 효과와 adaptive selection 효과를 구분할 수 있다. 보고서의 수치는 이전 버전과 공유한 simulator·catalog에서 나온 development evidence다. Baseline 두 개에 대한 평균 개선은 논문의 보조 결과로 넣었고, 실물 adaptive sensing이나 tail-risk 우위는 열린 가설로 남겼다.

## 재현 자료

정책과 scorer, timing을 별도로 구현하고 freeze 기록과 원시 paired 결과를 보관했다. 서버 작업 경로는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/adaptive_witness_v54`다. 배포되지 않는 latent oracle, noise별 결과, solver resolution별 수치와 코드 연결은 [정량 부록](RESULTS_KO.md), [정책 부록](POLICY_KO.md), [GPU 부록](GPU_KO.md)에 있다. 논문에는 본문과 Appendix C *Choosing the final calibration measurement*에 반영했다.

## 문헌 검토와 논문 심사

### 1. 원 논문별 질문과 근거

| 연구 | 질문과 방법 | 표본·평가 단위 | 결과와 결론 | v54에 적용할 때의 한계 |
|---|---|---|---|---|
| Fishel & Loeb, *Bayesian Exploration for Intelligent Identification of Textures* (2012) | 과거 tactile signal을 보고 다음 stroking movement를 고르면 texture를 효율적으로 식별하는가? BioTac 측정으로 후보별 posterior를 갱신하고 cycling·random 탐색과 비교했다. | Training 후보 117개. 16개 texture에서 별도로 수집한 측정으로 8,000회 virtual exploration. | Bayesian exploration은 median 5회, 95.4% 정분류를 기록했다. 목적에 맞춰 서로 다른 단서를 얻는 탐색의 사례다. | 8,000개 독립 재료의 성적이 아니다. 알려진 texture 집합의 ID가 목표이며 새로운 surface의 다음 impact를 채점하지 않는다. [원문](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00004/full) |
| Golovin, Krause & Ray, *Near-Optimal Bayesian Active Learning with Noisy Observations* (2010) | Noise가 있을 때 어떤 test가 decision-equivalent hypotheses를 빨리 가르는가? EC²의 greedy bound를 증명하고 빠른 EffECXtive를 실험했다. | 이론의 단위는 유한 hypothesis와 test. 행동 실험에서는 11명에게 각각 30개 lottery 선택을 제시했다. | 특정 equivalence-class identification 목적에서 단순 정보량 기준의 반례와 비용 보장을 제시했다. 인간 행동 실험은 모델 식별의 적용 예다. | 여기의 목표는 정답 class를 정할 때까지의 비용이다. Fixed horizon, probabilistic forecast 또는 session loss p90에 같은 보장을 적용할 수 없다. [원문](https://las.inf.ethz.ch/files/golovin10dtree.pdf) |
| Heath, Manolopoulou & Baio, *Efficient Monte Carlo Estimation of the Expected Value of Sample Information Using Moment Matching* (2018) | 아직 얻지 않은 표본의 decision value를 어떻게 저렴하게 평가하는가? Preposterior mean의 분포를 moment matching으로 근사했다. | 경제적 decision model 네 사례. 실제 환자 간 효과 시험과 다른 계산 단위다. | 반복적인 posterior 계산을 줄일 수 있고, 작은 미래 표본에서는 일부 근사에 bias가 생길 수 있음을 보였다. | 여기서 한 번의 probe는 극단적인 작은 표본이다. 이 논문의 근사 오차가 그대로 v54에 적용된다는 근거는 없으며 acquisition 순위의 검산이 필요하다. [원문](https://discovery.ucl.ac.uk/id/eprint/1528498/1/Baio_1611.01373v1.pdf) |
| Acosta, Yang & Posa, *Validating Robotics Simulators on Real-World Impacts* (2022) | Contact parameter를 맞춘 solver가 실물 impact trajectory를 어디까지 재현하는가? Cube에 세 solver, Cassie에 두 solver를 비교했다. | Cube toss trajectory 550개, Cassie jump 22개. | 전체 운동을 상당 부분 재현해도 restitution과 일부 edge/corner impact가 어려웠다. Cassie의 착지에서는 friction을 식별할 slip이 충분히 나타나지 않았다. | 이 자료에는 동일 표면에서 weak/reference와 weak/impact 중 무엇을 고를지 결정하기 위한 paired calibration prefix가 없다. [원문](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) |

### 2–3. 주제별 문헌 종합과 지식 지도

**탐색에서 구분할 대상**은 texture identity, decision-equivalence class, 다음 action의 expected utility처럼 과제별로 다르다. Texture posterior가 좁아지는 동작과 downstream Brier를 낮추는 동작의 순위가 같다는 보장은 없다. v54에서는 네 번째 센서만 바꾸고 forecast를 고정해 무엇이 좋아졌는지 측정할 수 있다. [Fishel & Loeb](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00004/full) · [Golovin et al.](https://las.inf.ethz.ch/files/golovin10dtree.pdf) · [Heath et al.](https://discovery.ucl.ac.uk/id/eprint/1528498/1/Baio_1611.01373v1.pdf)

연구 간 관계는 **과거 측정 → 가능한 미래 관측 → posterior 갱신 → 실행할 결정 → 평가할 손실**로 정리된다. Fishel & Loeb은 첫 두 단계와 재질 분류를 연결했고, Golovin 등은 결정에 영향을 주는 hypothesis 구분을 다뤘다. Heath 등은 관측 전 가능한 posterior를 평균하는 비용을 줄였다. 실제 결과가 모델과 어긋나는 마지막 단계는 Acosta 등의 실물 연구가 보여준다. [Fishel & Loeb](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00004/full) · [Golovin et al.](https://las.inf.ethz.ch/files/golovin10dtree.pdf) · [Heath et al.](https://discovery.ucl.ac.uk/id/eprint/1528498/1/Baio_1611.01373v1.pdf) · [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf)

### 4–7. 공백, 상충하는 듯한 결과, 방법과 증거 강도

- **잘 확립된 범위.** Test를 고르는 목적과 cost를 명시해야 어떤 acquisition이 좋은지 해석할 수 있다. 표본 수, 결정 결과, stopping 여부가 다른 연구의 성공률과 Brier 차이를 수치적으로 합치기 어렵다. [Golovin et al.](https://las.inf.ethz.ch/files/golovin10dtree.pdf) · [Heath et al.](https://discovery.ucl.ac.uk/id/eprint/1528498/1/Baio_1611.01373v1.pdf)
- **결과가 달라지는 이유.** 실제 texture 측정에서 adaptive exploration이 유리했던 결과와 정보량 기반 탐색이 실패할 수 있다는 이론은 양립한다. 전자는 후보 texture와 움직임을 고정한 재분류이고, 후자는 어느 hypothesis를 구분해야 결정을 바꿀 수 있는지에 대한 결과다. 둘 중 어느 것도 모든 prefix에서 optimal sensor가 바뀐다고 예측하지 않는다. [Fishel & Loeb](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00004/full) · [Golovin et al.](https://las.inf.ethz.ch/files/golovin10dtree.pdf)
- **제한적 근거와 공백.** 실제 impact trajectory에서 solver error가 조건에 따라 달랐던 결과는 surface와 contact regime을 나눠 검증할 이유를 제공한다. 그러나 세 reference로 감지할 수 있는 shift가 얼마나 자주 발생하는지, sensor 선택이 실물 session의 상위 손실을 낮추는지는 이 네 연구에서 확인되지 않았다. [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) · [Fishel & Loeb](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00004/full)

### 8. 후속 연구 질문 열 가지

| 질문 | 공백과 실험 | 근거 |
|---|---|---|
| 1. 첫 세 쌍에 impact 오류율에 대한 정보가 있는가? | 새 재료에서 reference spread와 실제 impact 불일치율을 따로 측정한다. Material·session을 분리한 validation에서 상관과 predictive value를 추정한다. | [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) |
| 2. 네 번째 물체의 readout 가능성을 먼저 알면 어떤가? | 가림·contrast·marker 신뢰도를 구입 전 관찰할 수 있는 조건을 추가하고 그 관찰 비용도 부과한다. | [Fishel & Loeb](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00004/full) |
| 3. Prefix에서 같은 부호가 반복된 이유를 분해할 수 있는가? | 같은 specimen 반복과 다른 specimen 반복을 비교해 session bias와 방향 유지율을 구분한다. | [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) |
| 4. Posterior를 가장 많이 바꾸는 probe와 downstream risk를 가장 줄이는 probe가 일치하는가? | 같은 prefix에서 entropy rule과 Brier rule의 disagreement와 paired loss를 평가한다. | [Golovin et al.](https://las.inf.ethz.ch/files/golovin10dtree.pdf) |
| 5. 두 종류의 센서가 언제 서로를 대체할 수 있는가? | Randomized reference/impact acquisition 후 식별된 error model과 terminal prediction을 함께 검증한다. | [Heath et al.](https://discovery.ucl.ac.uk/id/eprint/1528498/1/Baio_1611.01373v1.pdf) |
| 6. 평균용 policy가 tail까지 개선하는 조건은 무엇인가? | Session size와 공동 sensor bias를 분리하고 사전 지정 quantile·expected shortfall policy를 새 홀드아웃에서 비교한다. | [Heath et al.](https://discovery.ucl.ac.uk/id/eprint/1528498/1/Baio_1611.01373v1.pdf) |
| 7. 관측 세 번의 크기까지 쓰면 부호 규칙의 한계를 넘는가? | 같은 training catalog 수에서 분할 표와 연속 gain regression을 비교하며 다른 surface family로 옮긴다. | [Fishel & Loeb](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00004/full) |
| 8. 마지막 선택의 이득은 horizon이 길어져도 남는가? | Fixed slot 두 개 이상에서 myopic·look-ahead·fixed schedule을 같은 총 reset 예산으로 비교한다. | [Golovin et al.](https://las.inf.ethz.ch/files/golovin10dtree.pdf) |
| 9. 수치 오차가 작은 acquisition gap의 부호를 뒤집는가? | 서로 다른 solver와 직접 측정한 impact outcome에서 동일한 freeze된 결정을 재채점한다. | [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) |
| 10. 몇 초의 추가 reset 비용까지 정당화되는가? | 날짜·표면을 무작위화해 두 sensor의 reset time, 손상, 재시도와 held-out loss를 함께 기록한다. | [Heath et al.](https://discovery.ucl.ac.uk/id/eprint/1528498/1/Baio_1611.01373v1.pdf) · [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) |

### 9. 심사에서 남는 질문

EVSI나 adaptive sensing 자체는 이미 정립된 아이디어다. 이 프로젝트의 기여 후보는 같은 네 쌍 예산에서 **센서를 고르는 시점에 가용한 정보가 실제 선택을 바꾸고, 다른 물체의 확률 예측에 전달되는가**를 분리해 보이는 데 있다. Decision model의 예측, simulator에서 실현된 이득, 원래 사야 했던 관측의 포기 비용을 각각 보고해야 한다. 실제 물체·날짜·표면 단위에서 frozen policy와 양쪽 fixed policy를 무작위 배정해 검증하는 과정은 여전히 필요하다. [Heath et al.](https://discovery.ucl.ac.uk/id/eprint/1528498/1/Baio_1611.01373v1.pdf) · [Fishel & Loeb](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00004/full) · [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf)
