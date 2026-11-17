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
