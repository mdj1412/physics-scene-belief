## 선행연구 6편에서 본 v47의 위치

아래는 이 실험의 가설 또는 비교 방법을 바꾸는 6편을 선별한 검토다. 표의 표본 단위는 원 연구마다 다르다. 서로 다른 endpoint의 수치를 직접 순위 매기지 않았다.

| 연구 | 연구 질문·방법 | 표본·평가 | 결과와 해석의 경계 |
|---|---|---|---|
| Kim et al., *Im2Contact* | RGB-D만으로 로봇이 쥔 물체의 외부 contact를 국소화할 수 있는지 물었다. Simulation에서 학습한 visual model을 실제 로봇에서 평가했다. | Simulation 평가 500 episodes, 실제 상호작용 30 episodes·12,362 frames | 여러 형태의 contact를 국소화했다. 현재 외부 contact 위치를 찾는 성능으로 미래 stick에 대한 decoder 선택의 가치를 추론할 수는 없다. [원문](https://proceedings.mlr.press/v229/kim23b/kim23b.pdf) |
| Pfrommer et al., *ContactNets* | Signed distance와 contact-frame Jacobian을 trajectory로 학습해 impact·stiction의 불연속성을 표현했다. | 실제 cube toss 750회에서 정제한 570개 trajectory. 여러 training-set 크기로 unstructured baseline과 비교 | 구조가 있는 모델이 물리적으로 타당한 trajectory를 data-efficient하게 예측했다. 여러 compliant family에서 부분 profile 제거의 조건부 Brier는 별도 문제다. [원문](https://proceedings.mlr.press/v155/pfrommer21a/pfrommer21a.pdf) |
| Van der Merwe et al., *Neural Deforming Contact Fields* | Partial point cloud와 force/torque로 loaded shape와 contact patch를 함께 추정했다. Simulated supervision을 사용했다. | 실물 검증은 Franka에 장착한 46 mm cube sponge, 두 depth camera와 force/torque sensor. Material fitting에 flat press 16회 | Sim-to-real에서 point-cloud baseline보다 나은 결과를 보고했다. 접촉 **후**의 wrench는 v47의 pre-impact observer가 갖지 못하는 정보다. 이 차이가 matched-coupling 실험에 직접적인 후속 관측을 제시한다. [원문](https://www.roboticsproceedings.org/rss19/p080.pdf) |
| Acosta et al., *Validating Robotics Simulators on Real-World Impacts* | 실제 impact trajectory로 contact parameter를 맞춘 뒤 Drake·MuJoCo·Bullet을 대조했다. | Tossed cube와 착지하는 Cassie biped, 두 물리 시스템 | 비탄성·탄성 regime 및 전체 운동과 세부 운동에서 fidelity가 달랐다. 하나의 finite-pressure solver에서 식별한 손익의 크기를 실물 보정량으로 사용할 근거는 약하다. [원문](https://arxiv.org/abs/2110.00541) |
| Veitch et al., *Counterfactual Invariance to Spurious Correlations in Text Classification* | 어떤 feature 변형을 무시해야 하는지를 causal structure로 정의하고 OOD 조건을 분석했다. | 이론과 text-classification 실험. 물리 접촉 표본은 없다. | 어떤 부분을 바꿔도 되는지는 label과 feature가 생기는 경로에 따라 달라진다. v47에서 sensor 변형과 loaded shape 변형을 따로 조작한 이유다. 이 이론이 v47에서 특정 router의 개선을 예측하지는 않는다. [원문](https://proceedings.neurips.cc/paper_files/paper/2021/file/8710ef761bbb29a6f9d12e4ef8e4379c-Paper.pdf) |
| Luo et al., *PhysiCalWorld* (2026 preprint) | Exploratory interaction clip에서 physics token을 만들어 test scene의 action-conditioned rollout에 재사용했다. | 물성이 달라지는 네 환경에서 latent prediction과 downstream task를 평가 | Interaction이 알려주는 instance-specific dynamics를 활용할 수 있음을 보고했다. 추가 clip을 허용하는 setting과 같은 paid prefix에서 terminal decoder만 바꾸는 setting은 정보 예산이 다르다. [원문](https://arxiv.org/abs/2610.30125) |

**문헌의 구조와 겉보기 충돌.** 연구들을 하나의 사슬로 놓으면, unloaded geometry의 관측(Im2Contact의 visual localization과 연결) → load 이후 contact patch·force(NDCF) → impact outcome(ContactNets, Acosta et al.) → 특정 feature를 버릴 수 있는지에 대한 판단(Veitch et al.)으로 이어진다. PhysiCalWorld의 exploratory clip은 이 사슬에 추가 interaction을 제공한다. NDCF와 Im2Contact의 긍정적 결과와 v46의 transfer 실패는 같은 가설의 상반된 결과가 아니다. 센서가 사용 가능한 시점, geometry를 재구성하는지 future event를 채점하는지, 비교 모델과 접촉 재질이 다르기 때문이다. [Im2Contact](https://proceedings.mlr.press/v229/kim23b.html) · [NDCF](https://roboticsconference.org/2023/program/papers/080/) · [PhysiCalWorld](https://arxiv.org/abs/2610.30125)

**어디까지 근거가 있는가.** 휴지 상태의 contour와 하중을 받은 contact surface를 구분해야 한다는 점은 visuo-tactile 연구와 실물 simulator 검증 모두에 부합한다. Learned dynamics와 실제 exploratory measurement가 도움을 줄 수 있다는 근거는 과제와 데이터에 조건부다. 선정한 논문들은 same location·matched normal variance에서 profile axis와 loaded deformation을 교차해 동일 decoder의 Brier를 재는 비교를 제공하지 않는다. 이 실험 설계의 추가 가치는 그 조건부 차이를 직접 측정하는 데 있다. 문헌 전체에 대한 최초성, real-world 안전성, 일반적인 OOD 불가능성은 이 표에서 도출되지 않는다. [NDCF](https://roboticsconference.org/2023/program/papers/080/) · [ContactNets](https://proceedings.mlr.press/v155/pfrommer21a.html) · [실물 impact](https://arxiv.org/abs/2110.00541) · [Counterfactual invariance](https://proceedings.neurips.cc/paper_files/paper/2021/hash/8710ef761bbb29a6f9d12e4ef8e4379c-Abstract.html)

### 후속 연구 질문 10개

아래는 추가 실험을 고르기 위한 제안이다. 각 행의 오른쪽에 가장 직접적인 관련 문헌을 표시했다.

| 질문 | 미해결 부분과 필요한 실험 |
|---|---|
| 1. 작은 preload 하나로 hidden coupling을 구별하는가? | 동일 prefix에 load 전후 displacement나 wrench 한 번을 추가한다. Paired Brier gain과 sensing/reset 비용을 함께 재면 현재 정보 한계가 실험적으로 풀리는지 알 수 있다. [NDCF](https://roboticsconference.org/2023/program/papers/080/) |
| 2. 실제 물체에서도 sensor axis와 loaded mode가 분리되는가? | 서로 다른 표면 마감과 backing stiffness를 교차한 시험편으로 optical profile, force, slip을 독립 측정한다. 시험편·날짜 단위 holdout을 쓴다. [NDCF](https://roboticsconference.org/2023/program/papers/080/) |
| 3. Query height가 어긋나도 같은 결론인가? | Contact localization error를 독립 calibration session에서 추정한 분포로 주입해 route와 Brier의 민감도를 추적한다. [Im2Contact](https://proceedings.mlr.press/v229/kim23b.html) |
| 4. 물성이 달라져도 paired-mechanism 결과가 남는가? | Compliant simulator와 실제 elastic/inelastic material을 나눠 정렬×변형 interaction을 비교한다. [실물 impact](https://arxiv.org/abs/2110.00541) |
| 5. Contact-aware learned decoder가 projection의 이득을 흡수하는가? | 동일 paid measurement와 source training budget으로 contact Jacobian 기반 predictor를 학습한 뒤 held-out Brier를 비교한다. [ContactNets](https://proceedings.mlr.press/v155/pfrommer21a.html) |
| 6. 소수의 큰 오차는 sensor signal-to-noise ratio가 설명하는가? | Optical amplitude를 고정하고 range noise를 바꿔 true alignment에 대한 proxy discrimination과 route regret을 따로 잰다. [Im2Contact](https://proceedings.mlr.press/v229/kim23b.html) |
| 7. 자연스러운 object population에서도 같은 결정을 하는가? | 실제 독립 표본에서 alignment·coupling prevalence를 추정하고, 고정 policy의 Brier를 그 분포로 재가중한다. [실물 impact](https://arxiv.org/abs/2110.00541) |
| 8. 표면 session마다 소량의 calibration을 사는 것이 이득인가? | 같은 총 관측 예산으로 clip 기반 latent descriptor, 명시적 stiffness estimate, fixed decoder를 대조한다. [PhysiCalWorld](https://arxiv.org/abs/2610.30125) |
| 9. 동일한 proxy라도 causal pathway가 달라지면 선택이 달라져야 하는가? | Sensor-only 변형과 load-bearing 변형의 비율을 독립 randomization하고, source에서 정한 손익 예측의 부호가 target에서 재현되는지 확인한다. [Counterfactual invariance](https://proceedings.neurips.cc/paper_files/paper/2021/hash/8710ef761bbb29a6f9d12e4ef8e4379c-Abstract.html) |
| 10. Binary stick 기준이 결과를 결정하는가? | 사전 지정한 residual-speed threshold 범위와 continuous velocity error를 함께 평가하고, 독립 실물 velocity 계측의 해상도와 비교한다. [ContactNets](https://proceedings.mlr.press/v155/pfrommer21a.html) · [실물 impact](https://arxiv.org/abs/2110.00541) |
