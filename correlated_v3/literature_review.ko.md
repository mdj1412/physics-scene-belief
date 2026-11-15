# Correlated physical belief 연구의 선행연구와 가치 재검토

검토일: 2026-11-04. 아래 평가는 이번 `correlated_v3` 실험 결과를 전제로 한다. 2026년 논문의 상당수는 preprint이므로, 확인하지 않은 학회 acceptance를 가정하지 않는다. 검색으로 발견한 논문 중 가까운 연구의 원문 method와 관련 experiment 부분을 읽었으며, 전체 분야를 빠짐없이 검토했다는 의미는 아니다.

**문제의 중요성은 충분하지만, 현재 구현은 새로운 method의 필요성을 입증한 상태가 아니다.** Factual grazing contact에서 geometry–restitution coupling, 작은 grid의 under-resolution, prior importance sampling의 weight collapse를 재현했다. 그러나 standard SMC와 problem-specific conditional quadrature가 대부분의 gap을 없앴다. 또한 validation으로 고른 query-type rule과 비교한 adaptive selection의 MSE 차이는 작다. 따라서 지금 결과를 “새로운 joint scene belief algorithm의 성공”으로 포장하면 안 된다.

## 1. 이번에 특히 중요하게 확인한 최근 연구

### A Bayesian Approach for Task-Specific Next-Best-View Selection with Uncertain Geometry — Zhu, Sellán, Terenin, 2026

원문 §3.1–3.2.3, §4.3과 author project page를 확인했다. Author page는 SIGGRAPH 2026으로 표시한다. Stochastic Poisson Surface Reconstruction의 posterior samples로 가능한 scan을 생성하고, downstream utility의 expected improvement를 기준으로 viewpoint를 고른다. Classification·segmentation뿐 아니라 heat diffusion의 coldest-point 발견도 포함한다. 즉, **uncertain geometry를 유지하면서 physics task에 필요한 observation을 고르는 접근은 이미 직접적인 선행연구가 있다.** [원문](https://arxiv.org/html/2605.05095) · [author page](https://papers.avt.im/next-view-planning/)

우리의 남은 후보는 static surface scan을 넘어, factual motion이 geometry와 material을 결합해서 제약하고, discrete contact mode와 continuous parameter가 함께 counterfactual을 결정하는 설정이다. 다만 heat diffusion을 contact dynamics로 바꾸는 것만으로 충분한 method contribution이 된다고 볼 수는 없다. 이 논문을 단순 uncertainty-reduction baseline으로 묘사해서도 안 된다.

### Informationally Decoupled Trajectory Design for Sim-to-Real System Identification — Shin et al., 2026

2026-10-27의 v2 원문 §III–V를 확인했다. IDTD는 서로 다른 parameter의 효과가 보상되어 trajectory fit이 좋아도 parameter estimate가 틀리는 문제를 명시적으로 다룬다. Nominal parameter에서 계산한 Fisher Information Matrix의 Schur complement, normalization, segment-wise coverage로 exploration을 설계한다. Linear system, 여러 robot simulator, real K1 humanoid를 평가한다. [원문](https://arxiv.org/html/2610.10905)

우리의 차별점 후보는 parameter identification 전체가 아니라 **특정 counterfactual에 필요한 uncertainty를 남기거나 줄이는 것**, 그리고 local FIM으로 표현하기 어려운 여러 contact branch의 posterior다. 그러나 “parameter들이 서로 보상하므로 active experiment가 필요하다” 자체는 이 논문과 직접 겹친다. IDTD의 full trajectory-design algorithm을 이번 scalar sensor 실험에서 재현한 것은 아니다.

### Active Tactile Exploration for Rigid Body Pose and Shape Estimation — Gordon et al., 2025/2026

2026-02-24의 v2 원문 §III–VII를 중심으로 확인했다. 움직일 수 있는 rigid object의 pose와 continuous convex geometry를 tactile observation으로 함께 추정한다. Violation-implicit loss로 contact constraint를 다루고, local information approximation에 기반한 EIG로 exploration을 고른다. Simulation과 real robot 결과가 있다. 원문은 dynamic belief가 multimodal해져 단일 Gaussian propagation이 부적절할 수 있다는 문제도 논의한다. [원문](https://arxiv.org/html/2510.13595)

따라서 contact-aware geometry learning이나 active sensing 자체도 새 주장이 아니다. 우리에게 남는 검증은 local approximation이 놓치는 branch가 query-specific predictive distribution과 action ranking을 얼마나 바꾸는지다. 이번 결과는 그 가능성을 보여주는 작은 diagnostic이며, 해당 논문에 대한 성능 우위를 보인 것은 아니다.

### System Identification under Constraints and Disturbance: A Bayesian Estimation Approach — Martinez, Tonneau, Mastalli, 2026

원문 introduction, constrained formulation, hybrid dynamics와 results의 구성 및 설명을 확인했다. Joint state–parameter estimation, contact/loop constraints, friction, disturbance projection과 energy observation을 함께 다룬다. Physical model이나 constraint를 잘못 표현하면 parameter가 그 오류를 흡수한다는 지적이 중요하다. [원문](https://arxiv.org/html/2602.16358)

이번 box collision margin audit와 연결되는 교훈은, posterior를 정교하게 계산해도 잘못된 forward model을 믿고 있으면 inference가 틀릴 수 있다는 것이다. 이 논문의 constrained optimization을 arbitrary multimodal posterior sampling과 동일시하지 않는다. 우리에게는 **approximation error와 model discrepancy를 분리한 평가**가 필요하다.

## 2. Acquisition 원리와 numerical inference의 직접 선행연구

| 연구 | 원문에서 확인한 내용 | 이번 연구에 주는 제약 |
|---|---|---|
| [Action-BED, 2026](https://arxiv.org/html/2606.23662), §3–4·6.2–6.3 | Expected future loss를 기준으로 design policy와 downstream action policy를 공동 학습한다. Explicit posterior 없이도 학습할 수 있고 pendulum 및 visual patch acquisition을 평가한다. | Downstream loss를 acquisition objective로 쓰는 원리를 새롭게 주장할 수 없다. 이번 EVSI 계산은 Action-BED implementation이 아니다. |
| [GO-CBED, 2025 preprint](https://arxiv.org/html/2507.07359), §3–5·Appendix C.4 | Causal QoI에 대한 non-myopic EIG, variational posterior와 policy를 학습한다. Structural uncertainty와 기존 observational data도 다룬다. | Query-specific intervention, multimodal causal uncertainty, observational conditioning 자체는 이미 포함된다. 이번 one-step 실험과 horizon이 다르다. |
| [Amortized Bayesian Experimental Design for Decision-Making, 2024/2025](https://arxiv.org/html/2411.02064), §2.3–4 | Decision Utility Gain과 Transformer Neural Decision Process를 제안한다. Posterior approximation이 decision에 중요한 영역을 놓칠 수 있다는 점도 논의한다. | “좋은 posterior fit보다 좋은 decision이 중요하다”는 문제의식도 새 원리는 아니다. |
| [Nesting Particle Filters for Experimental Design in Dynamical Systems, ICML 2024](https://proceedings.mlr.press/v235/iqbal24a.html), [원문 §3–4](https://arxiv.org/html/2402.07868) | Inside-Out SMC²와 particle MCMC로 dynamical system의 sequential design을 다룬다. Naive prior weighting의 high variance를 설명한다. | Prior particles의 실패를 particle methods 전체의 한계로 일반화하면 안 된다. 이번 tempered SMC는 이 논문의 재현이 아니다. |
| [BayesSim, RSS 2019](https://arxiv.org/html/1906.01728), §IV–V | Simulator parameter의 likelihood-free posterior를 mixture density로 추정하고 domain randomization에 활용한다. | Physical parameter를 point estimate 대신 distribution으로 표현하는 것 자체는 오래된 접근이다. Diagonal mixture component도 여러 component의 결합으로 dependence를 표현할 수 있다. |
| [SPI-Active, 2025](https://arxiv.org/html/2505.14266), §3–5 | Parallel sampling 기반 physical parameter fitting과 Fisher Information 기반 active exploration을 결합한다. | GPU sampling과 active system identification의 결합만으로는 차별화되지 않는다. 우리의 작은 inference benchmark가 실제 locomotion transfer 성과를 대체하지 못한다. |

여기서 논문별 숫자를 가져와 우리 결과와 직접 비교하지 않았다. Observation interface, dynamics, task loss, horizon, training budget이 다르기 때문이다. 이번에 실제 실행한 strong baseline은 **tempered SMC, conditional posterior sampling, posterior-adapted conditional quadrature**다.

## 3. Video와 physical digital twin 방향에서 가까운 연구

| 연구 | 관련 원문과 핵심 | 우리가 주장할 수 있는 범위 |
|---|---|---|
| [VideoPhysEdit, 2026](https://arxiv.org/html/2609.35134v1), §3·Appendix C.2–C.3 | Video에서 executable rigid scene을 맞추고 physical edit 이후 trajectory를 생성한다. 초기 관측에 비슷하게 맞는 최대 3개 candidate를 이후 contact까지 유지하는 절차가 있다. | 기존 방법이 ambiguity를 전혀 다루지 않는다는 주장은 틀리다. Candidate selection과 calibrated posterior propagation의 차이를 실제 downstream 평가로 보여야 한다. |
| [Asking the World, 2026](https://arxiv.org/html/2609.39135v2), §3·Appendix A.2–A.3 | CEM으로 world를 fitting하고 question에 맞춰 simulation과 intervention을 선택한다. Representation과 world modeling도 수정할 수 있다. | Fitted simulator에서 virtual probe를 수행하는 것과 실제 environment의 새 measurement를 받는 acquisition을 구별해야 한다. 이번 실험은 후자의 이상화된 setting이다. |
| [PhysTwin, 2025](https://arxiv.org/html/2503.17973), §4–5 | Three RGBD videos에서 geometry, topology, physical parameters, appearance를 맞추고 unseen interaction을 평가한다. | 실제 visual twin의 어려움과 규모가 우리 2D toy보다 크다. 여기서 실패한 geometry grid를 PhysTwin의 실패 증거로 제시할 수 없다. |
| [Active System Identification for Physics Twins, 2026 ongoing project](https://jacobgit2.github.io/projects/phystwin-active-sysid/) | 같은 factual motion에 맞는 twin들이 unseen action에서 달라지고, mean-field uncertainty가 이를 숨길 수 있음을 분석한다. Ensemble disagreement가 실제 refit payoff를 잘 예측하지 못한 결과와 model error 문제도 공개한다. | Peer-reviewed paper로 취급하지 않는다. 다만 “factual fit의 비식별성→ensemble→active interaction”이라는 생각은 공개 선행 작업과도 겹친다. |

이번 survey에서 느끼는 흐름은 단순히 world model을 만드는 데서 **그 model이 어떤 intervention까지 믿을 만한지, 어떤 추가 observation이 그 신뢰를 바꾸는지**를 묻는 방향으로 넓어지고 있다는 것이다. 이는 위 primary sources를 종합한 내 해석이며, 학회 전체의 정량적인 trend analysis는 아니다.

## 4. 지금 연구가 좋은 연구인가

**좋은 출발점이지만, 현재 상태로는 독립적인 main-conference method paper의 근거가 약하다.** 다음 네 가지를 분리해서 평가해야 한다.

1. **Problem relevance는 높다.** Factual fit이 좋아도 unseen intervention의 prediction이 틀리는 문제는 reconstruction, robotics, simulation 모두에 실용적이다. 이번 실험은 그 원인을 측정 가능한 parameter coupling으로 만들었다.
2. **Diagnostic evidence는 유용하다.** Noise가 줄어들수록 부정확한 posterior approximation의 downstream 결과가 오히려 악화되고, sign branch를 잃으면 observation selection도 크게 틀어진다. 어떤 오류인지 검증할 수 있는 reference가 있다는 장점이 있다.
3. **Method novelty는 아직 없다.** Tempering, resampling, Metropolis rejuvenation, conditional integration은 기존 도구다. 우리의 conditional quadrature가 잘된다는 사실은 새로운 대형 model이 필요하다는 주장에 반대되는 증거다.
4. **Acquisition의 실용적 추가 이득은 아직 약하다.** Two-branch setting에서는 reference가 모든 query에서 view를 선택한다. One-branch에서는 query마다 action이 달라지지만 대부분 probe와 stop의 차이다. Query-type rule과의 MSE gap이 작다는 결과를 숨겨서는 안 된다.

여기서 toy를 더 어렵게 만들어 특정 baseline을 계속 실패시키는 방향은 권하지 않는다. 실패 조건을 만들기보다, 실제 scene reconstruction에서 자연스럽게 생기는 ambiguity를 수집하고 **SMC를 포함한 강한 기준선으로도 남는 비용·신뢰성 문제**를 찾아야 한다.

## 5. 내가 선택할 다음 연구 방향

우선 제목 수준의 방향을 **“Calibrated Counterfactual Prediction under Contact-Model Ambiguity”**로 좁히겠다. Active acquisition은 이 문제를 해결하는 수단으로 두고, 아직 입증되지 않은 새로운 acquisition 원리를 중심 주장으로 삼지 않겠다.

핵심 질문은 다음과 같다. “Factual observation에 모두 부합하는 geometry, material, contact model이 있을 때, 어떤 counterfactual을 믿을 수 있으며, 어떤 추가 measurement가 모델들 사이의 차이를 실제로 줄이는가?” 이번 audit의 1 mm collision margin도 이 질문을 구체화한다. Collision proxy의 작은 차이가 restitution 변화로 흡수될 수 있기 때문이다.

### Contribution 후보 A: 검증 가능한 benchmark와 failure taxonomy

Continuous parameter뿐 아니라 contact surface assignment, occluded support, sharp/rounded collision proxy, friction regime을 포함한다. Factual fit은 비슷하지만 counterfactual은 다른 scene을 만들고, 실제 추가 view 또는 probe를 얻을 수 있게 한다. 평가에는 proper predictive scores, calibration, event probability, acquisition cost, simulator-call budget을 함께 둔다.

이번 결과는 benchmark의 sanity check로 쓸 수 있다. 하지만 benchmark paper를 주장하려면 여러 scene family와 현실적인 visual observation, 가능하면 physical data가 필요하다. PCVE-RigidBench 등 기존 benchmark와의 구체적인 추가 기능도 비교해야 한다.

### Contribution 후보 B: Contact branch를 보존하는 효율적인 inference

Posterior density 전체를 균일하게 잘 맞추기보다, query에 영향을 주는 contact event와 observation response가 다른 hypotheses를 보존하는 compression 또는 allocation을 연구할 수 있다. 그러나 loss-calibrated inference와 decision-aware BED의 일반 원리를 새로 제안한다고 하면 안 된다.

논문 기여가 되려면 최소한 다음을 보여야 한다. 동일한 simulator-call budget에서 standard SMC, MCMC, conditional/adaptive quadrature, posterior surrogate보다 낮은 design regret 또는 높은 predictive coverage를 얻어야 한다. 또한 여러 query와 unseen contact configuration에 transfer되어야 한다. 지금 2D setting에서는 conditional quadrature가 너무 잘 풀기 때문에, 이 후보의 필요성은 아직 입증되지 않았다.

### Contribution 후보 C: Model discrepancy를 판별하는 closed loop

추가 observation이 parameter uncertainty를 줄이는 경우와, 기존 model class 자체가 observation을 설명하지 못하는 경우를 구분한다. Held-out probe의 posterior predictive check가 반복적으로 실패하면 noise를 임의로 크게 만드는 대신 contact proxy나 model class를 다시 검토한다. 필요한 경우 prediction을 보류하거나 범위를 넓힌다.

이 또한 generic Bayesian model checking 자체가 새 기여인 것은 아니다. Visual reconstruction에서 만들어진 collision geometry의 오류를 구체적으로 진단하고 수정하며, counterfactual reliability를 개선하는 실험이 있어야 한다. 이번 margin audit는 이 후보를 검토할 실질적인 이유를 제공하지만, 아직 closed-loop correction을 구현한 결과는 아니다.

## 6. 진행 여부를 결정할 다음 검증

| 검증 | 계속할 근거 | 방향을 바꿔야 할 신호 |
|---|---|---|
| Shape·pose·friction·restitution·contact mode를 함께 바꾼 3개 이상의 scene family | 동일 compute budget에서 strong posterior baseline과 실용적인 차이가 반복됨 | SMC만으로 모든 문제가 충분히 빠르게 해결됨 |
| Noisy RGB/RGBD observation | GT state 없이도 calibrated uncertainty와 acquisition 효과가 유지됨 | Sensor calibration 오류가 모든 이득을 압도함 |
| Query-type rule, parameter EIG, task-aware policy의 비교 | 여러 유용한 action 사이에서 의미 있는 선택 이득이 있음 | 차이가 거의 stop 여부에만 있고 fixed rule과 MSE가 비슷함 |
| Correct vs misspecified contact model | Parameter uncertainty와 model discrepancy를 구분해 대응할 수 있음 | Narrow posterior가 단순히 잘못된 simulator를 더 확신함 |
| Small real-world pilot | Held-out physical intervention에서 coverage와 prediction이 개선됨 | Factual fitting loss만 좋아지고 unseen action은 악화됨 |

내 판단은 **연구를 중단하기보다 중심 주장을 수정하는 편이 좋다**는 것이다. 현재 실험은 새로운 algorithm을 정당화하기보다, 어떤 일반 원리가 이미 알려져 있고 어떤 실용 문제가 아직 남는지 가려내는 데 성공했다. 다음 단계에서 현실적인 contact-model ambiguity와 calibrated counterfactual prediction을 입증하지 못한다면, 이 toy만 계속 확장하는 데 많은 시간을 쓰지는 않겠다.
