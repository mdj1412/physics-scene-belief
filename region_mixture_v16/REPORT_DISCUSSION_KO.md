## 8. 결과 해석과 논문 기여 가능성

### 8.1 V15에서 무엇이 달라졌는가

V15 counterexample에서는 label 0과 label 1의 observation distributions가 같아도 일부 hypotheses 사이의 거리가 컸다. Pair score는 이 거리를 information으로 잘못 계산했다. V16은 region 안의 likelihoods를 먼저 marginalize하므로 같은 distributions에 대해 정확히 같은 posterior label probability를 얻는다. Uninformative action의 Brier risk는 0.25이며 informative action의 compact estimate는 0.199175였다. Reference quadrature는 0.198986이다.

이 수정은 distribution의 multimodality를 유지할 이유도 보여준다. Mean과 variance가 같고 higher-order structure만 다른 두 label-conditioned mixtures에서도 mixture acquisition은 information을 감지했다. 하지만 이는 의도적으로 만든 mechanism check다. 실제 physical worlds에서 이런 구조가 충분히 자주 나타나고 실질적인 선택 이득으로 연결된다는 사실까지 이 예제만으로 증명할 수는 없다.

Equal-size global mixture가 두 counterexamples에서 region mixture와 같은 risk를 얻었다는 사실이 중요하다. **Counterexample을 고친 데 필요한 것은 mixture-aware risk 계산이며, 현재 실험은 region별 allocation이 필수라는 근거를 주지 않았다.** 같은 region의 hypotheses를 하나의 Gaussian으로 합쳐도 항상 충분하다는 주장 역시 higher-order example 때문에 성립하지 않는다.

### 8.2 Observation selection의 이득은 어디에 남는가

Matched contact-event에서 region32는 512 worlds 중 499개에 query view를 선택했다. Always-query의 loss가 0.044473으로 region32의 0.044491보다 오히려 조금 낮다. Action counts가 다르다는 사실만으로 adaptive selection의 가치가 생기지는 않는다. 현재 contact task에서는 query-type rule이 매우 강하다.

Stick-event에서는 region32가 query/factual/probe를 각각 70/95/347회 선택했고, global mixture는 225/72/215회, nested512는 148/43/321회였다. Ranking은 상당히 다르지만 mean loss는 가까웠다. 이는 여러 actions의 실제 utility 차이가 작거나 서로 다른 worlds에서의 개선과 악화가 평균에서 상쇄될 수 있음을 시사한다. 이 둘을 현재 aggregate result만으로 완전히 분리했다고 주장하지 않는다.

Region32는 LMMSE16보다 stick-event에서 유리했다. 그러나 비슷한 GPU compute를 쓰는 LMMSE64와 nested128도 강하다. LMMSE64는 4.438 ms, nested128은 4.767 ms이며 region32는 4.265 ms다. 두 baseline에 대한 secondary paired CI는 0을 포함한다. 특히 nested128의 stick loss 0.066739는 region32의 0.066922보다 낮다. 느린 nested512 하나와의 speedup만으로 compute frontier를 개선했다고 주장할 수 없다.

### 8.3 Ranking ambiguity와 compression의 한계

첫 32 matched worlds의 stick task에서는 acquisition noise를 Q4에서 Q8로 바꾸거나 Sobol seed를 바꿀 때 region32 action agreement가 각각 78.125%였다. Nested512와 independent-seed nested2048의 agreement도 78.125%다. 따라서 높은 sample count의 nested reference도 exact oracle로 취급할 수 없다. Q8/alternate-seed 결과는 primary policy를 수정하거나 best seed를 고르는 데 사용하지 않았다.

K를 32에서 64로 늘려도 matched stick loss가 단조 개선되지 않았다. Region64의 loss는 0.069161로 region32보다 높다. Cross-fitted region32는 0.067020이며 비용은 6.656 ms로 늘었다. 이것은 compression resolution, hypothetical-noise integration, finite posterior bank가 서로 얽혀 있다는 경고다. 이번 결과만으로 원인을 하나로 단정할 수 없다.

Actual decoder까지 compact mixture로 바꾼 경우 matched stick loss는 0.076230이었다. Full decoder 대비 difference는 **+0.009308**, secondary 95% CI **[+0.003312,+0.016184]**다. Shifted nominal/correct 조건에서도 악화의 CI가 0 위였다. Acquisition에 쓸 만큼의 approximation과 실제 posterior prediction에 필요한 approximation은 구분해야 한다. Factual ESS median은 약 109, selected-observation ESS medians는 대략 22–27이어서 finite QMC representation의 한계도 남는다.

CPU/GPU action 차이는 region64의 world 11, contact-velocity에서만 나타났다. 해당 compact risk는 actions 사이에 약 6×10⁻¹⁷ 수준으로 모두 사실상 0이었고, floating-point cancellation으로 tie-breaking이 달라졌다. 전체 100 unique timing cases에서 다른 methods의 action 차이는 없었다. CPU prediction을 실제 선택한 action의 GPU full-decoder reference와 대조한 maximum error는 전체 methods에서 1.8×10⁻¹⁴ 미만이었다. 이 현상을 substantive ranking uncertainty와 혼동하지 않으며 failure log를 보존했다.

### 8.4 Model mismatch와의 관계

Shifted data에 nominal inference를 쓴 stick-event loss는 region32 0.066489, native 0.069341, nested512 0.066478이었다. Correct-law diagnostic에서는 각각 0.060866, 0.058345, 0.058237이었다. 올바른 model을 제공했을 때에도 region32가 LMMSE보다 항상 좋아지는 것은 아니다. Correct-law condition은 law를 이미 안다는 privileged diagnostic이며 실용적인 mismatch adaptation 방법이 아니다.

Flexible-law worlds는 247개, unchanged base-law controls는 265개다. Controls의 factual data, observations, targets와 주요 methods의 losses가 paired nominal/shifted 조건에서 완전히 같은지 확인했다. 이는 model-mismatch manipulation의 구현 검증이다. 이 결과를 acquisition approximation만으로 model mismatch를 해결했다는 주장으로 확대하지 않는다.

### 8.5 가장 가까운 선행연구와 남는 차별점

Predictive quantity를 직접 겨냥한 experimental design은 이미 연구됐다. Zhong 등의 **Goal-Oriented Bayesian Optimal Experimental Design for Nonlinear Models using Markov Chain Monte Carlo**는 nonlinear QoI에 대한 EIG를 nested Monte Carlo, MCMC와 KDE로 추정한다. V16은 contact/stick prediction의 Brier loss를 직접 사용하는 compact estimator이지만, prediction-oriented acquisition이라는 목적 자체가 새롭지는 않다. [원문, v2](https://arxiv.org/html/2403.18072v2)

Zhu와 Hainy의 2026 preprint **Sequential Bayesian Experimental Design for Prediction in Physical Experiments Informed by Computer Models**는 KOH model의 predictive MI를 다루며 Gaussian mixture compression을 사용한다. Section 4.1에는 whitening, k-means moment matching, KNN-guided Runnalls merging이 제시된다. V16은 event-conditioned mass를 유지한 systematic representatives와 원래 sensor kernels를 사용한다는 구현 차이가 있다. 하지만 predictive mixture를 압축해 nested integration을 줄인다는 큰 아이디어는 선행연구와 겹친다. 해당 compression algorithm을 직접 구현해 비교한 것은 아니므로 성능 우위를 주장하지 않는다. [원문, Section 4.1](https://arxiv.org/html/2603.16756v1#S4.SS1)

Long의 **Multimodal Information Gain in Bayesian Design of Experiments**도 여러 modes를 mixture로 유지하는 Laplace approximation과 nested importance sampling을 다룬다. 따라서 “multimodality를 보존하면 single-Gaussian approximation보다 나을 수 있다”만으로는 충분한 novelty가 되지 않는다. V16의 conditional sensor-kernel mixture와는 구성 및 utility가 다르지만, 이 차이가 실용적 이득을 만드는지 별도로 입증해야 한다. [원문](https://arxiv.org/html/2108.07224v1)

위 논문들을 근거로 내리는 해석은 다음과 같다. **남을 수 있는 기여는 새로운 Bayes-risk objective가 아니라, downstream decision을 보존하는 compression을 언제·얼마나 수행해야 하는지를 검증 가능하게 정하는 방법**이다. 현재의 fixed K와 systematic resampling은 이를 위한 baseline이다.

### 8.6 다음 연구의 진행 기준

현재 구현은 reproducible negative/positive evidence를 함께 갖춘 실험으로 보존한다. 다음 방법을 개발한다면 fixed K를 사후 조정하는 방향보다 **acquisition error와 action gap에 따라 필요한 regions만 확장하는 adaptive compression**을 우선한다. 단, 다음 내용은 아직 구현하거나 성능을 입증한 결과가 아니라 후속 가설이다.

1. **Decision-loss-aware compression error:** 각 action의 compact risk와 independent held-out posterior reference 사이의 error를 추정한다. Top-two action gap이 error보다 작으면 components 또는 hypothetical samples를 추가한다. Confidence bound나 certificate를 주장하려면 sequential stopping과 reference error까지 반영한 별도 정당화가 필요하다.
2. **Region stratification의 필요성 검증:** Rare decision regions, 여러 contact modes, overlapping observation mixtures를 포함하는 새로운 predeclared distribution에서 equal-budget global resampling, generic GMM compression, optimized nested128 및 strong fixed rules와 비교한다. Counterexample를 알고 고른 몇 worlds만으로 성능을 주장하지 않는다.
3. **선택과 decoder를 분리한 compute frontier:** Acquisition만 compact하게 만들 때의 이득을 먼저 검증하고, compact decoder는 별도 가설로 다룬다. Full factual bank와 compression 비용을 계속 포함한다. GPU와 CPU 각각의 runtime–loss frontier를 보고, 두 event tasks의 사전 noninferiority 조건을 모두 통과해야 approximation claim을 한다.
4. **Simulation 밖의 검증:** 현재는 local impulse surrogate와 직접적인 noisy profiles다. Raw RGB, occlusion, geometry extraction error와 실제 contact trajectory를 포함한 검증이 없으므로 vision/robotics paper의 실증 수준에는 미치지 못한다. 별도 환경에서 동일한 conditional-mixture failure와 해결 효과가 나타나야 한다.

이번 단계에서 논문 기여를 정직하게 요약하면, **“pairwise hypothesis separation은 decision information을 과대평가할 수 있고, compact predictive mixtures로 이를 교정할 수 있다. 하지만 이 교정만으로 region-aware compression의 독자적 성능이나 nested-equivalent accuracy는 보장되지 않는다”**이다. 이 finding은 의미가 있으나 현재 수치만으로 넓은 범위의 새 method를 확정하기에는 부족하다.
