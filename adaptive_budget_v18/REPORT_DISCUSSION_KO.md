## 7. 결과 해석과 논문 기여 가능성

### 7.1 두 error sources를 구분하면 무엇이 달라졌는가

Matched contact에서는 426/512 worlds가 K16에서, 477/512가 M512에서 멈췄다. Adaptive의 평균 component–observation evaluations는 284,064, balanced는 399,264였다. Stick에서는 각각 3,727,104 대 4,325,760으로 adaptive가 더 적었다. 두 축을 무조건 함께 키우는 것보다 불필요한 likelihood evaluations를 줄이는 효과는 관찰됐다. 이 count는 준비·memory·error checks·kernel launches를 포함한 FLOPs나 latency가 아니다.

Component-only의 stick cap-unresolved는 332/512였고 adaptive는 260/512였다. Independent-seed agreement도 76.37%에서 82.62%로 높아졌다. 하지만 outer_only 82.81%, balanced 83.40%, fixed128/M4096 83.20%보다 높은 것은 아니다. 따라서 개선의 근거는 “outer refinement를 허용하면 도움이 된다”까지이며 “현재 두-error routing이 최선이다”까지는 아니다.

Adaptive stick의 278/512 worlds는 최대 K128/M4096 rectangle까지 계산했고 260/512는 stopping criterion을 만족하지 못했다. 이 unresolved 중 216은 compression proxy, 44는 outer proxy가 더 컸다. Outer sampling을 늘린 뒤에도 대부분의 unresolved cases에서 compression proxy 또는 작은 action gap이 병목으로 남는다. 이 숫자를 실제 compression error가 지배한다는 증명으로 읽어서는 안 된다.

Same-outer full-inner audit에서 contact의 observed gap discrepancy가 compression proxy보다 큰 비율은 30.97%, stick은 17.87%였다. Independent outer perturbation audit에서는 각각 22.22%, 21.31%였다. 특히 후자는 두 noisy estimates의 차이이므로 nominal confidence coverage의 실패율이 아니다. 이 결과와 shared-bias 가능성을 함께 보면, 현재 eC+eO를 reliable error certificate로 제시할 수 없다. Adaptive allocation은 heuristic으로 평가해야 한다.

### 7.2 Ranking 안정성과 decision loss가 일치하지 않는 이유

Ranking은 작은 risk 차이에도 action이 바뀌면 disagreement로 센다. 반면 actual decision loss는 실제 world와 observation noise에 대한 평균이다. 첫64 matched worlds의 independent nested8192 reference와 adaptive의 stick agreement는 65.63%이지만 reference risk regret는 약 5.19×10⁻⁵였다. Top-1 disagreement 모두가 큰 손실을 뜻하지 않는다는 수치다. 이 reference도 finite bank와 finite outer samples를 사용하므로 정확한 Bayes regret는 아니다.

Global128/M4096은 stick agreement 90.82%, mean loss 0.070957, GPU median 5.87ms로 강한 comparator다. Minority label support를 놓칠 수 있다는 구조적 약점만으로 이를 약한 baseline으로 제외할 수 없다. Region-preserving mixture가 필요한 조건을 별도 검증해야 하며, 이번 matched average에서는 adaptive의 우위를 보이지 않았다.

No observation의 stick loss 0.097282에 비해 adaptive는 0.071078로 낮았다. 그러나 always-probe도 0.072715이고 adaptive-minus-probe의 adjusted CI는 [−0.006948,+0.003351]이다. 추가 observation의 가치와 sophisticated observation selection의 추가 가치는 구분해야 한다. Contact에서는 query-only가 사실상 같은 loss를 훨씬 낮은 compute로 달성해, adaptive selection을 입증하기 어려운 ceiling 조건이다.

Shifted-law/nominal-inference stick loss는 adaptive 0.068577, global128/M4096 0.067193, nested2048 0.067621, always-probe 0.067616이었다. Privileged correct-law inference를 제공하면 adaptive는 0.063622였다. Budget refinement는 model mismatch를 자동으로 해결하지 않는다. Correct-law와 nominal의 차이는 acquisition뿐 아니라 posterior/decoder 차이도 포함하므로 순수한 selector 효과로 볼 수 없다.

### 7.3 Compute 결과를 어떻게 읽어야 하는가

GPU에서 cached likelihood evaluations를 줄여도 sequential stopping checks, synchronization과 작은 kernel launches가 든다. Adaptive stick은 평균 5.10 stages, balanced는 3.29 stages였다. Work count가 적은 adaptive의 stick GPU median이 18.52ms로 balanced 15.14ms보다 오히려 컸다. 이 관찰은 GPU에서 regular batched fixed computation이 유리할 수 있다는 해석을 지지한다. 상세한 profiler로 원인별 시간을 분해한 것은 아니므로 causal attribution은 추가 검증이 필요하다.

Original과 block-logsumexp kernel은 30,720 cases에서 선택 action, risks와 adaptive stopping이 일치했고 최대 risk 차이는 5.25×10⁻¹⁵였다. Fixed methods는 adaptive checks를 수행하지 않는 direct kernel로 측정했다. Original adaptive aggregate GPU median 7.16ms보다 optimized pass의 7.87ms가 오히려 높았다. 두 passes의 작은 시간 차이는 kernel 및 runtime variation을 함께 포함하므로 “optimization으로 adaptive를 가속했다”는 주장을 하지 않는다. 최종 비교에는 동일한 optimized pass의 모든 methods를 사용한다.

CPU aggregate median은 adaptive 5.52ms 대 fixed128/M4096 50.36ms지만, 두 tasks가 섞인 이 값은 hard-case 비용을 가린다. Stick의 감소는 50.81ms에서 43.64ms로 약14.1%이며, cheap fixed mixtures와의 tradeoff도 남는다. 따라서 GPU 결과를 CPU에 그대로 일반화하거나 aggregate median으로 일관된 9× CPU speedup을 주장하지 않는다.

이번 timing은 같은 hardware에서20 worlds×2tasks×3repeats다. 512-world loss 평가와는 표본 크기가 다르고 timing uncertainty CI를 제공하지 않는다. Batch1, float64, analytic cached forward model이라는 조건도 제한적이다. Expensive simulator나 real image encoder를 포함했을 때 budget allocation이 유리해질 가능성은 있지만 이번 결과가 그 가능성을 검증한 것은 아니다.

### 7.4 가장 가까운 선행연구와 novelty

Bartuska 등의 **Multilevel randomized quasi-Monte Carlo estimator for nested integration**는 nested integration에서 inner sample count와 inner integrand의 discretization accuracy를 levels로 다루고, bias와 statistical error를 분석하며 inner/outer sample allocation을 연구한다. V18은 discontinuous contact/stick event의 Brier-risk ranking에 conditional-mixture approximation을 쓴다는 차이가 있다. 그러나 inner approximation error와 outer integration error를 구별한다는 발상 자체는 새로운 기여가 아니다. 해당 multilevel estimator나 그 regularity assumptions를 이번 simulator에 적용해 이론적으로 검증한 것은 아니다. [원문](https://arxiv.org/html/2412.07723v1)

Sorokin 등의 **Fast Bayesian Multilevel Quasi-Monte Carlo**는 Bayesian cubature/GP uncertainty를 이용해, sample count를 두 배로 늘렸을 때 예상되는 uncertainty 감소와 비용을 비교하여 계산을 배분한다. V18은 단순히 현재 eC와 eO 중 큰 값을 기준으로 축을 고른다. Expected error reduction per unit cost를 추정하는 방법이 아니므로 optimal resource allocation이라고 부를 수 없다. 또한 V18의 4-scramble error proxy를 이 논문의 Bayesian error guarantee와 동일시할 수 없다. [원문](https://arxiv.org/html/2510.24604v1)

Dinkel 등의 2026 preprint **Efficient Bayesian Optimal Experimental Design for Expensive Computational Models over Finite Design Sets**는 EIG 차이를 common random numbers, Rao–Blackwellization과 paired bootstrap으로 비교하고, 불확실한 designs에 계산을 추가하는 adaptive elimination을 제시한다. V18은 event Brier loss와 compact conditional mixtures에 초점을 맞추지만, action gap의 불확실성에 따라 계산을 늘린다는 원리에는 선행연구가 있다. 해당 ADE method를 직접 구현해 이 데이터에서 비교한 것은 아니다. [원문](https://arxiv.org/html/2607.16933v1)

Zhu와 Hainy의 2026 preprint **Sequential Bayesian Experimental Design for Prediction in Physical Experiments Informed by Computer Models**는 predictive design에서 Gaussian mixture compression을 사용한다. V18은 label-conditioned nested representatives를 사용하지만, mixture를 압축해 nested acquisition 계산을 줄인다는 큰 아이디어만으로 novelty를 주장할 수는 없다. Generic GMM reduction과의 직접 비교도 남아 있다. [원문, Section 4.1](https://arxiv.org/html/2603.16756v1#S4.SS1)

### 7.5 주장할 수 있는 범위와 후속 연구의 조건

현재 loss는 event probability의 Brier loss다. 실제 robot control success, intervention cost, safety constraint나 task reward를 평가한 것은 아니다. 작은 analytic local-impulse model과 cached candidate forward predictions를 사용하며, raw images와 실제 영상에서의 geometry/contact-model error가 포함되지 않았다.

Method contribution을 주장하려면 fixed-budget mixtures와 balanced refinement를 넘는 accuracy–compute frontier, error proxy의 reliability, application-specific decision tolerance가 함께 필요하다. Marginal mean loss가 좋아 보인다는 이유만으로 acquisition error decomposition이 정확했다고 결론 내릴 수 없다. Conversely, ranking agreement가 높아져도 실제 loss나 compute가 나아지지 않으면 실용적인 contribution이 약하다.

추가 설계가 필요하다면 현재 error 크기만 비교하기보다 각 refinement의 **예상 decision-risk 감소 / incremental compute**를 추정하는 방향이 후보가 된다. 그러나 이 역시 cost-aware numerical integration과 adaptive experimental design의 기존 연구와 직접 비교해야 한다. Exact ranking 대신 application에서 정한 허용 regret를 사용하는 stopping도 검토할 수 있다. 이 보고서에서 그런 후속 방법을 구현하거나 성능을 입증했다고 주장하지 않는다.
