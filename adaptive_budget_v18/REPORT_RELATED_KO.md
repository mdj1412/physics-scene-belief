### 7.4 가장 가까운 선행연구와 novelty

Bartuska 등의 **Multilevel randomized quasi-Monte Carlo estimator for nested integration**는 nested integration에서 inner sample count와 inner integrand의 discretization accuracy를 levels로 다루고, bias와 statistical error를 분석하며 inner/outer sample allocation을 연구한다. V18은 discontinuous contact/stick event의 Brier-risk ranking에 conditional-mixture approximation을 쓴다는 차이가 있다. 그러나 inner approximation error와 outer integration error를 구별한다는 발상 자체는 새로운 기여가 아니다. 해당 multilevel estimator나 그 regularity assumptions를 이번 simulator에 적용해 이론적으로 검증한 것은 아니다. [원문](https://arxiv.org/html/2412.07723v1)

Sorokin 등의 **Fast Bayesian Multilevel Quasi-Monte Carlo**는 Bayesian cubature/GP uncertainty를 이용해, sample count를 두 배로 늘렸을 때 예상되는 uncertainty 감소와 비용을 비교하여 계산을 배분한다. V18은 단순히 현재 eC와 eO 중 큰 값을 기준으로 축을 고른다. Expected error reduction per unit cost를 추정하는 방법이 아니므로 optimal resource allocation이라고 부를 수 없다. 또한 V18의 4-scramble error proxy를 이 논문의 Bayesian error guarantee와 동일시할 수 없다. [원문](https://arxiv.org/html/2510.24604v1)

Dinkel 등의 2026 preprint **Efficient Bayesian Optimal Experimental Design for Expensive Computational Models over Finite Design Sets**는 EIG 차이를 common random numbers, Rao–Blackwellization과 paired bootstrap으로 비교하고, 불확실한 designs에 계산을 추가하는 adaptive elimination을 제시한다. V18은 event Brier loss와 compact conditional mixtures에 초점을 맞추지만, action gap의 불확실성에 따라 계산을 늘린다는 원리에는 선행연구가 있다. 해당 ADE method를 직접 구현해 이 데이터에서 비교한 것은 아니다. [원문](https://arxiv.org/html/2607.16933v1)

Zhu와 Hainy의 2026 preprint **Sequential Bayesian Experimental Design for Prediction in Physical Experiments Informed by Computer Models**는 predictive design에서 Gaussian mixture compression을 사용한다. V18은 label-conditioned nested representatives를 사용하지만, mixture를 압축해 nested acquisition 계산을 줄인다는 큰 아이디어만으로 novelty를 주장할 수는 없다. Generic GMM reduction과의 직접 비교도 남아 있다. [원문, Section 4.1](https://arxiv.org/html/2603.16756v1#S4.SS1)

### 7.5 주장할 수 있는 범위와 후속 연구의 조건

현재 loss는 event probability의 Brier loss다. 실제 robot control success, intervention cost, safety constraint나 task reward를 평가한 것은 아니다. 작은 analytic local-impulse model과 cached candidate forward predictions를 사용하며, raw images와 실제 영상에서의 geometry/contact-model error가 포함되지 않았다.

Method contribution을 주장하려면 fixed-budget mixtures와 balanced refinement를 넘는 accuracy–compute frontier, error proxy의 reliability, application-specific decision tolerance가 함께 필요하다. Marginal mean loss가 좋아 보인다는 이유만으로 acquisition error decomposition이 정확했다고 결론 내릴 수 없다. Conversely, ranking agreement가 높아져도 실제 loss나 compute가 나아지지 않으면 실용적인 contribution이 약하다.

추가 설계가 필요하다면 현재 error 크기만 비교하기보다 각 refinement의 **예상 decision-risk 감소 / incremental compute**를 추정하는 방향이 후보가 된다. 그러나 이 역시 cost-aware numerical integration과 adaptive experimental design의 기존 연구와 직접 비교해야 한다. Exact ranking 대신 application에서 정한 허용 regret를 사용하는 stopping도 검토할 수 있다. 이 보고서에서 그런 후속 방법을 구현하거나 성능을 입증했다고 주장하지 않는다.
