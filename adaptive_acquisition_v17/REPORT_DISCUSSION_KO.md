## 7. 해석과 논문 기여 가능성

### 7.1 Early stopping은 작동했지만 전체 compute는 줄지 않았다

Contact-event의 평균 최종 K는 27.0으로 cap128보다 훨씬 작았다. 두 replicas와 모든 stages에서 처리한 components의 평균 합도 약 184개였다. 그러나 stages 사이에 compression, likelihood evaluation, risk comparison을 반복하므로 최종 K가 작다는 사실만으로 latency 절약이 보장되지는 않는다.

Stick-event에서는 평균 최종 K가 105.23, 평균 stages가 4.49였다. 총 처리한 components는 평균 약 810개로 fixed128의 최대 512개보다 많다. Sampling-error term이 action gap보다 큰 상태에서 representation만 확장하면 비용을 쓰고도 stopping 조건을 만족시키지 못한다.

Original adaptive GPU median은 15.81 ms였다. Region indices/CDF를 재사용하고 logging용 GPU scalar의 CPU 변환을 줄인 equivalent implementation에서는 14.58 ms였다. 세 cells × 512 worlds × 2 tasks × 5 mixture methods, **15,360 cases 전체에서 actions, stopping metadata와 histories가 일치했고 maximum risk difference는 0**이었다. 최적화로 policy를 바꾸거나 좋은 결과를 다시 선택하지 않았다. Baselines도 같은 새 timing pass에서 다시 측정했다.

이러한 비용은 최소 가능한 algorithmic lower bound가 아니다. Batched stages, asynchronous stopping 또는 GPU-specific fusion을 더 개발할 여지는 있다. 다만 현재의 measured implementation에서 adaptive speed advantage는 없으며, “components 수가 줄었으니 compute도 절약했다”는 주장은 성립하지 않는다. Exact kernel-level profiling으로 원인을 완전히 분해한 것은 아니다.

### 7.2 Acquisition error의 한 종류만 줄인다는 한계

이 실험의 numerical error는 적어도 세 종류다. 첫째는 compact inner mixture가 full posterior predictive response를 얼마나 잘 표현하는지에 따른 compression error다. 둘째는 hypothetical outcomes를 512개로 적분하는 outer integration error다. 셋째는 finite QMC factual posterior와 contact model 자체의 error다.

K를 늘리는 것은 주로 첫 번째를 겨냥한다. 두 번째는 고정된 상태이며 세 번째 역시 변화하지 않는다. 따라서 components를 늘릴수록 모든 error가 줄어든다는 전제가 맞지 않는다. Matched stick의 357 cap-unresolved cases 중 278개에서 sampling term만으로 gap 이상이었던 결과는 이 한계를 직접 보여 준다. 단, heuristic SE의 크기만으로 true error의 인과적 비중을 정확히 추정했다고 주장하지는 않는다.

Coarse stages에서 gap이 충분히 큰 cases는 일찍 종료하고, ambiguous cases는 cap까지 간다는 동작 자체는 구현됐다. 첫 64 matched worlds 중 gap-separated인 contact 61개, stick 15개는 모두 nested2048와 action이 같았다. 그러나 이는 작은 diagnostic subset이며 valid confidence coverage나 false-certification probability의 증명이 아니다.

### 7.3 안정성과 정확성은 다른 평가다

Adaptive region의 first64 matched stick reference agreement는 68.75%, nested512는 82.81%였다. 반면 independent acquisition seed 사이에서는 adaptive가 77.54%, nested512가 76.56%로 가까웠다. Reference agreement, self-repeatability, actual decision loss는 서로 다른 질문에 답한다.

Adaptive global이 더 stable하다는 수치도 주의해서 읽어야 한다. Global compression은 rare label region의 mass를 정확히 유지하지 않는다. 이 때문에 compact representation에서 prediction이 거의 constant가 되면 actions가 동률이 되고 query view로 결정될 수 있다. Section 5의 post-analysis label-support audit에서 matched stick의 global risk ties 210개 중 209개는 full factual posterior에 두 regions가 있는데 두 compact replicas가 single-label이 된 경우였다. 이 subset의 minority mass median은 약 0.000656, maximum은 약 0.06044였다. Contact에서는 41개 ties 중 39개가 해당했고 minority mass maximum은 약 0.13156이었다. 높은 agreement를 정확한 uncertainty representation으로 곧바로 해석하지 않는다.

동시에 global methods의 actual losses가 adaptive region보다 명확히 나쁘다는 근거도 없었다. Region mass를 보존한다는 구조적 장점이 실제 decision loss를 줄이는 충분한 이유는 아니다. 현재 결과는 global mixture에 대한 독자적 superiority를 지지하지 않는다.

Independent-bank audit에서도 stick Top-1 agreement는 adaptive 85.94%, global128 87.50%, LMMSE16 90.63%, nested512 81.25%였다. 이 audit는 64 worlds에 한정되며 full multi-seed generalization을 보장하지 않는다. Bank가 바뀌어도 같은 action을 택하는 LMMSE가 항상 더 정확하다는 뜻도 아니다.

### 7.4 Model mismatch와 실제 검증 범위

Shifted data/nominal inference의 stick loss는 adaptive 0.077189, native 0.077388, nested512 0.077818이었다. Correct-law diagnostic에서는 0.068911, 0.068037, 0.069161이었다. Model을 올바르게 제공해도 adaptive region이 LMMSE를 일관되게 이기는 것은 아니다.

현재 연구는 noisy boundary profiles와 local impulse surrogate를 사용한다. Raw image의 ambiguity, geometry reconstruction error, full collision trajectories와 실제 영상의 domain gap을 이 실험에서 검증하지 않았다. Factual bank도 계속 필요하다. 따라서 이 결과를 real-world visual active perception이나 bank-free scene belief의 성과로 확대하지 않는다.

### 7.5 관련 연구와 novelty 판단

Dinkel 등의 2026 preprint **Efficient Bayesian Optimal Experimental Design for Expensive Computational Models over Finite Design Sets**는 design 간 EIG 차이에 common random numbers와 Rao–Blackwellization을 적용하고, paired bootstrap comparisons에 따라 designs를 제거하거나 sample budget을 늘린다. Sections 2.3–2.4와 Algorithm 1을 확인했다. V17은 EIG 대신 event Brier risk를 사용하고 inner conditional-mixture size를 조절한다는 차이가 있다. 그러나 action 간 차이가 불확실할 때 계산을 더 쓴다는 원리 자체는 새롭지 않다. 해당 ADE algorithm을 이 physics setting에서 재현해 비교한 것은 아니다. [원문](https://arxiv.org/html/2607.16933v1)

Zhu와 Hainy의 2026 preprint **Sequential Bayesian Experimental Design for Prediction in Physical Experiments Informed by Computer Models**는 predictive MI 계산에 Gaussian mixture compression을 사용한다. Whitening, k-means moment matching과 KNN-guided Runnalls reduction을 제시한다. V17의 label-conditioned systematic representatives와 gap-based expansion은 구성상 다르지만, predictive mixture를 줄여 acquisition 비용을 낮춘다는 큰 아이디어 역시 선행연구와 겹친다. Generic GMM compression에 대한 직접적인 superiority는 아직 검증하지 않았다. [원문, Section 4.1](https://arxiv.org/html/2603.16756v1#S4.SS1)

Garivier와 Kaufmann의 **Optimal Best Arm Identification with Fixed Confidence**는 Track-and-Stop의 sampling/stopping strategy와 asymptotic optimality를 제시한다. V17의 sequential heuristic은 해당 theoretical guarantees를 따르지 않는다. Biased nested approximation, dependent QMC samples와 compression error를 가진 여기의 설정에 그 보장을 그대로 적용할 수 없다. [원문 출판 페이지](https://proceedings.mlr.press/v49/garivier16a.html)

따라서 “adaptive compression을 넣었다”만으로는 method novelty가 약하다. 기존 adaptive design 및 mixture reduction과 비교했을 때, representation error와 integration error를 구별하여 필요한 계산만 늘리고, 실제 downstream risk를 보존한다는 추가 근거가 있어야 한다.

### 7.6 다음 단계의 우선순위

현재 데이터를 보고 cap만 높이거나 threshold를 낮추는 방식으로 성공 여부를 바꾸지 않는다. 다음 가설을 평가한다면 새로운 protocol과 worlds를 사용해야 한다.

1. **Error-source-aware budget allocation:** Compression drift나 replica discrepancy가 큰 경우 K를, randomized outer integration variation이 큰 경우 outer sample count를 늘린다. Posterior-bank discrepancy가 지배적이면 acquisition integration을 더 해도 해결되지 않으므로 별도로 다룬다. 이 분기는 아직 구현·검증한 방법이 아니다.
2. **Exact ranking보다 bounded decision regret:** Actions가 사실상 동등하면 ranking을 끝까지 분리하는 데 비용을 쓰지 않는 stopping을 고려한다. Allowed risk tolerance는 test 결과가 아니라 application cost에서 정해야 한다. 이 역시 새로운 원리라고 주장하기 전에 best-arm/good-arm identification 및 adaptive OED와의 차이를 확인해야 한다.
3. **강한 fixed baseline을 먼저 넘는 검증:** Fixed32/128, global compression, nested128, task-specific query/probe rules를 동시에 비교한다. Loss와 actual latency의 Pareto frontier를 개선하지 못하면 adaptive method claim을 중단한다. 실제 optimized CPU median은 adaptive 6.52 ms, nested512 89.63 ms로 약 13.8× 빠르다. 그러나 fixed128은 3.49 ms, global128은 3.42 ms로 더 저렴하다. CPU에서 nested512 대비 빨라도 더 싼 fixed mixture가 같은 loss를 내면 adaptivity의 필요성은 입증되지 않는다.
4. **실제 geometry/observation으로의 연결:** 여러 hidden contact modes와 rare but decision-relevant regions가 실제 시각적 관측에서도 나타나는지 확인한다. Synthetic corner 사례만으로 region stratification의 일반적 가치를 주장하지 않는다.

이번 실험의 가장 명확한 finding은 **“acquisition uncertainty가 남았다는 이유만으로 components를 늘리는 것은 충분하지 않으며, 고정된 outer integration error가 병목이면 더 큰 compact mixture가 오히려 비용만 늘릴 수 있다”**이다. 유용한 failure analysis이지만, 현재 adaptive rule 자체의 우수성을 입증한 결과는 아니다.
