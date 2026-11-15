## 문헌 검토와 연구 설계에 대한 평가

### 1. 원문별 비교

| 연구 | Research question과 방법 | Sample·data source | 결과와 결론 | V24에 적용할 때의 한계 |
|---|---|---|---|---|
| Teerapittayanon, McDanel & Kung, *BranchyNet* | 쉬운 입력에서 추론을 끝낼 수 있는가? 여러 깊이에 side classifiers를 두고 confidence로 early exit한다. | MNIST·CIFAR-10, LeNet·AlexNet·ResNet 계열 | 실험한 네트워크에서 많은 입력이 일찍 종료됐고 inference time을 줄일 수 있었다. | 최종 image classification과 one-step observation selection은 목표가 다르다. 입력별 분기가 V24 p90을 줄이는지는 별도 측정이 필요하다. [원문](https://bradmcdanel.com/wp-content/uploads/2016/10/2016-icpr-teerapittayanon-mcdanel-kung.pdf) |
| Douc, Cappé & Moulines, *Comparison of Resampling Schemes for Particle Filtering* | Particle resampling 방식에 따라 conditional variance가 달라지는가? 이론적 비교와 counterexample을 제시했다. | Weighted empirical measures, asymptotic analysis. 실제 장비 표본 없음 | Residual 및 stratified resampling은 제시한 조건에서 multinomial보다 낮거나 같은 variance를 갖는다. Systematic resampling에는 해당 ordering의 반례가 있다. | Particle 통계량의 variance 보장을 action 선택 후의 Brier noninferiority로 옮길 수 없다. [원문](https://arxiv.org/pdf/cs/0507025) |
| Dinkel et al., *Efficient Bayesian Optimal Experimental Design for Expensive Computational Models over Finite Design Sets* | 제한된 forward evaluations로 가장 유용한 design을 찾는가? Common random numbers, sample reuse, adaptive elimination을 비교했다. | Synthetic, pharmacokinetic, lung, viscoplastic 네 사례. 일부 ranking curves는 1,000 independent runs | Estimator와 design별 budget 배정에 따라 selection accuracy를 높이는 데 드는 평가 수가 달라졌다. | EIG 순위와 model evaluations를 비교한다. 실행시간 분포의 p90 및 physical query의 예측 오차는 다른 측정량이다. [원문 §3](https://arxiv.org/html/2607.16933v1#S3) |
| Zhu & Hainy, *Sequential Bayesian Experimental Design for Prediction in Physical Experiments Informed by Computer Models* | 어떤 physical observation이 미래 예측의 불확실성을 더 줄이는가? Predictive mutual information과 local complexity adjustment, Gaussian mixture compression을 사용했다. | Toy simulation, JAK–STAT5 시계열. 실제 사례에서는 초기 4 time points에서 10 rounds를 평가 | Field prediction에서 predictive criterion과 평가 지표에 따라 방법의 상대적 순위가 달라졌다. | Continuous time-series와 보간한 reference를 사용했다. 새 sensor의 binary contact prediction에 적용하려면 다른 검증이 필요하다. [원문 §§4–5](https://arxiv.org/html/2603.16756v1#S5) |
| Piaggio et al., *Reporting of Noninferiority and Equivalence Randomized Trials* | Noninferiority 주장을 해석 가능하게 보고하려면 무엇을 밝혀야 하는가? CONSORT reporting extension과 예시를 제시했다. | 임상시험 보고 방법에 관한 원문. 새 GPU나 물리 실험 cohort가 없다. | Margin의 사전 명시와 근거, confidence interval 방향, active comparator를 명시하도록 제시한다. | 여기서 차용한 것은 판정·보고 원칙이다. +0.003이나 5%를 권장하는 논문은 아니다. [원문](https://jamanetwork.com/journals/jama/fullarticle/1487502) |

### 2–3. Literature review와 knowledge map

**예측 목표와 계산 배정.** Zhu·Hainy는 관측이 최종 예측에 주는 가치를 다루고, Dinkel 등은 여러 관측 후보에 대한 numerical ranking 비용을 다룬다. BranchyNet은 입력별로 추가 계산이 필요한지를 앞단에서 판단한다. V24는 같은 action 후보에 대해 public factual evidence를 읽고 posterior support의 크기를 정한다. 이 세 문헌은 목표와 계산이 만나는 위치를 설명하지만, 동일한 contact simulator의 Brier와 p90을 함께 평가하지는 않았다. [Zhu & Hainy](https://arxiv.org/html/2603.16756v1#S3) · [Dinkel et al.](https://arxiv.org/html/2607.16933v1#S2) · [BranchyNet](https://arxiv.org/abs/1709.01686)

**Knowledge map.** Factual evidence → cheap approximation-error predictor → support budget → resampled label posterior → action별 expected Brier → selected observation → realized Brier로 이어진다. 별도 경로인 budget → GPU latency distribution은 같은 입력에서 측정한다. Weighted resampling의 일부 variance 성질과 margin을 사용한 판정은 알려져 있다. Error predictor가 새로운 calibration에서도 decision-relevant worlds를 찾아내는지, routing overhead를 포함해 p90을 줄이는지는 경험적으로 검증해야 한다. [Douc et al.](https://arxiv.org/pdf/cs/0507025) · [Dinkel et al.](https://arxiv.org/html/2607.16933v1#S2) · [Piaggio et al.](https://jamanetwork.com/journals/jama/fullarticle/1487502)

### 4–7. Research gaps, 상충하는 결과, methodology, evidence synthesis

| Evidence와 비교 축 | 남은 gap 및 해석 |
|---|---|
| BranchyNet의 classification accuracy와 inference time, Dinkel 등의 design 선택 정확도와 forward calls는 sample, hardware, endpoint가 다르다. [BranchyNet](https://arxiv.org/abs/1709.01686) · [Dinkel et al.](https://arxiv.org/html/2607.16933v1#S3) | 평균 계산량 감소와 p90 감소가 같은 결론을 내릴 필요는 없다. 두 연구가 같은 지표에 대해 충돌한다는 근거는 없다. V24에서는 gate와 always-high의 비용을 같은 호출 경로에서 잰다. |
| Douc 등은 estimator의 반복 변동을 다루고, Zhu·Hainy는 design 선택 후 다른 metric에서의 predictive performance를 비교한다. [Douc et al.](https://arxiv.org/pdf/cs/0507025) · [Zhu & Hainy](https://arxiv.org/html/2603.16756v1#S5) | Posterior approximation 개선과 binary loss 개선의 전달 과정에 대한 직접 근거는 약하다. Continuous risk distortion과 실제로 action이 달라진 횟수, 그때의 loss를 따로 측정한다. |
| Piaggio 등의 margin은 특정 적용에서 허용 가능한 저하와 관련된다. 문헌끼리 trial, algorithm run, empirical measure의 표본 단위가 다르다. [Piaggio et al.](https://jamanetwork.com/journals/jama/fullarticle/1487502) · [Dinkel et al.](https://arxiv.org/html/2607.16933v1#S3) | 이 corpus의 효과를 하나의 평균으로 합치지 않았다. 새 장비와 contact law에 허용 가능한 차이는 불확실하다. V24의 crossed bootstrap은 simulator의 world·bank 변동을 평가한다. |

### 8. 다음 연구 질문 10개

| 질문과 연구가 필요한 이유 | Variables와 적합한 검증 | 근거 |
|---|---|---|
| 1. 실제 high-budget 배정이 같은 사용 빈도의 blind routing보다 유용한가? 저렴한 branch 자체와 배정 가치를 분리해야 한다. | Firing rate를 유지해 기대 Brier를 비교한다. | [BranchyNet](https://arxiv.org/abs/1709.01686) |
| 2. 같은 평균 support를 전부에게 주면 더 나은가? Adaptive computation의 overhead를 반영해야 한다. | 정적 K와 gated K를 같은 M과 decoder에서 interleaved timing한다. | [Dinkel et al.](https://arxiv.org/html/2607.16933v1#S2) |
| 3. 값이 작은 action-risk error가 실제로 중요한가? Ranking과 loss는 다를 수 있다. | Centered risk error, action flip, paired Brier를 함께 추적한다. | [Dinkel et al.](https://arxiv.org/html/2607.16933v1#S2) |
| 4. Tail의 label별 slots 대신 sensor likelihood별 slots를 배정할 수 있는가? 같은 크기로도 남는 bias가 다를 수 있다. | Label stratification과 response clustering을 같은 K에서 비교한다. | [Douc et al.](https://arxiv.org/pdf/cs/0507025) · [Zhu & Hainy](https://arxiv.org/html/2603.16756v1#S4) |
| 5. Curvature와 friction의 joint effect가 특정 셀에 피해를 집중시키는가? 전체 평균으로는 가려질 수 있다. | Factorial cell별 Brier와 action 빈도를 보고, 독립 object sample에서 interaction을 재검증한다. | [Zhu & Hainy](https://arxiv.org/html/2603.16756v1#S5) |
| 6. Sensor calibration을 잘못 알면 gate는 얼마나 민감한가? 알려진 noise scale을 사용하는 검증만으로는 부족하다. | 실제 noise와 inference calibration을 독립적으로 바꿔 calibration-error sweep을 한다. | [Zhu & Hainy](https://arxiv.org/html/2603.16756v1#S2) |
| 7. Lab에서 정한 noninferiority margin이 실제 사용 비용에 맞는가? | 재촬영·reset·실패 비용과 Brier 차이를 측정해 새 margin을 실험 전에 확정한다. | [Piaggio et al.](https://jamanetwork.com/journals/jama/fullarticle/1487502) |
| 8. Training에 보지 않은 contact law에서도 방법이 성립하는가? Matched prior 실험으로는 충분하지 않다. | Coulomb, speed-dependent friction, damping을 train/test에서 교차한다. | [Zhu & Hainy](https://arxiv.org/html/2603.16756v1#S2) |
| 9. GPU-resident vision encoder를 쓰면 host-side routing이 여전히 싼가? Raw input의 시작 위치가 달라진다. | Host/device input, batch size, device-to-host synchronization을 요인으로 p90을 다시 잰다. | [BranchyNet](https://arxiv.org/abs/1709.01686) |
| 10. One-step에서 줄인 비용이 여러 관측을 고를 때 누적되는가? | Sequential horizon, posterior updates, trajectory loss와 wall-clock budget을 함께 평가한다. | [Zhu & Hainy](https://arxiv.org/html/2603.16756v1#S3) |

### 9. Research assistant review

입력별 계산 예산 조절, predictive experimental design, stratified resampling은 각각 선례가 있다. 이 다섯 편은 V24의 방법과 측정에 직접 관련된 corpus이며 포괄적인 novelty search는 아니다. 알고리즘 기여를 주장하려면 강한 정적 budget과의 차이, route가 달라졌을 때의 이득, 새로운 물체에서의 재현을 보여야 한다. Simulator에서 동등한 sensing 비용을 가정한 end-to-end latency는 물리 센서 동작까지 포함한 robot latency로 해석할 수 없다. [BranchyNet](https://arxiv.org/abs/1709.01686) · [Dinkel et al.](https://arxiv.org/html/2607.16933v1#S3) · [Zhu & Hainy](https://arxiv.org/html/2603.16756v1#S5)
