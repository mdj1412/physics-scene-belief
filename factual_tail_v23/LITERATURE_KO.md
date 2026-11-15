## 문헌 분석과 project review

### 1. 원문별 research question, 방법 및 적용 범위

| 연구 | Research question · methodology | Sample · data source | 결과와 결론 | V23 적용 한계 |
|---|---|---|---|---|
| Roy & McCallum, *Toward Optimal Active Learning through Monte Carlo Estimation of Error Reduction* | 다음 label을 얻으면 future classification error가 얼마나 줄어드는가? 후보 label을 적분해 재학습 후 error를 추정했다. | Real-world classification datasets 3개. | 일부 비교에서 동일 accuracy에 필요한 labeled examples를 크게 줄였다. | Training label을 사는 문제다. 한 물체의 sensor action과 GPU inference 비용은 직접 측정하지 않았다. [원문](https://groups.csail.mit.edu/rrg/papers/icml01.pdf) |
| Attia, Alexanderian & Saibaba, *Goal-Oriented Optimal Design of Experiments for Large-Scale Bayesian Linear Inverse Problems* | 최종 quantity of interest의 posterior uncertainty를 줄이는 sensor placement는 무엇인가? A-/D-optimal criteria와 gradient optimization을 구성했다. | Advection–diffusion 초기값 역문제. 건물 주변 농도 예측에 대해 위치별 placement를 비교했다. | Goal에 따라 informative sensor의 위치가 달라지며, 저차원 goal을 계산에 활용할 수 있다. | Linear forward model과 Gaussian uncertainty에 의존한다. Binary contact transition의 Brier 차이로 직접 환산할 수 없다. [원문](https://arxiv.org/pdf/1802.06517) |
| Douc, Cappé & Moulines, *Comparison of Resampling Schemes for Particle Filtering* | Weighted particles의 resampling에서 conditional variance를 줄일 수 있는가? 수학적 비교와 counterexample을 제시했다. | Empirical measures와 이론적 비교. 실제 물체 표본은 없다. | Residual 및 stratified resampling에는 multinomial 대비 variance ordering이 있다. Systematic resampling에는 반례가 있다. | 해당 보장은 discrete action 선택이나 held-out Brier의 개선까지 포함하지 않는다. [원문](https://arxiv.org/pdf/cs/0507025) |
| Dinkel et al., *Efficient Bayesian Optimal Experimental Design for Expensive Computational Models over Finite Design Sets* | 여러 design의 expected information gain을 적은 forward calls로 비교할 수 있는가? Sample reuse, common random numbers, adaptive design elimination을 평가했다. | Synthetic, pharmacokinetic, lung, viscoplastic model의 네 사례. Synthetic 및 pharmacokinetic selection curve는 각각 1,000회 independent runs를 사용했다. | 동일한 design을 반복 평가하는 비용과 선택 오류를 줄일 수 있는 조건을 제시했다. | Ranking accuracy 및 forward calls와 Brier 및 GPU p90는 서로 다른 측정량이다. [원문 §§2–3](https://arxiv.org/html/2607.16933v1#S3) |
| Zhu & Hainy, *Sequential Bayesian Experimental Design for Prediction in Physical Experiments Informed by Computer Models* | Future prediction을 개선할 physical observation은 무엇인가? Predictive mutual information, local complexity, Gaussian mixture compression을 연구했다. | Toy simulation 및 JAK–STAT5 field time series. 실제 사례에서 초기 4 time points로 10차례 design을 평가했다. | Prediction 평가 지표에 따라 유리한 criterion이 달랐다. | Time-series prediction과 보간한 reference를 one-step stick decision에 일반화하려면 추가 검증이 필요하다. [원문 §§4–5](https://arxiv.org/html/2603.16756v1#S5) |

### 2–3. 주제별 literature review와 knowledge map

**최종 prediction에서 observation으로.** Expected error reduction은 다음 observation을 고르는 목적을 후속 prediction의 손실에 두며, goal-oriented design은 sensor의 유용성을 예측 대상에 연결한다. V23의 query 위치를 고정한 채 geometry와 friction이 서로 다른 observation을 요구하게 만든 것은 이 연결을 시험하는 설계다. Latency에서의 우위는 이 선행연구로부터 도출되지 않는다. [Roy & McCallum](https://groups.csail.mit.edu/rrg/papers/icml01.pdf) · [Attia et al.](https://arxiv.org/pdf/1802.06517) · [Zhu & Hainy](https://arxiv.org/html/2603.16756v1#S3)

**근사에서 decision으로.** Douc 등의 연구가 다루는 것은 weighted measure에서 표본을 선택할 때의 variance다. Dinkel 등은 numerical uncertainty 아래에서 designs 사이의 순위를 선택하는 단계까지 다룬다. 이 두 단계에 후속 physical outcome을 붙이면 **factual weights → tail support → conditional stick probability → action별 risk → 선택된 observation → realized Brier**로 이어진다. 첫 단계의 이론적 성질은 비교적 잘 알려져 있지만, 같은 GPU budget에서 마지막 단계까지 이득이 전파되는 조건은 이 corpus에서 확립되지 않았다. [Douc et al.](https://arxiv.org/pdf/cs/0507025) · [Dinkel et al.](https://arxiv.org/html/2607.16933v1#S2)

### 4–7. Research gaps, 상충하는 결과, methodology, evidence synthesis

| 검토할 주장 | 근거와 방법 차이 | 판단 및 남은 gap |
|---|---|---|
| “Informative observation은 언제나 같다.” | Attia 등은 goal 위치를 바꾸고, Zhu·Hainy는 predictive criteria와 평가 지표를 바꿨다. | Goal dependence에는 일관된 근거가 있다. 동일 binary target에서 family를 숨긴 채 독립적인 friction·curvature variation을 평가하는 문제는 별도다. [Attia et al.](https://arxiv.org/pdf/1802.06517) · [Zhu & Hainy](https://arxiv.org/html/2603.16756v1#S5) |
| “정확한 ranking은 동등한 크기의 downstream gain을 낳는다.” | Dinkel 등의 반복 단위는 algorithm runs이고, Roy·McCallum의 관측 단위는 classification examples다. 측정량도 selection accuracy와 future classifier error로 다르다. | 같은 estimand에 대한 상반된 결론을 내린 연구로 볼 근거는 부족하다. Runner-up gap과 실제 policy disagreements의 loss를 함께 측정해야 한다. [Dinkel et al.](https://arxiv.org/html/2607.16933v1#S3) · [Roy & McCallum](https://groups.csail.mit.edu/rrg/papers/icml01.pdf) |
| “Variance가 낮으면 adaptive tail이 Brier를 줄인다.” | Conditional variance ordering은 특정 resampling 단계의 이론이며, physical design 사례의 predictive error는 model·criterion·metric에 의존한다. | Resampling variance에 대한 근거는 강하다. Factual gate의 downstream 이득에 대한 직접 증거는 부족하다. 동일 K에서 always/never/random routing을 비교할 이유다. [Douc et al.](https://arxiv.org/pdf/cs/0507025) · [Zhu & Hainy](https://arxiv.org/html/2603.16756v1#S4) |

Numerical case studies는 mechanism과 compute의 trade-off를 보여주며, 실제 classification 및 physical time series는 제한된 domain에서 predictive utility를 뒷받침한다. Sample, endpoint, observation 비용이 달라 effect sizes를 합산하지 않았다. V23의 gate-on subgroup은 factual score에 의해 선택되므로 그 집단의 평균 loss만으로 gate의 효과를 추론하지 않는다. [Roy & McCallum](https://groups.csail.mit.edu/rrg/papers/icml01.pdf) · [Dinkel et al.](https://arxiv.org/html/2607.16933v1#S3) · [Zhu & Hainy](https://arxiv.org/html/2603.16756v1#S5)

### 8. 다음 research questions 10개

| 질문과 gap | 왜 중요한가 · 변수와 방법 | 근거 |
|---|---|---|
| 1. 추가 sensor precision이 실제 runner-up 비용을 키우는가? | 같은 world·noise에서 precision을 바꾸고 split-repeat Brier를 비교한다. V23에서 수행했다. | [Attia et al.](https://arxiv.org/pdf/1802.06517) |
| 2. Gate는 동일한 빈도의 무작위 allocation을 이기는가? | Tail을 많이 쓰는 효과와 적절한 world에 쓰는 효과를 구분한다. Bank별 firing rate를 맞춘 random routing expectation을 비교한다. | [Dinkel et al.](https://arxiv.org/html/2607.16933v1#S2) |
| 3. Tail recovery가 sensor likelihood에서 보이지만 query label에는 무관한 경우가 얼마나 되는가? | Tail mass와 full-vs-truncated label error를 따로 측정해 computation의 표적을 확인한다. | [Douc et al.](https://arxiv.org/pdf/cs/0507025) · [Attia et al.](https://arxiv.org/pdf/1802.06517) |
| 4. Label mass 대신 sensor response별 support를 보존하면 좋은가? | Conditional distribution의 여러 modes를 보존하도록 cluster 축을 바꾸고 같은 p90에서 평가한다. | [Zhu & Hainy](https://arxiv.org/html/2603.16756v1#S4) |
| 5. 저비용 factual slope rule이 충분한가? | 단순 cue로 policy를 대체할 가능성을 확인한다. Slope sign rule과 same decoder의 Brier를 비교한다. | [Roy & McCallum](https://groups.csail.mit.edu/rrg/papers/icml01.pdf) |
| 6. Family cue의 신뢰도가 낮아져도 유지되는가? | Slope distributions의 overlap을 조절하고 gate calibration과 held-out Brier를 측정한다. | [Attia et al.](https://arxiv.org/pdf/1802.06517) |
| 7. Action gap이 큰 동시에 두 selectors가 갈리는 영역을 사전에 예측할 수 있는가? | Held-out noise로 사후 선별하면 비용을 과장할 수 있다. 독립적인 학습 split에서 action-value 차이를 추정하고 새 worlds 전부에서 검증한다. | [Roy & McCallum](https://groups.csail.mit.edu/rrg/papers/icml01.pdf) · [Dinkel et al.](https://arxiv.org/html/2607.16933v1#S2) |
| 8. 큰 model mismatch에서도 tail이 도움이 되는가? | Out-of-support physics는 resampling으로 복원할 수 없다. Friction law를 교체하고 matched/model-expanded 조건을 교차한다. | [Douc et al.](https://arxiv.org/pdf/cs/0507025) · [Zhu & Hainy](https://arxiv.org/html/2603.16756v1#S2) |
| 9. Sensor 비용과 GPU budget을 함께 반영하면 정책이 바뀌는가? | Probe 시간·reset 비용을 실제로 측정하고 Brier–cost frontier를 평가한다. | [Dinkel et al.](https://arxiv.org/html/2607.16933v1#S3) · [Zhu & Hainy](https://arxiv.org/html/2603.16756v1#S5) |
| 10. 실제 물체의 반복 intervention에서 효과가 재현되는가? | Reset 가능한 물체에서 query view·probe를 교차 수집하고 object-level split으로 평가한다. | [Zhu & Hainy](https://arxiv.org/html/2603.16756v1#S5) |

### 9. Research assistant review

예측 목표 중심의 acquisition, variance를 줄이는 resampling, mixture compression 모두 선례가 있다. 이번 corpus는 이들 연결을 검토하기 위한 다섯 편으로 범위를 정했으므로 분야 전체의 novelty를 단정할 수 없다. 논문에서 검증해야 할 주장은 세 축이다. 잘못된 action을 택할 때 정말 손실이 커지는지, factual gate가 강한 두 baseline이 놓치는 상황을 예측하는지, 그 예측이 online 비용을 지불하고도 남는지다. 첫 번째가 성립해도 나머지는 실측으로 판단해야 한다. Real contact의 actuator·image noise와 distribution shift에 대한 증거도 별도로 필요하다. [Douc et al.](https://arxiv.org/pdf/cs/0507025) · [Dinkel et al.](https://arxiv.org/html/2607.16933v1#S2) · [Zhu & Hainy](https://arxiv.org/html/2603.16756v1#S5)
