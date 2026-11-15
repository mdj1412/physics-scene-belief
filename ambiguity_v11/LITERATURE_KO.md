## 선행연구와 논문 기여의 경계

2026-11-12 기준으로 아래 primary sources의 objective와 관련 설명을 확인했다. 이번 실험에서 이 논문들의 전체 system을 재구현하거나 성능을 직접 비교한 것은 아니다.

| 선행연구 | 이미 다룬 부분 | v11에서 주장하면 안 되는 novelty |
| --- | --- | --- |
| [BayesSim, RSS2019](https://www.roboticsproceedings.org/rss15/p29.html) | Simulator parameters의 posterior inference와 Sim2Real domain randomization | Physics parameters를 point estimate 대신 distribution으로 유지한다는 원리 자체 |
| [Amortized Bayesian Experimental Design for Decision-Making, NeurIPS2024](https://proceedings.nips.cc/paper_files/paper/2024/hash/c59f05d7ab3638b138cc61f32e1a7cd1-Abstract-Conference.html) | Downstream decision utility를 고려한 experimental design과 amortized design/prediction | Prediction에 중요한 observation을 고른다는 일반적 formulation |
| [Action-BED, 2026 arXiv preprint](https://arxiv.org/html/2606.23662v1) | Expected future loss를 design policy와 downstream action policy의 joint optimization으로 표현; explicit posterior 없이도 학습 가능한 formulation | Expected decision loss acquisition이나 explicit posterior가 모든 좋은 decision에 필수라는 주장 |
| [Active Learning of Model Discrepancy with Bayesian Experimental Design, 2025 arXiv](https://arxiv.org/html/2502.05372v2) | Model discrepancy가 BED를 해칠 수 있는 문제와 sequential discrepancy correction | Model mismatch를 고려해야 한다는 문제 제기 자체 |

Action-BED와 TNDP를 보면 **최종 prediction에 필요한 정보와 full parameter posterior는 같은 요구사항이 아니다.** 이번 연구에서도 “full posterior를 유지하니까 더 원리적이다”는 설명만으로는 contribution이 성립하지 않는다. 동일한 decision quality를 더 작은 mixture나 amortized predictor로 얻을 수 있다면 그것이 중요한 비교 기준이다. 이는 해당 연구들의 objective를 현재 실험에 적용해 해석한 판단이다. [TNDP 원문](https://arxiv.org/html/2411.02064v1), [Action-BED 원문](https://arxiv.org/html/2606.23662v1).

Model discrepancy 연구는 candidate model set에 true model이 들어 있어야 한다는 가정의 한계를 직접 설명한다. v11의 primary는 바로 이 closed-world 가정을 사용한다. 별도 stress test를 통과하더라도 임의의 unknown physics를 다뤘다는 의미가 아니며, 실패한다면 단순히 particles나 quadrature points를 늘려 해결할 문제인지부터 다시 나눠야 한다. [Model Discrepancy 원문](https://arxiv.org/html/2502.05372v2).

### 이 실험이 기여로 발전하려면 필요한 증거

현재 가능한 contribution 후보는 새로운 Bayes objective보다 **geometry와 contact explanations가 서로 보상하는 상황에서 어떤 belief approximation이 decision에 충분한지 측정하는 diagnostic**이다. 이를 논문으로 키우려면 다음 증거가 필요하다.

1. **Generator와 inference의 분리:** 같은 analytic family에서만 truth와 posterior를 만드는 조건을 넘어, 독립 simulator와 다른 contact laws에서도 관찰된 효과가 남아야 한다.
2. **Observation 비용과 실행 가능성:** Camera movement, calibration error, probe/reset 비용을 포함한 cost-normalized decision loss로 비교해야 한다. Query contact point를 직접 지정할 수 있다는 현재 가정도 검토해야 한다.
3. **강한 compact baseline:** Mixture plug-in, LMMSE, 더 많은 training worlds로 학습한 amortized rule을 비교하고, selector와 decoder를 교차 평가해야 한다. 작은 tree를 이겼다는 것만으로 복잡한 inference가 필수라고 결론낼 수 없다.
4. **Contact에 중요한 geometry 정보의 실제 측정:** Natural images나 실제 boundary measurements에서 slope/curvature의 uncertainty를 calibration하고, geometry error가 intervention prediction으로 전파되는 과정을 검증해야 한다.
5. **Robustness의 원인 분리:** Unseen boundary family와 unseen contact law를 한 번에 바꾸지 않고 각각 바꾸어, 실패가 geometry mismatch인지 contact mismatch인지 규명해야 한다.

이 항목은 이번 turn에서 모두 실행한 결과가 아니라, 현재 diagnostic을 넘어 논문 기여를 판단하기 위한 다음 기준이다. 본 보고서의 numerical 결과만으로 real-world performance나 주요 학회 채택 가능성을 확정하지 않는다.
