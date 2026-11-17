## 문헌과 논문 기여 심사

### 1. 논문별 질문·방법·표본·근거·한계

| 연구 | 질문과 방법 | 표본·단위 | 결과와 결론 | v57에 적용할 때의 한계 |
|---|---|---|---|---|
| Vandermeulen & Scott, *On the Identifiability of Mixture Models from Grouped Samples* | 같은 latent component에서 나온 여러 관측을 묶으면 mixture가 식별되는가? Grouped distribution을 연산자로 표현해 충분조건과 반례를 증명했다. | m개 probability measure에 대한 수학적 정리. 물리 피험 표본은 없다. | 일반적인 mixture에는 group당 2m−1개의 관측으로 식별이 가능하고 더 적을 때 구별할 수 없는 사례가 있다. | v57은 두 개의 주어진 후보를 비교하고 e의 support도 안다. 일반 정리는 두 번이면 모든 contact mixture를 학습한다는 근거가 아니다. [원문](https://arxiv.org/pdf/1502.06644) |
| Stoye, *Minimax Regret Treatment Choice with Finite Samples* (2009) | 유한 표본으로 두 행동을 고를 때 어떤 규칙이 최대 regret을 줄이는가? Small-sample decision risk를 직접 계산했다. | 두 treatment와 여러 표본 배정에 대한 이론적 실험. 실물 cohort는 없다. | 일부 소표본 설계에서 empirical success와 minimax regret 규칙의 행동·regret이 달라진다. | v57의 source-risk 규칙은 두 환경과 prior를 고정한 expected Brier 기준이다. 이 논문은 그 규칙의 minimax 최적성이나 실제 slip 비용을 보장하지 않는다. [원문](https://cemmap.ac.uk/wp-content/legacy/resources/files/manski09_7.pdf) |
| Luo et al., *PhysiCalWorld* (2026, preprint) | Test scene 안에서 exploratory interaction clip을 얻어 새로운 물리에 맞게 예측을 조절할 수 있는가? 영상에서 physics descriptor를 추출해 action-conditioned rollout에 재사용한다. | 네 환경의 rollout prediction과 visual planning. 각 환경에서 변화시키는 물리 속성이 다르다. | 탐색 영상을 이용하면 test scene의 dynamics와 미래 예측을 더 잘 맞춘 사례를 보고했다. | Clip의 frame은 하나의 session에 속할 수 있다. 그것이 곧 독립 물체 두 개의 mismatch 표본에 해당하지 않는다. 해당 연구에는 v57의 동등 8회 예산·Brier 대조가 없다. [원문](https://arxiv.org/html/2610.30125v1) |
| Acosta, Yang & Posa, *Validating Robotics Simulators on Real-World Impacts* (2022) | Parameter를 맞춘 simulator가 실제 충돌을 얼마나 재현하는가? 실측 trajectory를 simulator 간 대조했다. | 550개 cube toss와 Cassie의 22개 jump. Cube는 3개 engine, Cassie는 2개. | 비탄성 충돌과 전체 운동은 재현되기도 하지만 elasticity 및 착지 dynamics에서 차이가 남았다. | 이 자료에는 같은 표면의 짝지어진 2.0 N reference–impact 방향과 이어지는 독립 deployment outcome이 없다. v57 정책의 hardware 검증을 대체하지 못한다. [원문](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) |

### 2–3. 주제별 종합과 knowledge map

이 문제는 **같은 상태를 공유하는 물체의 범위 → 그 범위 안에서 수집한 관측의 분포 → 상태 또는 환경에 대한 evidence → 비용을 고려한 행동 → 실제 impact outcome**으로 연결된다. Grouped-sample theory는 앞의 두 단계를, finite-sample decision theory는 posterior evidence를 행동으로 바꿀 때 생기는 손실을 연구한다. Exploratory interaction 연구는 물리적 observation을 실제 prediction에 넣는 방법을 제공하고, simulator validation은 마지막 비교에 외부 측정이 필요한 이유를 보여준다. [Vandermeulen & Scott](https://arxiv.org/pdf/1502.06644) · [Stoye](https://cemmap.ac.uk/wp-content/legacy/resources/files/manski09_7.pdf) · [Luo et al.](https://arxiv.org/html/2610.30125v1) · [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf)

### 4–7. 연구 공백, 비교되는 가정과 근거 강도

- **식별 가능성과 유용성.** 동일 component에서 독립적으로 얻은 관측 여러 개가 mixture를 구분하는 정보를 갖는다는 점은 이론에 근거한다. 한 표면당 8회에서 얻을 수 있는 accuracy나 Brier 개선 크기는 별도로 계산해야 한다. 같은 상태를 더 잘 맞혀도 특정 오판의 비용이 크면 downstream decision은 악화될 수 있다. [Vandermeulen & Scott](https://arxiv.org/pdf/1502.06644) · [Stoye](https://cemmap.ac.uk/wp-content/legacy/resources/files/manski09_7.pdf)
- **방법에 따른 결과 차이.** Latent rollout을 학습하는 방법과 부호의 joint likelihood를 쓰는 방법은 정보량도 endpoint도 다르다. 후자는 어떤 두 물체가 같은 session이었는지 교란할 수 있어 pairing 효과를 분리한다. 전자는 속도·변형 크기 등 binary sign 외의 단서를 활용할 수 있다. 두 논문 사이에 동일 실험 조건에서 관찰한 직접적인 모순은 없다. [Luo et al.](https://arxiv.org/html/2610.30125v1) · [Vandermeulen & Scott](https://arxiv.org/pdf/1502.06644)
- **미해결 부분.** 같은 session에서 두 물체가 같은 e를 공유하는지, sensor drift·wear·첫 impact가 다음 물체의 판독을 바꾸는지, 그 의존성이 deployment로 전이되는지는 이 네 연구로 결정할 수 없다. 실물 반복에서 센서와 실제 dynamics를 별도로 읽어야 한다. [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) · [Vandermeulen & Scott](https://arxiv.org/pdf/1502.06644)
- **근거의 범위.** 일반 이론, 네 환경의 rollout 평가, 실제 cube와 로봇의 trajectory 검증에는 서로 다른 독립 단위와 가정이 있다. 이 연구들의 sample size 또는 성공률을 합치거나 한 방법의 성능을 다른 방법의 Brier 효과로 환산할 근거가 없다. [Stoye](https://cemmap.ac.uk/wp-content/legacy/resources/files/manski09_7.pdf) · [Luo et al.](https://arxiv.org/html/2610.30125v1) · [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf)

### 8. 검증 가치가 있는 후속 질문 열 가지

| 질문 | 다루는 공백과 의미 | 변수·연구 방법 |
|---|---|---|
| 1. 두 물체가 공유하는 상태는 어느 시간 간격까지 유지되는가? | Latent component 공유가 grouping의 전제다. [Vandermeulen & Scott](https://arxiv.org/pdf/1502.06644) | 간격과 물체 순서를 무작위화하고 조건부 joint mismatch를 측정. |
| 2. 두 부호 판독의 상관이 센서에서 생기는가? | Shared sensor noise가 physical persistence와 비슷한 통계를 줄 수 있다. [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) | 독립 camera와 gauge를 교차하고 동시 오판 확률을 추정. |
| 3. 두 환경의 정의를 모를 때도 grouping이 도움이 되는가? | 알려진 두 서명 중 고르는 문제보다 unknown mixture 학습이 어렵다. [Vandermeulen & Scott](https://arxiv.org/pdf/1502.06644) | Source에서 mixture support를 추정해 고정하고 새 material에서 검증. |
| 4. 8회 중 어느 grouping이 실제 decision loss를 최소화하는가? | 명목상 식별과 작은 표본의 regret이 다르다. [Stoye](https://cemmap.ac.uk/wp-content/legacy/resources/files/manski09_7.pdf) | 8×1·4×2·2×4 설계의 held-out loss를 같은 시간·횟수에서 비교. |
| 5. Active 판독 가능성이 e에 따라 달라지는가? | 비무작위 누락은 관측 likelihood를 바꾼다. [Vandermeulen & Scott](https://arxiv.org/pdf/1502.06644) | 독립 reference instrument로 모든 시도에 audit label을 붙여 수율을 stratify. |
| 6. 오판 비용을 source에서 배운 threshold는 다른 surface에도 맞는가? | 판별 신호와 cost transfer를 분리한다. [Stoye](https://cemmap.ac.uk/wp-content/legacy/resources/files/manski09_7.pdf) | 동일 pilot에 frozen source cost와 known-target cost를 교차. |
| 7. 같은 횟수의 영상에서 부호 이상의 단서를 쓸 수 있는가? | Sign-only pilot이 버리는 정보의 가치를 잰다. [Luo et al.](https://arxiv.org/html/2610.30125v1) | 동일한 여덟 contact를 영상·부호로 표현하고 공통 terminal에서 Brier 비교. |
| 8. 여러 deployment session이 하나의 pilot을 공유하면 tail이 어떻게 변하는가? | Margin에서 평균한 session p90과 fleet 전체 위험은 다르다. [Stoye](https://cemmap.ac.uk/wp-content/legacy/resources/files/manski09_7.pdf) | Surface당 배포 수를 바꾸고 worst-session 및 surface 평균을 따로 평가. |
| 9. Session setup 시간이 grouping의 경제성을 바꾸는가? | 같은 contact 횟수에서도 작업자는 표면을 세팅하는 시간을 지불한다. [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) | Setup·reset·readout을 독립 계측하고 deployment 횟수별 실제 시간을 보고. |
| 10. 다른 contact solver 및 실물에서도 같은 action이 이기는가? | Readout이 같아도 target outcome의 정의가 달라질 수 있다. [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) | 동일 물체·surface의 두 센서와 미래 stick을 독립 측정해 simulator와 비교. |

### 9. 연구 심사

Grouping으로 mixture를 식별하는 원리 자체와 실험으로 test-scene physics를 보정하는 아이디어는 이미 연구됐다. v57의 상대적인 기여는 **같은 수의 물리 관측으로 얻은 두 번째 moment가 환경 판별과 실제 sensor 선택에 각각 얼마의 가치를 갖는지, 두 값이 언제 다른 방향으로 움직이는지**를 한 과제에서 검증하는 데 있다. Grouping을 모르는 permutation control, 같은 비율로 전환하는 random control, 두 기본 정책, source cost를 쓰는 selector를 유지해야 그 주장을 평가할 수 있다. [Vandermeulen & Scott](https://arxiv.org/pdf/1502.06644) · [Stoye](https://cemmap.ac.uk/wp-content/legacy/resources/files/manski09_7.pdf) · [Luo et al.](https://arxiv.org/html/2610.30125v1)

실물에서 동일 session의 latent state 공유, 센서의 조건부 독립, setup·reset의 재현성 중 어느 것이 성립하는지 아직 검증해야 한다. 주장을 현재의 binary pilot, 두 알려진 후보 환경과 합성 contact task로 한정한다. 이 문헌 검토는 네 원문에 대한 목적별 비교이며 분야 전체의 최초성을 판정하지 않는다. [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) · [Vandermeulen & Scott](https://arxiv.org/pdf/1502.06644)
