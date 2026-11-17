## 문헌 검토와 심사

### 1. 논문별 질문, 방법, 표본 및 결론

| 연구 | 질문·방법 | 표본과 평가 단위 | 결과와 결론 | v56에 적용할 때의 한계 |
|---|---|---|---|---|
| Vandermeulen & Scott, *On the Identifiability of Mixture Models from Grouped Samples* | 같은 latent component에서 나온 관측들을 묶으면 mixture가 유일하게 정해지는가? 분포를 표현하는 연산자로 충분조건과 반례를 증명한다. | 임의의 m개 probability measure. 수집한 로봇 trial은 없다. | Group당 2m−1개 관측이면 일반적인 식별이 가능하며 더 작은 group에 대한 반례가 존재한다. | Known e 값을 세 개로 제한한 v56의 최소 필요 횟수를 이 일반 정리에서 그대로 읽을 수 없다. 여기서는 하나의 Bernoulli mismatch가 전달하는 moment를 직접 계산한다. [원문](https://arxiv.org/pdf/1502.06644) |
| Stoye, *Minimax Regret Treatment Choice with Finite Samples* (2009) | 유한한 표본을 보고 두 행동 중 고를 때 worst-case regret이 가장 작은 규칙은 무엇인가? 작은 표본의 risk를 직접 계산한다. | 두 treatment와 다양한 표본 배정으로 정의한 이론적 실험. 실제 치료 환자 cohort를 평가한 연구는 아니다. | Empirical success rule이 자주 근접하지만 작은 표본과 일부 설계에서는 minimax 규칙과 regret이 달라진다. | 이 논문은 v56의 Brier나 sensor noise를 연구하지 않았다. 여기에서 frozen threshold가 minimax라는 결론은 나오지 않는다. [원문](https://cemmap.ac.uk/wp-content/legacy/resources/files/manski09_7.pdf) |
| Heath, Manolopoulou & Baio, *Estimating the Expected Value of Sample Information across Different Sample Sizes* | 표본 크기에 따른 EVSI를 효율적으로 추정할 수 있는가? posterior variance와 nonlinear regression을 결합해 moment matching을 확장했다. | Chronic pain intervention의 cost-effectiveness model에서 복수 trial design을 비교했다. 실제 환자 치료 효과를 새로 측정하지 않았다. | 예제에서 여러 표본 크기를 효율적으로 비교했다. Simulation 자체가 비싼 model에서는 추가 계산 부담이 있다. | Brier 1단위와 장비 시간 1초의 교환 가치가 주어지지 않으면 여기서 최적 K를 경제적으로 결정할 수 없다. [원문](https://doi.org/10.1177/0272989X19837983) |
| Acosta, Yang & Posa, *Validating Robotics Simulators on Real-World Impacts* (2022) | Contact parameter를 맞춘 simulator가 실제 impact를 얼마나 재현하는가? 동일 관측과 Drake·MuJoCo·Bullet의 궤적을 대조했다. | Cube toss 550개 trajectory, Cassie jump 22회. 로봇에는 Drake와 MuJoCo를 비교했다. | 비탄성 충돌과 전체 움직임을 잘 재현한 사례가 있지만 elasticity 및 세부 dynamics에서 오차가 남았다. | 동일 surface에서 reference–impact sign을 쌍으로 읽고 sensor 구매 정책을 비교하는 데이터가 아니다. [원문](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) |

### 2–3. Literature review와 knowledge map

**Surface의 latent law → 같은 session에서 얻는 관측 수 → 식별할 수 있는 mixture 정보 → action별 expected loss → 측정 비용** 순서로 연구를 연결할 수 있다. 앞의 두 단계는 grouped sampling에 관한 이론, 가운데 두 단계는 finite-sample decision theory, 마지막 단계는 EVSI와 trial design의 문제다. 이 연결을 따르면 e를 더 정확히 추정한 정책이라도 두 행동의 Brier 차이가 작으면 지불한 impact의 가치가 작을 수 있다. 이는 v56에 대한 해석이다. [Vandermeulen & Scott](https://arxiv.org/pdf/1502.06644) · [Stoye](https://cemmap.ac.uk/wp-content/legacy/resources/files/manski09_7.pdf) · [Heath et al.](https://doi.org/10.1177/0272989X19837983)

### 4–7. 공백, 연구 간 차이, 방법론과 근거 강도

- **확립된 부분.** Mixture의 component를 식별하려면 어떤 관측들이 같은 component에서 나왔는지에 대한 실험 설계가 중요하다. 통계적 추정 정확도와 행동을 잘못 골랐을 때 치르는 regret은 구별할 수 있다. [Vandermeulen & Scott](https://arxiv.org/pdf/1502.06644) · [Stoye](https://cemmap.ac.uk/wp-content/legacy/resources/files/manski09_7.pdf)
- **상충하는 듯한 결과.** 표본을 더 얻을 때 기대 정보 가치가 생긴다는 EVSI 결과와, 실제 threshold rule이 작은 K에서 한쪽 상수 정책보다 나쁠 수 있다는 결과는 양립한다. EVSI의 decision maker는 정보를 받은 뒤 기대 효용이 가장 큰 행동을 고를 수 있다. Frozen rule에는 그 최적성이 주어지지 않는다. 문헌에 동일한 contact task에서 상반된 결과를 낸 직접 비교는 없다. [Heath et al.](https://doi.org/10.1177/0272989X19837983) · [Stoye](https://cemmap.ac.uk/wp-content/legacy/resources/files/manski09_7.pdf)
- **측정의 공백.** 물리 simulator와 실물 사이에 남는 오차는 중요한데, 기존 실물 impact trajectory에는 v56에 필요한 독립 pilot·향후 session·동일 예산의 두 action label이 갖춰져 있지 않다. 직접 포착되지 않은 impact sign을 solver state로 대신할 때 generalization의 대상이 좁아진다. [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf)
- **방법별 해석.** 이론적 식별 보장은 무한히 많은 group에서 가능한 것에 대한 진술이다. Exact small-sample risk는 지정한 observation law에서의 성능이며, 실물 trajectory 비교는 센서와 solver에 대한 현장 증거를 준다. 이 세 종류의 표본 수를 합산할 수 없다. 실제 surface의 mixture 빈도와 reset 비용에 대한 근거는 부족하다. [Vandermeulen & Scott](https://arxiv.org/pdf/1502.06644) · [Stoye](https://cemmap.ac.uk/wp-content/legacy/resources/files/manski09_7.pdf) · [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf)

### 8. 공백에서 도출한 후속 연구 질문 열 가지

| 질문 | 중요한 이유 | 변수와 검증 방법 |
|---|---|---|
| 1. 독립 session당 한 번 읽으면 행동에 필요한 mixture 정보가 충분한가? | 조성이 다른 population이 같은 관측을 줄 수 있다. [Vandermeulen & Scott](https://arxiv.org/pdf/1502.06644) | 평균 mismatch를 고정한 두 mixture의 최적 행동을 fresh Brier로 비교. v56에서 진단. |
| 2. 같은 session에서 두 물체를 읽는 편이 나은가? | Mismatch의 첫 두 moment를 분리할 수 있다. [Vandermeulen & Scott](https://arxiv.org/pdf/1502.06644) | 총 impact 수 고정, 한 번씩 넓게 읽기와 두 번씩 묶어 읽기 비교. |
| 3. 같은 두 건의 mismatch도 어느 쪽 행동의 손실이 더 큰가에 따라 결정이 달라지는가? | 틀린 선택의 비용이 비대칭일 수 있다. [Stoye](https://cemmap.ac.uk/wp-content/legacy/resources/files/manski09_7.pdf) | Source의 conditional cost로 고정한 rule을 새 surface에서 검증. |
| 4. 몇 번의 impact를 사야 측정 시간이 정당화되는가? | 정확도 개선을 장비 비용에 연결한다. [Heath et al.](https://doi.org/10.1177/0272989X19837983) | K별 loss와 장비 시간을 동시 기록해 cost frontier 추정. |
| 5. 한 표면에서 뒤이어 운용할 session 수가 적어도 pilot가 유리한가? | Setup 비용이 여러 배포에 걸쳐 분담된다. [Heath et al.](https://doi.org/10.1177/0272989X19837983) | 같은 surface 재사용 횟수별 no-pilot 정책과 net benefit 비교. |
| 6. 판독 실패가 mismatch와 연관되는가? | 쉽게 읽히는 impact만 남기면 빈도가 편향될 수 있다. [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) | Independent high-speed readout과 대조해 censoring을 e·속도·표면별 추정. |
| 7. 동일 표면의 sign이 시간에 따라 변하는가? | 고정된 구성비를 공유한다는 가정을 시험한다. [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) | Pilot과 deployment 시간 간격을 무작위화하고 연속 날짜별 외부 검증. |
| 8. Near-crossing이 아닌 surface만 골라 더 빨리 멈출 수 있는가? | 의사결정 가치가 작은 구간에 불필요한 probing을 줄인다. [Heath et al.](https://doi.org/10.1177/0272989X19837983) | Dynamic stopping을 source에서 고정하고 equal total contact budget에서 비교. |
| 9. Tail loss의 최적 switch point가 mean과 다른가? | 평균에 맞춘 threshold가 드문 큰 손실을 줄이지 못할 수 있다. [Stoye](https://cemmap.ac.uk/wp-content/legacy/resources/files/manski09_7.pdf) | Source에서 quantile criterion을 고정하고 독립 session의 p90과 mean을 함께 평가. |
| 10. 다른 contact solver·실물에서 최적 행동의 부호가 유지되는가? | Sensor 신호가 충분해도 outcome model이 달라지면 정책이 바뀐다. [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) | 동일 fixture의 paired impact·trajectory, 서로 다른 solver, binary Brier를 대조. |

### 9. 논문 심사 관점

단순히 mismatch 빈도를 세거나 threshold 정책을 제안하는 것의 독창성은 낮다. 이 작업이 추가하는 증거는 독립적인 물리 outcome에서 **통계적으로 식별 가능한 정보와 정책 전환으로 회수할 수 있는 Brier gain을 같은 sensing budget에서 함께 정량화**하는 데 있다. v55에서 정한 교차점이 개별 surface에 정확히 맞는지, 물체의 판독 가능성이 mismatch와 독립인지, 몇 개의 실제 impact로 값을 측정할 수 있는지가 남는다. 같은 session에서 여러 독립 물체를 probing하고 실측 sensor timing을 넣는 비교가 이 한계를 직접 시험한다. 이 네 편은 연구 방향을 판단하기 위한 선택적 검토이며 전체 분야의 최초성을 확증하지 않는다. [Vandermeulen & Scott](https://arxiv.org/pdf/1502.06644) · [Stoye](https://cemmap.ac.uk/wp-content/legacy/resources/files/manski09_7.pdf) · [Heath et al.](https://doi.org/10.1177/0272989X19837983) · [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf)
