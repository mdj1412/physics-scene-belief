## 문헌과 연구 심사

### 1. 논문별 질문·방법·표본·결론

| 논문 | 연구 질문과 방법 | 표본과 측정 단위 | 결과와 결론 | v55 적용의 한계 |
|---|---|---|---|---|
| Fishel & Loeb (2012) | BioTac이 얻은 tactile evidence를 바탕으로 다음 stroking movement를 고르면 texture를 더 잘 식별하는가? Bayesian exploration을 cycling·random과 비교했다. | 후보 117개 중 16개 texture의 별도 validation 측정으로 8,000회 virtual exploration을 수행했다. | Median 5회 동작으로 정분류율 95.4%를 얻었다. 무엇을 다음에 측정하는지가 texture identification에 영향을 줬다. | 반복 탐색은 독립 재질 8,000개에 해당하지 않는다. 목표가 texture ID여서 downstream slip probability의 개선 크기를 가져올 수 없다. [원문](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00004/full) |
| Luo et al., *PhysiCalWorld* (2026, preprint) | Test scene에서 얻은 interaction clip을 physics descriptor token으로 압축해 model weight를 고정한 채 rollout을 개선할 수 있는가? | 서로 다른 dynamics variation을 가진 네 환경에서 rollout prediction과 visual planning을 평가했다. | Scene별 탐색 정보를 다시 사용해 rollout·planning 성능을 개선했다. 같은 물체의 추가 상호작용으로 물리를 추정한다는 발상은 이미 선행연구에 있다. | 평가 단위는 주로 trajectory와 planning task다. 네 번째 센서의 구매 횟수를 맞춰 slip의 conditional Brier gain을 분리한 실험과 직접 환산하기 어렵다. [원문](https://arxiv.org/html/2610.30125v1) |
| Koh et al., *WILDS* (ICML 2021) | 실제 발생하는 distribution shift에서 기존 방법이 얼마나 전이되는가? 현장 환경별 split을 표준화했다. | Hospital, camera trap, 촬영 장소와 시점 등 domain이 구별되는 10개 dataset. | 여러 dataset에서 OOD 성능이 떨어졌고 기존 robust training도 격차를 남겼다. 어떤 단위를 분리했는지가 평가 해석에 중요하다. | Contact policy나 Brier gain을 실험하지 않았다. 합성 surface의 family split에 외부 타당성을 자동으로 부여할 수 없다. [원문](https://proceedings.mlr.press/v139/koh21a.html) |
| Acosta, Yang & Posa (RA-L 2022) | 추정한 contact parameter로 실제 impact를 얼마나 재현하는가? 동일한 실물 trajectory와 simulator를 대조했다. | Acrylic cube의 550개 toss trajectory와 Cassie의 22개 jump. Cube는 Drake·MuJoCo·Bullet, Cassie는 두 solver. | 비탄성 충돌과 전체 움직임을 잘 재현하는 사례가 있었지만 elasticity 및 기체 모델 차이에서 오차가 남았다. | 실물 trajectory에는 동일 prefix에서 reference 또는 impact sensing을 고른 뒤의 paired forecast가 없다. v55의 acquisition 정책 자체를 이 데이터로 검증할 수 없다. [원문](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) |

### 2–3. 문헌 종합과 지식 지도

연구 질문을 따라가면 **prefix 관측 → 다음 측정 선택 → 미래 운동 예측 → 평가 환경의 변화**로 연결된다. Fishel & Loeb은 앞의 두 단계에서 탐색을 바꿨다. Luo 등은 이미 얻은 scene-specific interaction이 세 번째 단계의 정보를 제공함을 보였다. 두 설계는 상보적이다. 달라진 측정으로 학습한 decoder만 비교하면, 어떤 sensor를 샀기 때문에 개선되었는지 식별하기 어렵다. [Fishel & Loeb](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00004/full) · [Luo et al.](https://arxiv.org/html/2610.30125v1)

검증의 마지막 연결은 **training에서 분리한 단위 → 새 domain의 error → 물리 측정과의 일치**다. WILDS의 domain split은 같은 환경 속 trial만 바꾸는 방법의 한계를 보여준다. Acosta 등의 결과는 물체와 trial을 새로 생성해도 공통 solver가 실물과 다를 수 있음을 보여준다. v55는 새 synthetic surface에서 gate를 고정하는 검증을 담당한다. [Koh et al.](https://proceedings.mlr.press/v139/koh21a.html) · [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf)

### 4–7. 연구 공백·상충하는 결과·방법론·근거의 강도

- **어떤 입력이 필요한가.** 원 논문들에서 물리 탐색이 분류나 rollout을 도운 사례는 일관된다. 그러나 texture class·latent trajectory가 목표일 때와 calibrated slip probability가 목표일 때 유용한 센서가 같다는 근거는 없다. 미래 forecast를 고정하고 action을 바꾸는 것이 v55의 직접 비교다. [Fishel & Loeb](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00004/full) · [Luo et al.](https://arxiv.org/html/2610.30125v1)
- **성공과 실패가 양립하는 이유.** 개별 scene에서 contextual calibration이 성공할 수 있는 것과 새 domain에서 전체 성능이 떨어질 수 있는 것은 서로 다른 모집단과 endpoint에 대한 결과다. 이 논문들 사이에서 동일한 task·sample에 대한 직접적인 상반 증거는 확인하지 못했다. v55는 aggregate mean, upper-tail quantile과 환경 조성별 mean을 구분해 보고한다. [Luo et al.](https://arxiv.org/html/2610.30125v1) · [Koh et al.](https://proceedings.mlr.press/v139/koh21a.html)
- **표본과 한계.** Virtual 재탐색, simulated trajectory, real-world dataset, 실물 impact는 각기 다른 표본 단위다. Sensor 선택의 효과를 실물에서 주장하려면 같은 물체·표면 session 안에서 두 행동을 비교하고 label과 시간도 별도로 측정해야 한다. 네 원문은 그런 조건에서의 tail gain, sensor 복구 비용, reference–impact sign mismatch의 현장 빈도를 결정하지 못한다. [Fishel & Loeb](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00004/full) · [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf)

### 8. 후속 질문 열 가지

| 질문 | 메우는 공백과 중요성 | 변수와 검증 방법 |
|---|---|---|
| 1. 구매율이 같아도 prefix의 가치는 새로운 surface에서 유지되는가? | 탐색 자체와 구매 총량을 분리한다. [Fishel & Loeb](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00004/full) | 구매율 고정, paired mean Brier 비교. v55에서 수행. |
| 2. Reference–impact mismatch의 현장 빈도는 얼마인가? | 합성 평가의 regime 가중치에 실증 근거를 준다. [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) | 표면별 paired reference·impact readout을 별도 표본에서 수집하고 domain별 빈도를 추정. |
| 3. 같은 접촉 예산에서 learned descriptor가 analytical gate보다 유리한가? | 표현을 학습할 때의 추가 가치를 측정한다. [Luo et al.](https://arxiv.org/html/2610.30125v1) | Source-only encoder, 동일한 세 쌍·공통 decoder·동일한 평가 session. |
| 4. Readout noise가 집단별 이득의 부호를 바꾸는가? | 탐색 신호의 판독 오차가 어느 정도까지 허용되는지 알 수 있다. [Fishel & Loeb](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00004/full) | 동일 물리 trial의 복수 센서 판독, jitter 수준별 risk. |
| 5. 다른 solver에서도 같은 prefix가 같은 행동을 정당화하는가? | Solver error가 큰 regime의 순위 반전을 찾는다. [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) | 엔진과 restitution을 교차하고 같은 action의 realized gain을 대조. |
| 6. 교체 뒤 표면 마모가 gate를 무효화하는가? | 독립 session 효과와 시간 drift를 구분한다. [Koh et al.](https://proceedings.mlr.press/v139/koh21a.html) | 같은 표면을 날짜·마모 차수로 분리해 순차 holdout. |
| 7. Tail을 직접 목표로 한 acquisition은 평균을 얼마나 희생하는가? | Prediction 개선과 나쁜 session 보호의 차이를 측정한다. [Luo et al.](https://arxiv.org/html/2610.30125v1) | Source에서 quantile objective를 고정하고 새 domain의 mean·p90을 함께 비교. |
| 8. 첫 세 쌍의 위치 다양성이 한 쌍의 강도보다 중요한가? | 탐색 순서를 설계할 때 바꿔야 할 동작을 찾는다. [Fishel & Loeb](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00004/full) | Site spacing과 preload를 factorial design으로 교차하고 probing 횟수 고정. |
| 9. 실물 reset 시간까지 포함하면 어느 정책이 유리한가? | Sensor 가격 차이가 작은 Brier 이득의 적용 여부를 바꾼다. [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) | Readout과 reset을 별도 timestamp로 재고 time-matched risk frontier 산출. |
| 10. Marginal purchase rate를 새 fleet에서 알 수 없어도 비교가 유지되는가? | 평가용 대조와 현장 배포 기준을 연결한다. [Koh et al.](https://proceedings.mlr.press/v139/koh21a.html) | 구매율은 이전 fleet에서 추정해 고정한 뒤 시간순 새 fleet에서 calibration과 비교. |

### 9. 연구 심사

새로울 수 있는 부분은 세 번의 저렴한 관측이 **어느 session에 더 강한 관측을 배정할지** 알려주는 효과를, 같은 구매율·동일 decoder·분리된 surface라는 대조로 식별하는 것이다. Active tactile exploration이나 test-time physics calibration 자체는 이미 발표되었다. 현재 가장 가까운 learned 대안은 source interaction으로 학습해 test scene의 dynamics를 요약하는 descriptor다. 동일한 입력·label·terminal decoder에서 직접 비교하기 전에는 학습 방법보다 우수하다는 주장이 성립하지 않는다. [Fishel & Loeb](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00004/full) · [Luo et al.](https://arxiv.org/html/2610.30125v1)

독립 synthetic catalog에서 재현되는 효과와 다른 물리 엔진·실물에서 재현되는 효과를 구분해야 한다. 서로 다른 표면 재료와 설치 날짜에 대해 독립적인 holdout을 구성하고, 결과를 보지 않고 정한 실용적 loss margin과 실제 elapsed sensing time을 함께 보고하는 것이 다음 검증이다. 네 편에 근거한 관련 연구 검토이므로 최초성에 대한 포괄적인 판단은 유보한다. [Koh et al.](https://proceedings.mlr.press/v139/koh21a.html) · [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf)
