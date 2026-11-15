## 동일한 네 번째 물체의 slip 전환별 분해

전체 target outcome을 보기 전에 정한 secondary 분석이다. 명령 속도가 +0.50에서 +0.75 또는 +1.10으로 바뀔 때, 같은 환경·surface regime에서 동일한 네 번째 calibration 물체가 stick을 유지한 경우, stick에서 slip으로 전환한 경우, 두 조건 모두 slip인 경우, slip에서 stick으로 전환한 경우를 구분한다. Physics gate가 볼 수 없는 latent simulator state로 정한 분석용 집단이다.

Primary local 강·약·0을 균등 가중하고 weak/null 균등과 global-null을 별도로 표시한다. 각 분할의 표본 비율, gate가 on/off를 바꾼 비율, 고정 shrinkage 대비 gate 이득의 속도별 변화에 기여한 양을 계산한다. 각 속도의 실제 physics probability를 사용할 때와 +0.50에서 선택한 동일한 deployment action 및 원래 확률을 사용할 때를 모두 평가한다. 고정 deployment 조건의 기여는 분할 indicator × (빠른 속도 gate − 기준 속도 gate) × 기준의 (상시 보정 Brier − 무보정 Brier)이다. 네 집단 기여의 합은 전체 차이와 일치한다. Conditional slope와 friction에 따라 slip 전환 가능성이 달라지는 paired diagnostic이며, slip 자체를 독립 조작한 causal mediation estimate로 해석하지 않는다. CI는 기존의 geometry × 시작 부호 층화와 같은 block bootstrap이며 각 secondary contrast에 pointwise 95% interval을 적용한다.
