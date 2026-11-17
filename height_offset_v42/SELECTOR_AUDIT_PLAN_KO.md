# v42 세 보정 선택의 독립 진단

동일한 128개 deployment 물체에 대해 보정 후보들의 Brier 차이를 원 확률과 label로 다시 계산한다. Hidden offset이 단지 실제 사건을 바꾸므로 한 환경·noise realization에 대한 운영 선택은 9개 offset에서 같아야 한다. 주 operational 비교는 0을 제외한 여덟 signed offset을 균등하게 가중한다. 아래는 보조 진단이며 12개 독립 panel 단위의 개별 양측 95% t interval을 사용한다.

1. 0%·fixed·optimizer 선택 빈도, 실제 selector의 zero 대비 Brier 차이, 같은 사용 빈도로 관측과 무관하게 무작위 선택했을 때의 차이를 비교한다. Blind mixing의 빈도는 panel과 24개 cell 각각에서 보존한다.
2. 128개의 정답을 사후에 모두 본 세 가지 진단을 비교한다. 해당 batch에서 (a) 실제 offset마다 최선 action을 다시 고르거나 (b) 숨긴 offset 여덟 개의 평균 risk가 최소인 action 하나를 모두에 쓰거나 (c) offset 0에서만 최선인 action을 골라 옮긴다. (a)와 (b)의 empirical risk 차이는 동일 observations에서 실제 offset이 추가로 알려졌을 때의 같은 표본 oracle 개선량이다. Finite-sample optimistic 값이며 deployment에서 실행할 수 있는 이론적 Bayes bound와 구분한다.
3. 같은 환경에서 얻은 다른 물체의 정답으로도 이 효과가 재현되는지 본다. 중복 prototype ID를 같은 half에 두고, 4개의 사전 지정된 난수 분할에서 한 half의 정답으로 (a), (b)를 고른 뒤 반대편 half의 대상에서 채점하고 두 역할을 맞바꾼다. 이 방식은 실제 event별 label을 사용한 환경 식별 진단이며 운영 selector 입력에는 사용하지 않는다. 분할 수를 독립 표본 수에 더하지 않는다.
