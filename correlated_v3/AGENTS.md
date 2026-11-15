# Correlated posterior extension

- 사용자 요청에 따라 actual experiments는 remote allocation에서 실행한다. CUDA numerical inference를 허용한다.
- 이전 continuous_v2 source와 결과를 변경하지 않는다. 이 실험의 모든 새 결과는 correlated_v3 아래에 저장한다.
- Known shape, frictionless rigid contact, calibrated scalar sensors의 controlled diagnostic이다. RGB perception 또는 real robot 결과로 표현하지 않는다.
- Hidden truth는 data generation과 evaluation에만 사용한다. 모든 policy는 factual measurement와 선택된 observation만 받는다.
- Factual noise sweep와 sign ambiguity를 사전에 명시하고, 실패를 보이지 않는 baseline도 그대로 보고한다.
- Conditional quadrature와 conditional sampling은 model structure를 이용하는 strong baseline이며 새로운 general-purpose method로 주장하지 않는다.
