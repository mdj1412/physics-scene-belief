# 같은 환경의 모델 선택 이득은 다른 물체로 옮겨가는가

V41 주 접촉 실험 시작 뒤 target score를 읽기 전에 정한 CPU 보조 분석이다. 같은 환경에 128개 deployment 물체가 있다. 한쪽 물체의 label에서 optimizer와 fixed 중 우세한 방법을 고르고, 다른 쪽 물체에서 그 선택의 Brier를 평가하며 역할을 맞바꾼다. 세 decoder·두 방법·noise 반복마다 수행한다. 같은 prototype이 두 번 뽑힌 경우 같은 half에만 속하게 한다. 서로 다른 8개 deterministic random split 결과를 평균하며, 이 split 반복은 독립 표본 수에 넣지 않는다. 서로 다른 condition에는 동일 split을 적용한다.

평가 half의 크기에 따라 가중해 원래 128개 물체의 빈도를 보존한다. 같은 자료에서 true mean loss가 작은 방법을 즉석 선택한 값은 낙관적인 lower bound다. 반면 cross-fitted 결과는 유한한 환경 안에서 약 64개 물체의 실제 정답을 추가로 알 때 다른 물체에 전달될 수 있는 선택 이득을 추정한다. Label을 모르는 실제 정책의 상한도, 실물에서 지급 가능한 관측 비용도 아니다. 비교는 12개 independent panel에서 paired interval을 계산한다. 실사용 selector에는 해당 label을 전달하지 않는다.
