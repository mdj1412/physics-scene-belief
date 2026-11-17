# v54 실험 계획

두 정책의 차이가 기존 보정의 재사용 방식에서 생기지 않도록, 네 번째 측정을 산 뒤에는 모든 정책에 같은 v53 continuous joint를 쓴다. Acquisition은 교체 후 첫 세 weak/reference 쌍만 받는다. 교체 전 네 쌍, 네 번째 weak, family, catalog 번호, surface, e, stick/slip outcome을 acquisition 입력에서 제외한다. 평가자는 세 정책을 같은 prefix, 같은 두 잠재적 네 번째 측정, 같은 128개 deployment 물체에 짝짓는다. 네 번째 측정 두 개를 모두 구매한 정책은 허용하지 않는다.

Primary는 model-based expected value selector와 fixed 4R, fixed 3R+I의 2×2 endpoint 비교다. 평균 Brier와 session loss p90 모두의 강한 개선은 네 차이의 simultaneous interval이 모두 0보다 낮을 때 인정한다. 같거나 작은 차이도 그대로 기록한다. 기존 12 held-out catalog를 재사용한다. Solver resolution, deployment noise, impact readout 및 prior 민감도는 주 비교와 구분한다. GPU p90은 고정된 512개 물체 단위로 비교한다.

별도 학습형 진단은 첫 세 쌍 중 readable pair 수 n과 부호 일치·불일치 중 적은 수 m으로 6개 group을 고정한다. Catalog 0–5에서 group별 평균 Brier 차이를 추정해 6–11에서 평가하고 반대 방향도 별도로 보고한다. 0을 threshold로 하고 tie는 reference, 빈 group은 training 쪽 전체 평균으로 결정한다. 이 진단만 다른 catalog의 simulated deployment outcome에 접근한다. Sensor calibration 세 쌍을 본 뒤 true e를 맞히는 모델로 해석하지 않는다. 같은 관측·decoder라도 fixed sensor 대비 개선이 표본, prior, readout에 따라 바뀔 수 있다는 것이 기존 실험에서 나온 예측이다.
