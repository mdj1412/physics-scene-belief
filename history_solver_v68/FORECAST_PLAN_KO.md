# v68 예측과 통계 계획

V66의 12개 catalog와 각 4 × 128개 deployment object, surface와 noise seed를 고정한다. V67의 두 contact law와 네 source axis·8개 head를 고정한다. 같은 형상에서 two independent solver implementations × two laws를 교차한다. 각 solver/law로 fixed 4 calibration 쌍의 factual contact와 paid probe, deployment factual contact, 구매한 probe, 마지막 impact를 생성한다. Calibration 3개는 높이 0의 방향 교대, 마지막은 23 mm +1.10 m/s이다. 측정의 접촉 radius는 0.25 mm이고 label에서는 v67과 같은 .25/2 mm×세 offset을 평균한다. 새로운 noise나 센서 budget은 추가하지 않는다.

Acquisition은 같은 13 probe/128의 frozen v43 규칙이고 height-specific camera route도 v66 그대로다. Terminal predictor에는 알려지지 않은 law, solver, shape, sign, regime, prototype, 미구매 센서를 전달하지 않는다. Primary는 v67 probability에 v38 source continuous optimizer가 관측한 네 calibration 쌍에서 추정한 같은 log-odds shift를 더한다. 네 번째 observation을 포함한 이 보정은 forecast 모두에 동일하며 신규 target 학습은 없다. 기존 calibration-ignoring predictor와 prefix-3 cue도 따로 저장한다. v67 raw와 이 predictor의 정의가 다르므로 동일 predictor의 ablation을 별도로 비교한다. 모든 신규 forecast를 봉인한 뒤 label을 계산한다.

독립 compliant solver의 각 새 법칙에서 Wendland soft075−full <0, Dispersive soft075−full >0, 두 형상 평균 restored−dropped <−.002를 사전 가설로 둔다. 12 independent catalog에서 4 axis를 먼저 평균하고, 2 law × 3 contrasts의 동시 95% Bonferroni paired t interval을 보고한다. 양쪽 형상의 부호가 각각 확인되어야 반전 재현이라 판단한다. Prescribed solver, raw predictor, 세 개의 probe만 넣는 보정은 secondary다. 4 axis는 별도 target sample로 세지 않는다.
