## 같은 환경에서 표면의 방향 특성만 바꾼 대조

네 번째 접촉의 높이는 23 mm로 두고 speed +0.50, +0.75, +1.10을 비교한다. 세 local regime에서 같은 물체·orientation·입력 및 noise를 대응시킨다. 한 속도에서 세 regime의 gate 선택 G를 그 세 regime 사이에 균등하게 다시 배정한다. 보정할 때와 건너뛸 때의 환경별 deployment Brier 차이를 D라 할 때, 실제 배정 − 재배정의 기대값은 mean[(G−mean_regime G)D]다. 같은 물체와 signed proposal에서 평균 보정률을 고정했을 때 표면별 선택의 가치를 측정한다.

세 regime 전부, weak와 local-null의 두 집단, strong과 weak/null을 짝지은 두 대조의 평균을 각각 계산한다. 세 속도의 결과는 동일한 4개 object–surface cell과 첫 부호를 층화한 paired block bootstrap으로 동시에 구간을 낸다. 첫 세 번으로 얻은 signed proposal의 orientation 정확도와 orientation이 맞거나 틀렸을 때의 보정 적용률, 적용 대 건너뜀의 실제 Brier gain 및 고정 shrinkage와의 차이도 계산한다. held-out label에서 정한 gain의 부호에는 유한 deployment objects에 따른 오차가 있다.

같은 물체와 noise를 지닌 두 regime에서 강도가 더 큰 표면이 더 높은 posterior strength를 받는 비율을 알아본다. 2개 score가 동률이면 0.5로 계산한다. 이는 같은 환경의 반복을 비교한 counterfactual 진단이다. 새 물체 하나만 관찰하는 deployment의 식별력은 독립 물체를 포함한 통상 AUROC로 따로 보고한다.

같은 speed-conditioned source model의 강도 요약 방식이 weak/null 구분에 미치는 영향을 보조로 시험한다. 먼저 4개 regime 전체의 posterior mean strength를 계산한다. 두 번째 score는 가능한 상태를 weak와 local-null로 좁히고 family·orientation을 적분한 두 source marginal likelihood의 log ratio다. 이 비교는 두 상태가 후보라는 사전 정보를 이용하므로 측정된 gate 성능과 별도의 진단이다. 데이터를 재학습하거나 속도에서 threshold를 맞추지 않는다. 속도별 AUROC와 각 속도에서 두 score 차이의 paired interval을 기록한다.

추가 보조 정책은 같은 source expected Brier gain 표에서 log-odds correction multiplier 0, 0.325, 1에 해당하는 세 값을 비교한다. Source gain이 가장 큰 행동을 선택하고 동률이면 고정 shrinkage를 우선한다. 이 규칙은 v37의 target Brier를 보기 전에 정했다. 고정 shrinkage와 binary gate에 대한 차이를 각 속도에서 비교하고, .50에서 .75, 1.10으로 바꾼 차이와 difference-in-differences를 paired interval로 평가한다. 추가 GPU p90은 측정하지 않는다.
