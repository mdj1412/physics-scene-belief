# 보정 선택의 사전 보조 분석

추정한 projection에서 selector가 이기는 경우, 실제로 environment 순위를 구별한 것인지 단순히 두 고정 보정 방법을 섞은 것인지 검산한다. 각 panel과 condition에서 결정을 내린 비율 π와 평균 Δ를 계산한다. 같은 확률 π로 무작위로 optimizer를 택했을 때의 기대 loss는 fixed+π·mean(Δ)다. 실제 선택 loss와의 차이는 해당 집단 안에서의 선택과 Δ의 covariance다. π는 평가 정책이 현재 관측에서 실제 선택한 빈도만 사용하며 label은 이 대조를 채점할 때만 사용한다.

각 decoder에 대해 원래 decoder에서 내린 결정을 모든 환경에 그대로 쓰는 선택, 두 후보 중 더 높은 stick probability를 고르는 v39 보조 규칙, 환경별 실제 label 집합을 본 뒤 더 낮은 방향을 고르는 offline oracle도 같은 예측으로 재채점한다. 세 번째 값은 환경별 정답을 훔쳐본 하한으로 operational 방법이 아니다. Target에서 가중치나 threshold를 다시 고르지 않는다. Cell 및 panel 단위로 추정해 같은 latent draw의 중복을 피하고, 보조 interval은 개별 비교의 95% 구간으로 보고한다. Shape나 contact law 이름은 실제 selector 및 v39 baseline의 입력에 쓰이지 않는다.
