# 독립 full-budget 재학습과 policy 평가

최초 두 model 모두 아직 학습 중이고 target 결과가 생성되기 전에, 별도 training data와 bank를 사용해 **총 3쌍**의 540 s/arm full-budget 학습을 진행하기로 했다. Seed pair 1의 다섯 primary Brier contrasts와 실제 GPU timing은 원래 설계대로 보고한다. 나머지 두 쌍은 각 source seed에서 전체 label 생성·scorer fit를 새로 실행한 reproducibility check다. Source block을 나누어 다시 fit하는 저예산 진단과 구분한다.

새로 학습한 scorer들은 처리하지 않은 과거 prefix 정보를 열 수 없고, 최초 평가와 동일한 48 target blocks에서 stage 1 뒤 선별한 네 환경의 stage 2 score만으로 최종 할당을 결정한다. Stage별 risk는 scorer의 identity에 의존하지 않고 동일한 query/probe 후보, source-law inference, 공통 decoder, 같은 underlying 2·4·8회 history에 조건부다. 그러므로 저장된 배열로 policy를 실행하고 선택한 2·4·8회 예측의 Brier를 가져오면 online policy와 같은 결과다. 첫 번째 쌍에서 실제 GPU policy와 replay가 일치하는지 확인한다. 외부 input은 동일해도 새 seed마다 예산 배분과 최종 loss를 새로 계산한다. 고정 policy 두 개도 같은 target block에서 재사용한다.

Seed별로 broad, repeated, fixed 4, uncertainty-only의 Brier와 다섯 paired 차이를 보고하고 평균을 계산한다. 3개 seed를 고정했을 때의 target-block bootstrap과, training seed pair를 함께 재표집하고 같은 법칙 내 target block을 네 정책에 공통 재표집하는 crossed bootstrap을 분리한다. 모든 policy가 같은 evaluation blocks를 쓰므로 세 seed의 evaluation을 144개의 독립 target blocks로 집계할 수 없다. 독립 training seed가 세 쌍뿐이므로 crossed interval은 탐색적이다. 한 seed가 빠질 때의 범위와 세 차이의 부호도 제시한다. 양성 주장에는 primary baselines 대비 실용적 margin 및 seed 간 일관성이 모두 중요하다. GPU p90은 최초로 고정한 두 scorer와 두 baseline에서 측정한다. 재학습 seed의 policy 효과에 timing을 임의 복제하지 않는다.
