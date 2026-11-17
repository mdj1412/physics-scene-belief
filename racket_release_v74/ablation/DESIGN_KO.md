# 입사 normal speed에 대한 feature ablation

결과를 보기 전에 v73 elastic response의 4개 계수, 계수별 clipping, source별 회귀와 공분산 floor를 유지하고, alpha의 기울기에 들어가는 feature 하나만 바꾼다. 기존 |s_in|와 대안 |s_in| / |v_n,in|을 각각 source에서 표준화한다. 후자는 normal load를 나타내는 속도가 커질수록 달라지는 상대 접선 운동을 허용하는 단순한 비율이다. Gossard의 Eq. 10을 복제한 모델은 아니다. Source prior를 각 feature로 다시 추정하고, 두 모델 모두 같은 K=4/8 paid impact만으로 네 계수를 갱신한다. 동일 v73 fast test와 margin 0.20을 쓴다. 이는 모델의 특정 feature와 함께 적합한 prior가 오차를 설명하는지 확인하는 보조 실험이다.
