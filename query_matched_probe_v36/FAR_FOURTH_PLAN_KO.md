## 같은 세 번의 접촉 이후 네 번째 측정의 위치를 비교하는 보조 대조

v36 GPU worker가 첫 block을 계산하던 시점에, target loss를 열기 전에 정했다. 네 번째로 high-speed conventional probe를 사용해도 gate의 적용 여부를 개선할 수 있는지 비교한다. 새 학습은 없다. 처음 세 번의 자료로 정한 동일한 z와 같은 source expected Brier gain curve를 사용하며 네 번째 conventional action의 실제 부호에 해당하는 source likelihood로 posterior를 한 번 갱신한다. Target 정보로 threshold를 고르지 않는다.

두 비교를 계산한다. (1) 실제 선택한 contact의 물리 likelihood에 맞춰 acquisition까지 실행한 far-fourth gate와 near-fourth gate를 비교한다. 이는 네 번째 측정이 posterior acquisition과 gate에 준 효과를 합친다. (2) 동일한 near deployment action과 공통 base prediction 위에 두 gate를 각각 적용한다. 이 paired counterfactual에서는 네 번째 측정이 terminal gate에 준 효과를 비교할 수 있다. 두 비교 모두 세 localized target regime을 균등 평균하고, global-null을 따로 보고한다. Paired seed block을 cell과 시작 부호 안에서 함께 12,000회 resample한다. 두 집단과 두 비교에 simultaneous interval을 적용한다. 이 대조에는 별도의 online timing을 실시하지 않는다.
