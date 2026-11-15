# 전체 stopping policy의 결과를 해석할 보조 기준

Primary held-out 결과를 열기 전에 추가한다. Policy가 fixed 4와 달라지는 경로는 일부 환경을 두 번에서 멈추는 결정과 일부 환경을 여덟 번까지 계속하는 결정이다. 각 block에서 실제 배정을 그대로 놓고, fixed 4 대비 Brier 차이를 두 경로의 부호 있는 기여로 나눈다. 또한 2·4·8회에 4·2·2곳을 할당하는 균등 무작위 배분의 정확한 기대 Brier와 비교한다. 0.5 R2 + 0.25 R4 + 0.25 R8을 여덟 환경에 평균하면 이 기준이 된다. 같은 budget이어도 fixed 4와는 보유량의 분포가 다르므로, 무작위 배분 대비 차이는 환경을 고른 효과를 진단한다.

Primary 평가는 한 번의 실제 stopping trajectory만 사용한다. 보조 진단은 사전 고정한 action과 risk[k]로 사후 계산한다. Learned 두 방법이 stage 1과 최종 8회 대상에서 얼마나 자주 같은 환경을 고르는지, stable sort에 기대는 exact score ties가 있는지도 기록한다. 이 지표는 Brier와 별개다. 16개 block씩의 paired law-stratified bootstrap으로 pointwise 95% 구간을 제시한다.
