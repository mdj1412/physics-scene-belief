# Near measurement의 고유 효과 및 regime 대응 대조

Primary target loss를 검토하기 전에 정한 secondary analysis다. Frozen gate에서 네 번째 측정의 factual feature만 제공한 posterior를 analytically marginalize한다. Gate4 − gate3+x4는 같은 factual rows와 첫 세 paid responses가 주어진 상태에서 네 번째 noisy velocity가 decision을 실제로 바꾼 효과를 보여준다. Paired loss, 적용률, 결정을 바꾼 비율을 보고한다. Local strong과 weak/null을 분류하는 continuous posterior strength의 AUROC도 함께 계산한다.

같은 물체, orientation, probe noise의 local strong/weak/null을 하나로 묶어 세 gate decision을 이 세 regime 사이에서 uniform permutation했을 때의 expected Brier와 실제 대응의 Brier를 비교한다. 각 target에서 보정의 비용 d는 해당 target의 원래 deployment action 및 outcome으로 평가하며, 평균 적용률은 같은 환경 안에서 고정한다. 식은 mean_r[(G_r−mean_k G_k)d_r]이다. 이 값과 세 regime에서 동일한 결정을 내리는 비율이 fourth probe가 실제 구별에 기여한 정도를 정량화한다. Interval은 object–surface cell 및 첫 motion을 층화한 32개 block의 paired bootstrap(12,000회)으로 구한다.
