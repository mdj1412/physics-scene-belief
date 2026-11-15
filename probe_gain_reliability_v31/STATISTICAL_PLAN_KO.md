# v31 · 사전 통계 분석 계획

평가 파일을 열기 전에 고정했다. 한 block에는 같은 접촉 법칙에 속한 서로 다른 8개 환경이 있고, 각 환경의 factual evidence와 이미 구입한 calibration prefix는 모든 비교에서 동일하다. 독립적인 continuation vector를 f, 독립적인 deployment batch를 b라고 하자. 환경 e에서 G[e,f,b]는 같은 batch와 common decoder에서 중지했을 때의 Brier에서 continuation 이후의 Brier를 뺀 값이다. 양수가 이득이다. 한 cell 안의 world와 반복 sensor noise를 독립적인 block으로 세지 않는다.

## 주 추정량

1. Stage별로 세 law에 동일한 가중치를 주어 **deployment batch main-effect variance − calibration future main-effect variance**를 추정한다. 양수는 같은 횟수의 두 종류 반복에서 batch 쪽 주변분산이 크다는 뜻이다. Interaction을 어느 축에 귀속시키지 않는다.
2. 같은 stage에서 두 축을 **모두 독립적으로** 다시 뽑아 얻은 두 번의 환경 순위가 environment pair마다 같은지 확인한다. 동률은 0.5점을 주며 무작위 기준은 0.5다. 두 stage의 variance difference와 두 stage의 rank concordance − 0.5에 대해 전체 4개 대비의 simultaneous 95% interval을 산출한다. 해당 interval이 0을 포함하면 그 대비의 방향은 결정하지 않는다.
3. 각 law 안에서 독립적인 block 전체를 paired bootstrap 10,000회 재표집한다. 같은 block에 들어간 stages, 환경, futures, batches 및 두 공통 decoder 조건이 있으면 함께 유지한다. 서로 다른 law가 같은 독립 block seed를 공유하면 그 block을 law 사이에서도 같이 재표집한다. 사전 고정한 물리 법칙과 수치 설계에 조건부인 interval이다. Family에 대해 bootstrap 표준오차로 표준화한 절대편차의 최대값으로 simultaneous 구간을 구하고, 그 밖에는 pointwise percentile interval을 쓴다. 법칙당 block 수가 작으면 CI의 꼬리와 우열 판정은 불안정할 수 있다.

## Crossed variance decomposition

각 prefix 안에서 F×B cell의 grand mean을 m, row/column means를 r_f,c_b로 두고 interaction residual을 u_fb = G_fb − r_f − c_b + m으로 정한다. MS_F = B Σ(r_f−m)²/(F−1), MS_B = F Σ(c_b−m)²/(B−1), MS_I = Σu_fb²/[(F−1)(B−1)]. 독립적인 두 축의 무작위 표집을 가정하면 **V_F=(MS_F−MS_I)/B, V_B=(MS_B−MS_I)/F, V_I=MS_I**는 각 population variance component의 method-of-moments 추정량이다. 이는 Gaussian·additive 응답을 필요로 하지 않는 two-factor functional ANOVA다. 매 cell을 복제하지 않아 별도의 cell measurement error와 future×batch interaction을 더 세분할 수 없다. 난수·비용이 고정된 decoder에서 지속되는 finite-world 및 sensor-noise 변동은 deployment draw 또는 그 interaction에 들어간다. 특정 calibration future로 조건화한 단순 batch sample variance는 V_B+V_I를 추정하므로 독립적인 batch main effect와 구분한다.

추정값은 prefix별로 먼저 계산하고, 그 다음 block·law에 동일 가중치를 주어 평균한다. 음수 추정치를 0으로 바꿔 winner를 정하지 않는다. 추정된 한 cell의 total variance는 V_F+V_B+V_I, F'×B' 독립 교차평균의 조건부 variance는 V_F/F'+V_B/B'+V_I/(F'B')다. 결과는 raw variance(Brier²), 분산이 양수인 경우 square root(Brier 단위), component share로 제시한다. Share나 sqrt는 해석용이며 음수 분산이 물리적으로 존재한다는 뜻이 아니다. 추가 future와 추가 batch의 효율 비교는 같은 cell 수의 sampling precision을 비교하며 실제 GPU/센서 비용의 동등성을 가정하지 않는다. 0.003은 v29–v30의 참조 효과 크기이며 v31에서 새로운 significance threshold로 쓰지 않는다.

## 재현성과 추가 검산

- 환경 두 개의 gain 차이 부호를 비교한다. 모든 unordered environment pair와 조건을 만족하는 ordered cell 쌍을 균등하게 사용한다. (a) future는 같고 batch만 다름, (b) batch는 같고 future만 다름, (c) 두 축 모두 다름을 구별한다. 같은 한 묶음을 공유하는 (a),(b)의 높은 일치율을 독립 재현성으로 해석하지 않는다.
- 두 개의 futures 평균, 두 개의 batches 평균, 2×2 평균을 서로 겹치지 않는 두 반복에서 각각 비교한다. 예를 들어 1×2 평균의 두 반복은 다른 future를 사용하고 두 batch 집합이 겹치지 않는다. 모두 무작위 예측 0.5가 기준인 descriptive concordance다. 불확실성은 중복 사용된 cell pair가 아닌 원래 block으로 평가한다. 2×2가 불가능한 설계에서는 해당 endpoint를 산출하지 않는다.
- 같은 조건부 분산 구조가 예측하는 값과 독립 cell 간 squared difference 평균의 1/2이 맞는지 수치로 확인한다. Law·stage별 mean signed gain, SD of gain와 환경 평균에 대한 SE를 구별한다. 환경마다 하나의 prefix이므로 환경 순위에 population 차이와 실제 prefix 이력이 모두 들어 있다. 진정한 환경 고유값의 순위나 일반적인 다른 contact law로 단정하지 않는다.
- 2→4회와 4→8회는 prefix마다 가능한 continuation horizon에 대한 평가다. Eight environments all-prefix diagnostic과 이전 learned stopping의 survivor-conditioned 결과를 구별한다.

실행 크기는 각 law 6 blocks, 각 block 8 environments, F=B=4로 고정되었다. Law 사이에도 seeds가 분리되어 stratified bootstrap을 적용한다. 서로 다른 환경의 f=0 또는 b=0에는 물리적으로 공유하는 사례가 없다. 따라서 rank 계산에서는 서로 다른 환경 두 곳의 합법적인 반복 index 쌍을 독립적으로 조합하여 index 재배열에 불변인 concordance를 구한다. 한 환경 내에서 gain의 반복 표본 간 평균 절대 차이와 그 차이가 0.003을 넘는 빈도도 기술한다. 이는 정책 효과가 0.003을 넘을 비율과 다르다. F×B 전체에서 환경 간 표본평균의 분산을 그대로 신호로 쓰면 finite-grid noise가 더해진다. 원할 경우 독립 replicate의 환경 간 centered covariance에 해당하는 noisy-mean 보정값을 부가 진단으로만 보고한다.
