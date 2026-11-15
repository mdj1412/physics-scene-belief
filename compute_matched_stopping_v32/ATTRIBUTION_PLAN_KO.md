# 추가 분석 · 같은 prefix에서 averaging만 바꾸면 무엇이 달라지는가?

두 primary arm은 각각 동일한 GPU 예산으로 독립 source pools를 생성한다. 그래서 두 최종 model 사이에는 prefix 수, label의 정밀도, 실제로 뽑힌 training sample의 차이가 모두 들어간다. 평가 결과를 열기 전에 다음 보조 비교를 정했다.

Repeated arm의 모든 source prefix와 15개 feature를 그대로 고정한다. 각 stage의 2×2 gain matrix에서 (F1,B1), (F1,B2), (F2,B1), (F2,B2) 중 한 위치만 일관되게 골라 네 벌의 1×1 training labels를 만든다. 네 scorer와 2×2 평균 scorer에 같은 λ=3, train-only standardization, intercept를 적용한다. 네 scorer는 서로 겹치는 행·열을 사용하므로 독립 training trials로 집계하지 않는다. 검증용 추가 fit에는 CPU를 사용하고 primary GPU 예산의 모델은 원래의 GPU-trained lock만 쓴다.

같은 디자인과 규제를 쓴 ridge solution은 target에 선형이므로 네 single-cell scorer의 intercept와 coefficients를 평균하면 2×2 scorer와 수치적으로 같아야 한다. 반면 top-half stopping은 비선형이라 먼저 score를 평균한 결과와 각 policy loss의 평균은 다를 수 있다. 각 model의 target stage 1과 survivor stage 2를 다시 실행한다. 원래 예측을 낼 때 쓰는 stage별 held-out Brier table로 policy 결과를 채점하고 평균 single-cell policy와 2×2 policy의 차이를 본다. 이는 같은 prefix에서 averaging에 따른 변화를 보는 보조 질문이다. 1×1에서 절감된 시간을 새 prefix에 투자하는 primary comparison과 비용이 다르다.

표에는 동일한 source X를 사용한 네 후보의 held-out Brier 범위, signed difference (repeated − single 평균), 전체 세 law와 개별 law를 제시한다. 각 law 안에서 evaluation block을 함께 재표집한 pointwise interval을 붙인다. 부호와 크기가 달라져도 training design 일반의 우열은 독립적인 full-budget 재학습에서 판단해야 한다.
