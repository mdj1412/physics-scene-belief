# 2→4 gain을 학습하는 것과 전체 stopping 목표의 차이

모델이 틀릴 수 있는 이유 하나는 stage 1의 target이다. 4회까지 계속하면 지금 2→4회의 gain뿐 아니라 나중에 8회까지 갈 기회도 생긴다. 결과에 맞춰 법칙이나 모델을 재선택하지 않고 이 차이가 얼마나 클 수 있는지 진단한다.

동일 law·block의 여덟 환경에서 허용된 모든 배분(4 × k=2, 2 × k=4, 2 × k=8)은 420개다. Same-block retrospective full-information 선택과 same-block myopic 선택(먼저 R2−R4로 네 곳, 그중 R4−R8로 두 곳)을 계산한다. 둘 다 미래를 본 진단이라 policy 성능으로 제시하지 않는다. 별도로 각 law 안에서 독립 block을 하나 완전히 떼어놓고, 다른 15개 block의 동일 여덟 coupling에서 stage별 평균 Brier를 구한다. 이 값으로 myopic 배분과 420개 중 최소 expected Brier 배분을 각각 정해 떼어 둔 block에서 평가한다. 이는 해당 law와 실제 coupling을 아는 참조값으로 새 prefix sensor evidence를 활용하지 않는다. 이 모형과 평가 단위의 제한을 명시한다. 목표의 차이가 중요한지에 대한 기술통계로만 보고한다. 훈련과 평가 fold가 겹치므로 leave-one-block-out 차이를 독립 표본으로 간주한 신뢰구간이나 일반적 최적성 주장은 제시하지 않는다.
