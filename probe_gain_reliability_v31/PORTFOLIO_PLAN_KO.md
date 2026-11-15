# v31 · 순위가 실제 접촉 배분에 미치는 영향

주 분석에 덧붙일 사전 지정 보조 진단이다. 모든 비교는 같은 law, block, 동일 prefix의 후보 8개를 사용한다. 한 stage에서 여덟 곳 중 정확히 네 곳만 계속할 때 배분 정책의 이득은 선택한 네 곳의 signed Brier gain 합계를 전체 여덟 곳으로 나눈 값이다. 균등 무작위로 네 곳을 뽑을 때의 기대 이득은 전체 평균 gain의 절반이다. 둘 사이 차이(양수일수록 유리함)를 보고한다. Stage별 의사결정 한 번을 모의하므로 v30의 전체 halving policy에 대응하는 성능으로 해석하지 않는다.

- 하나의 future × batch 또는 두 futures × 두 batches의 realized gain으로 상위 네 환경을 정한다. 선택한 부분표본과 future 및 batch 양쪽이 겹치지 않는 반대쪽 부분표본으로 계산한 이득과 비교한다. 모든 순서 있는 서로소 분할을 평균한다. 선택에 쓴 부분표본에서 다시 평가한 값도 제시해 hindsight optimism을 드러낸다. 같은 손실이면 tie에 걸친 환경들에 fractional selection probability를 적용한다.
- V30의 frozen ridge score와 paid-prefix friction uncertainty로 네 곳을 골라 16개 crossed cells 전체의 평균으로 채점한다. 이 두 규칙은 prefix 뒤의 결과를 볼 수 없다. 실현 gain으로 고른 앞선 규칙은 hindsight 사용에 따른 재현성 진단이며 배포할 수 없다.
- Law 별 independent 6 blocks 전체를 10,000회 재표집한 descriptive pointwise interval을 붙인다. 같은 block의 반복 조합 수는 독립 표본 수가 아니다.
- 환경을 단지 population coupling으로 해석했을 때의 차이도 살핀다. Law·stage 고정 후 여섯 blocks에서 같은 8개 coupling을 정렬한다. 서로 다른 block에서 얻은 4×4 환경평균의 pairwise concordance, 서로 겹치지 않는 3개 blocks씩 평균한 순위 일치율을 기술 지표로 기록한다. 새 block에서는 paid prefix·factual pool·bank도 함께 바뀌므로 이 지표는 prefix 단독 효과가 아니다. 6개밖에 없는 blocks의 중복 pair/partition을 독립 표본처럼 수치 검정하지 않는다.
