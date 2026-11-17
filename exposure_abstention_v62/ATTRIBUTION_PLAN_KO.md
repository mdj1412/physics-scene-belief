# v62 · 보류한 정판독과 오판독의 기여를 분리하는 보조 분석

채점 전 별도 검증을 정한다. 실제 paired mismatch가 틀렸던 광학 관측을 보류한 효과와, 옳았던 관측을 보류한 효과를 구분한다. 물리적 부호가 반전되어야 하는지는 원인 P/C나 장치 판별이 아니라 각 유료 물체의 실제 simulator deformation으로 정의한다. Original v61 reset이 판독했으나 selective reset이 버린 reference–impact 쌍만 대상이다.

두 privileged counterfactual을 구성한다. 첫째는 selective가 버리기로 한 오판독만 missing으로 만들고 나머지는 v61 그대로 쓴다. 둘째는 버리기로 한 정판독만 missing으로 만든다. 전자 sensor digit, 순서와 reset, 물리 현상, 유료 예산, frozen likelihood 및 후속 decoder는 모두 동일하다. 원인 정답은 배포 reader에 주지 않는다. 네 arm(original, wrong-only, correct-only, both)의 conditional risk로 각 종류를 먼저/나중에 제거하는 두 순서의 평균 marginal effect를 계산한다. 두 기여의 합이 selective-minus-v61의 exact change인지 독립 검산한다. Board·catalog 6,000회 paired crossed bootstrap, 보조 95% 구간을 보고한다. 제거할 때 sign 자체를 고쳐 주지는 않는다. 이 계산은 관측 묶음과 손익 구조가 주어진 simulator 안의 진단이다.
