# v61 · P/C를 정확히 알려주었을 때 원인 정보의 가치

Primary에서 C의 high/low sensor fault는 A/B와 독립이며 두 regime이 같은 비중이다. C라는 정보만 알면 pilot의 관측은 A/B의 odds를 바꾸지 않는다. P에서는 두 물체의 물리적 association 차이가 남는다. 이 가설을 같은 새 24개 catalog × 8개 board의 dual reader에서 검산한다. Target label을 사용하는 selector fitting은 없다.

- Source frozen: v59 P/C·A/B joint likelihood에서 A/B odds에 v56 loss cutoff 2.016을 적용한다. 주 분석과 동일해야 한다.
- Fault-marginalized: C의 high/low likelihood를 같은 비중으로 적분해 양 target에 동일하게 준다. P의 likelihood 및 행동 cutoff는 그대로다. True cause는 모른다.
- Privileged cause + paid evidence: true cause P를 받으면 P 조건부 A/B odds를 사용한다. C를 받으면 gate를 유지한다. 이는 P/C 정보의 잠재적 증분가치를 진단한다. Policy에 실제 반영할 수 있는 정보는 아니다.
- Cause alone: P/C와 A/B가 독립이고 실제 Brier 손익의 분포가 P/C에 따라 변하지 않으므로 P/C를 완벽히 받아도 동일한 gate를 선택한다.
- Full target: 사후 각 surface × A/B에서 두 행동 중 더 낮은 Brier를 고르는 상한. Primary와 같은 독립 배정에서 정책별 mean, fixed 선택 빈도, gate 대비 차이 및 보조 비교를 보고한다. 24개 catalog와 8개 board를 각각 재추출하는 crossed bootstrap 8,000회를 사용한다. 추가 진단은 주평가를 대체하지 않는다.
