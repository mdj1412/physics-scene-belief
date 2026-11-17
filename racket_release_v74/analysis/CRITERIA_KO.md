# v74에서 동시 이득을 판정하는 기준

기존 v73의 A-B-S II soft 평균과 fast subgroup 결과는 알고 있었다. 아래 판정 규칙은 v74의 target 보정 결과가 나오기 전에 정한다. 새 시편에 대한 사전등록 실험은 아니다.

- 주 평가는 v73의 공식 test에서 source가 정한 fast cutoff 이상인 37개와 그중 관측한 |s'|>0.20 m/s인 22개를 사용한다. 모든 방법·예산에서 같은 참조 판독을 쓴다. 모델의 confidence로 어려운 test 행을 제외하지 않는다.
- 주 보정법의 방향 적중률이 v73 majority vote보다 높고, velocity·spin 오차 각각이 majority vote보다 낮으면 요청한 세 조건을 수치상 충족한다. 두 연속 endpoint를 각각 보고한다. 추가로 보정 전 elastic과의 차이, elastic의 원래 이득 중 보정 후 남는 비율, single affine EB에 대한 순위를 제시한다. 이득 보존율은 (majority 오차 − 보정 후 오차)/(majority 오차 − elastic 오차)다. 분모가 양수일 때만 해석한다. 보정 전 elastic에 대한 동등성이나 noninferiority는 별도의 오차 한계가 없어 확정하지 않는다.
- Calibration draw마다 같은 held impact를 재사용한다. 우선 같은 impact에 대한 256 draw의 결과를 평균한 후 고유 held impact를 4,096회 paired bootstrap한다. 방향은 판독한 impact들에서, 연속 오차는 모든 fast impact에서 계산한다. 구간은 이 시편·관측 calibration bank·속도 조건에 한정한다. 부호 개선의 구간이 0을 포함하면 점추정 우위를 독립 재현으로 표현하지 않는다. 보조 split의 판정도 같은 규칙을 따른다.
- 부호가 모호한 충돌을 포함시킨다면 전체 fast 비교가 어느 쪽으로 얼마나 움직일 수 있는지 양 끝을 계산한다. 각 ambiguous impact의 미확정 참 부호를 하나로 고정하고 같은 행의 모든 draw와 두 방법에 공통 적용한다. 어떤 방법의 positive 예측 비율이 p이고 상대 방법의 비율이 q이면 실제 부호에 따라 그 행의 적중률 차이는 ±(p−q)다. 행별 최솟값과 최댓값을 평균한다. 이는 감도 분석이며 새로운 부호 label을 생성하지 않는다.
