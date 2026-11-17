# v62 · 같은 수의 판독을 보류하는 대조

Confidence에 따라 어떤 관측을 보류하는 것이 가치가 있는지 확인한다. 주 모집단과 동일한 source-frozen reset likelihood, 독립 fault–target, 균등 LED pulse × shutter와 board·surface 가중치를 쓴다. 모든 arm의 입력은 표면당 8회 impact에서 얻는다.

- 기존 v61이 판독한 reference–impact object pair의 수를 pilot마다 n, selective reader가 채택한 쌍의 수를 k로 둔다. Selective가 v61의 부호를 바꾸거나 신규 쌍을 발급하지 않는지 검증한다.
- 같은 pilot의 기존 n쌍 중 정확히 k쌍을 무작위로 유지하는 모든 subset을 열거해 균등 평균한다. Source 표면이나 장치 이름, 실제 부호, future label에 따른 재추첨은 없다. 이미 annotation된 결과는 scoring에만 사용한다. 이 대조는 건수와 초기 난이도를 정확히 맞춘 조건부 진단이다. 실제로 별도 장치를 사용한 trial로 보지 않는다.
- 기존 reset, selective reset, 같은 수를 무작위로 남긴 reset, camera 정보를 모두 보류하고 전자센서만 쓰는 reset 및 v54 gate를 비교한다. 전체 pilot을 점수화하고, 판독한 쌍의 오류율과 별도로 P/C 및 정책 balanced accuracy·mean held-out Brier·fixed 선택률을 계산한다. 같은 유료 관측에서 행동이 바뀐 비율 및 마지막 판독을 보류한 pilot 비율도 보고한다.
- Exact expectation을 집계한 뒤 24개 catalog × 8개 board를 각각 재추출하는 crossed bootstrap 8,000회로 selective − random-thinning 및 selective − no-camera의 평균 차이와 개별 95% CI를 계산한다. 사후 이득에 따라 confidence cutoff를 선택하지 않는다. Primary 세 대조의 판정은 별도 계획을 따른다.
