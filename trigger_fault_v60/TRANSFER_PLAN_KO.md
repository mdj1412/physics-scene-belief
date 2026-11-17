# v60 · 표면별 연속 fault 빈도와 선택 threshold의 별도 검증

Primary 평가와 별도로 48개 새 surface에 독립적인 fault 빈도를 하나씩 부여한다. 48개 균등 구간의 중앙값을 seed 60071901로 무작위 치환해 surface별 high-hazard mixing probability w를 정한다. 혼합은 8회 pilot 단위가 아니라 session 단위에서 수행하며, 한 표면의 w는 모든 session·A/B·delay에 대해 고정한다. Common-mode branch에서 기대 hazard는 w·c_high+(1−w)·c_low다. 두 공통오류 potential outcome의 같은 session 중 하나를 seed 60071902로 고르고, 2개 물체와 두 전자센서의 correlated readout을 같이 취한다. 동일 선택 coin을 두 환경, 두 reset 조건, 모든 delay에 공유한다. 원인 P의 경우에는 실제 물리 가설의 A/B를 유지한다. 이 추가 표본은 기존 pixel 및 contact를 재사용하는 synthetic counterfactual이다.

동결된 source selector와 별도로 C에서 장치의 오류 빈도가 A/B에 무관하다는 대안을 설계한다. C의 high/low likelihood를 동일 prior로 합산해 A와 B의 C likelihood를 동일하게 만든다. P의 A/B likelihood와 v56의 비용 threshold는 유지한다. 이 대안은 true cause, actual w, target labels를 입력받지 않는다. 4×2의 8회 예산과 행동 두 개는 동일하다. 이 비교는 주 평가를 대체하지 않는다. 두 selector에서 mean held-out Brier와 선택률, cause/정책 판별률 및 gate 대비 paired catalog interval을 보고한다. 개별 표면에 알려진 oracle w는 필요할 때만 진단 상한으로 표시한다.

Source의 fault–A/B 연결에 대한 의존 정도를 수치화하기 위해 C에서 표면-환경 대응을 matched로 유지할 확률 α를 0~1로 바꾸고, 반대는 reversed로 설정한다. α=0.5가 independent 조건이다. 동결된 selector의 평균 Brier는 α에 대해 선형이므로 corrected reset과 v54 gate, corrected sham의 risk가 같아지는 교차점을 계산한다. 교차점이 [0,1] 밖이면 관측 가능한 범위 내 교차 없음으로 표시한다. 24개 catalog bootstrap으로 교차점의 불확실성 및 두 대조 모두를 이기는 영역의 존재를 평가한다. Target label로 threshold·selector·하드웨어 파라미터를 조정하지 않는다.
