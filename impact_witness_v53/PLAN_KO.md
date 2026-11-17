# v53 · 네 쌍 중 한 쌍을 impact에 쓰는 실험

교체 전 네 쌍을 모두 공유한다. 교체 후 4R 조건은 네 물체에서 weak/reference를 읽고, 3R+I 조건은 첫 세 물체의 관측을 유지한 뒤 네 번째 물체에서 weak/impact direction을 읽는다. 어떤 조건에서도 네 번째 reference와 impact direction을 함께 얻지 않는다. 교체 단계의 구입은 8 contacts·8 resets다. Impact는 희생용 calibration 물체에서만 일으키며 deployment stick/slip 정답은 공개하지 않는다. 같은 v50–52 holdout을 고정하여 정보 교체의 효과를 짝짓는다. 이 실험은 신규 물리 session에서의 replication이 아니다.

주분석은 같은 3R+I를 받은 continuous joint mixture에서 같은 정보에 맞춘 v52 세 행동 selector와 같은 행동을 쓰는 sign vote의 loss를 각각 뺀다. Paired catalog를 독립 단위로 평균 held-out Brier와 128개 물체 session 평균의 p90을 평가하고 4개 contrast에 대한 simultaneous 95% interval을 보고한다. Post-only simple vote도 비교한다. 추가로 기존 4R selector와 4R continuous, 같은 prior를 사용해 마지막 impact를 가린 대조를 포함한다. 실제 budget을 들여 교체할 가치가 있는지는 3R+I continuous와 4R continuous의 차이로 판단한다. GPU p90은 synchronized 512-object inference이며 물리적 acquisition 시간은 별도다.

예상은 active 물체에서 κ가 0.5에서 멀 때 impact sign이 유용할 수 있고, κ=0.5나 inactive일 때 한 번의 관측으로 e를 식별하기 어렵다는 것이다. 2N reference의 신뢰도가 높으면 네 번째 reference가 주는 정보까지 포기해 평균 손실이 늘 수 있다. 개선 여부 또는 크기는 사전에 단정하지 않는다. 독립 deployment noise와 solver resolution에서 민감도를 평가한다.

추가 진단으로 이상적 ternary sign을 두 시점·두 마커의 합성 line-scan에서 복원한 sign으로 교체한다. 0.1 mm/pixel, 0.15 mm dead zone을 고정하고 marker localization jitter를 0.2/0.85 pixel로 둔다. 이 검증은 contrast가 보이는 marker의 virtual sensor이며 실물 관측이 아니다. Timing과 onset localization, surface damage 및 촬영 실패가 남는다.
