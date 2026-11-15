# v36 결과를 열기 전에 정한 분석

목표는 환경마다 지불하는 네 번의 접촉 중 하나를 query 부근으로 옮겼을 때, 이후 별도 물체의 stick probability를 언제 보정할지 더 잘 고르는지를 측정하는 것이다. Primary schedule은 v35의 양방향 교대를 사용하고 최초 부호를 균형화한다. 원래 high-speed 네 번을 받은 physics 및 v35 보정, 마지막에 near-query probe를 받은 physics 및 보정을 비교한다. Matched arm 안에서는 같은 correction proposal과 동일한 deployment actions 및 outcomes로 gate의 입력만 바꾼다. Gate는 source 자료에서 고정하며 target의 표면명, sign, theta, deployment label과 평가용 oracle 정보를 받지 않는다.

세 가지 질문을 구분한다. (1) Matched gate가 conventional physics와 v35보다 낮은 전체 held-out Brier를 내는가? (2) 같은 matched observations에서 gate가 always apply보다 weak/null에서 손해를 줄이는가? (3) 같은 예측값과 action에서 네 번째 측정을 제공한 gate가 가린 gate보다 나은가? 세 번째 대조가 네 번째 probe의 고유한 decision utility를 평가한다. Weak/null에서만 이기거나 전체 Brier가 내려간 것만으로 세 번째가 성립했다고 주장하지 않는다. Primary paired contrasts에는 family-wise 95% simultaneous interval을 적용하고 절대 Brier 0.003을 v35와 동일한 실용 기준으로 함께 제시한다. GPU p90의 기술적 허용 상한 +2%는 동일 장치 및 session에 적용한다.

각 material regime에 같은 비중을 주고 그 안에서는 object × surface, orientation을 균등 가중한다. 새로운 seed block이 통계 단위다. 같은 geometry와 noise로 만든 regime, mirror orientation, method, sensor repeat는 함께 재표집한다. 시작 부호가 균형화되면 해당 부호로 층화한다. 고정된 law definitions와 서로 독립인 seed 사이의 변동을 추정한다. 별도 global isotropic control은 가중 방법과 함께 따로 제시한다. 각 regime의 apply rate는 binary gate의 선택을 뜻한다. Soft gate라면 평균 보정 강도와 기준을 명시한다.

두 국소 regime에서 원래 factual data와 conventional prefix가 일치하는지 검사한다. 동일한 calibration 물체의 near-query noiseless means μa, μb와 공통 sensor covariance Σ가 있으면 equal-prior binary Gaussian experiment의 privileged Bayes accuracy를 Φ(||Σ^(−1/2)(μa−μb)||/2)로 독립 계산한다. 이 값은 실제 관측 가능한 classifier의 달성 성능이 아니라 simulator의 latent 물체·orientation·두 후보 law까지 알려주었을 때 한 번의 추가 측정이 낼 수 있는 정확도다. 실제 gate decision에는 이 계산을 쓰지 않는다.

Primary target panel은 두 contour와 두 directional response law를 교차한 4개 cell, cell당 8개의 새 block을 쓴다. 세 local regime에서는 far calibration site에서의 amplitude가 같고 query site에서만 1, 0.3, 0으로 갈린다. 물체와 양·음 orientation을 대응시킨다. Global-null에서는 calibration부터 방향성이 없다. 주분석은 세 local regime의 균등 평균이다. Global-null 및 네 regime을 같은 비중으로 둔 배포 혼합은 추가로 제시한다.

Correction 빈도 자체와 환경을 구별한 효과를 분해한다. 환경당 always-apply − never-apply Brier를 d, 실제 gate를 G∈{0,1}라 하면 적용률만 맞춘 무작위 정책 대비 차이는 E[Gd]−E[G]E[d]다. G와 d를 environment 단위로 계산한다. Pooled와 regime별 결과에 같은 cluster resampling을 적용한다. True label을 본 oracle은 별도 설명용이다. 추가로 평가 물체와 공통 noise를 유지한 채 동일 near 위치에서 높이와 속도를 바꾼 privileged upper bound를 계산한다. 측정된 gate 성능과 구별해 다음 probe 설계의 후보를 해석한다.

보조 비교로 βz의 32.5%를 어떤 환경에도 똑같이 적용한 constant shrinkage를 추가한다. 0.325는 네 source prior에서의 평균 near-field multiplier (1 + .3 + 0 + 0)/4이며 target Brier로 고르지 않는다. 같은 proposal에서 posterior mean multiplier를 사용한 soft correction도 세 측정과 네 측정에 각각 비교한다. 이 수치들은 source와 target에서 동일한 연속 calibration law가 최적이라는 가정을 하지 않으며, 알고리즘의 고유 기여를 확인하기 위한 저비용 대조다. 비교 범위는 Brier다. 또한 세 local regime 중 strong의 빈도를 바꿀 때 두 방법 차이가 어떻게 달라지는지 이미 측정한 regime별 loss를 재가중해 보고한다.
