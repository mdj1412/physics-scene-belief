# v37 사전 분석 계획

Fixed 4의 1–3번째 probe와 마지막 probe의 높이 23 mm를 고정한다. Target query는 25 mm, incoming tangential speed +0.425다. 네 번째 probe의 tangential speed만 +0.50, +0.75, +1.10으로 바꾼다. +0.50은 v36 reference, +0.75는 사전에 선택한 주 비교, +1.10은 가장 빠른 추가 조건이다. 같은 물체, 첫 방향, sensor noise, orientation, deployment batch, bank를 세 속도에 재사용한다. 속도별 likelihood는 실제 동작에 맞추되 v36의 첫 세 관측 기반 signed proposal z, β, source gain lookup, priors, gate threshold는 유지한다. Gate에는 true slip과 표면 정보가 주어지지 않는다.

Local strong, weak(원래 방향 효과 30%), null(0%)의 균등 혼합에서 주 비교를 한다. 모두 먼 probe 영역에서 같고 query 부근에서 다르다. Global null은 전체 높이에서 방향성이 없으며 별도 control이다. 어떤 표면에서 보정해야 할지 판별하는 성능은 (a) strong 대 weak/null, (b) 미세한 구별인 weak 대 local null의 posterior strength AUROC로 측정한다. Slip은 네 번째 접촉 전체 중 실제로 미끄러진 비율이다. 같은 world에서 속도별 paired Δ를 계산한다. 속도와 함께 velocity response 크기도 달라지므로 slip 발생 자체의 causal mediation으로 해석하지 않는다.

Primary Brier 비교는 각 속도에서 gate − 고정 32.5% shrinkage, gate(.75) − gate(.50), gate(1.10) − gate(.50), 그리고 각 고속에서 baseline 대비 이점이 .50에서보다 커졌는지 difference in differences를 포함한다. Delta가 작을수록 유리하다. Gate와 shrinkage는 같은 speed-conditioned acquisition 및 original probability, βz를 사용한다. Shrinkage는 확률의 log odds 보정량을 항상 0.325배로 적용한다. 각 scope의 7개 대조에 simultaneous 95% interval을 사용한다. Brier의 absolute 0.003, 같은 속도 두 방법의 p90 ratio +2%를 v35–36의 실용적 기준으로 유지한다. 원래 physics, always apply, soft 보정은 가능한 경우 offline 비교로 구분한다.

4개의 고정 object × surface cell 안에서 각각 8개의 독립 seed block을 첫 probe 방향별로 층화해 paired bootstrap 12,000회를 시행한다. 같은 block 내 environment 8개, mirrored orientation, regime, 관측 noise, methods, speeds는 함께 묶는다. 재표집 구간은 이 네 cell에 조건부이다. AUROC는 block 쌍 사이의 concordance kernel로 재계산하고 두 구별 과제와 두 speed change의 interval을 동시에 보정한다. 각 속도의 절대 AUROC 구간은 pointwise다. 국소 강도가 다른 짝에서 noise-free response가 같아지는 비율과 latent object 및 orientation, 후보 surface를 아는 privileged Bayes accuracy도 계산한다. Decision utility와 score 순위는 별개로 판정한다.

같은 latent 물체·orientation에서 두 후보 표면의 slip 상태를 대응시킨다. 속도 변화 뒤에도 두 후보 모두 slip인 짝, 둘 중 하나 이상이 stick에서 slip으로 바뀐 짝, 나머지 짝에서 privileged discrimination 증가분을 분해한다. 동일 측정 noise 0.027의 Gaussian likelihood를 가정해 두 후보의 outgoing velocity mean 차이로 Bayes ceiling을 계산한다. 이는 outcome을 보고 정책을 고르는 oracle이 아니라, 같은 한 번의 sensor로 얻을 수 있는 정보의 이론적 상한이다. 실제 posterior가 이 순위를 얼마만큼 구현했는지와 gate/shrinkage의 Brier는 각각 실측한다.

속도에 따른 acquisition 선택 변화와 gate의 네 번째 근접 측정 효과를 분리하기 위해, 세 속도의 accept/skip만 +0.50에서 선택된 동일한 deployment actions와 probabilities 위에 놓는 offline control을 추가한다. Binary policy family의 남은 가능성은 환경별 항상 적용/건너뛰기 중 하나를 독립 deployment 물체 절반의 label로 정한 후 나머지 절반에서 채점해 진단한다. 정답을 이용하는 이 진단은 실제 gate의 성능과 구별한다.

Weak와 local null을 반반으로 둔 평균과 각 조건의 결과도 보조 추정치로 보고한다. Global-null은 먼 위치에서 이미 방향성이 없어 처음 세 conventional 측정도 달라지므로 구분해 제시한다.

추가로 v36의 기존 +0.50 archive에서 세 local regime의 평균 Brier를 최소화하는 고정 multiplier λ∈[0,1]를 하나 선택한 뒤, v37에서는 모든 속도에 같은 λ를 적용한다. 이 보조 baseline은 v37의 물체와 label을 사용하지 않지만 이미 target과 같은 simulator family의 과거 labeled validation 자료를 가진 조건이다. 사용자가 요청한 고정 32.5% 비교는 그대로 primary다.
