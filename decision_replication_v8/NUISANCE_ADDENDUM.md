# Nuisance-dimension control 사전 기록

2026-11-11 KST. Primary acquisition 40/64, actual evaluation 0/64인 상태에서 source inspection으로 추가했다. 어떤 actual loss도 읽지 않았다. 기존 four primary contrasts는 변경하지 않는다. 아래는 supplementary diagnostic이다.

Corner lower boundary와 lower-left arc는 `cy=.014+d+g`, `hy=d`이므로 d가 소거된다. 모든 incoming source-y shifts는 최대 .028m이고 upper core의 최소 y는 .044m다. 따라서 현재 actions/queries에서는 upper corner/edge를 방문하지 않아 half-height d는 dynamics에 무관할 것으로 예상한다. GPU에서 prior samples 4096개와 d endpoints/midpoint에 대해 모든 factual/action/query forward means의 invariance를 검증한다. 실패하면 marginalization baseline을 실행하지 않고 실패를 보고한다.

View mean의 두 번째 coordinate는 d(mm)+calibrated bias다. Independent uniform prior d∈[22,34]mm를 Gaussian view likelihood에서 analytically integrate한다. Covariance correlations와 finite prior bounds를 유지한 Gaussian CDF difference를 사용하며 단순 channel deletion은 아니다. 1D quadrature로 likelihood를 검증한다.

고정16k baseline과 동일한 four independent SMC4096 populations/seeds를 재생성하고 d를 analytically marginalized view weights만 변경한다. Dynamic-action posterior means가 기존 fixed16k와 일치하는지 확인한다. Standard 및 Cross-fitted Rao–Blackwellized SNIS16k choices를 public outer data만으로 정한다. Private truth/actual arrays를 읽지 않으며 auxiliary choices를 checksum으로 고정한다. 이미 생성된 common decoder predictions에 이 choices를 적용해 descriptive paired world-cluster95% CI를 구한다. Primary multiplicity family에는 포함하지 않는다.

이 control은 known-family 2D simulator의 analytic structure를 활용한다. Unknown geometry의 real-image task에서 바로 쓸 수 있는 일반적 baseline이라고 주장하지 않는다. 다만 이 toy benchmark에서 nuisance-induced importance collapse를 Bridge의 보편적 physics advantage로 오해하지 않기 위한 강한 model-aware baseline이다.
