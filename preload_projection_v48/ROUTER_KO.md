# v48 · Single-readout policy와 source calibration

## 같은 비용으로 어떤 정보를 비교했나

두 방법 모두 v43에서 이어진 공통 observation acquisition이 끝난 뒤 weak preload 한 번을 한다. **Query-local**은 명령된 미래 접촉 높이에, **fixed probe**는 source validation에서 한 번 고른 높이에 둔다. 한 번의 contact에서 3 mm gauge 양쪽의 변화 차이를 scalar displacement(mm) 하나로 받는다. 각 정책이 갖는 입력은 이 scalar, 명령한 높이, 이미 구입한 seven-point range profile이다. 다른 위치에서 얻었을 측정값, 대상의 family와 compliance state, 다른 signed variant 및 미래 label은 보지 않는다. 기존 구매가 deployment probe이면 두 geometry decoder가 일치하므로 original을 선택한다.

## Source 학습과 목적함수

v47의 12개 catalog에서 저장된 두 forecast와 finite-pressure label을 사용했다. Optical variant와 mechanical state의 부호를 독립적으로 교차했다. State 비율은 inert 1/2, 음·양 active 각각 1/4다. 두 radius와 세 offset을 고르게 평균했으며 매 source example의 학습 목표는 (p_projected − p_original) × (p_projected + p_original − 2y)였다. 이는 projected Brier에서 original Brier를 뺀 값과 정확히 같다. 같은 sample의 반복 measurement는 독립적인 held-out family로 계산하지 않았다.

각 profile에서 quadratic least-squares constant·linear·quadratic 계수, 원래 source axis 방향의 signed coordinate, quadratic residual의 signed coordinate와 RMS, residual cosine²를 뽑았다. 일곱 값을 tanh로 제한하고 28개의 2차 항과 intercept를 더했다. 높이별로 signed displacement 구간의 conditional relative risk를 ridge regression(계수 0.001, intercept 제외)으로 추정했다. 음수이면 projection을 선택한다. Cutpoints는 사전에 정한 −0.060, −0.025, +0.025, +0.060 mm다. 귀무 구간 주변과 양·음 방향을 다르게 볼 수 있다. Query-local 자료, other-site 자료에 같은 모델 형태와 규제를 적용했다. Fixed site가 query와 같은 경우에는 두 정책이 **완전히 같은 measurement와 local model**을 사용한다. [훈련](fit_route48.py) · [배포 구현](route48.py)

Source catalog 700–707로 학습하고 708–711로 19/31 mm를 비교한 뒤 전체 12개로 계수만 다시 적합했다. Source validation의 Brier는 always-original 0.165149, 추가 관측 없는 same-capacity profile model 0.159681, query-local에서 displacement 크기만 쓴 경우 0.159305, signed query-local 0.155067이었다. Fixed 19 mm는 0.156280, 31 mm는 0.156243이어서 31 mm를 동결했다. 두 고정 후보의 차이는 0.000037이며, 31 mm가 다른 환경에서도 가장 좋은 고정 위치라는 주장은 지원하지 않는다. 이 수치는 v47을 v48 state mixture로 재채점한 source validation으로 v47의 기존 primary endpoint와 분포가 다르다. [Frozen artifact](policy48.json) · [Source freeze](router_freeze48.sha256)

## Transfer sensitivity 해석

Fixed와 query-local의 성능 차이는 두 위치의 실제 response가 약 60% 확률로 공통 state를 공유하도록 한 generator에 의존한다. 나머지 40%에서는 independent state를 뽑으므로 실제 state 일치 비율은 일반적으로 60%보다 높다. 별도의 **secondary replay**는 다른 station의 공통 state 확률만 0·0.30·0.60·0.85·1.00으로 바꾸고, 같은 물체와 noise, unchanged target label을 사용한다. 두 가지 fixed decision을 채점한다. 첫 번째는 원래 60% 환경에서 학습한 remote-bin 계수를 고정한다. 두 번째는 transfer가 완전하다고 가정하고 source에서 학습한 local-bin 계수를 사용한다. 100%에서 station별 noise·gain·profile response는 여전히 다르다. 이 비교는 한 generator 가정에 대한 sensitivity이며 물리적 상한 또는 실제 장비의 spatial dependence를 추정하지 않는다. [재채점 코드](transfer48.py)

| Other-site common-state 확률 | Query-local Brier | Fixed Brier (frozen remote weights) | Local − fixed [각 95% CI] |
|---:|---:|---:|---:|
| 0% | 0.154652 | 0.156672 | −0.002020 [−0.002243, −0.001797] |
| 30% | 0.154652 | 0.156156 | −0.001504 [−0.001688, −0.001320] |
| 60% (primary) | 0.154652 | 0.155636 | −0.000984 [−0.001149, −0.000819] |
| 85% | 0.154652 | 0.155102 | −0.000450 [−0.000574, −0.000325] |
| 100% | 0.154652 | 0.154809 | −0.000157 [−0.000259, −0.000054] |

100%에서 fixed 계수를 완전 전달에 맞는 기존 source-local 계수로 바꾸면 fixed Brier는 **0.154623**, local−fixed는 **+0.000029 [−0.000055, +0.000112]**였다. 공간적으로 state가 완전히 공유되고 이에 맞는 calibration을 쓰는 환경에서도 query-local 전략이 우세하다는 근거는 없었다. 60% CPU replay에서 archived primary fixed 결정과 차이는 0건이었다. 구간은 12개 catalog에서 각 대비별로 산출한 보조 95% t interval이다. [수치 및 height별 결과](transfer48.json)
