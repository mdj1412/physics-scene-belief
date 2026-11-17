# v50 사전 계획 · 소수의 paired calibration probe로 보정 방향을 옮길 수 있는가

**Primary hypothesis.** 같은 환경 내 calibration object와 deployment object가 weak-to-full polarity law를 공유할 때, K=4회의 paired calibration trial에서 유지율을 추정해 query-local direct의 보정 부호를 선택하면 같은 예산의 frozen direct, fixed-site direct, sign-blind direct보다 held-out Brier loss를 줄일 수 있다. 매우 작은 K, κ≈0.5, calibration 이후 환경의 polarity가 바뀌는 경우는 예상 실패 조건이다.

## 관측 계약

12개의 새로운 catalog와 새로운 analytic contour family 4개, surface 2종을 생성한다. 각 catalog에 4 surface/law environment가 있으며, 각 environment에 대해 별도 prototype support에서 추출한 calibration object와 배포 object를 분리한다. Factual acquisition과 original decoder는 동결한다. Calibration stage에서 새로운 물체를 약하게 눌러 signed displacement를 측정하고 같은 위치의 별도 reference full-load contact로 signed displacement를 측정한다. 각 contact/reset 및 reference 측정 비용을 모두 산입한다. 배포 물체의 full-load label이나 진짜 변형 부호는 policy에 제공하지 않는다. Weak-only contact로 weak-to-full retention을 식별할 수 있는지는 별도의 식별성 문제다.

## 설계와 분석

- κ ∈ {0,.25,.5,.75,1}는 active일 때 weak 부호가 full 부호로 유지될 확률이다. Inert는 유지된다. ρ ∈ {0,.5,1}은 off-site weak state가 query weak state를 복사할 확률이다. 나머지는 같은 prior에서 독립적으로 draw한다. 이 두 조건은 policy 입력에서 숨기고 primary 배포 성능은 15개 조건에 같은 가중치를 둔다. 실제 환경의 κ 분포를 추정했다고 해석하지 않는다.
- Paid prefix를 K ∈ {0,1,4,8}로 바꾸고 같은 session에서 prefix를 중첩한다. 독립적인 calibration 반복을 catalog·environment·κ별로 평균한다. Posterior mean이 0.5 미만일 때 d의 부호를 뒤집고 동률은 유지한다. K=0에서 hard policy는 frozen direct다. Soft policy는 posterior mean으로 p(d)와 p(−d)를 혼합한다. Calibrated fixed는 같은 calibration으로 31 mm fixed-site의 부호를 정한다. 두 정책의 원래 source calibration coefficients 및 location은 v48에서만 선택했다.
- **사전 지정 Primary 비교**는 K=4의 query-local hard sign policy와 (1) frozen query-local direct, (2) frozen fixed direct, (3) 같은 K로 부호를 정한 fixed direct, (4) source-only로 fit한 unsigned direct, (5) p(d)와 p(−d)의 단순 평균이다. 동일한 calibration prefix, 하나의 deployment preload 및 reset을 각 정책에 제공한다. Calibration을 읽지 않을 수 있는 baseline의 절약 가능성도 따로 계산한다.
- 모든 primary 차이는 catalog별 paired 평균으로 만든 뒤 12개 catalog에 대해 Student t + Bonferroni simultaneous 95% CI (5개)로 판정한다. 가설별 음수면 이득이다. 15개 cell별 승패는 direct local hard와 주요 네 comparator(frozen local, frozen fixed, calibrated fixed, fitted unsigned)에 대해 60개 simultaneous contrast를 적용한다. K별 경향, soft policy, oracle κ, 같은 calibration을 다른 κ의 배포에 쓰는 교차 평가는 secondary다. Target에 맞춰 threshold를 다시 고르지 않는다.
- Estimator 자체는 κ 대비 bias/MAE/RMSE, sign 선택률을 별도로 검증한다. κ=.5에서 어느 한 부호를 정답으로 정의하지 않는다. Hypothesis와 sensor model의 독립성을 명시한다. Paired vs unpaired object, 다른 surface session으로 전이 가능한지와 진짜 κ를 알려준 diagnostic의 차이를 구별한다.
- GPU end-to-end p90은 실제 하나의 GPU에서 action·구매한 readout·원래 posterior·보정까지 측정한다. Calibration posterior 업데이트의 물리적 비용과 계산 비용, 재사용하는 단위를 명시한다. 실제 로봇이 없으므로 contact와 reset 횟수를 먼저 비교하고 물리 시간은 명시적 시나리오로만 보고한다. 계산은 Slurm compute node에 한정한다.
