# v58 · 두 센서의 readout을 통한 physical persistence와 sensor error의 분해

## 어떤 비교가 원인을 구별하는가

한 물체에서 동일한 2.0 N reference 방향과 impact 방향을 두 독립 계측 채널이 동시에 읽는다. 각 sensor에서 두 부호가 어긋나면 mismatch다. 같은 session의 두 물체가 모두 mismatch일 때, 동일 sensor를 사용한 두 판독(0→0 또는 1→1)을 **same-sensor**, 다른 sensor를 사용한 두 판독(0→1 또는 1→0)을 **cross-sensor**로 구분했다. 두 물체에서 모두 변형 부호를 읽은 경우만 분모에 넣었다. 두 sensor가 보는 물리적 contact와 missing 여부는 동일하다. 새 24개 catalog, 48개 surface instance 각각에서 2,048회 독립 pilot을 생성해 factorial cell마다 98,304번의 8-impact episode를 확보했다. Episode마다 여덟 sacrificial specimen 모두 서로 다른 개체이고 primary 분석에서 sensor 측정을 위해 추가 contact를 사지 않았다. 이 pilot은 analytic instrument simulation이다. [사전 계획](PILOT_PLAN_KO.md) · [생성기](pilot58.py)

**Physical intervention.** A에서 session mismatch 확률 e는 0과 0.5를 각각 42%, 58% 확률로, B에서는 0.25와 0.5를 84%, 16% 확률로 취한다. 두 환경 모두 latent q=0.29이다. 4×2에서 persistence를 켜면 두 물체의 e가 같고 끄면 매 물체마다 e를 다시 뽑는다. 같은 e에서 두 물체의 mismatch 여부는 독립으로 뽑는다. 8×1은 각 물체에 별도 session을 쓴다. [생성 코드](pilot58.py)

**Sensor intervention.** 각 sensor에서 session별 확률 h로 상대적 변형 방향의 readout이 한 값 Z에 고정되는 latch를 넣었다. Z는 확률 q로 mismatch를 가리키므로 이 오류가 있어도 marginal mismatch는 같다. 두 sensor의 latch와 Z는 독립이다. 정상일 때는 물리 부호를 따르고, reference 및 impact의 마지막 sign readout에는 각각 별도로 2% 오류를 주었다. 미리 정한 두 조건 h=0, h=26.582%와 physical persistence off/on을 완전히 교차시켰다. 유효 판독의 noisy mismatch 확률은 어느 조건이나 30.6464%다. 센서의 electrical bias를 실측해 보정한 수치는 없다. [동결한 가정](PILOT_PLAN_KO.md)

**물체 순서.** 독립적인 393,216개 fair coin으로 specimen ID 두 개의 chronological position을 정했다. 원래 첫 물체가 먼저 배정된 비율은 50.075%다. 4×2 각 cell에서 order에 따른 cross-sensor double mismatch 차이의 24-catalog bootstrap interval은 0을 포함했다. 이는 구현된 exchangeable simulator의 audit이다. Wear, reset 실패나 첫 impact가 다음 물체의 물리 상태를 바꾸는 carryover를 실물에서 배제할 수는 없다. [순서 및 분모](pilot_results58.json)

## 물리 상태의 지속은 cross-sensor double mismatch를 높였다

다음 표는 **양쪽 물체를 읽었을 때의** double mismatch 백분율이다. 어느 cell에서나 393,216개 가능한 group 중 98,233개가 complete였고 98,304개 episode 중 31,153개에는 complete group이 없었다.

| 4×2의 원인 | A same | A cross | B same | B cross |
|---|---:|---:|---:|---:|
| 두 요인 off | 9.33% | 9.33% | 9.41% | 9.39% |
| Sensor latch만 on | 14.00% | 9.37% | 14.04% | 9.43% |
| Physical persistence만 on | 14.45% | 14.47% | 10.14% | 10.13% |
| 두 요인 on | 17.75% | 12.12% | 14.58% | 9.84% |

A에서 same-sensor double mismatch만 보면 14.00%와 14.45%가 비슷하다. Cross-sensor 통계에서는 9.37%와 14.47%로 나뉜다. 각 position에서의 observed marginal 곱을 빼면 sensor만 on일 때 A/B의 cross covariance는 각각 +0.005 [−0.116,+0.132], +0.021 [−0.090,+0.139] percentage points였다. 둘 다 on일 때는 +2.738 [+2.607,+2.872], +0.433 [+0.336,+0.528]이었다. 8×1의 인접한 독립 물체를 같은 방식으로 짝지으면 cross covariance의 모든 interval이 0을 포함했다. reported interval은 4,000회 paired catalog bootstrap의 pointwise 95% 구간이다. [모든 factorial cell·clean readout 결과](pilot_results58.json)

## 두 원인이 함께 있을 때의 정량 귀속

추가 분석에서는 latent 물리 label을 보지 않고 **같은 물체를 두 sensor가 읽은 covariance** W를 구했다. 다른 물체를 다른 sensor가 읽은 covariance를 C라 하자. q는 latent mismatch 확률, d는 한 sensor의 reference와 impact 사이에서 순 부호 반전이 발생할 확률 0.0392, t=1−2d다. 두 sensor가 독립이고 채널이 stable이면 W=t²(1−h)²q(1−q), C=t²(1−h)²rho Var(e)이다. 따라서 **C/W는 진짜 두 물체 간 mismatch의 intraclass correlation**을 추정한다. 이 비율은 sign error와 latch에 따른 감쇠를 상쇄한다. Latch 비율 자체를 추정할 때는 q와 d의 값을 알고 있어야 한다. [관측 기반 분석](attribution58.py)

| 두 요인 on | 실제 physical ICC | 관측 추정 ICC [95% CI] | 관측 추정 h [95% CI], 설정 26.58% | Same-sensor covariance | 추정 physical 기여 | 추정 sensor 기여 |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.2958 | 0.2909 [0.2782, 0.3039] | 26.64% [26.20, 27.08] | 8.360 pp | 3.732 pp | 4.659 pp |
| B | 0.0408 | 0.0459 [0.0356, 0.0561] | 26.55% [26.16, 26.97] | 5.174 pp | 0.589 pp | 4.644 pp |

Physical 기여는 C/(1−h), sensor 기여는 h t² q(1−q)로 추정했다. 두 합과 실제 same-sensor covariance 사이에는 A −0.032 pp, B −0.059 pp의 표본 잔차가 남는다. 이 표는 환경별 많은 pilot에서 얻은 population-level 분해다. 한 사용자가 산 단일 8-impact pilot에서 두 비율을 이 정밀도로 추정한다는 뜻으로 해석할 수 없다. CI는 24개 독립 catalog에 대한 5,000회 bootstrap이며 pointwise이다. [관측 moment와 전체 CI](moment_results58.json)

## 두 독립 센서라는 가정이 깨질 때

실제 A처럼 physical e가 완전히 공유되는 경우와, 물리 mismatch는 완전히 독립이지만 확률 **29.5775%**로 그 session의 두 sensor가 모두 같은 가짜 상태를 읽는 경우를 구성했다. 후자의 session-wide readout Z는 확률 0.29로 mismatch를 낸다. 그러면 두 경우에서 기저 mismatch pair의 (00,01,10,11) 확률이 정확히 (0.565,0.145,0.145,0.145)로 같다. 동일한 missingness 및 per-sign noise를 적용한 뒤 가능한 관측 pair 25개의 확률도 모두 같았다. 물체 순서 무작위화 또는 동일 sensor 두 개만 추가하는 것으로는 이 common-mode 대안을 배제할 수 없다. 실제 장비에서 서로 다른 물리량을 읽는 camera와 gauge, 독립 clock 및 drift audit이 필요한 이유다. [Exact counterexample](pilot_results58.json)

## 접근 경계와 재현

서버의 `dual_sensor_v58/pilots`에는 policy가 볼 수 있는 여덟 물체의 2-channel mismatch code만 저장했다. `pilot_audit`에는 별도 처리한 두 reference sign, 두 impact sign, 누락과 순서 coin, audit count가 있다. 실제 online timing은 raw sign에서 다시 계산할 수 있다. 48개 archive에는 각 파일의 SHA-256 digest가 있다. 모든 saved code를 raw 부호에서 복원해 원래와 완전히 일치함을 확인했다. [Freeze](pilot_freeze58.sha256) · [파일별 digest](pilot_results58.json)
