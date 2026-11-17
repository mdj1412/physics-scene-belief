# v50 · 새 물체에서 frozen decoder forecast와 GPU 비용

## 어떤 측정을 실제로 사용했나

Factual acquisition은 v43의 frozen Sobol bank와 per-environment budget을 사용한다. 128개 물체 가운데 115개는 optical query profile, 13개는 기존 physical probe를 구매한다. 미선택 관측을 mask한 뒤 original decoder가 posterior stick probability를 계산한다. 이 공통 acquisition에 v50의 calibration prefix 및 별도 약한 deployment preload를 더한다. K개의 calibration pair에서 얻은 두 displacement만 estimator에 들어간다. Query를 채점할 full-load label, 환경의 진짜 κ, 참 deformation state와 물체 family 이름은 decision feature에 포함하지 않는다.

Signed 보정은 v47 source에서 학습하고 v48에서 동결한 계수를 썼다. 같은 물체에서 local은 query 높이, fixed는 31 mm를 측정한다. Fixed가 19 mm query에서 쓰는 계수는 v48 source의 remote calibration이다. Hard rule은 session posterior mean이 0.5 미만이면 관측한 scalar의 부호를 한 번 뒤집는다. Soft rule은 같은 scalar에서 두 부호별 예측 확률을 얻고 posterior mean을 가중치로 평균한다. Hard와 soft 모두 original physics decoder는 한 번만 실행한다. [정책 구현](policy50.py) · [Retention estimator](estimator50.py)

두 sign-invariant 대조를 구분했다. Symmetric policy는 두 부호를 적용한 최종 확률을 1/2씩 평균한다. Fitted unsigned policy는 original 확률 p를 변환한 x=2(p−0.5)와 정규화한 절댓값 z=min(|d|/0.084,2)를 써서 1,x,x²,z,xz를 ridge regression으로 직접 학습한다. 학습에는 v47 source 12개 catalog만 사용했고, v48과 같은 0.001 regularization과 높이별·기존 구매 arm별 계수를 사용했다. 원래 signed coefficient에서 일부 항을 임의로 제거하면 sign-invariant predictor의 재학습 이득이 빠질 수 있어 이 대조를 별도로 넣었다. [Source-only 학습](unsigned50.py) · [Frozen artifact](unsigned_source50.json)

## 서버 실행과 독립 검산

두 GPU allocation의 memory 사용량, utilization, compute process를 확인하고 빈 장치에만 guarded worker를 실행했다. 두 worker가 새로운 12개 catalog를 나눠 factual acquisition과 original posterior를 계산했다. 같은 31 mm station을 지정했을 때 local과 fixed의 positive·negative 예측이 모든 항목에서 bitwise 일치하고, fitted unsigned는 d와 −d에서 bitwise 일치함을 확인했다. 별도 코드로 direct48의 7-term design matrix와 sign-invariant 5-term contraction을 다시 계산해 5개 경로별 2,359,296개 예측을 검산했다. [Forecast 코드](policy50.py) · [독립 검산](audit_policy50.json) · [실험 파일 checksum](policy_freeze50.sha256)

## GPU 시간 측정 방법

한 장의 RTX PRO 6000 Blackwell에서 K=4를 사용했다. 12개 catalog마다 512개 물체(4개 독립 session × 128개), noise realization 1개를 처리했다. 순서를 무작위화한 일곱 방법을 warmup 한 번과 4회 반복으로 교차 실행했다. 각 방법은 총 48개 timed batch를 가진다. 다른 transfer 법칙을 가진 session도 같은 batch로 처리할 수 있게 실제 paired displacement 4쌍을 각 session에 전달한다. 방법이 이를 쓸 때는 session별 estimator를 매번 다시 계산해 cold-session overhead를 포함했다. 여기서 cold는 새 환경 prefix posterior가 캐시되지 않았다는 뜻이고, CUDA context와 model template은 모든 방법에서 warm 상태다.

시작과 끝에서 GPU synchronize하고 factual action, 구매한 관측 구성, original posterior, host의 κ 추정과 확률 보정을 포함한 전체 elapsed time을 기록했다. Frozen·sign-invariant 방법은 동일한 접촉 예산을 지출하는 primary 비교에서도 불필요한 κ 계산을 생략한다. 가상 접촉 생성 및 실제 장비의 접근·가압·재설정은 계산시간 외부에 있다. 기간 간 clock drift가 점추정을 바꾸지 않도록 paired catalog block 16,000회 bootstrap으로 p90 차이의 pointwise interval을 구했다. 다른 K의 host estimator 시간도 4-session별로 분리해 측정했다. [GPU benchmark](bench50.py)

## Primary timing 결과

| K4 방법 | p90, ms / 512 objects | Local hard에서 이 방법을 뺀 p90 차이 [pointwise 95% CI], ms |
|---|---:|---:|
| Local hard | 470.26 | 기준 |
| Frozen local | 463.56 | +6.70 [−2.28, 15.85] |
| Frozen fixed | 464.44 | +5.82 [−10.76, 14.08] |
| Fixed hard | 461.68 | +8.58 [0.85, 15.21] |
| Fitted unsigned | 462.15 | +8.11 [−0.98, 16.00] |
| Symmetric unsigned | 462.76 | +7.50 [−4.04, 15.80] |
| Local soft | 462.32 | +7.94 [−1.51, 15.47] |

네 session 전체가 다른 environment에서 시작하는 estimator의 host-only p90은 K0·1·4·8에서 각각 0.00268, 0.04362, 0.04543, 0.04805 ms였다. 이 범위에서는 κ 추정보다 공통 original decoder가 전체 deployment 시간을 지배한다. Local hard와 fixed hard는 같은 수의 estimator와 decoder를 실행하는데도 첫 crossover에서 p90 차이가 나타났다. 따라서 이 차이를 방법 고유의 compute cost로 해석하기 전 별도 순서 통제를 실시했다. [Primary 결과](timing50.json)

## 순서를 균형화한 secondary crossover

같은 GPU·12개 catalog·K=4·물체와 sensor noise를 그대로 사용했다. 각 catalog에서 4×4 Williams design을 적용해 모든 정책이 각 순서를 한 번씩 차지하게 하고 바로 앞 방법도 균형화했다. Method label과 row 순서는 catalog별로 무작위화했다. 정책당 48개 timed batch다.

| 방법 | p90, ms/512 objects | Local hard에서 이 방법을 뺀 값 [95% CI], ms |
|---|---:|---:|
| local_hard | 469.92 | +0.00 [+0.00, +0.00] |
| local_frozen | 465.53 | +4.39 [+0.37, +12.64] |
| fixed_hard | 470.36 | -0.44 [-14.81, +9.56] |
| local_soft | 465.83 | +4.09 [+0.07, +12.43] |

Local hard와 fixed hard 차이는 primary +8.58 ms에서 secondary −0.44 ms로 바뀌고 후자의 interval에 0이 포함됐다. 동일 estimator를 사용하는 두 정책 사이에 일정한 latency 차이가 있다는 근거는 재현되지 않았다. Frozen local·local soft와의 비교는 두 실험 모두 local hard가 더 느린 점추정이지만 CI가 0을 제외한 것은 secondary뿐이다. 따라서 latency 우월성이나 비열등성을 주기여로 해석하지 않았다. [Balanced 결과](balance50.json) · [측정 코드](balance50.py)
