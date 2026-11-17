# 국소 2차식으로 요약한 paid geometry는 어느 예측력을 보존하는가?

기존 terrace/dimple archive에서 **세 개의 국소 계수를 사용한 readout의 평균 Brier는 0.139355**, 7개 원좌표를 사용하면 0.139455였다. 같은 세 source seed와 CPU 학습 환경으로 비교했을 때 차이의 simultaneous 95% CI는 [−0.000593, +0.000392]였다. Source deformation 축을 지우고 6개 직교 좌표를 유지한 readout보다 세 계수의 예측 손실이 0.027077 낮았다. 같은 관측에서 어느 방향을 남겼는지가 차원의 개수보다 중요했던 한 사례다. 이번 수치를 결정하면서 기존 archive와 v66 main readout의 결과는 이미 알고 있었다. [사전 계획](moment_plan.md) · [서버 집계](moment_results.json)

## 무엇을 압축했는가?

Nominal contact height 중심에서 −6, −4, −2, 0, 2, 4, 6 mm에 예약된 7개의 관측값에 q(x)=b₀+b₁x+½b₂x²를 least squares로 적합했다. 세 좌표는 fitted center ordinate(mm), slope, second derivative(mm⁻¹)다. 마지막 값은 유한 구간의 quadratic fit이며 실제 pressure patch의 curvature 또는 contact area를 측정하지 않는다. Factual 11채널, nominal height와 hard-projected physics forecast의 log odds는 세 방법 모두에 제공된다. 따라서 **추가 geometry 입력만 세 계수로 줄었다.** 참 inertia, contact Jacobian, force, solver label은 입력에 없다. Paid probe를 받은 trial에서는 세 방법이 같은 예측을 유지한다. 물리적 feature 선택은 Fazeli 등의 예에서 동기를 얻었다. 저자들의 contact inertia·momentum GP와 다른 baseline이다. [Fazeli et al. §§3–4](https://proceedings.mlr.press/v78/fazeli17a/fazeli17a.pdf) · [구현](moment_readout.py)

세 방법 모두 64-unit SiLU residual head, AdamW, batch size 4,096, source fit/validation catalog 8/4, 같은 seed 66101–66103을 사용했다. 3개 계수 또는 6개 잔여축은 zero padding으로 7개 profile 입력 폭을 맞췄다. 공통 source validation에서 각 seed별 epoch를 선택했다. 모든 평가에서 seed별 loss를 평균했다. 두 contrast에는 12개 paired catalog 기준 Bonferroni simultaneous 95% t interval을 적용했다. CPU와 GPU는 같은 정수 seed라도 shuffle·연산 특성이 달라 full과 dropped도 CPU에서 다시 fit했다. Complete risk의 CPU−GPU 차이는 full +0.000003, dropped −0.000152였다. [학습 기록](moment_fit.json) · [봉인 예측](moment_forecasts.json)

| 표현 | 전체 Brier | Query-only Brier | Terrace 전체 | Dimple 전체 |
|---|---:|---:|---:|---:|
| Moment | 0.139355 | 0.131301 | 0.131462 | 0.147248 |
| Full | 0.139455 | 0.131412 | 0.131343 | 0.147567 |
| Dropped | 0.166432 | 0.161438 | 0.174095 | 0.158768 |

Moment−full −0.000100 [−0.000593,+0.000392], moment−dropped −0.027077 [−0.029125,−0.025028]. V46에서 사전 등가성 기준은 두지 않았다. Seven-point polynomial subspace와 source axis 간 squared overlap은 19 mm에서 0.669, 31 mm에서 0.825였다. 그중 linear slope 성분은 각각 0.621, 0.566이다. 따라서 저차 표현이 stick/slip을 잘 구별했던 가능한 이유로 후보 축의 일부분이 slope에 함께 남은 점을 들 수 있다. 그 overlap 수치로 Brier 차이의 얼마를 설명하는지는 식별할 수 없다. [Outcome과 독립적인 축 검산](moment_axis.json)

## 미관측 두 contour family로 옮겼을 때

위 v46 결과를 확인한 뒤, 새 wendland/dispersive에서 **별도의 secondary transfer 비교**를 사전 지정했다. 기존 source의 CPU-trained weights 아홉 개를 그대로 적용했다. Label directory가 없을 때 전체 12개 catalog·576개 cell의 예측을 작성하고 SHA-256 manifest를 봉인했다. 봉인 직후(01:32:06 UTC) independent solver 담당자에게 label 생성을 요청했다. 검정한 두 contrast는 moment−full과 moment−dropped다. 전자의 동시 CI 상한이 +0.002 아래면 설계상의 noninferiority, 후자의 상한이 0 아래면 superiority로 두었다. +0.002는 설계자가 정한 absolute Brier margin이고 현장 비용에서 추정한 값은 아니다. [추가 사전 계획](moment_lock_plan.md) · [예측 해시](moment_locked_forecasts.json) · [봉인 시각](moment_lock_sealed.utc)

| 새 모집단 | Moment | Full | Dropped | Moment−full |
|---|---:|---:|---:|---:|
| 새 12 catalog 평균 | 0.139439 | 0.135853 | 0.181857 | **+0.003586** |
| Wendland | 0.169709 | 0.161686 | 0.185925 | +0.008023 |
| Dispersive | 0.109169 | 0.110020 | 0.177789 | −0.000852 |

전체 구매를 포함한 moment−full의 simultaneous 95% CI는 **[+0.003261,+0.003910]**으로 양 끝이 +0.002보다 높았다. 새 family의 pooled mean에서 세 계수로 줄인 비용이 사전 허용치를 초과했다. Moment−dropped는 **−0.042418 [−0.044018,−0.040818]**로 개선이 재현됐다. Family별 차이는 기술용이며, 단일 surface·기기·실물 일반화에 대한 추정량은 얻지 않았다. 현 결과는 full이 보유한 추가 좌표가 이 source-trained head에서 도움이 된다는 근거다. 특정 고차 물리 성분의 인과 효과나 Bayes-optimal information loss 크기는 식별하지 않는다. [서버 집계](moment_locked_results.json) · [분리된 forecast·score 코드](moment_lock.py)
