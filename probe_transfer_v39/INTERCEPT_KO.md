# z, w 기반 정책의 해석과 single-intercept control

보조 회귀식의 계수는 (-0.00575117, -0.00311468, -0.00086904)다. a=z(w−0.325), |z|≤1, |a|≤0.675를 대입하면 Δ̂/a=b₁+b₂z+b₃a는 -0.002050 이하다. 따라서 Δ̂<0인 조건은 정확히 a>0이다. 즉 보조 정책은 항상 두 stick probability 중 더 큰 값을 채택한다. a=0이면 두 예측이 같다. 실제 전체 archive에서도 이 항등식을 수치 허용오차 2×10⁻¹⁵ 이내로 확인했다.

이 성질을 보고 v39 target별 결과를 읽기 전 추가 비교를 동결했다. V38의 전체 fixed 32.5% 예측에 log-odds intercept 하나를 더해 Brier가 가장 낮아지는 값을 정했다. 그 값 +0.159820를 모든 새 형상·물체에 공통 적용했다. V38의 평균 fixed forecast−incidence는 -0.017459였다. 새 데이터를 이용한 재학습은 없다. 이 비교는 GPU 실행 시작 이후에 추가한 보조 실험이다. [보조 계획](INTERCEPT_PLAN_KO.md) · [v38 fitting](intercept_fit39.json) · [시각](intercept_frozen_utc.txt)

| V39 진단 | Held-out Brier | 평균 forecast − incidence |
|---|---:|---:|
| fixed | 0.113638 | -0.006616 |
| optimizer | 0.113252 | — |
| route | 0.113339 | +0.001942 |
| global_intercept_fixed | 0.113838 | +0.010064 |

| 대조 | Δ Brier | 95% pointwise interval |
|---|---:|---|
| intercept − fixed | +0.000200 | [+0.000065, +0.000335] |
| route − intercept | -0.000499 | [-0.000604, -0.000390] |
| optimizer − intercept | -0.000586 | [-0.000781, -0.000377] |

| 보조 정책의 선택 | 고정값의 평균 forecast − incidence | 보조 정책 | Intercept 적용 |
|---|---:|---:|---:|
| optimizer | -0.018754 | -0.002552 | -0.002394 |
| fixed | +0.006974 | +0.006974 | +0.024012 |

같은 source에서 배운 단일 계수의 성능은 새 집단 전체의 calibration bias를 고치는 것만으로 얻는 효과를 보여준다. 보조 정책이 그보다 좋은 경우에도 두 예측 중 큰 값을 고르는 간단한 규칙과 완전히 같으므로 물성별 regime 식별 능력을 입증하지 않는다. 한편 두 방법은 같은 방향으로 높은 확률을 낼 수 있어도 물체별 probability 이동량은 다르다. Brier superiority, 평균 calibration residual 감소와 물성 정보의 이용을 각각 구분한다. [구간과 원표](intercept39.json)

Router와 single intercept는 모두 v38 label로 계수를 학습했다. 반면 고정 32.5% 및 continuous gain optimizer는 v38 label을 학습에 사용하지 않은 사전 비교 방법이다. 따라서 새로운 v39 target label의 사용량은 네 방법 모두 0이지만, 기존 label 사용량까지 동일한 대조는 router와 intercept 사이의 비교다.

독립 환자 표본에서 intercept·slope recalibration과 더 복잡한 모델 갱신을 대조한 선례가 있다. Steyerberg et al.은 TIMI-II(3,339명)에서 개발한 예측을 GUSTO-I(40,830명)로 옮겨 갱신 표본 크기와 방법을 비교했다. 한 계수만 추가하는 baseline을 시험하는 방법론에 해당하며, 물체 충돌에서 어느 방식이 이길지는 우리 데이터로 평가해야 한다. [Steyerberg et al., Statistics in Medicine](https://onlinelibrary.wiley.com/doi/10.1002/sim.1844)
