# 물리 및 사전 예측의 독립 검산

12개 panel의 576개 archive에서 label 2,654,208개를 displaced contour의 수치 미분과 독립 NumPy impulse inequality로 검산했다. 불일치가 없었다. 모든 예측으로 재구성한 Brier의 최대 오차는 0.00e+00였다. Base·hollow·bulge 사이의 factual evidence, 선택한 관측, 네 번의 calibration, 명목 label은 물체별로 일치했다.

| Offset (mm) | 같은 높이에서 contour만 바꿨을 때 flip, % [panel 95%] | Privileged−original Brier (변형 물체만) | Contour label 개입 항 | 기본물체 placement 항 |
|---:|---:|---:|---:|---:|
| -1 | 19.14 [18.38, 19.91] | +0.004547 | +0.005090 | -0.000159 |
| -0.5 | 14.84 [14.09, 15.60] | +0.003790 | +0.004264 | -0.000090 |
| -0.25 | 9.78 [9.15, 10.42] | +0.003407 | +0.003838 | -0.000048 |
| -0.125 | 5.63 [5.18, 6.07] | +0.002354 | +0.002755 | -0.000016 |
| +0 | 0.00 [0.00, 0.00] | -0.000384 | +0.000000 | +0.000000 |
| +0.125 | 9.27 [8.70, 9.83] | +0.005845 | +0.006216 | +0.000012 |
| +0.25 | 13.36 [12.71, 14.01] | +0.007982 | +0.008341 | +0.000025 |
| +0.5 | 20.90 [20.31, 21.49] | +0.012462 | +0.012810 | +0.000035 |
| +1 | 20.42 [19.94, 20.91] | +0.011075 | +0.011387 | +0.000072 |

같은 물체·같은 offset에서 base와 hollow/bulge를 비교했다. 마지막 두 열의 합이 명목 높이 대비 Brier 차이의 변화다. 정확한 식은 −2·mean[(p_priv−p_orig)·(Y_variant,δ−Y_base,δ)]와 −2·mean[(p_priv−p_orig)·(Y_base,δ−Y_base,0)]다. 예측은 offset을 보지 않아 이 분해가 성립한다.

관측된 flip 중 impulse demand만 바꿔 재현한 비율은 99.16% [99.11%, 99.21%], 속도에 따른 마찰 한계만 바꿔 재현한 비율은 1.35%다. 두 비율은 같은 충돌을 중복 포함할 수 있다. 각 경로 모두 단독으론 재현하지 못한 비율은 0.84%다. 구간은 panel bootstrap이다.

Label을 보기 전 7점 profile의 apparent slope와 실제 tangent 변화로 예측한 projection 손익 부호는 비영 조건 32개 중 31개에서 Brier 점추정과 일치했다. 항상 손해로 두면 28개가 맞았다. 이 보조 비교는 deformed contour만 포함하며 family·offset은 서로 독립적인 32개의 trial이 아니다. 실제 offset과 simulator 기하를 알아야 하는 예측이어서 blind policy accuracy가 아니다.

| Shape / 명목 높이 | 8개 offset 부호 일치 | 예측과 달랐던 signed offset (mm) |
|---|---:|---|
| fold / 19 | 8/8 | 없음 |
| fold / 31 | 8/8 | 없음 |
| ripple / 19 | 7/8 | +1 |
| ripple / 31 | 8/8 | 없음 |

[동결된 사전 예측](geometry_forecast43.json) · [동결 기록](geometry_freeze43.sha256) · [검산 수치](physics_results43.json) · [검산 코드](physics_results43.py)
