# v42 실제 물리 event와 숨긴 높이의 비용

서버 node56의 CPU에서 총 15,925,248개 저장 stick/slip label을 별도의 NumPy rigid collision 식으로 다시 계산했다. 불일치는 0이었다. Calibration probe의 네 속도 관측과 factual evidence도 seed에서 복원했다. 최대 절대오차는 factual 4.44e-16, 첫 세 probe 5.55e-16, 네 번째 6.66e-16였다. 모든 panel의 passive projection은 독립 covariance eigensolver와 대조했다. [독립 검산 수치](physics_audit42.json) · [물리 구현](physics_audit42.py)

새 catalog의 contact support는 고유 prototype 393,216개, passive profile 물체는 1,536개였다. V41 training에 들어간 이전 12개 panel까지 포함한 exact latent tuple 789,504개에 중복이 없었다. 새 실험의 deployment draw 12,288회에는 서로 다른 prototype 11,924개가 실제 사용됐다. 동일 draw가 offset·shape·law에 대응되므로 Brier와 flip interval의 독립 단위는 여전히 catalog panel 12개다.

## 같은 물체에서 실제로 무엇이 바뀌었나

각 행에서 물체와 contact law를 고정했다. 같은 실제 높이에서 base가 slip인데 변형된 물체가 stick인 경우, 그리고 그 역방향을 따로 세었다. 두 비율을 합한 것이 contour에 의한 event flip이다. Base의 nominal 대비 flip은 contour lobe가 없어도 접촉 위치 변화만으로 결과가 바뀌는 비율이다. 변형은 hollow·bulge를 같은 비중으로, 나머지 설계 조건과 panel도 동일하게 가중했다.

| 실제 − nominal, mm | Base slip→변형 stick | Base stick→변형 slip | Contour flip 합 [개별 95% CI] | 변형 자체의 nominal 대비 flip | Base 자체의 nominal 대비 flip |
|---:|---:|---:|---:|---:|---:|
| -1.000 | 9.92% | 13.74% | 23.66% [23.18%, 24.14%] | 23.66% | 0.59% |
| -0.500 | 6.68% | 8.18% | 14.86% [14.53%, 15.20%] | 14.86% | 0.29% |
| -0.250 | 3.95% | 5.00% | 8.95% [8.69%, 9.21%] | 8.95% | 0.14% |
| -0.125 | 2.31% | 2.80% | 5.11% [4.95%, 5.26%] | 5.11% | 0.07% |
| +0.000 | 0.00% | 0.00% | 0.00% [0.00%, 0.00%] | 0.00% | 0.00% |
| +0.125 | 5.17% | 6.78% | 11.96% [11.68%, 12.23%] | 11.96% | 0.06% |
| +0.250 | 11.01% | 10.65% | 21.66% [21.34%, 21.99%] | 21.66% | 0.13% |
| +0.500 | 13.33% | 16.55% | 29.88% [29.55%, 30.22%] | 29.80% | 0.26% |
| +1.000 | 16.04% | 19.40% | 35.45% [35.02%, 35.88%] | 35.26% | 0.50% |

## 실제 높이의 방향을 숨겼을 때 남는 Brier floor

동일한 물체에서 부호가 +인 경우와 −인 경우를 정확히 절반씩 섞고, 두 경우에 같은 nominal evidence를 제공했다. True geometry·friction·offset의 절대 크기를 아는 관찰자도 그 부호가 없으면 실제 label을 항상 맞힐 수 없다. 두 경우의 label이 같으면 최적 Brier가 0이고 서로 다르면 0.25이므로, 이 조건의 최소 Brier는 0.25 × sign flip rate다. 아래에서는 실제로 변형한 두 contour만 이용했다. 이 이상적 하한은 측정값 자체에 부호가 남는 실제 장비나 branch별 유한 표본 calibration uncertainty에 그대로 적용할 수 없다.

| Shape·nominal height | abs(δ)=0.125 mm | 0.25 mm | 0.5 mm | 1 mm | 아홉 offset을 균등 혼합한 floor |
|---|---:|---:|---:|---:|---:|
| flare / 19 | 0.03107 | 0.06635 | 0.07651 | 0.05126 | 0.09846 |
| flare / 31 | 0.02177 | 0.03889 | 0.06574 | 0.09624 | 0.06734 |
| roll / 19 | 0.03134 | 0.05302 | 0.06981 | 0.06406 | 0.08363 |
| roll / 31 | 0.00488 | 0.01083 | 0.02317 | 0.03733 | 0.05156 |
| deformed | 0.02226 | 0.04227 | 0.05881 | 0.06222 | 0.07525 |

아홉 offset 혼합의 floor는 각 물체에서 아홉 label의 stick 비율 m을 구해 m(1−m)을 평균한 값이다. 변형된 물체에 대한 이론값이며, 각 방법의 전체 held-out Brier에는 다른 물체·가설·보정 오차도 포함된다. 두 값의 차이가 sensor 높이 측정으로 즉시 얻는 개선량은 아니다.

## Geometry만 사용한 projection 손익 예측

GPU run을 시작한 뒤 결과를 열기 전에 생성식으로 예측한 보조 분석이다. 이 시점과 36개 예측값을 따로 기록했다. Nominal 7점 profile에서 2차 곡선의 기울기 s를 추정하고, 실제 접촉점의 기울기 변화 t를 계산했다. t²−(t−s)²가 양수면 추정한 방향을 지운 decoder가 손해일 것으로 정했다. 이는 friction 또는 stick label을 입력받지 않는 간단한 tangent 예측이며, 물리 가설의 posterior 전체나 셀별 Brier 크기를 예측하지 않는다. [동결 예측](slope_forecast42.json) · [생성 UTC](slope_forecast_frozen_utc.txt)

| Shape·height | Apparent slope s | 손해를 예상한 signed offset, mm | 실제 손해인 signed offset, mm | 부호 일치 (비영 offset 8개) |
|---|---:|---|---|---:|
| flare / 19 mm | -0.06017 | -1, -0.5, -0.25, -0.125, +0.125, +0.25, +0.5, +1 | -1, -0.5, -0.25, -0.125, +0.125, +0.25, +0.5, +1 | 8/8 |
| flare / 31 mm | +0.04440 | +0.125, +0.25, +0.5, +1 | +0.125, +0.25, +0.5, +1 | 8/8 |
| roll / 19 mm | +0.06534 | -1, -0.5, -0.25, +0.125, +0.25, +0.5, +1 | -1, -0.5, -0.25, -0.125, +0.125, +0.25, +0.5, +1 | 7/8 |
| roll / 31 mm | -0.10246 | -1, -0.5, -0.25, +0.25, +0.5, +1 | -1, -0.5, -0.25, -0.125, +0.25, +0.5, +1 | 7/8 |

비영 signed offset의 held-out Brier 점추정에서 예측과 부호가 같은 경우는 30/32였다. 모든 조건에서 손해라고 답하면 27/32다. 하나의 설계에서 여러 셀을 반복 비교했으므로 이 수치를 32개 독립 사례의 일반화 정확도로 해석할 수 없다. 실제 손해 판정의 다중 비교 보정 결과와 비교할 때는 본 보고서의 사전 기준을 사용한다.

각 셀에서 estimated−original Brier 차이 D가 μ−2E[(p_est−p_orig)Y]임을 이용해, 실제 variant 대신 같은 위치의 base label로 채점하는 counterfactual 대조를 검산했다. 두 대조 사이의 차이는 −2E[(p_est−p_orig)(Y_variant−Y_base)]와 부동소수점 오차 안에서 일치한다. 이 항은 단순 높이 이동만으로 설명되지 않는 contour-specific event 변화의 방향과 크기를 기록한다. [Panel별 비교](physics_audit42.json)

![Flare의 nominal 7점 profile과 actual contact의 local slope 비교](geometry42.png)

왼쪽은 bulge−base의 실제 변위와 nominal 7점에 적합한 quadratic이고, 오른쪽은 같은 contour의 actual contact slope 변화다. 31 mm flare에서는 nominal profile로 얻은 apparent slope가 양수지만 actual offset이 음수이면 진짜 slope 변화는 음수가 된다. 19 mm flare의 +1 mm에서는 slope 변화가 −0.574 mm/mm에 이른다. 생성식의 가장 좁은 lobe는 폭 4 mm, 최대 displacement 크기는 해당 lobe에서 0.765 mm다. Flare의 이 구간 경계에서 one-sided second derivative의 크기는 1.53 mm⁻¹이다. 실제 접촉 면적이 있고 표면이 더 완만하면 같은 0.125 mm가 전혀 다른 slope 변화를 만들 수 있다. Distance 숫자는 이 profile의 scale과 함께 해석해야 한다.

## 어떤 물리 경로가 event flip을 만드는가

이 simulator의 stick 조건은 R(g)≤μ_static(vₜ,h)다. Local slope를 g, query 속도를 v=0.425, restitution을 e라 두면 R(g)=|v+g|/[(1+e)(1−vg)]이고 vₜ=(v+g)/√(1+g²)다. 새 contour에서는 기하학적인 impulse demand R과 미끄러지는 속도에 따른 μ_static이 함께 바뀐다. 각 변형 물체가 실제로 base의 label을 뒤집은 사례에 대해 한 경로씩 base 값으로 고정한 physics counterfactual을 채점했다. 이 분석은 사용된 analytic constitutive law 안에서의 경로 분해다.

| Law | μ_static을 base에 고정해도 재현한 실제 flip | R을 base에 고정해도 재현한 실제 flip | 어느 하나만 바꿔서는 재현 못한 flip |
|---|---:|---:|---:|
| all | 98.30% [98.27%, 98.33%] | 3.44% | 1.68% |
| saturating | 99.33% [99.32%, 99.34%] | 1.33% | 0.67% |
| crossover | 97.27% [97.22%, 97.31%] | 5.54% | 2.70% |

두 경로 모두 단독으로 같은 실제 flip을 재현할 수 있어 두 열은 상호 배타적이지 않다. 분모는 8개 비영 signed offset에서 실제로 뒤집힌 hollow·bulge의 총 사례 수다. 첫 열의 95% 구간은 12개 paired panel bootstrap을 사용했다. [경로별 계수와 원자료](friction_path42.json) · [독립 식](friction_path42.py)
