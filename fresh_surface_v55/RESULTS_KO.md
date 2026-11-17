# v55 · 새 surface와 session에서 frozen acquisition을 채점한 결과

## 동일한 terminal에서 평균 개선이 재현되는가

새로운 두 procedural surface family에서 24개 catalog를 생성했다. 각 catalog는 2개 material instance와 기존 2개 contact law의 조합 4개를 포함한다. 총 48개 material instance, 96개 catalog × material × law 조건과 122,880개 post-swap calibration prefix다. 각 조건의 새 calibration pool, 이전 표면의 calibration pool, 128개 deployment 물체의 pool이 분리되어 있다. 통계의 독립 단위는 24개 catalog다. 여러 law에 같은 물질을 배치한 횟수, 같은 prefix에 서로 다른 mismatch·drift를 적용한 횟수 또는 calibration trial 수를 독립적인 surface 수로 늘리지 않았다. [설계와 판정 기준](SCORE_PLAN_KO.md) · [생성 자료의 독립성](independence55.json)

| 정책 | 네 번째 impact 구매율 | Mean held-out Brier | Session loss p90 | 두 번째 deployment seed의 p90 |
|---|---:|---:|---:|---:|
| Frozen v54 source-risk gate | 34.6590% | **0.156878466** | 0.188775987 | 0.188756526 |
| Fixed 3R+I | 100% | 0.156896966 | 0.188775987 | 0.188776151 |
| 같은 구매율의 random | 34.6590% | 0.156901644 | 0.188802511 | 0.188790784 |
| Fixed 4R · 진단 | 0% | 0.156904126 | 0.188818560 | 0.188794896 |
| Catalog 안에서 구매율을 맞춘 random · 진단 | 34.6590% | 0.156901609 | 0.188802705 | 0.188790873 |

**평균 개선은 두 대조 모두에서 재현되었다.** 아래 차이는 gate에서 각 대조를 뺀 절대 Brier 차이다. Mean과 p90 각각에서 두 대조에 대한 simultaneous 95% bootstrap interval을 제시했다. 음수일수록 gate가 유리하다. Catalog를 5,000회 쌍으로 재추출하고 매번 gate의 marginal 구매율을 다시 계산했다.

| 비교 | Mean 차이 [simultaneous 95% CI] | Session p90 차이 [simultaneous 95% CI] | 평균이 낮은 catalog |
|---|---|---|---:|
| Gate − fixed 3R+I | −0.00001850 [−0.00002408, −0.00001359] | 0.00000000 [−0.00003647, +0.00002810] | 23/24 |
| Gate − 같은 구매율의 random | −0.00002318 [−0.00002486, −0.00002145] | −0.00002652 [−0.00004626, −0.00000748] | 24/24 |

두 mean interval 모두 사전 지정한 −0.00001보다 아래에 있었다. 이 참고선은 v54 효과의 약 절반을 새 자료에서도 유지하는지 묻기 위해 두었다. 실물에서 어느 차이가 경제적으로 가치 있는지는 실측 결과가 필요하다. P90 자체가 완전히 같은 것은 양쪽이 일부 같은 slot에서 같은 예측을 하고 두 분포의 90% 경계가 같은 loss 값에 걸렸기 때문이다. 독립적으로 실현한 두 번째 deployment batch에서는 gate가 두 대조보다 작았다. Fixed 대비 p90 차이의 동시 interval이 양쪽 부호를 모두 포함하므로 tail superiority를 주장할 수 없다. [Endpoint와 confidence interval](score55.json)

## 어떤 risk를 평균하고 어느 session의 tail을 측정했는가

각 catalog × environment에서 128개 새 deployment 물체를 균등하게 사용했다. 각 물체는 frozen decoder의 두 예측과 두 개의 deterministic Bullet outcome을 갖는다. 동일 충돌에서 weak 방향을 유지하거나 반전시켜 잔류 접선 속도가 0.020 m/s 이하가 되는지 계산한다. Positive label은 stick이다. 같은 실제 label의 확률을 보완하여 slip으로 표현하면 Brier는 동일하다. Weak→reference 유지율을 k, reference→impact 반전 확률을 e로 두면 weak→impact 유지율은 q=e+(1−2e)k다. Mean에서는 square loss를 두 binary outcome에 각각 계산한 뒤 q로 가중했다. 기대 label과 예측의 차이만 제곱하면 생략되는 outcome variance까지 포함한다. 각 경로의 final forecast에는 그 경로가 구매한 네 번째 readout 하나만 반영했다. [Scorer](score55.py)

Session p90에서는 같은 128개 물체 pool에서 복원 추출한 128회 deployment와 독립적인 유지·반전 변수를 session마다 새로 뽑았다. e·drift event·k와 두 policy 간에 같은 deployment draws를 짝지었다. 별도 seed로 전체 128회 batch를 다시 그렸다. Realized-label mean은 첫 seed에서 gate / fixed impact / random이 0.156898466 / 0.156917838 / 0.156921166, 두 번째 seed에서 0.157067453 / 0.157084669 / 0.157090492였다. 두 seed 모두 mean 차이의 interval이 두 대조에서 0보다 작았다. Time step과 contact resolution을 높인 별도 solver 설정에서도 integrated mean 차이가 fixed 대비 −0.00002067 [−0.00002563, −0.00001595], random 대비 −0.00002358 [−0.00002509, −0.00002200]이었다. 두 setting은 같은 Bullet 계열이므로 독립 solver 검증은 아니다. [Catalog별 risk와 두 deployment seed](clusters55.npz)

## 구매율 효과, material 효과, prefix 효과를 구분하는 대조

동일 구매율 대조는 각 session의 잠재적 reference loss와 impact loss를 각각 1−r과 r로 가중한다. r은 평가 label을 사용하지 않고 세 쌍으로 고른 42,589개 impact / 122,880개 prefix다. Finite evaluation pool에서 42,589장만 impact ticket으로 만들어 무작위로 섞어도 각 slot이 impact를 받을 확률은 r이다. 표의 mean은 이 lottery를 정확히 적분한 값이다. Session p90은 각 경로의 loss CDF를 섞어 계산한 marginal p90이다. 한 번의 quota shuffle에서 나온 표본 p90의 기대값과는 다른 estimand다. 전체 loss를 실제 정렬하고 weighted inverse CDF를 구한 독립 계산으로 세 primary p90을 검산했다.

같은 **catalog × material × law 안에서** 구매율을 일치시킨 random과 비교해도 mean 차이는 −0.00002314 [탐색적 95% CI −0.00002459, −0.00002162]였다. Gate가 impact를 평균적으로 더 많이 배정한 특정 material에서 이득이 난다는 설명만으로는 이 차이를 설명하기 어렵다. 실제로 다음 네 행은 서로 다른 24개 catalog에서 같은 material family × law의 결과를 모은 것이다. 각 행의 두 차이는 mean Brier × 10⁻⁶이다.

| Material × contact law | Impact 구매율 | Gate − fixed 3R+I | Gate − same-rate random | 더 작은 mean을 낸 catalog 수, 각각 |
|---|---:|---:|---:|---:|
| Braid × rational | 34.60% | −23.59 | −24.86 | 21/24, 24/24 |
| Braid × gaussian_rate | 34.65% | −14.85 | −20.27 | 20/24, 24/24 |
| Seam × rational | 34.77% | −14.37 | −24.14 | 21/24, 24/24 |
| Seam × gaussian_rate | 34.62% | −21.19 | −23.45 | 22/24, 24/24 |

서로 다른 e를 따로 보면 fixed 3R+I 대비 gate의 mean 차이는 e=0에서 −0.00004187, e=.25에서 −0.00011667, e=.5에서 **+0.00010304**였다. 반면 같은 구매율의 random 대비 차이는 각각 −0.00001421, −0.00002355, −0.00003177이었다. 같은 수를 구매할 때 어느 session에 배정하는지의 효과와, impact를 얼마나 많이 사는 편이 좋은지가 분리된다. 첫 세 쌍으로 직접 관측되지 않는 e가 큰 모집단에는 fixed impact가 유리할 수 있다. [조건별 분석과 weighted p90 검산](audit55.json) · [Bootstrap draws](bootstrap55.npz)

## 계산과 물리 예산의 경계

모든 policy는 교체 후 4 calibration 쌍, 8 contacts와 8 resets를 쓴다. v54 cost model에서 reference slot은 1.04초이며 impact 측정과 reset까지의 시간을 T초라 두면 gate와 random의 기대 비용은 세션당 5.92 + 0.346590(T−1.04)초, fixed impact는 5.92 + (T−1.04)초다. T=1.04일 때 모두 5.92초다. T=2.08이면 gate와 random은 6.2805초, fixed impact는 6.9600초다. 물리 장비에서 T를 측정하면 그 값을 대입해야 한다. GPU latency와 이 물리 비용은 다른 자원이다. [GPU timing 계획](TIMING_PLAN_KO.md)
