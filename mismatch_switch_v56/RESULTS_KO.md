# v56 · Mismatch 추정, 관측 정책 전환과 held-out loss

V55의 source-risk gate와 fixed 3R+I 중 하나를 새 표면의 pilot으로 고른다. 별개 session의 물체 한 개마다 reference 방향과 impact 방향을 한 쌍으로 관측한다. 두 부호가 모두 읽힌 쌍 중 방향이 다른 비율을 q̂로 둔다. High-mismatch session 비중 w가 주어지면 이 generator에서 물체 하나의 mismatch 확률은 q=.125+.375w다. 따라서 ŵ=clip((q̂−.125)/.375,0,1)이 v55의 고정 교차점 .43480744를 넘으면 fixed 3R+I를 쓴다. 이는 읽을 수 있는 두 부호가 28.805% 넘게 다를 때와 같다. 유효 쌍이 0개이면 gate를 유지한다. 판독값 이외에 e, w, catalog·surface 이름, 아직 구입하지 않은 네 번째 측정, deployment 결과를 outer selector에 제공하지 않았다. [구현](switch56.py) · [사전 계획](SCORE_PLAN_KO.md)

## 8쌍으로 무엇을 구별했는가

관심 모집단은 두 종류의 표면 환경(w=.35 또는 .52)을 같은 비율로 포함한다. 각 조건의 q는 .25625와 .32000으로 불일치 확률 차이가 **6.375 percentage points**다. 24개 새 catalog × 2개 material, 512개 pilot group씩 평가했다. Group 안의 여덟 쌍 중 평균 **4.009쌍**이 유효하고, 10/50/90번째 백분위수가 **2/4/6쌍**이었다. 유효 쌍이 0개인 group은 0.407%였다. 각 group을 독립적인 새 surface로 세지 않았다. 아래 추정 오차는 유효 쌍이 하나 이상인 경우다. [Pilot-only 진단](pilot_stats56.json)

| High-mismatch session 비중 w | 실제 q | 판독 전체를 합친 q̂ | Pilot group별 q̂ RMSE | ŵ RMSE | Fixed 3R+I 선택률 |
|---:|---:|---:|---:|---:|---:|
| .35 | .25625 | .25686 | .2380 | .3899 | 40.86% |
| .52 | .32000 | .32202 | .2557 | .4008 | 52.34% |

두 표면군 모두 ŵ의 10·90번째 백분위수가 0·1이었다. 전체 pilot을 합치면 평균 mismatch 빈도를 되찾았지만, 한 surface episode에서 어느 쪽 정책을 써야 할지 맞힌 비율은 두 군을 균등 가중해 **55.74%**였다. 이 수치는 true w 기준 v55 threshold와의 일치율이다. 해당 material의 실제 contingent Brier가 더 낮은 정책을 맞힌 분율과는 다르다.

## Primary 결과

| 네 번째 측정 정책 | 평균 held-out Brier ↓ | Session loss p90 ↓ | 두 번째 deployment replay의 p90 | 실제 네 번째 impact 구매율 |
|---|---:|---:|---:|---:|
| K=8 pilot switch | 0.158239094 | 0.188808307 | 0.188635588 | 65.42% |
| 항상 v54 source-risk gate | 0.158239423 | 0.188826740 | 0.188646838 | 35.25% |
| 항상 fixed 3R+I | 0.158240573 | 0.188794509 | 0.188620955 | 100% |

| Switch − 대조 (×10⁻⁶ Brier) | 평균 차이 [동시 95% CI] | Session p90 차이 [동시 95% CI] | 평균 우세 catalog |
|---|---:|---:|---:|
| 항상 v54 gate | −0.33 [−2.30, +1.60] | −18.43 [−38.10, +2.41] | 13/24 |
| 항상 fixed 3R+I | −1.48 [−3.72, +0.82] | +13.80 [−5.57, +40.28] | 14/24 |

평균의 두 point estimate는 감소했지만 모두 동시 interval에 0을 포함했다. 실용적 equivalence나 두 대조군에 대한 우위를 입증하지 못했다. Session loss p90은 gate와 fixed 사이에 위치했고 두 대조 역시 유의한 차이를 보이지 않았다. 각 arm의 평균은 deployment orientation과 object population에 대해 적분했다. P90은 128개 binary outcome로 만든 session 평균의 분포를 사용했다. 각 material·w의 독립 pilot action 확률로 각 arm의 realized-session CDF를 혼합한 뒤 역함수를 구했다. 해당 표면의 두 law에 동일한 outer decision을 적용했다. 24개 catalog를 paired bootstrap 5,000회 반복하고 mean과 p90을 별도 family로 두어 각각 두 대조를 Bonferroni 보정했다. 모든 observation의 가중치를 직접 펼친 독립 정렬로도 같은 세 p90을 얻었다. [채점 결과](score56.json) · [직접 검산](audit_risk56.json)

두 번째 deployment replay에서 mean의 switch−gate 차이는 +1.29×10⁻⁶, switch−fixed 차이는 −3.21×10⁻⁶이었다. 더 세밀한 solver에서 기대 mean 차이는 각각 +0.40×10⁻⁶과 −2.28×10⁻⁶이고 두 동시 interval 모두 0을 포함했다. 작은 우위의 방향은 solver 또는 deployment draw에 민감했다. [두 replay와 refined solver](score56.json)

## 이득의 상한과 환경 조성

동일한 v55 threshold에 올바른 w만 대입하는 privileged policy는 평균 0.158232124였다. 가장 좋은 constant policy인 gate에서 개선할 수 있던 차이는 **7.30×10⁻⁶**였다. K=8 switch가 그 중 회수한 점추정은 **4.50%**였다. 양쪽 정책의 target conditional mean을 material·w별로 모두 본 뒤 더 작은 것을 고르면 0.158231610이다. 이 진단은 배포 시 관측할 수 없는 target risk도 사용한다. 전체 joint posterior 없이 얻을 수 있는 보장된 개선량으로 해석할 수 없다. 새 catalog의 평균 risk로 사후 계산한 교차점은 44.12%였다. 실제 정책은 v55의 43.48%에서 변경하지 않았다. [Oracle과 교차점](audit_risk56.json)

독립적으로 switch-to-fixed 횟수의 비율만 맞춘 무작위 정책 대비 평균 차이는 −0.864×10⁻⁶ [탐색적 95% CI −0.957, −0.769]였다. 동일한 비율에 정보를 써서 배정한 효과는 작게 검출되었다. 더 큰 이득을 위한 질문은 적응적 전환이 더 나은 constant를 이기는가이며, 이 primary 판정은 통과하지 못했다. [구매율 통제](score56.json)

| K (지불한 pilot 쌍) | w=.35에서 fixed 선택 | w=.52에서 fixed 선택 | 평균 Brier | Gate 대비 mean 차이 (×10⁻⁶) |
|---:|---:|---:|---:|---:|
| 0 | 0% | 0% | 0.158239423 | 0 |
| 1 | 12.81% | 16.35% | 0.158239326 | −0.10 |
| 4 | 39.31% | 47.84% | 0.158239272 | −0.15 |
| 8 | 40.86% | 52.34% | 0.158239094 | −0.33 |
| 16 | 37.61% | 53.40% | 0.158238736 | −0.69 |

각 K 안에서는 세 정책이 같은 수의 pilot을 지불했다. K 간 절대 비용은 다르다. K=16도 두 constant 대비 mean 차이의 interval이 각각 0을 포함했다. 반면 secondary 양끝 환경(w=.10, .80)을 균등 가중하면 K=8은 gate보다 15.30×10⁻⁶, fixed보다 13.68×10⁻⁶ 낮았다. 이 두 평균 차이의 탐색적 동시 95% CI는 각각 [−17.42, −13.23], [−16.42, −11.24]×10⁻⁶였다. 두 점만 섞은 모집단은 등록된 primary 모집단과 다르며, 어느 한 환경에서 혼합 규칙이 그 환경의 더 나은 constant를 이겼다는 의미도 아니다. [Budget과 w별 원자료](score56_cells.npz) · [Secondary 계산](audit_risk56.py)

## 비용과 해석

각 K=8 arm은 16 pilot contacts와 16 resets, 이어지는 post-exchange 네 쌍에서 8 contacts와 8 resets를 지불한다. 총 24 contacts·24 resets에 공통 교체 전 history 수집비를 더할 수 있다. 고정된 sensing time까지 맞추려면 reference와 impact가 같은 slot을 쓰거나 남은 시간을 padding해야 한다. 두 고정 정책은 실제 운영에서는 pilot을 생략할 수 있으므로, 공통 pilot을 청구한 성능 비교만으로 새 pilot 구매의 경제성을 주장할 수 없다. 한 pilot batch를 여러 deployment session에 재사용하면 amortized cost는 낮아지지만 당시의 w가 계속 같은지도 확인해야 한다. Sensor의 오류, 물체당 e 조건부 구조, calibration과 deployment가 같은 mixture를 쓰는 가정은 실물 검증 과제로 남는다. [사전 정보 조건](SCORE_PLAN_KO.md)
