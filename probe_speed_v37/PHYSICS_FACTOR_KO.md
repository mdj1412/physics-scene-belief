## 속도 변화가 sensor separation에 기여하는 두 경로

같은 calibration 물체가 두 friction law 중 어느 쪽인지 구분한다고 생각하자. Kinematics 속도는 접촉 직전 tangential velocity와 normal impulse를 정하고, modulation 속도는 directional friction coefficient의 velocity 항에 들어간다. 두 입력을 독립적으로 .50 또는 높은 속도로 바꿔 같은 물체의 outgoing velocity를 계산했다. 두 속도를 같게 둔 경우에는 192개 local raw archive의 세 속도에서 clean velocity와 stick label이 모두 일치했다. 교차한 조건은 simulator에만 넣은 반사실이다. [설계](PHYSICS_FACTOR_PLAN_KO.md) · [계산 코드](physics_factor37.py)

표의 단위는 accuracy percentage point다. 두 후보를 같은 확률로 가정하고 latent geometry와 orientation까지 제공한 privileged Bayes accuracy의 증가분을 표시한다. Kinematics를 먼저 바꿨을 때와 modulation을 먼저 바꿨을 때의 증가분을 평균해 두 성분에 배분했다. 양쪽 합은 총 증가량과 같다. 마지막 열은 두 경로의 interaction이며 총 증가량에 다시 더하지 않는다. 여덟 성분 추정에 simultaneous 95% interval을 적용했다.

| 구별할 regime | 높은 속도 | 총 증가 (pp) | Kinematics (pp) [95%] | Friction modulation (pp) [95%] | Interaction (pp) |
|---|---:|---:|---:|---:|---:|
| strong_vs_null | 0.75 | +9.49 | +7.66 [+5.73, +9.59] | +1.83 [+1.65, +2.00] | +0.74 |
| strong_vs_null | 1.10 | +16.70 | +12.63 [+10.19, +15.06] | +4.08 [+3.76, +4.40] | +3.38 |
| weak_vs_null | 0.75 | +4.09 | +3.19 [+2.25, +4.14] | +0.89 [+0.81, +0.98] | +0.46 |
| weak_vs_null | 1.10 | +7.31 | +5.17 [+4.07, +6.27] | +2.13 [+1.98, +2.29] | +2.00 |

같은 물체의 두 표면을 비교하면 normal impulse Jₙ과 접선 방향 단위벡터가 같아, 두 outgoing velocity mean 사이의 거리는 |Jₜ₁−Jₜ₂|다. 두 충돌이 모두 stick이면 Jₜ가 tangential demand로 같아진다. 두 충돌이 모두 slip이고 접선 방향이 같으면 그 거리는 Jₙ μ |a₁−a₂| |f(vₜ)|가 된다. 여기서 a는 해당 높이에서의 방향 효과 세기, f는 속도에 따른 방향 효과의 modulation이다. 따라서 미끄러지는 물체 수가 더 늘지 않는 구간에서도 Jₙ이나 f가 달라지면 두 velocity 분포를 더 쉽게 또는 어렵게 구별하게 된다.

Kinematics 성분에는 stick에서 slip으로 넘어간 물체뿐 아니라, 이미 미끄러졌어도 normal impulse 등이 달라진 물체가 포함된다. Modulation 성분 역시 바뀐 friction coefficient가 contact state에 주는 피드백을 포함한다. 두 경로는 이 simulator가 생성한 두 Gaussian sensor likelihood에 관한 계산이다. 독립 물체의 query Brier와 실제 gate의 식별 성능은 본문의 held-out 결과로 판단한다. [수치 원본](physics_factor37.json)
