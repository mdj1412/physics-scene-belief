# v39의 사전 예측

예측량 Δ는 네 번째 접촉까지 읽고 계산한 **frozen continuous gain optimizer의 held-out Brier에서 고정 32.5% shrinkage의 held-out Brier를 뺀 차이**다. 음수이면 optimizer가 유리하다. 아래 수치 산출에는 설계자가 정한 새 형상의 방정식, contact law 이름, 기존 v38의 관측·정답만 사용했다. 새 catalog의 관측, latent prototype, query label은 사용하지 않았다.

| v39 형상과 law | 대응하는 v38 형상 | 사전 Δ | 같은 v38 catalog 내 block bootstrap 95% 구간 | 사전 방향 |
|---|---|---:|---:|---|
| Shelf / weakening | Shoulder | −0.000003 | [−0.000610, +0.000494] | 수치상 미세한 optimizer 우세. 방향의 신뢰도 낮음 |
| Shelf / strengthening | Shoulder | +0.000638 | [+0.000156, +0.001114] | 고정값 우세 |
| Ridge / weakening | Bead | −0.000769 | [−0.001121, −0.000452] | Optimizer 우세 |
| Ridge / strengthening | Bead | −0.001354 | [−0.002027, −0.000603] | Optimizer 우세, 네 cell 중 가장 큰 이득 예상 |

네 cell을 같은 비중으로 둔 평균 예측은 −0.000372다. 어느 cell에서도 기존 실용 기준 0.003에 이르는 평균 감소를 예상하지 않는다. 위 구간은 과거 데이터 내 변동만 나타낸다. 새 형상·새 catalog의 prediction interval로 해석할 수 없다.

**형상 대응.** 마지막 probe와 query의 높이 23·25 mm에서 contour의 추가 local slope 두 개를 계산했다. v38 shoulder는 (+0.0330,+0.0360), bead는 (−0.0370,−0.0357)이다. v39 shelf는 (+0.0376,+0.0430), ridge는 (−0.0413,−0.0383)이다. 두 값의 Euclidean distance가 가장 작은 이전 형상을 대응시킨 뒤 같은 law에서 얻은 Δ를 옮겼다. Local slope는 normal 및 tangential component에 영향을 주지만 curvature 분포, 전체 contour, catalog까지 일치시키지는 않는다. 대응 규칙이 유효한지 자체가 검증 대상이다. v38은 cell·α당 catalog가 하나였으므로 당시에는 독립 catalog로 이 규칙을 검증할 수 없었다.

**판정.** 네 부호를 모두 공개하고, 신뢰도가 낮다고 미리 표시한 shelf/weakening까지 포함해 point estimate 적중 수를 센다. 동시에 다른 세 방향에 대해서도 따로 적중 수와 interval을 보고한다. 네 개 대조의 simultaneous 95% interval이 예측한 편에 모두 있으면 해당 방향을 지지하고, 반대편에 있으면 해당 방향을 반증하며, 0을 포함하면 방향 미확정이다. 단순히 부호가 일치한 cell 수와 interval로 지지된 수를 따로 센다. 크기 예측은 cell별 Δ의 평균 absolute error로 평가하며, 모든 cell에 v38 전체 평균 −0.000372를 주는 단순 예측과도 비교한다. 네 cell만으로 일반적인 정확도에 대한 모집단 비율은 추정하지 않는다.

### 형상 이름 없이 적용하는 보조 예측

정책 차이가 0이면 두 방법의 확률과 Brier가 정확히 같다는 제약을 사용했다. 처음 세 접촉에서 얻은 signed cue를 z, 네 번째 관측 후 frozen optimizer가 정한 보정 비율을 w로 놓고 a=z(w−0.325)를 구한다. 사전 예측식은 **Δ̂=b₁a+b₂az+b₃a²**다. z와 w만 있으면 계산할 수 있다. 따라서 개별 환경에서 사용 가능한 시점은 네 번째 접촉을 관찰한 뒤이며, 추가 접촉 자체를 미리 결정하는 정책으로 해석할 수 없다.

v38 네 cell·32개 block에서 표면 강도와 부호를 짝지어 계산한 1,536개 환경별 paired loss를 회귀했다. 셀마다 같은 비중을 주고 intercept는 0으로 고정했다. 세 열의 root-mean-square로 표준화한 뒤 0.01 ridge penalty를 적용했다. 최종 계수와 정밀한 부호·자료 hash는 [동결된 예측](forecast39.json)에 기록했다. Δ̂<0이면 optimizer, 그 외에는 고정 32.5%를 고르는 보조 정책도 이 시점에 동결했다.

V38 leave-one-cell-out의 합산 결과에서 optimizer를 고른 비율은 54.2%였다. 그 집단의 실제 Δ는 −0.001387, 고정값을 고른 집단의 실제 Δ는 +0.000827이었다. 다만 shoulder/weakening에서는 두 방향이 모두 뒤집혔다. 이 과거 수치는 모델의 입력·설계를 선택한 자료 안에서 계산했으므로 v39의 independent validation을 대신하지 않는다. v39에서는 조건별 평균 δ, 두 집단의 차이, 보조 정책과 두 고정 전략의 paired Brier를 같은 catalog 단위로 평가한다. 원래 optimizer와 fixed의 GPU latency 주 비교는 그대로 둔다.
