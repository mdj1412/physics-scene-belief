# Frozen single-intercept control

이 대조는 v39 주 실험과 별도로, GPU 실행을 시작한 뒤 target별 score·label을 읽기 전에 정했다. 앞서 동결한 회귀식에서 |z|≤1이고 0≤w≤1이면 세 계수의 합성항 b₁+b₂z+b₃a가 항상 −0.00205 이하이다. 따라서 Δ̂<0은 정확히 a=z(w−.325)>0과 같은 조건이다. 이때만 optimizer가 고정값보다 높은 stick probability를 반환하므로 이 보조 정책은 두 확률 중 큰 값을 고른다. 자료를 읽으면 전 행에서 이 항등식을 확인한다.

이를 설명할 수 있는 단순 대안으로 v38의 fixed 32.5% 확률에 하나의 global log-odds intercept를 더해 동일한 held-out Brier를 평가한다. 계수 +0.15981991164881015는 v38에만 있는 fixed 예측과 실제 label로 squared loss를 최소화했다. (-2,2) 내부해이며, 이 값에서 v38 Brier는 고정값 대비 0.000548 낮아졌다. V38 평균 fixed 예측−발생률은 −0.01746이었다. 데이터 전체에 대해 한 번 적합하고 모든 v39 cell과 환경에 같은 값을 적용한다. V38의 감소량은 fitting 자료에서 얻은 것이다.

V39에서 single intercept와 fixed·optimizer·observation-only router를 비교한다. 보고할 것은 paired Brier, 평균 forecast−incidence, cell 내 catalog bootstrap의 pointwise interval이다. Supplementary policy의 GPU p90은 측정하지 않는다. 이 대조는 unconditional calibration bias가 단순 정책으로 얼마나 해결되는지 보는 용도다. 원래의 두 정책과 prospective cell 예측에 대한 판정은 그대로 유지한다.
