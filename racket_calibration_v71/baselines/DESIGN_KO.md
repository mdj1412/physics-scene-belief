# v71 같은 예산의 baseline

모든 방법은 같은 target 공식 train의 paid impact prefix만 보고, 해당 제품 공식 test를 평가한다. 6 outer leave-one-rubber-out, 256개 무작위 순서, budgets 0/1/2/4/8/16/32, 독립 draw 안 비복원 추출. 예측 오차 단위는 XZ velocity vector의 norm(m/s)과 y spin 절대오차(rad/s)이다.

- Inclined rigid는 v70의 source-only 기울기·restitution·friction에서 시작한다. Target fit은 0–35°(5° 간격), e=0–1(.05), μ=0–1(.05), 1.1/1.25/1.5/2와 정확한 source parameter를 포함하는 grid다. Calibration의 3개 endpoint squared loss 합에, source train incoming state의 128개 참조점에서 original solver prediction과 달라진 크기에 해당하는 penalty를 더한다. λ는 0.25/1/4/16 pseudo-impacts 중 선택한다. None 또는 zero budget에서는 정확히 source fit을 사용한다.
- Single affine ridge는 source 판별 없이 하나의 affine head만 쓰고, source-standardized x와 y=(vx,vz,Rωy)에서 계수 이동에 ridge를 준다. λ=0.03/0.3/3/30 중 선택한다.
- Single affine empirical Bayes도 하나의 affine head다. Source rubber마다 fit한 계수들의 변동과 within-rubber residual covariance를 source train만으로 추정한다. Coefficient covariance의 85%는 full matrix, 15%는 regularized diagonal로 구성한다. 하나의 paid impact에서 식별된 계수와 함께 움직인 다른 계수도 이 prior가 갱신할 수 있다. Prior precision multiplier는 0.1/1/10 중 선택한다.
- 세 방법의 λ는 바깥 target을 제외한 5개 source 제품에서 각각 하나를 pseudo-target으로 삼고 나머지 4개로 source fit하여 선택한다. Pseudo-target official train에서 calibration과 별도 validation을 분리해 24 draws, 최대 192개 validation shots, 6개 nonzero budget에 동일 비중을 줘 squared error(vx, vz, Rωy)를 집계한다. 한 outer fold 내 hyperparameter는 모든 budget에 공통이다. 각 inner/outer target의 test는 tuning에 쓰이지 않는다.
