# v71 추가 비교 설계

동일한 256개의 random calibration prefix와 공식 test에서 gate가 없는 state-dependent elastic impulse model을 추가 평가한다. Thin-shell k=2/3에서 s=vx−Rωy, Δvx=−αs, Δ(Rωy)=αs/k, vz'=−e vz를 쓴다. e는 |vz|, α는 |s|에 대해 각각 affine로 둔다. 네 회귀계수의 prior mean·covariance와 3성분 error covariance는 outer fold의 다섯 source rubber로부터만 추정한다. 각 target prefix의 이탈 세 성분으로 conjugate Gaussian update를 한 뒤 predictive mean 계수를 쓴다. e∈[0,1], α∈[0,0.8]로 예측 시 제한한다. 비교 목적은 gate가 없어도 연속적인 impulse 보정만으로 release 방향이 변하는 설명을 검증하는 것이다. Hyperparameter tuning 없이 source variability에 floor를 둔다.

Grass D.TecS의 두 configuration에 대해서는 한 configuration의 공식 train에서만 calibration을 산 뒤 상대 configuration의 공식 test로 배포한다. 나머지 split과 source fit은 v70과 동일하다. 같은 시험 표본에 같은 configuration에서 calibration을 산 조건도 대조한다. 이 검증은 서로 다른 sponge 두께에서의 transfer이며, 다른 물리적 long-pips 제품을 독립 재현하는 것은 아니다.

두 expert 쪽에도 source rubber 간 coefficient covariance를 주는 별도 진단을 한다. 5개 source 제품에서 release 부호로 분리한 각 affine head 24계수의 covariance와 residual covariance를 추정한다. v70의 frozen expert를 prior 중심으로 쓰고 85% sample covariance와 15% diagonal 및 작은 variance floor를 사용한다. 업데이트는 모든 출력 성분을 결합하며 prior multiplier는 1로 고정한다. Source validation으로 multiplier까지 선택한 single affine baseline과 이 진단의 튜닝 범위는 다르다. 모델은 test label을 보지 않는다.
