# v71 calibration protocol

v70의 여섯 leave-one-rubber-out source fit을 고정한다. Incoming state는 (vx,vz,0.02ωy), outgoing release 방향은 vx′−0.02ωy′의 부호다. 각 target 제품의 공식 train에서만 무복원 무작위로 길이 32인 prefix를 얻는다. 같은 draw의 앞 0·1·2·4·8·16·32개를 모든 방법에 똑같이 공개한다. 제품당 256 draws, 공식 test는 모두 고정한다. 여러 구성을 가진 제품에서는 원 파일의 shot 빈도대로 섞어 sampling한다. 독립 물성 실험의 수는 여섯 제품이다.

분류기는 각 calibration shot의 실측 부호가 source gate 예측과 다르면 1표를 얻는다. 반대가 과반이면 두 expert routing을 뒤집으며 동률이면 원래 방향을 유지한다. 이 결정을 확실한 반전 검출과 구분하기 위해, p=0.5인 binomial의 상측 tail ≤0.05인 단측 검출률 또한 함께 기록한다. 이 개별 prefix 진단의 반복 점검에는 별도 sequential 보장이 없다. 무작위 sampling은 이미 기록된 반응을 구매하는 offline replay다.

Source-standardized affine feature a=(1,zx,zz,zspin)로 source expert의 output residual을 fit한다. 공통 coefficient와 두 expert 사이 contrast를 결합한다. Calibration의 실측 signed release branch에 따라 feature는 [a,+a] 또는 [a,−a]가 된다. Ridge penalty λ,4λ를 적용한다. Expert-only와 gate+expert는 동일한 coefficient를 쓰고 test routing만 바꾼다. Frozen gate가 정한 branch로 calibration부터 residual을 fit하는 routed expert-only도 평가한다.

Outer target 외부의 다섯 source 제품에서 한 제품씩 inner pseudo-target으로 놓고 나머지 네 제품의 train으로 두 source affine expert 및 gate를 v70 방식대로 다시 fit한다. Inner pseudo-target **공식 train만** 사용해 0-based index % 5 = 0인 row를 inner evaluation으로, 나머지를 calibration pool로 놓는다. 각 제품 32개 고정 무작위 calibration sequences, 1·2·4·8·16·32 예산, 제품·예산 동등 가중으로 joint output (vx,vz,Rωy) 제곱오차를 최소화하는 λ∈{0.25,1,4,16}를 gate+expert용으로 정한다. 동일 λ를 모든 expert ablation에 전달한다. Primary target labels은 해당 prefix 이외에는 parameter 추정에 접근하지 않는다.

평가는 test별 XZ velocity vector Euclidean error의 평균(m/s), Y spin MAE(rad/s). 제품 동등 macro를 우선하고 shot-weighted micro도 계산한다. Draw간 분산은 calibration의 sampling 변동이며 추가 제품 표본은 아니다.
