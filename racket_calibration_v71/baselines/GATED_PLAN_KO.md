# Same-prior two-expert control

Single-head empirical Bayes의 이득이 효과적인 prior인지 expert 수 차이인지 구분하기 위해, v71 결과 확인 후 추가한 탐색적 비교이다. 각 rubber에서 두 branch의 source-standardized (4×3) affine coefficient를 sign 실측으로 각각 fit한 후 이어 붙인 24차원 covariance를 추정한다. Single-head와 같은 full covariance 85% + floor를 둔 diagonal 15%, within-rubber residual 3×3 covariance와 precision multiplier grid 0.1/1/10을 사용한다. Inner 5 pseudo-targets, 24 draws, 192 verification shots, 6개 nonzero budget의 joint loss, 모든 split·budget·seed가 동일하다. 차원 차이에 맞춰 residual degree of freedom만 4에서 8로 늘린다.

Target calibration은 기존 source gate 부호와 실제 release direction의 불일치를 다수결로 집계하며 동률이면 원래 부호다. 두 expert의 affine 계수는 calibration의 실제 부호별로 업데이트한다. Target test에서는 incoming state와 paid prefix만으로 routing한다.
