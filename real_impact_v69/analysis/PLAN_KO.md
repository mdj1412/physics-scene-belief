# v69 empirical control와 비교 계획

Biber et al.의 Online Resource 2에서 한 campaign 전체를 target으로 교대 사용한다. Source 원래 행 번호를 0부터 세어 24행 단위 block 번호 mod 4가 1이면 residual calibration, 나머지는 fit에 배정한다. Target의 post-impact 측정치를 fit, threshold 또는 보정에 쓰지 않는다.

- 연속 지표는 center-of-mass tangential velocity와 최저점 rigid-kinematic velocity의 MAE이다. 부호 convention은 counterclockwise-positive이다.
- Primary event는 exit에서 |u| ≤ 0.50 m/s. 0.25, 1.00은 sensitivity이고, 입출구 u의 sign reversal도 원문의 물리적 해석을 점검하는 별도 endpoint로 계산한다. 두 event 모두 patch 내부 stick/slip의 직접 측정과 구별한다.
- Source에서만 fit한 affine와 two-regime affine를 사용한다. 후자는 source fit에서 release u의 부호로 나눈 두 affine expert와 incoming velocity/spin으로 학습한 ridge logistic gate를 쓴다. Negative-control은 residual calibration 분포의 source frequency다. 두 실험의 radius와 확률 보정식을 통일한다.
- 차이의 불확실성은 target 원래 순서의 24행 block을 짝지어 재표집한다. 12, 48행은 sensitivity이다. 생성된 block은 실제 launcher/session ID와 다르다. 두 campaign으로 새 표면 모집단 추론은 하지 않는다.
