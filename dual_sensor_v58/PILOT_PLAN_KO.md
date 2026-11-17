# v58 dual-sensor pilot 사전 계획

24개 fresh catalog × 2개 surface × 2,048회 독립 8-impact pilot을 생성한다. 같은 여덟 specimen seed를 설계·환경에 짝짓는다. 두 물체의 원래 ID (0,1)를 physical state, 센서 error, mismatch 여부를 보기 전 fair coin으로 섞고, 두 실제 센서가 각 물체의 동일 reference와 impact를 동시에 측정하는 합성 readout을 쓴다. 방향이 0인 물체 50%는 양쪽에서 읽을 수 없고, 판독을 재시도하지 않는다. 센서별 reference 및 impact sign은 확률 2%로 각각 독립 반전한다. 별도 clean 조건은 이 반전을 제거한다.

환경 A의 e는 0(.42), .5(.58), B는 .25(.84), .5(.16)이며 q=.29다. 8×1은 매번 새 session이다. 4×2는 두 slot이 session을 공유하지만 physical persistence rho를 0 또는 1로 조절한다. rho=0은 각 slot에서 e를 다시 뽑고, rho=1이면 같은 e를 쓴다. Sensor-specific persistent readout error는 h=0, 0.26582278481012656로 교차시킨다. 각 session/센서의 독립 latch와 Z~Bern(q)를 뽑고, latch가 켜지면 해당 sensor의 상대 변형 방향은 Z를 따른다. 꺼지면 물리 mismatch X를 따른다. 이진 방향의 대칭 독립 판독 noise는 마지막에 붙인다. 이 설정에서는 같은 센서의 error가 이어져도 서로 다른 sensor와 물체를 택한 cross-sensor pair에는 물리 상태를 통하지 않는 의존성이 없다.

주요 diagnostic은 둘 다 판독된 물체에서 same-sensor와 cross-sensor double mismatch 및 marginal q*의 제곱을 뺀 covariance다. h에 무관한 q*=d+(1−2d)q, d=2(.02)(.98). t=1−2d, v=Var(e), u=q(1−q)이면 4×2의 same covariance는 t²[(1−h)rho v+h u], cross는 t²(1−h)²rho v이다. 이 식으로 관측 패턴과 생성한 mechanism을 검사한다. 관측이 없는 pair 발생률, object order balance와 slot별 mismatch, clean/noisy, 물체 순서별 분해를 기록한다. CI는 독립 catalog 24개를 재추출한다. 진짜 원인은 audit에서만 접근할 수 있다. Primary의 이 결론을 깨는 sensitivity로, rho=h=0에서 두 센서가 같은 common-mode session error를 공유할 때 rho=1인 A의 pair 분포를 복제할 수 있음을 평균과 second moment를 맞춰 검증한다. 하드웨어 검증은 포함하지 않는다.

Policy용 codes 순서는 layout, rho, h, regime, material, episode, adjacent pair이며 각 pair=5*s1+s2, s=0 missing 또는 1+b0+2*b1이다. b는 해당 sensor의 reference와 impact의 mismatch다. Common-mode 및 clean은 별도 diagnostic이며 primary policy 입력에는 제공하지 않는다.
