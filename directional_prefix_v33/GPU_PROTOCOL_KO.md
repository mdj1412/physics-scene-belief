# v33 GPU 평가의 고정 조건과 재현

Source Coulomb에서 만들어 둔 v32의 세 model pairs를 사용한다. 각 pair는 같은 15개 features와 stage별 ridge penalty 3, 같은 training allowance 540초를 썼다. 세 pair 각각의 1×1 및 2×2 source 데이터는 독립이다. v33의 target에서 추가 학습 비용은 0초다. 첫 pair의 네 policy는 서버 GPU가 전체 경로를 실행하고 실제로 고른 관측으로 Brier를 산출한다. 나머지 두 pair는 같은 새 환경에서 stage별로 갈 수 있는 2·4·8회 결과를 저장한 뒤, 해당 scorer의 순차 결정으로 고른 결과를 재생한다. 이 때 두 번째 stage에는 첫 stage에서 통과한 네 환경의 score만 들어간다.

물체의 표면 윤곽은 offset + slope × y + curvature × y²/(2h) + d × y³/(3h²)이며 h=0.025다. Rounded와 ridged는 원래 curvature의 0.90·1.10배와 d=−0.060·+0.060을 각각 사용한다. 이름은 이 두 개의 합성 곡선 계열을 가리킨다. 곡면상의 contact normal은 같은 윤곽의 미분으로 구한다. 새 표면의 유효 마찰계수는 μ_eff = aμ[1 + Dη tanh(vₜ/τ)]다. Fine에서는 (a,η,τ)=(0.98,0.17,0.34), coarse에서는 (1.04,0.22,0.50)이다. D는 시작 전에 +1 또는 −1로 정했다. 운동 방향에 따라 마찰이 대칭인 표면의 기준은 같은 geometry와 aμ를 유지한 상태에서 η=0으로 구한다. factual intervention에서 접촉이 sticking인 조건, calibration probe에서 sliding인 조건은 매 생성마다 검사했다. 측정 전 모델의 가설 공간에는 cubic contour도 방향 효과도 들어 있지 않다.

하나의 object/surface block은 독립 calibration pool 여덟 개와 calibration과 겹치지 않는 deployment objects 1,024개를 포함한다. 같은 block의 +/− 방향에 동일한 물체, latent variables, noise, posterior bank, 순서가 쓰인다. Random generator lane을 나눠 calibration, deployment, bank를 구분한다. +/−에서 probe response와 intervention의 stick label은 물리에 따라 달라질 수 있다. 네 조합마다 8개 blocks를 따로 뽑고 2개 방향을 각각 평가한다. 총 독립 block 수는 32개다. 모든 policy에서 calibration은 block당 32회이고, 별도 deployment probe는 128개 물체마다 13회, block당 104회다.

GPU-synchronized full-call p90 측정은 각 조합의 첫 2개 block에서 두 방향 모두 실시한다. 각 평가 block에서 한 번 warm-up 후 세 번씩 방법 순서를 무작위로 섞으므로 policy당 48개의 시간 측정과 8개의 독립 timing block이 생긴다. 한 call에서 factual evidence와 사용할 수 있는 probe 값부터 각 policy의 posterior, 순차 allocation, selector, 1,024개 선택 관측의 최종 probability를 다시 구한다. 공통 detector와 privileged source decoder 초기화, 표면 전환, reset, sensor wait, simulation, 파일 기록은 제외한다. 전용 GPU guard가 시작 때 물리 장치가 비어 있는지 확인하고 외부 사용자 작업이 나타나면 자신의 child process만 정지시킨다.

독립 구현은 별도 seeds의 32,768개 합성 물체에서 유한 차분으로 contour normal을 구하고 Newton impulse와 tangential impulse를 NumPy로 다시 계산했다. Torch와의 최대 outgoing velocity 오차는 1.90×10⁻¹¹이고 query stick label은 모두 일치했다. 별도 검사에서 +/−의 factual evidence·query view·isotropic label이 정확히 일치했다. [물리 검증 원자료](physics_audit33.json)
