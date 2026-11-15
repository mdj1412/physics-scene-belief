# v38의 독립 target population과 관측 생성

v37에서 source gain table과 calibration policy를 모두 고정한 뒤, 새 형상 두 종류와 새 표면 법칙 두 종류를 교차했다. Geometry와 contact law는 target generator와 명시적 oracle에만 전달된다. 비교 정책이 받는 정보는 기존 policy가 처리하는 관측 및 공통 decoder 조건이다. Shape와 surface의 이름은 모두 synthetic curve의 이름이다. 실제 재료에 적합한 수치는 아니다.

## 형상 두 종류

높이 단위는 m이며 u=y/0.025다. 기존 local profile은 offset + slope·y + curvature·y²/0.05이다. 아래 표의 profile 변위를 여기에 더하고, 그 정확한 도함수를 contact normal 계산에도 사용한다.

| Shape | Profile 변위 | 추가되는 local slope | 기본 curvature 배율 |
|---|---|---|---|
| shoulder | 0.072·0.025·(u−atan u) | 0.072·u²/(1+u²) | 0.97 |
| bead | −0.078·0.025·0.80²·(1−exp(−u²/(2·0.80²))) | −0.078·u·exp(−u²/(2·0.80²)) | 1.03 |

두 profile과 slope 변위는 y=0에서 0이다. Shoulder는 양·음 상대 높이에서 포화하는 완만한 slope를, bead는 중심에서 멀어지면 다시 줄어드는 비대칭 부호의 slope를 만든다. v35–v37의 sinusoidal slope와 cubic slope, frozen source의 quadratic slope와 함수형이 다르다.

## 접촉 법칙 두 종류

v는 local tangent로 분해한 상대 입사 속도, s=|v|, o는 숨은 orientation 자체의 ±1 값이다. A는 위치별 방향 강도다. h=min(|y|/0.023,1)일 때 A=1−(1−a)(1−cos(πh))/2이며, a는 strong=1, weak=0.3, local null=0이다. 세 부류 모두 y=0에서 A=1이므로 첫 세 접촉은 부류 자체를 구분하지 못한다. y=0.023과 y=0.025에서는 A=a다. A를 0으로 두는 full-isotropic global null도 코드에서 지원하나 primary target에는 들어가지 않는다.

| Surface | 방향 변조 d(v) | Static threshold 배율 S(s) | Sliding 배율 K(s) |
|---|---|---|---|
| weakening | 0.19·sign(v)·(1−exp(−s/0.47)) | 1+0.04/(1+(s/0.40)²) | 0.89+0.08·exp(−s/0.48) |
| strengthening | 0.192·2r/(1+r²), r=v/0.75 | 1+0.05·s/(0.45+s) | 0.88+0.11·s/(0.50+s) |

μ_s=μ(1+oAd)S, μ_k=μ(1+oAd)K다. Normal impulse J_n=(1+e)max(v_n,0)로 두며 |v|≤μ_s J_n이면 stick, 아니면 slip이다. Slip일 때 tangential impulse의 크기는 μ_k J_n이다. 두 law 모두 S>K>0이고 방향 배율은 양수다. 따라서 slip branch에서 허용 마찰 impulse가 접촉 전 tangential momentum보다 커지는 경우는 없다. 첫 law는 방향 차이가 포화하면서 kinetic friction이 속도와 함께 줄고, 둘째 law는 방향 차이가 중간 속도에서 최대가 되면서 kinetic friction은 증가한다. 두 law 모두 v35–v37에서 동일한 coefficient로 stick 판정과 sliding impulse를 계산하던 방식과 다르다. v30에서도 static과 kinetic을 분리한 선례가 있으므로, 새로운 물리 기전의 제안으로 해석할 수는 없다.

Factual 동작의 접촉은 항상 stick이며, 기존 현장 probe(높이 0, 속도 ±1.10)는 항상 slip이다. 높이 0의 |slope|는 0.035 이하, μ는 [0.1055,0.3965], e는 [0.515,0.885]다. Factual의 |v|≤0.035는 가장 작은 μ_s J_n보다 작다. 반면 conventional probe의 |v|≥1.064는 보수적인 최대 μ_s J_n (<0.974)보다 크다. Target의 네 번째 +1.10 probe는 23 mm에 있으므로 contact mode는 별도로 계산한다. 두 개의 효과를 섞어 비교하지 않도록 query는 25 mm, +0.425 그대로 유지한다.

## 유한 집단과 실험적 독립성

- 4개 geometry × law cell, 사전에 고정된 8개 population proxy coupling 값 α, 두 역할(calibration/deployment)에 대해 독립 RNG catalog를 만든다. 각 catalog에는 연속 latent parameter vector 2,048개가 있다. 각 행은 난수로 뽑은 특정 geometry/물성 조합이며 번호는 policy에 들어가지 않는다.
- 각 split의 prototype 수는 4×8×2,048=65,536개다. Sign이나 local regime을 바꿔도 같은 catalog를 쓴다. Calibration과 deployment의 support RNG는 서로 다르다. 같은 환경의 object draws는 해당 catalog에서 독립 복원추출한다. 따라서 calibration 네 접촉은 서로 다른 pool 행이고 같은 prototype이 우연히 다시 뽑힐 수 있다. Deployment의 서로 다른 추출 횟수는 반드시 서로 다른 물성 조합 수와 같지 않다.
- Catalog 생성에는 base seed 1,838,210,000에 shape, law, alpha, split의 명시적 offset을 더한다. 평가 seed로 catalog를 갱신하지 않는다. Label에 맞춘 particle 선택도 하지 않는다. Draw index RNG, factual Gaussian noise RNG, 4개 pool row 선택 RNG, paid response noise RNG, deployment selected sensor noise RNG는 분리한다.
- Factual은 아홉 profile 변위와 factual collision 두 velocity component다. 관측 noise의 표준편차는 각각 0.0016, 0.095다. Conventional/near probe의 outgoing velocity에는 성분당 0.027, 선택 가능한 near query view의 7개 profile에는 성분당 0.00030을 쓴다. Noise는 서로 다른 world 행, 서로 다른 성분에서 독립이다. 같은 환경·같은 물체의 sign/regime에 common random numbers를 재사용하므로 부호 및 부류별 결과는 paired contrasts로 취급한다.
- Evaluation oracle은 고정 cell과 α, 후보 prototype의 값과 질량, sign/regime prior, 센서 분포를 안다. 어느 prototype이 실제 관측을 냈는지는 모른다. Continuous distribution을 유한 bank로 근사한 뒤 그 결과를 연속 population의 exact oracle로 선언하는 방식과 구별하기 위해 실제 생성 분포부터 uniform finite support로 정의했다. 결과의 uncertainty는 이 catalog를 조건으로 한다. Catalog 교체, 실제 장비의 noise, 새로운 표면 계열에 대한 uncertainty는 별도 검증이 필요하다.

구현과 동결 분석 조건은 [physics38.py](physics38.py), [PLAN_KO.md](PLAN_KO.md)에 있다.

## GPU 원자료의 독립 검산 결과

별도 node56의 CPU 1개에서 Cartesian impulse 투영과 local normal/tangent 변환을 NumPy로 독립 구현했다. 64개 catalog의 모든 행에 2개 sign, 3개 regime, 5개 접촉을 적용한 3,932,160건에서 normal restitution, tangential momentum balance, stick threshold, 에너지 비증가를 확인했다. 같은 전체 catalog에 대한 PyTorch generator와 contact state 차이는 0건, velocity의 최대 성분 차이는 1.45×10⁻¹⁵ 이하였다. 전체 131,072개의 prototype vector가 서로 달랐다.

32개 block의 seed로 latent draw, sensor noise, 네 유료 접촉 및 각 deployment 물체의 선택 관측을 재생성했다. Held-out stick label과 네 번째 slip/stick의 불일치는 0건, 기록된 관측의 최대 성분 차이는 7.22×10⁻¹⁶ 이하였다. 262,144개 calibration factual draw는 64,295개 prototype, 네 번씩 256개 환경에 시행한 총 1,024개 유료 접촉은 1,019개 prototype, 32,768개 deployment draw는 25,816개 prototype에 해당했다. 부호·regime·observation noise를 바꾼 반복은 이 수에 중복 산입하지 않았다. [검산 원자료](physical_summary.json) · [독립 구현](audit_physics38.py)
