# v50 · 새로운 물체와 표면, held-out contact labels

12개 catalog에서 **calibration용 물체 6,144개와 deployment용 물체 6,144개의 물리 parameter vector 사이에 중복이 0개**였다. 같은 물체·표면에서 full-load 변형 방향만 바꾸면 primary stick/slip label의 **36.72%**가 바뀌었다(12개 catalog의 paired 95% CI 35.90–37.55%). 방향에 관한 추가 관측이 미래 outcome을 구별할 여지가 있는 환경이다. [수치 검산](world50_audit.json)

| 조사 대상 | 구현과 검산 |
|---|---|
| 물체 형상 | saddleback, terrace, swale, scallop이라는 새 contour 네 종류에 서로 다른 asymmetric shoulder 및 bending profile을 지정했다. 9–40 mm의 공통 801개 좌표에서 v47·48의 기존 여덟 contour와의 최소 profile RMS 차이는 0.1495 mm, 새 family 사이의 최소 차이는 0.3118 mm였다. 19·31 mm 중심에서 ±2.5 mm 안쪽 unloaded geometry는 네 family가 정확히 같아 같은 국소 외관에서도 signed bending이 달라질 수 있다. |
| 표면 | Mosaic에는 폭과 부호가 다른 두 material insert를 넣었다. 19→31 mm에서 isotropic friction multiplier가 1.063→0.947로 내려간다. Nap은 brushed surface의 방향성 전환을 구현했고 같은 위치에서 0.967→1.063으로 올라간다. Anisotropy multiplier는 각각 0.991→0.244 및 0.897→−0.397이다. Factual apparatus와 PyBullet이 같은 spatial field를 공유한다. 실제 재질 측정에서 fitting한 계수는 없다. |
| 독립 추출 | 환경별 128개씩 복원 없이 추출했다. Deployment는 support 900–911·lane 20, calibration은 920–931·lane 10의 독립 seed를 썼다. 각 catalog의 네 환경은 두 surface와 기존 rate law 두 개를 균형 있게 포함한다. 같은 batch에서 calibration이 deployment로 전달되는 상황을 만들기 위해 upper-level population parameter를 공유한다. 같은 prototype의 다른 camera noise가 calibration과 평가 양쪽에 들어갈 수 없음을 6개 physical coordinate로도 검산했다. |
| Preload 신호 | 0.4 N weak command의 two-fiducial signed displacement는 positive state에서 family별 0.07876–0.08527 mm다. 두 query 높이의 family별 무잡음 차이는 수치 정밀도 이내다. 실제 센서에는 gain, bias, readout noise가 더해진다. Full-load deformation의 기준 geometry에 대해 weak displacement를 0.2배로 둔 것은 재료 응답 가정이다. |
| Contact solver | PyBullet DIRECT에 동일한 locked-spin sphere가 닿는 순간부터 normal·friction impulse를 계산하게 했다. Primary는 0.0125 ms time step·0.025 mm mesh·80 iterations, ultra는 0.00625 ms·0.0125 mm·160 iterations다. 0.25/2 mm sphere, 세 offset, 두 height, 세 full states를 교차했다. 각 해상도에서 884,736 label slot에 해당하며 inert geometry의 family 중복을 접어 실제로는 663,552 trajectory를 계산했다. |

Primary에서 전체 weighted stick 비율은 **51.19% [50.50, 51.89]**, ultra는 **47.91% [47.25, 48.58]**였다. 동일한 조건에서 **3.365% [3.295, 3.434]**의 label이 바뀌었다. 두 해상도 모두에서 event threshold는 첫 contact tangent 방향 잔류 속도 ≤0.020 m/s로 유지했다. Primary에서 inert/negative/positive state의 stick 비율은 52.81/67.71/31.44%, ultra에서 49.32/64.97/28.03%였다. 부호별 차이는 재현되며 절대 확률은 해상도에 민감하다. Weak-to-full polarity retention은 실험에서 지정한 관계다. PyBullet의 rigid impact는 주어진 full-load geometry에서 outcome을 계산하며, 실물의 탄성·잔류 변형·reset 성공률까지 검증하지 않는다. [물리 protocol](PHYSICS_PLAN_KO.md) · [물체 구현](world50.py) · [Population](population50.py) · [Solver 구현](solver50.py) · [검산 코드](audit_world50.py)

서버의 allocation 2744468에서 6개 CPU worker를 사용했다. GPU inference와 label generation은 별도 실행했다. 두 해상도 모두 12개 catalog가 정상 종료됐고, label과 보관한 slip speed의 threshold를 모든 슬롯에서 대조했다. [동결한 코드](solver50.sha256) · [Label·population checksum](physics_results50.sha256)
