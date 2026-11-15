# v40 물리 검산 및 contact law 변화

서버 node56의 CPU에서 288개 archive를 검산했다. 각 latent catalog를 고정한 뒤 별도 NumPy 구현으로 normal과 tangential 좌표의 충돌을 풀어 Cartesian velocity와 stick state를 재계산했다. 독립 catalog는 12개다. 2개 split과 8개 α를 포함한 192개 support의 393,216개 latent vector에서 중복은 0건이었다.

| 검산 항목 | 결과 |
|---|---:|
| 전체 support에서 접촉 속도와 binary state를 대조한 횟수 | 28,311,552 |
| Cartesian velocity의 최대 절대 오차 | 1.55e-15 |
| 접촉 state 불일치 | 0 |
| 실제 query label 검산 수 / 불일치 | 1,769,472 / 0 |
| 실제 paid factual 및 velocity 재생성의 최대 절대 오차 | 8.88e-16 |
| 독립 물체 추출 횟수(복원추출, 조건 간 공유) / 고유 deployment prototype | 24,576 / 23,078 |
| Calibration pool의 고유 prototype / 네 probe에 실제 쓰인 고유 prototype | 124,205 / 767 |

각 support에서 강체의 접근·마찰 양수 조건, sliding에서의 접선방향 반전 부재, 충돌 후 운동에너지의 비증가를 함께 검사했다. Profile과 도함수는 위치를 중심에 맞춘 별도 polynomial로 계산했다. 두 구현의 최대 도함수 차이는 5.55e-17였다. [물리 검산 원자료](physics_audit40.json) · [독립 CPU 계산](audit_physics40.py)

## 실제 접촉 속도와 두 law의 차이

아래는 새 catalog의 base contour에서 실제 관측된 물리 상태다. 각 family에서 같은 latent 물체와 noise를 쓰고 law만 바꿨다. 세 contour의 접촉 위치와 tangent가 같아 이 물리적 대조는 세 변형에 공통이다. Fourth contact와 query contact는 서로 다른 물체에서 측정한다.

| Tangent family | 접촉 | 속력이 crossover(0.70)보다 낮은 비율 | Saturating stick | Crossover stick | Law 교체 시 stick/slip 전환 |
|---|---|---:|---:|---:|---:|
| shelf | fourth | 0.00% | 4.86% | 4.69% | 0.35% |
| shelf | query | 75.42% | 43.68% | 44.66% | 5.35% |
| ridge | fourth | 12.50% | 7.12% | 6.86% | 0.43% |
| ridge | query | 89.66% | 59.56% | 58.16% | 4.00% |

속력 열은 각 실제 draw당 한 번 계산했다. Stick과 전환율은 orientation 두 가지와 near-field regime 세 가지를 균등 가중했다. 이 빈도는 contact law가 실제로 작동한 범위를 보여준다. 어떤 센서의 보정이 target residual과 정렬되는지는 별도로 Brier를 비교해야 알 수 있다. [실제 속도와 law 대조](law_shift40.json) · [계산 코드](law_audit40.py)

실물과 연결되는 부분은 방향과 속도의 상호작용이다. Datta et al.은 약 57° 기울어진 titania nanorod 표면에 대해 네 sliding speed에서 force microscopy를 수행해 방향별 velocity weakening의 차이를 관측했다. Tramsen et al.은 sawtooth 표면에서 stiffness·aspect ratio·roughness를 조작해 특정 조합에서 friction anisotropy가 뒤집히는 것을 측정했다. 두 결과는 방향 효과의 크기와 부호가 유동적일 근거를 주지만, v40에서 속력 하나에 설정한 0.70 crossover나 rigid impact 중 동일 현상의 발생률을 실물로 검증한 자료는 아니다. [Datta et al.](https://arxiv.org/abs/2602.13343) · [Tramsen et al.](https://doi.org/10.1098/rsif.2017.0629)
