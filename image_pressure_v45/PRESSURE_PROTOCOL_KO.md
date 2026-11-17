# 최초 접촉점과 총 normal impulse를 맞춘 pressure intervention

## 실험이 바꾸는 물리량

각 물체에서 실제 최초 접촉 좌표를 명목 위치(19 또는 31 mm)와 숨긴 offset(−0.5, 0 또는 +0.5 mm)의 합으로 정한다. Torque와 회전을 억제한 질량 1 kg 물체가 수평 +0.425, 수직 −1 m/s로 접근한다. 세 비교군 모두 같은 점에서 닿기 시작한다. 넓이가 0인 point와 peak 반지름 0.25·2.0 mm인 두 원형 contact patch를 조사한다. 이는 v44의 rigid tip을 바꾸는 조작과 달리, 최초 접촉과 누적 normal impulse를 유지한 채 **모형 안의 pressure field**를 직접 지정하는 실험이다. 같은 pressure law에서 면적을 바꾸면 peak pressure도 함께 달라져야 총하중이 보존된다.

Cumulative scalar normal impulse의 비율을 t∈[0,1]로 두고 radius를 a(t)=a_max sin(πt)로 둔다. 식의 너비는 처음과 끝에서 0이고 중간에 가장 크다. Projected plane의 하중 밀도는 disk 안에서 sqrt(1−r²/a²)에 비례한다. 매 순간 적분값을 1로 정규화해 최초 접촉 위치와 normal-load centroid가 세 군에서 같다. 곡면상 물리 밀도는 이 projected density를 local arc stretch로 나눈 값이다. 두 유한 patch의 peak projected area는 0.19635와 12.56637 mm²다. Effective pressure area는 (∫p dA)²/∫p² dA로 정의하며 projected plane에서는 기하 면적의 8/9이다. 실제 곡면의 effective pressure area도 local slope를 적분해 별도 저장한다. y 방향으로 일정한 extruded contour를 사용한다. 압력 폭을 서로 다른 값으로 고정했으므로 압력 분포의 endogenous formation이나 실제 material stiffness를 추론하는 검증은 아니다.

## 접선 운동과 outcome

모든 arm의 normal impulse magnitude J는 중심에서의 closing speed × (1+restitution) × 질량으로 고정한다. 국소 normal과 tangent는 surface derivative에서 직접 구하며 local friction은 동일 v43 surface·law의 접근 속도 계수를 사용한다. 같은 물체에서는 동일한 외력·중심·material field·law·속도·restitution과 동일 입력 확률을 유지하고 radius만 바꾼다. Disk 전체의 normal force를 합쳤을 때 방향까지 같게 고정하는 것은 아니다. 법선 분포 변화는 조사하려는 물리 경로다.

1. 접촉점마다 local normal impulse의 중심 접선 방향 성분을 누적해 incoming central tangential momentum에 더한다.
2. 하나의 중심 tangent translational mode를 가정한다. Local Coulomb traction이 이 mode에 가할 수 있는 impulse magnitude는 µ dJ × |t_local·t_center|다. 이를 면과 loading history에 적분한 값이 가용한 friction impulse다.
3. 번역 운동 최대 소산 해에서 terminal speed는 max(|driving momentum|−friction capacity,0)/mass다. 0.020 m/s 이하면 stick proxy로 판정한다. 0.010 및 0.040은 threshold 감도 분석이다. No spin·no torsion·단일 slip mode이므로 국소 incipient slip, separation이나 hysteresis까지 측정하지 않는다.

## 검사와 보관

- Primary는 12-node Gauss–Legendre로 x 방향에 투영된 pressure를 적분하고, 대칭을 이용해 반 loading cycle에 6-node quadrature를 사용한다. 전 물체에서 24×12로 독립 재계산한다. Peak surface effective area도 두 정밀도에서 각각 16·24 nodes로 평가한다. Point control은 접촉점 하나로 계산한다.
- Target 실행 전 일정 경사·일정 마찰 200개의 analytic negative control(모든 radius에서 같은 답), 평면 위 독립적인 quadratic friction positive control(반지름 제곱에 따른 정확한 잔류 속도), unit load와 centroid, 마찰식의 독립 scalar 구현을 검산한다.
- 12개 catalog × 4환경 × 환경당 128개 물체. 각 panel의 48개 shape/surface/law/variant/height 조합을 동일 물체에서 평가한다. 결과 archive는 [radius, signed offset, environment, object] 축 [3,3,4,128]의 label과 잔류 속도, peak effective surface area, 중심별 scalar normal impulse를 보관한다. Outcome을 이용해 radius나 threshold를 선택하지 않는다.
- 이 pressure solver는 지정한 물리 개입을 분리하기 위한 reduced-order impulse model이다. V44의 Bullet 위에 별도의 압력 모형을 설치한 것이 아니다. Static Coulomb friction에서는 평면·균일 물성의 접선 motion이 총하중에만 좌우되므로, 면적만 바꾼 차이는 곡률에 따른 normal 변화와 공간별 friction coefficient의 가중이 달라질 때 생긴다. [구현](pressure45.py) · [독립 fixture](pressure_neutral45.json)
