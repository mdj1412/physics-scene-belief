# v41 물리 검산과 contact height 민감도

서버 node56의 CPU에서 576개 archive의 SHA-256을 확인한 뒤, 별도의 NumPy 구현으로 접촉을 normal/tangential 좌표에서 풀어 Cartesian velocity와 stick state를 대조했다. 측정 noise, 물체 추출, probe 순서, 표면 방향도 재생성했다. 모든 계산에 적용한 절대 오차 허용치는 3×10⁻¹²다.

| 검산 대상 | 결과 |
|---|---:|
| Calibration·deployment support 고유 물체 | 393,216 |
| 별도 passive pair 물체 / 전체 중복 | 1,536 / 0 |
| Support의 독립 충돌 재계산 / stick state 불일치 | 37,748,736 / 0 |
| Cartesian velocity / contour slope 최대 절대 오차 | 1.78e-15 / 1.36e-15 |
| 저장된 query label / 불일치 | 3,538,944 / 0 |
| Paid factual·velocity 재계산 최대 절대 오차 | 1.33e-15 |
| Deployment draw(복원 추출) / 실제 고유 prototype | 24,576 / 23,153 |
| Calibration pool / 실제 네 probe에서 쓰인 고유 prototype | 124,199 / 765 |

모든 support에서 접근 속도, kinetic/static friction의 순서, sliding 도중 예측된 접선 속도의 부호와 충돌 직후 kinetic energy의 비증가를 검사했다. 물체와 반복 noise가 여러 조건에 대응되므로 저장된 label 수는 독립 표본 수와 다르다. [검산 수치](physics_audit41.json) · [독립 구현](physics_audit41.py)

## 같은 물체에서 바뀐 실제 contact outcome

Base contour의 보관된 query label을 사용했다. 두 orientation, 세 regime, 여덟 환경, 128개 물체를 같게 가중한 뒤 12 panel·2 block을 평균했다. 나머지 두 contour variant는 설계상 같은 접촉 결과를 낸다. Height flip은 law별로 같은 latent object에서 두 높이의 결과가 다른 비율을 구한 뒤 두 law를 평균했다. 두 높이에서 같은 대조를 읽으므로 이 열은 행 쌍마다 동일하다.

| Shape | Height | Saturating stick | Crossover stick | Law flip | Height flip |
|---|---:|---:|---:|---:|---:|
| flare | 19 mm | 43.44% | 44.43% | 7.66% | 5.15% |
| flare | 31 mm | 41.96% | 42.87% | 4.42% | 5.15% |
| roll | 19 mm | 62.50% | 60.65% | 5.64% | 2.78% |
| roll | 31 mm | 60.63% | 59.55% | 2.74% | 2.78% |

[Archive 집계](contact_counts41.json) · [집계 코드](describe_contacts41.py)

## 실제 접촉이 명목 높이에서 벗어나는 경우

기본 설정은 수직 위치가 정확한 point contact다. 디자인한 quartic을 미분하여 각 명목 높이에서 실제 hit가 위 또는 아래로 δ만큼 벗어났을 때 가능한 두 |bulge slope − base slope| 중 큰 값을 계산했다. 표는 slope(dx/dy)의 차이를 나타낸다. Hollow의 차이도 절대값이 같다.

| Shape | 명목 높이 | δ=0 mm | δ=0.10 mm | δ=0.25 mm | δ=0.50 mm |
|---|---:|---:|---:|---:|---:|
| flare | 19 mm | 0.000 | 0.142 | 0.314 | 0.502 |
| flare | 23 mm | 0.000 | 0.142 | 0.314 | 0.502 |
| flare | 31 mm | 0.000 | 0.041 | 0.097 | 0.175 |
| roll | 19 mm | 0.000 | 0.092 | 0.203 | 0.325 |
| roll | 23 mm | 0.000 | 0.092 | 0.203 | 0.325 |
| roll | 31 mm | 0.000 | 0.041 | 0.097 | 0.175 |

이는 사후의 geometry-only 민감도 분석이다. Nominal height에서 보존되는 contact state가 오차가 있는 contact center나 유한 contact patch에서도 보존되는지는 별도의 실험이 필요하다. 이 오차에서 Brier loss는 측정하지 않았다.
