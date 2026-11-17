# Normal impact에서 contact radius가 커지는 시간

Cross의 별도 normal impact에서, v69의 compliant solver에 **측정된 접촉시간**을 입력하고 spherical-cap geometry로 읽은 반경은 네 wire를 모두 실측보다 일찍 통과했다. 기준 restitution 0.75에서 20 mm에 도달한 시점은 관측 1.408 ms, 예측 0.964 ms였다. 최대 직경은 관측 42±1 mm, 예측 47.16 mm였다. 하나의 pressurized Wilson Australian Open ball이 수직으로 떨어진 대표 impact이므로, 이 결과로 oblique trial의 slip mechanism이나 solver 간 Brier loss 차이를 설명할 수 있는 범위는 제한된다. [Cross 원문](https://doi.org/10.1007/s12283-014-0159-x) · [저자 원고 §2, Fig. 4b](https://www.researchgate.net/publication/271948217_The_footprint_of_a_tennis_ball)

| 관측량 | 실측 | e=0.65 | e=0.75 | e=0.85 |
|---|---:|---:|---:|---:|
| 5 mm 최초 도달 시점 (ms) | 0.070 | 0.0495 | 0.0495 | 0.0495 |
| 10 mm (ms) | 0.218 | 0.2025 | 0.2023 | 0.2022 |
| 15 mm (ms) | 0.698 | 0.4815 | 0.4793 | 0.4778 |
| 20 mm (ms) | 1.408 | 0.9873 | 0.9644 | 0.9500 |
| 최대 직경 (mm) | 42±1 | 45.54 | 47.16 | 48.52 |

공 지름은 66.0 mm, 입사 속도는 7.7±0.1 m/s, measured contact duration은 4.72±0.01 ms이다. 논문은 aluminum block 위 약 1 mm에 배치한 직경 0.6 mm copper wire를 5 mm 간격으로 사용했다. 중앙 wire의 첫 접촉이 시간 원점이고 다른 wire들의 회로가 닫힌 시점으로 반경 증가를 추정했다. 측정값은 개별 wire에 도달한 시간이므로, 네 행을 독립적인 impact로 셀 수 없다. 연구자가 밝힌 약 1 mm의 curve-fit 공간 분해능과 실측 최대 직경 오차가 있다. [원 논문 §2](https://www.researchgate.net/publication/271948217_The_footprint_of_a_tennis_ball)

Compression angular frequency는 Ω=π(1+1/e)/(2T), unloading frequency는 eΩ이다. Loading의 indentation은 δ(t)=v sin(Ωt)/Ω이다. Spherical-cap 가정 a²=2Rδ−δ²에서 각 반경의 δ를 역산한 뒤 첫 도달 시점과 최대 반경을 계산했다. 이 readout은 v69의 normal impulse model에 추가하는 기하학적 가정이다. 얇은 shell의 bending, cloth 압축, wire를 누르는 조건과 COM indentation 사이의 관계에 따라 결과가 달라질 수 있다. 이에 대한 원인 분해는 아직 가능하지 않다. 실제 접촉시간을 조건으로 사용했으므로 duration prediction 정확도 역시 검증 대상 밖이다. 입사 속도와 기간의 공표 오차 양 끝 조합에 세 e값을 모두 적용해도 최대 직경은 45.25–48.82 mm였다. 이 범위는 parameter sensitivity이며 반복 실험에서 얻은 confidence interval이 아니다. [실행 코드](wire_grid.py) · [전체 결과](wire_grid_results.csv)

Slurm allocation 2744471의 node56에서 GPU를 숨긴 CPU step 272로 실행했다. 숫자는 적분의 근사 time step 없이 analytic solution에서 얻었다. Rigid sphere-plane PyBullet 설정은 이 관측에 대응할 finite footprint readout을 갖고 있지 않다. [실행 기록](run.json) · [실험 조건](PLAN_KO.md)
