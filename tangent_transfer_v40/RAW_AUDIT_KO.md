# v40 독립 scoring 검산과 선택된 sensor별 Brier

이번 검증에서는 288개 block의 SHA-256, 예측과 생성 코드의 동결 기록, calibration과 deployment의 seed, QMC bank seed를 원자료와 대조했다. 실제 독립 단위는 12개 catalog panel이다. 각 시작 방향에서 모든 12개 조건이 같은 bank·물체 추출을 썼다. 384개 고유 evaluation seed와 48개 고유 bank seed를 확인했다.

CPU에서 첫 세 calibration 관측의 source likelihood부터 signed cue와 네 번째 관측의 최적 보정량을 다시 구했다. GPU worker와의 최대 절대 차이는 cue 2.44e-15, weight 8.10e-15였다. 확률과 held-out label로 복원한 두 정책 Brier의 최대 절대 차이는 8.33e-17였다. 각 block에서 115/128개에는 query view, 13/128개에는 probe가 배정됐다.

동일 family·law의 base 대 hollow/bulge에서 1,179,648개 matched label과 1,277,952개 probe-branch probability를 비교했고 모두 정확히 일치했다. 두 값은 각각 동일 표본의 paired comparison 수다. 행들이 noise 및 물체를 공유한다.

| 조건 | Source expected Δ | 실제 전체 Δ | Query view 선택 물체 Δ | Probe 선택 물체 Δ |
|---|---:|---:|---:|---:|
| shelf / base / saturating | -0.001437 | +0.000041 | +0.000149 | -0.000909 |
| shelf / base / crossover | -0.001495 | +0.001896 | +0.002008 | +0.000899 |
| shelf / hollow / saturating | -0.001437 | -0.001138 | -0.001164 | -0.000909 |
| shelf / hollow / crossover | -0.001495 | +0.000870 | +0.000866 | +0.000899 |
| shelf / bulge / saturating | -0.001437 | +0.000929 | +0.001137 | -0.000909 |
| shelf / bulge / crossover | -0.001495 | +0.002603 | +0.002795 | +0.000899 |
| ridge / base / saturating | -0.001048 | -0.000848 | -0.000474 | -0.004158 |
| ridge / base / crossover | -0.001124 | +0.000093 | +0.000542 | -0.003877 |
| ridge / hollow / saturating | -0.001048 | -0.000236 | +0.000208 | -0.004158 |
| ridge / hollow / crossover | -0.001124 | +0.000589 | +0.001094 | -0.003877 |
| ridge / bulge / saturating | -0.001048 | -0.001676 | -0.001395 | -0.004158 |
| ridge / bulge / crossover | -0.001124 | -0.000621 | -0.000253 | -0.003877 |

모든 열에서 Δ=optimizer−fixed이며 음수가 optimizer의 이득이다. 고정값을 후보에 포함한 source gain table은 정의상 expected Δ가 항상 0 이하이지만, 새 물체의 실제 손실 차이 부호는 어느 쪽도 가능하다. 한 cell의 전체 Δ는 해당 sensor 열을 115/128 및 13/128로 가중한 합이다. Sensor가 무작위 배정되지 않았으므로 두 열의 차이를 observation을 강제로 바꿨을 때의 효과로 해석할 수 없다. 같은 tangent family와 law에서 contour를 바꾼 비교는 같은 물체의 두 측정을 대응시켰다. [독립 검산 결과](audit40.json) · [검산 코드](audit40.py)
