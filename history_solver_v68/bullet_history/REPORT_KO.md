# v68 · PyBullet로 첫 calibration부터 마지막 impact까지 실행한 보조 검정

두 friction law 모두에서 여섯 사전 방향·margin을 충족했다. Wendland와 Dispersive 각각에서 likelihood 억제의 부호를 검사했고, 두 family 평균에서 좌표 복원 이득을 별도로 검사했다. 네 source axis의 family별 억제 부호는 각 법칙에서 동일했다.

두 family, 두 새 law, 12개 catalog 전부를 PyBullet의 rigid frictional contact로 재생했다. 각 환경의 128개 candidate 전체로 13회의 probe 배정을 결정하고, 사전에 정한 환경별 첫 8개 prototype에서 평가했다. 4개 source axis와 8개 head는 v67에서 학습한 값을 고정했다. 네 차례 calibration에 대한 v38 log-odds correction도 다시 학습하지 않았다. [사전 계획](PLAN_KO.md) · [기존 PyBullet terminal 설계](../BULLET_PLAN_KO.md)

## 방법과 판정

같은 12개 catalog × 4개 환경에는 서로 다른 노이즈 반복과 shape·surface·law의 짝지은 개입이 있다. 비교의 독립 단위는 12개 catalog다. 둘째 engine을 사용하는 단계는 객관적으로 확인할 수 있다. ① calibration에 쓸 네 물체의 factual contact, ② 세 far probe와 네 번째 query-matched probe, ③ deployment candidate의 factual contact, ④ 선택된 물리 probe, ⑤ terminal impact velocity를 모두 PyBullet에서 계산했다. Query 영상 geometry와 optical route, sigma 및 noise seed는 공통이다. Deployment용 미구매 센서는 GPU 입력에서 NaN으로 가렸다. Calibration은 최종 log odds를 조절하고 관측 선택은 factual channel에 반응한다. [센서 생성](history.py) · [GPU 예측](forecast.py)

Factual/paid 접촉은 semi-infinite mesh 위 회전 고정 sphere(반경 0.25 mm), 20 μs step, 0.80 ms horizon, 0.10 mm mesh로 모사했다. 새 law는 최초 접촉 위치와 진입 접선 속도에서 계수를 정한다. Height 0의 10 mm 표면 mesh ordinate가 완전히 일치할 때만 계산을 공유했다. Signed velocity의 순서는 이전 factual feature와 같다. Geometry channel은 기존 tape에서 가져왔다. 센서용 simulated shot는 총 110,976회였고 미접촉 0회, 마지막 step에 여전히 하중이 걸린 경우 0회였다. 이 수에는 동등한 mesh의 계산 재사용이 반영돼 있다. [입력과 forecast checksum](forecasts_sealed.json) · [모든 counts](validity.json)

## Held-out Brier loss

같은 terminal label 및 같은 paid budget으로 비교했다. 차이는 낮을수록 왼쪽 방법에 유리하다. 6개 secondary contrast에 Bonferroni 동시 95% paired t interval을 사용했다. 두 family 반전은 각 법칙에서 Wendland의 구간 전체가 0 아래, Dispersive의 구간 전체가 0 위일 때 확인한다. 두 family를 평균한 복원 이득은 restored−dropped 구간 전체가 사전 기준 −0.002보다 낮을 때 확인한다.

| PyBullet law | Wendland soft − full | Dispersive soft − full | 두 family 평균 restored − dropped |
|---|---:|---:|---:|
| strengthening | -0.002339 [-0.003850, -0.000829] | +0.005745 [+0.003823, +0.007667] | -0.016961 [-0.020639, -0.013283] |
| weakening | -0.003062 [-0.004247, -0.001877] | +0.006553 [+0.004570, +0.008536] | -0.023440 [-0.027601, -0.019279] |

| Family | Law | Full | Soft | Hard | Dropped | Restored | stick / valid / total |
|---|---|---:|---:|---:|---:|---:|---:|
| Wendland | strengthening | 0.180810 | 0.178470 | 0.178083 | 0.175088 | 0.169949 | 17,717 / 27,648 / 27,648 |
| Wendland | weakening | 0.167266 | 0.164204 | 0.163664 | 0.161554 | 0.152893 | 17,629 / 27,648 / 27,648 |
| Dispersive | strengthening | 0.151287 | 0.157031 | 0.159031 | 0.149588 | 0.120805 | 12,587 / 27,648 / 27,648 |
| Dispersive | weakening | 0.130192 | 0.136744 | 0.138909 | 0.129711 | 0.091493 | 11,928 / 27,648 / 27,648 |

| 같은 PyBullet terminal, 다른 이력 | Law | Wendland soft − full | Dispersive soft − full | 두 family 평균 restored − dropped |
|---|---|---:|---:|---:|
| Bullet full history without calibration correction | strengthening | -0.002869 | +0.005994 | -0.016526 |
| Bullet full history without calibration correction | weakening | -0.003352 | +0.006506 | -0.022704 |
| compliant history and 4cal | strengthening | -0.002223 | +0.006240 | -0.017636 |
| compliant history and 4cal | weakening | -0.002979 | +0.007245 | -0.024254 |

동일 후보군에서 compliant와 PyBullet history 사이에 query/probe 선택이 바뀐 비율은 14.347%였다. 세 이력의 절대 risk, 12개 catalog별 mean, source seed별 차이를 JSON에 보관했다. [채점 결과](external_history_result.json) · [채점 코드](score.py)

이 분석은 기존 primary 설계 이후, 새 외부 terminal outcome을 이 분석자가 열기 전에 추가로 계획했다. 다른 프로세스가 같은 표본의 외부 terminal label을 생성하고 있었으며 GPU 예측기는 terminal 파일을 읽지 않았다. Sensor scene은 두 엔진 간 비교를 가능하게 하는 동일 생성기·law와 제한된 sphere geometry를 사용한다. 실물 접촉이나 새로운 object population으로의 전이를 평가하려면 별도의 holdout이 필요하다.
