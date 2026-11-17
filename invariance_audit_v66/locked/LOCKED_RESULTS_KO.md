# 새 contour 두 종류에서 고정된 projection 강도를 확인한 실험

V45의 fold/ripple에서 선택한 α=0.75를 **새로운 family와 물체**에 그대로 적용했다. Wendland family는 국소적이고 유한한 범위에만 높이 변화가 있는 C2 radial basis를, dispersive family는 중심 양쪽에서 모양이 바뀌고 꼬리가 길게 이어지는 rational basis를 사용한다. 동일한 위치·너비·진폭 계수표를 사용하고 두 꼭짓점 부근을 변형시켰다. 두 family 모두 원점의 높이·기울기를 공유한다. 기존 48개 camera profile variant 각각과 5–40 mm의 701개 지점에서 높이와 tilt를 맞춰도 새 곡선의 최소 RMS 차이는 0.248–0.360 mm였다. 12개 새 catalog에서 6,144회 뽑은 물체 중 고유한 6차원 prototype은 5,945개다. 보관된 이전 실험 15종의 288개 deployment manifest와 각 calibration archive의 같은 파라미터를 대조해 정확히 중복된 물체는 없었다. [두 family의 함수](locked_geometry.py) · [독립성 검산](independence.json)

## 무엇을 결과 전에 고정했는가

새 support 980–991의 첫 label을 만들기 전, 기존 v45의 acquisition과 위치 gate, 두 source axis, α=0/0.75/1, 9개 source-trained readout 가중치, label cutoff와 균등 집계, 2개의 primary contrast와 방향을 고정했다. 각 catalog의 4개 환경당 128개 draw를 세 가지 shape variant, 두 명목 높이, 3개 접촉 offset, 2개 surface와 2개 마찰 법칙에서 비교했다. Query와 probe 중 구매하지 않은 관측은 NaN으로 숨겼다. 구매가 probe이면 세 물리 forecast는 같다. 광학 위치 추정은 221,184개 clip의 세 pre-impact frame에서 산출했다. 두 finite pressure radius(0.25 및 2 mm)의 결과를 평균하고, 같은 Brier를 모든 방법에 적용했다. 평균과 신뢰구간의 독립 단위는 12개 catalog다. α=0.75가 full보다 높은 loss, hard보다 낮은 loss를 낼 것으로 사전에 예측했다. [사전 계획](LOCK_KO.md) · [구현·source freeze hash](lock.sha256)

원본 evidence·세 physics forecast 576개 파일, route 144개, population 12개를 합친 732개 파일의 SHA manifest를 봉인했다. Source-trained readout과 추가 compression baseline의 prediction도 각각 봉인한 뒤 별도 CPU에서 label을 생성하는 설계다. Primary solver는 기존의 prescribed finite-pressure model을 사용한다. 두 family, 이 solver, 두 마찰 법칙에 조건부인 결과이며 독립적인 실물 접촉 성능을 추정하지는 않는다. [모든 입력·forecast hash](paid_routes.sha256)

## 사전 방향 두 개가 모두 재현됐고, 가족별 손익은 달랐다

| 평가 범위 | Full | α=0.75 | Hard | α=0.75 − Full | α=0.75 − Hard |
|---|---:|---:|---:|---:|---:|
| 12개 catalog, 두 family 균등 | **0.166229** | 0.175941 | 0.178108 | **+0.009712** | **−0.002167** |
| Wendland | 0.183312 | 0.182821 | 0.182843 | −0.000491 | −0.000022 |
| Dispersive | 0.149145 | 0.169060 | 0.173373 | +0.019915 | −0.004312 |

Pooled α=0.75 minus full의 simultaneous 95% interval은 [+0.009246, +0.010178], minus hard는 [−0.002279, −0.002055]다. 12개 catalog 각각에서 두 대조의 방향이 pooled 결과와 같았다. **같은 방향을 모든 형상에 적용할 수 있다는 주장에는 Wendland가 반례다.** Wendland에서 α=0.75는 full보다 0.000491 낮았고 Dispersive에서는 0.019915 높았다. 가족별 α=0.75 minus full의 탐색용 pointwise 95% interval은 Wendland [−0.000708, −0.000273], Dispersive [+0.019089, +0.020741]이다. [새 원자료 채점](locked_results.json) · [재생성 코드](locked_run.py)

같은 prototype·재료·높이·면적에서 hollow를 bulge로 바꾸자 stick/slip label이 달라진 비율은 Wendland **33.32% [32.46, 34.19]**, Dispersive **7.82% [7.40, 8.24]**였다. 두 interval은 family 두 곳에 대해 동시 보정했다. Label이 더 자주 바뀌는 family에서 이 고정된 decoder는 오히려 soft 처리로 mean Brier가 조금 낮아졌다. 이 관찰은 두 서로 다른 개입을 비교한다. 형상 개입은 simulator의 geometry와 label을 함께 바꾸고, likelihood 개입은 유료 관측과 label을 고정한 채 한 축의 가중치만 바꾼다. 한 개입의 변화율만으로 다른 개입의 손익을 예측하기 어렵다. [물리 개입 및 독립 검산](locked_score_audit.json)

Pressure label 생성 직전에 모든 forecast의 hash를 확인했다. Separate access가 필요한 label 파일 576개의 해시를 보존했고, 제곱합을 별도의 식으로 다시 계산한 모든 catalog·family·방법의 Brier 최대 절대오차는 5.56×10⁻¹⁷ 미만이었다. 576개 prediction·576개 label 등 전체 NPZ는 서버의 이 실험 디렉터리에 보존했다. [Label release 시점](FORECASTS_SEALED) · [Label checksum](labels.sha256)
