# v68 · 처음부터 contact law가 달라질 때의 GPU 예측

### 사전 고정과 측정 경로

v66의 Wendland·Dispersive object와 공통 noise를 재생하고 처음 factual contact부터 calibration과 deployment의 모든 물리 측정에 같은 contact law와 solver를 사용했다. ROI·height·surface·속도·probe 수는 같았다. v67의 source axis 네 개와 8개 readout weight를 그대로 불러왔고 checkpoint hash를 확인했다. 각 128개 deployment object 중 13개에서 물리 probe를 구매하고 나머지에서 contour query를 샀다. Q/P 중 고르지 않은 observation은 GPU inference 직전에 NaN으로 대체하고, 구현 안에서 모든 행의 mask를 검산했다. 각 법칙에 도착한 관측을 넣어 acquisition은 다시 계산했다.

Calibration의 물리적 경로를 예측에 실제 연결하기 위해, 사전에 정한 v38 continuous optimizer를 4개 측정 쌍에 적용했다. 높이 0의 초기 세 방향 probe로 원시 부호 cue를 추정하고 23 mm의 마지막 +1.10 m/s probe로 source gain table의 가중치를 선택했다. 네 개의 쌍에서 얻은 log-odds shift를 다섯 predictor 모두에 동일하게 적용했다. 별도의 raw 비교에서는 이 값을 0으로 두었고, prefix3 비교에서는 네 번째 측정 이전의 source posterior를 썼다. 모두 동일한 frozen acquisition과 source head를 쓴다. Calibrated forecast는 v67 predictor에 추가된 변환이다. Solver/law·실제 sign·contact outcome은 predictor 입력에 없다. [사전 계획](FORECAST_PLAN_KO.md) · [테이프 생성](tapes68.py) · [예측 코드](predict68.py)

### Held-out Brier

| Solver | Law | Wendland soft−full | Dispersive soft−full | 평균 restored−dropped |
|---|---|---:|---:|---:|
| prescribed | strengthening | -0.003089 [-0.003612, -0.002567] | +0.006714 [+0.005555, +0.007873] | -0.012642 [-0.014441, -0.010842] |
| prescribed | weakening | -0.003596 [-0.004125, -0.003068] | +0.007442 [+0.006443, +0.008441] | -0.022122 [-0.024144, -0.020100] |
| compliant | strengthening | -0.001036 [-0.001426, -0.000646] | -0.000128 [-0.000818, +0.000561] | +0.000559 [-0.000548, +0.001667] |
| compliant | weakening | -0.001525 [-0.001903, -0.001148] | +0.000965 [+0.000150, +0.001780] | -0.006015 [-0.007432, -0.004599] |

각 row는 12개 paired catalog에서 네 source axis를 먼저 평균했다. 새 independent compliant solver의 두 law × 세 primary contrast는 Bonferroni 동시 95% 구간이다. Prescribed row는 같은 폭을 적용한 secondary 대조다. 각 contact의 binary Brier는 두 radius와 세 offset에 동일 가중을 두었다. Held-out family는 고정된 두 종류다.

| Solver | Law | Family | Full | Soft | Hard | Dropped | Restored |
|---|---|---|---:|---:|---:|---:|---:|
| prescribed | strengthening | wendland | 0.196416 | 0.193326 | 0.192808 | 0.184409 | 0.179918 |
| prescribed | strengthening | dispersive | 0.183233 | 0.189947 | 0.191974 | 0.171378 | 0.150585 |
| prescribed | weakening | wendland | 0.169496 | 0.165900 | 0.165266 | 0.161375 | 0.153228 |
| prescribed | weakening | dispersive | 0.128696 | 0.136138 | 0.138343 | 0.126779 | 0.090681 |
| compliant | strengthening | wendland | 0.270945 | 0.269909 | 0.269799 | 0.258528 | 0.263239 |
| compliant | strengthening | dispersive | 0.230536 | 0.230408 | 0.230601 | 0.231238 | 0.227646 |
| compliant | weakening | wendland | 0.245223 | 0.243697 | 0.243488 | 0.233880 | 0.236469 |
| compliant | weakening | dispersive | 0.194959 | 0.195924 | 0.196413 | 0.201818 | 0.187199 |

### Calibration 입력과 source axis별 민감도

| Compliant law | 보정 입력 | Wendland soft−full | Dispersive soft−full | 평균 restored−dropped |
|---|---|---:|---:|---:|
| strengthening | raw | -0.001403 | -0.000245 | +0.001198 |
| strengthening | prefix3 | -0.001283 | -0.000348 | +0.000701 |
| strengthening | cal4 | -0.001036 | -0.000128 | +0.000559 |
| weakening | raw | -0.001671 | +0.000808 | -0.005485 |
| weakening | prefix3 | -0.001556 | +0.000679 | -0.005928 |
| weakening | cal4 | -0.001525 | +0.000965 | -0.006015 |

| Law | Source seed | Wendland soft−full | Dispersive soft−full | 평균 restored−dropped |
|---|---:|---:|---:|---:|
| strengthening | 670101 | -0.001010 | -0.000254 | +0.003621 |
| strengthening | 670102 | -0.001181 | +0.000185 | +0.003246 |
| strengthening | 670103 | -0.001111 | -0.000071 | -0.004083 |
| strengthening | 670104 | -0.000844 | -0.000373 | -0.000547 |

| weakening | 670101 | -0.001566 | +0.000723 | -0.003263 |
| weakening | 670102 | -0.001670 | +0.001733 | -0.004064 |
| weakening | 670103 | -0.001571 | +0.001140 | -0.010242 |
| weakening | 670104 | -0.001294 | +0.000265 | -0.006492 |


| Solver | Law | 평균 절대 4쌍 shift | 네 번째 관측 추가 후 평균 절대 shift 변화 | v66 원래 acquisition과 다른 물체 비율 |
|---|---|---:|---:|---:|
| prescribed | strengthening | 0.228371 | 0.167497 | 0.184% |
| prescribed | weakening | 0.165660 | 0.147394 | 0.184% |
| compliant | strengthening | 0.228537 | 0.167233 | 6.532% |
| compliant | weakening | 0.165612 | 0.147346 | 6.543% |

두 solver에서 measured calibration probe 2,304행과 deployment probe 73,728행 모두에서 law 교체가 응답을 바꿨다. Compliant solver에서 두 법칙의 쌍별 차이를 성분별 RMS로 보면 calibration probe는 0.13948 m/s, deployment probe는 0.14515 m/s였다. 원점의 0 lateral speed factual contact에서는 0.00000608 m/s에 그쳤다. 같은 물체 표면의 광학 profile은 두 law에서 동일했다. [입력 검산](input_audit68.json)

서버 NVIDIA RTX PRO 6000 Blackwell Server Edition, Slurm allocation 2744471에서 소스 검증·GPU inference·저장을 포함해 283.5초가 걸렸다. v67의 원래 tape와 frozen head를 재생한 최대 probability 차이는 3.28e-06이었다. 독립 직접 Brier 재계산 최대 오차는 2.78e-17이었다. [실행 기록](gpu_run.json) · [24개 forecast shard 봉인](FORECASTS_SEALED) · [catalog별 원자료](score_main68.json) · [채점 코드](score_main68.py)

별도 release audit에서 일부 terminal trajectory가 엄격한 global separation 조건을 통과하지 않아 33.33 ms의 finite-time velocity를 nominal label로 쓰고 별도 unresolved mask를 저장했다. 이 표는 해당 nominal label의 결과다. 최초 한도 내 완료된 trajectory와 모든 history tape는 유지했다. [수치 종료 규약](SOLVER_STATUS_KO.md) · [미확정 label 민감도 계산](unresolved_bounds68.py)
