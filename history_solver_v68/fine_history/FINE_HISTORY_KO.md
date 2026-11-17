# 전체 fine history와 전체 fine terminal outcome의 교차 검산

기본 해상도와 부분 표본의 fine 결과를 확인한 뒤 추가한 민감도 분석이다. 같은 12개 catalog 전체, 네 source axis, source weight, 네 개 calibration pair, acquisition 및 noise를 고정했다. Contact support를 9→17로 늘리고 step size를 절반으로 줄여 calibration, deployment factual, paid probe, terminal contact를 모두 다시 생성했다. 표의 9/17은 앞에서부터 관측 이력/terminal solver의 공간 해상도이고 temporal resolution도 함께 변한다. 두 hybrid row는 두 경로를 분리하기 위한 계산이다. 판정 단위는 이번 sensitivity에서 지정한 6개 paired contrast이며 각 interval은 해당 6개에 대해 보정했다. [계획](PLAN_KO.md) · [예측 봉인](FORECASTS_SEALED) · [원자료](fine_scores.json)

| History/terminal | Law | Wendland soft−full | Dispersive soft−full | 두 형상 평균 restored−dropped |
|---|---|---:|---:|---:|
| coarse/coarse | strengthening | -0.001036 [-0.001426, -0.000646] | -0.000128 [-0.000818, +0.000561] | +0.000559 [-0.000548, +0.001667] |
| coarse/coarse | weakening | -0.001525 [-0.001903, -0.001148] | +0.000965 [+0.000150, +0.001780] | -0.006015 [-0.007432, -0.004599] |
| coarse/fine | strengthening | -0.000680 [-0.001037, -0.000323] | +0.000175 [-0.000542, +0.000892] | +0.000024 [-0.001091, +0.001139] |
| coarse/fine | weakening | +0.000376 [+0.000040, +0.000713] | +0.001124 [+0.000328, +0.001919] | -0.006621 [-0.007990, -0.005252] |
| fine/coarse | strengthening | -0.001083 [-0.001451, -0.000715] | -0.000129 [-0.000776, +0.000518] | +0.000576 [-0.000587, +0.001740] |
| fine/coarse | weakening | -0.001567 [-0.001945, -0.001188] | +0.000952 [+0.000185, +0.001718] | -0.005927 [-0.007380, -0.004474] |
| fine/fine | strengthening | -0.000724 [-0.001057, -0.000390] | +0.000162 [-0.000506, +0.000830] | +0.000044 [-0.001114, +0.001202] |
| fine/fine | weakening | +0.000351 [+0.000014, +0.000687] | +0.001091 [+0.000338, +0.001844] | -0.006546 [-0.007952, -0.005140] |

| 해상도 변화, paired Δ | Law | Wendland soft−full | Dispersive soft−full | 두 형상 평균 restored−dropped |
|---|---|---:|---:|---:|
| fine/fine minus coarse/coarse | strengthening | +0.000313 [+0.000237, +0.000389] | +0.000290 [+0.000163, +0.000417] | -0.000515 [-0.000656, -0.000375] |
| fine/fine minus coarse/coarse | weakening | +0.001876 [+0.001662, +0.002090] | +0.000126 [-0.000003, +0.000255] | -0.000531 [-0.000738, -0.000323] |
| fine/fine minus coarse/fine | strengthening | -0.000044 [-0.000099, +0.000011] | -0.000013 [-0.000108, +0.000081] | +0.000020 [-0.000126, +0.000166] |
| fine/fine minus coarse/fine | weakening | -0.000025 [-0.000046, -0.000005] | -0.000032 [-0.000126, +0.000062] | +0.000075 [-0.000060, +0.000211] |
| coarse/fine minus coarse/coarse | strengthening | +0.000356 [+0.000306, +0.000407] | +0.000303 [+0.000195, +0.000411] | -0.000536 [-0.000617, -0.000454] |
| coarse/fine minus coarse/coarse | weakening | +0.001902 [+0.001695, +0.002108] | +0.000158 [+0.000042, +0.000275] | -0.000606 [-0.000798, -0.000415] |

| Fine/fine law | 형상 | Full | Soft | Hard | Dropped | Restored |
|---|---|---:|---:|---:|---:|---:|
| strengthening | Wendland | 0.273469 | 0.272745 | 0.272699 | 0.261486 | 0.266119 |
| weakening | Wendland | 0.249640 | 0.249991 | 0.250186 | 0.242292 | 0.244891 |
| strengthening | Dispersive | 0.224862 | 0.225024 | 0.225319 | 0.225582 | 0.221038 |
| weakening | Dispersive | 0.189577 | 0.190668 | 0.191224 | 0.196488 | 0.180798 |

미분리 trajectory의 binary label을 각각 두 가지 모두로 두어 얻은 동시 95% interval의 외곽이다. 각 법칙의 순서는 Wendland, Dispersive, 두 형상 평균 복원이다.

- strengthening_0: [-0.001057, -0.000390]
- strengthening_1: [-0.000506, +0.000830]
- strengthening_2: [-0.001114, +0.001204]
- weakening_0: [+0.000014, +0.000687]
- weakening_1: [+0.000338, +0.001844]
- weakening_2: [-0.007952, -0.005137]

측정용 fine contact 호출 304,128행 중 horizon 연장 0행, 여전히 미분리 0행이다. 원 coarse catalog 위험 재계산 최대 차이는 1.1e-16였다. Fine GPU 예측 실행은 130.5초였다. 이미 첫 32개 fine label의 경향을 알고 있었고 전체 fine label 생성은 별도 CPU에서 병행했다. 전체 fine history 예측을 별도 봉인한 후 scoring을 시작했다. [파이프라인](fine_predict.py) · [2×2 검산](score_fine.py)
