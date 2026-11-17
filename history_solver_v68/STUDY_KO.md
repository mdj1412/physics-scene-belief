# v68 · 무엇을 바꿨고 무엇을 비교했는가

v67에서는 새로운 마찰 법칙이 적용되는 시점이 마지막 충돌이었다. 앞선 시행에서 관측한 운동은 같은 상태로 두고 정답을 바꾸는 실험이었다. v68에서는 **표면에 처음 닿는 순간부터** 같은 법칙을 적용한다. 같은 물체라도 calibration, 구매한 probe, 마지막 충돌의 관측값이나 정답이 함께 달라질 수 있다. [v67 설계](../law_axis_v67/PLAN_KO.md) · [v68 사전 계획](FORECAST_PLAN_KO.md)

## 사전 가설과 비교 단위

- **Projection 손익 반전.** 고정된 위치 규칙에서 하나의 source axis에 대한 likelihood를 부분적으로 약화한 soft 예측의 Brier에서 full 예측의 Brier를 뺀다. 각 law에서 Wendland의 차이는 음수이고 Dispersive의 차이는 양수여야 한다.
- **좌표 복원.** v67에서 학습한 같은 구조의 readout을 가져온다. 한 모델은 학습된 축의 좌표를 제거한 입력을 받고 다른 모델은 부호까지 보존한 입력을 받는다. 각 law에서 두 형상 평균 restored−dropped의 동시 95% 구간 상한이 사전 지정한 −0.002보다 낮으면 실용 기준도 충족한 것으로 판정한다. 0.002는 연구 설계값이며 실제 robot의 효용 단위는 별도다.
- 동일한 12개 catalog에서 4개 environment × 128개 draw를 사용한다. v66의 source와 분리된 물체지만 v66·v67에서 분석한 같은 target 물체를 이번에도 의도적으로 짝지었다. Source recalibration seed 네 개, 위치 offset 세 개, tip/pressure scale 두 개와 관측 noise repeat는 독립 target catalog 수를 늘리지 않는다. 네 seed는 서로 다른 축·readout의 Brier를 각각 구한 뒤 평균했다.
- Confirmatory arm은 independent compliant solver다. 두 law × 세 contrast에 Bonferroni 동시 95% paired t interval을 사용한다. Prescribed solver, calibration ablation, PyBullet, cutoff와 해상도 대조는 진단용이다.

## 예측과 정답에 허용한 정보

| 단계 | 같은 물체에서 달라질 수 있는 값 | 예측에 들어가는 정보 |
|---|---|---|
| Calibration | 각 물체의 factual 충돌과 4회의 probe 응답. 앞 세 번은 높이 0에서 방향을 교대하고 네 번째는 23 mm, +1.10 m/s | 네 쌍의 관측으로 이전 source에서 정한 v38 optimizer가 공통 log-odds 보정량을 계산 |
| Deployment factual | 같은 world의 9개 광학 측정과 2개 충돌 후 속도 성분에서 속도 성분이 달라질 수 있음 | 기존 acquisition을 이 factual evidence로 다시 실행. 환경당 128개 물체에서 13개 물리 probe, 나머지 115개 contour query |
| Paid observation | 산 물체의 물리 probe는 새 law와 solver로 생성. 광학 측정은 형상이 같아 동일 | 코드가 미구매 채널을 NaN으로 가린 뒤 decoder를 실행. 네 calibration의 보정은 terminal predictor에 공통 적용 |
| Terminal impact | 같은 물체에서 두 law 및 각 solver로 접촉 후 slip speed를 계산 | 기준은 0.020 m/s 이하의 stick. Outcome은 scoring에만 사용 |

Source model을 고정한 이유는 target에서 새로 적합한 학습이 원래 효과를 복구하는 경로를 닫기 위해서다. v67 predictor에는 별도 calibration 보정이 없었으므로 v68에서도 업데이트를 생략한 같은 predictor를 돌려 이 변경을 따로 확인했다. Calibration은 다음에 무엇을 살지 정하는 입력으로 직접 들어가지는 않는다. 새 법칙으로 생성한 factual evidence가 acquisition에 들어간다. [입력 생성](tapes68.py) · [GPU 예측](predict68.py) · [추가 교차 검산 계획](CROSSOVER_PLAN_KO.md)

## 독립 solver에 대한 해석

두 법칙은 접촉점의 진입 접선 속도에 따라 반대로 변하는 마찰계수를 사용하고 충돌 한 번 동안 그 계수를 유지한다. 두 엔진이 공유하는 입력은 contour, 물체와 재료, 속도, noise, 기준 높이 및 surface의 공간별 마찰 함수다. Prescribed 모델은 미리 정한 압력 가중치로 impulse를 합친다. Independent compliant 모델은 이동 중 각 지점의 침투로부터 매 step의 힘과 하중 분배를 계산하므로 prescribed pressure 경로나 최종 속도 계산식을 재사용하지 않는다. 두 모델은 2D 병진 운동과 고정된 orientation을 가정한다. [새 solver 설계](SOLVER_PLAN_KO.md) · [접촉 모델 간 실험 비교](https://simple-robotics.github.io/publications/contact-models/) · [실측 평면 충돌 비교](https://proceedings.mlr.press/v78/fazeli17a.html)

문헌 검토에서는 실제 impact와 pushing, simulator별 수치 오차, 관측으로 물리 parameter를 보정하는 방법, representation의 정보 손실을 구분했다. 원 연구 6편의 질문·표본·방법·결과·한계, 서로 다른 측정량 때문에 생기는 견해 차이, evidence map, 연구 공백, 검증 가능한 후속 질문 10개를 부록에 정리했다. [문헌 및 연구 심사](LITERATURE_KO.md)
