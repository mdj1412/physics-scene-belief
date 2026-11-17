# v44 연구 보고서 · 접촉 도구가 달라지면 어떤 contour 정보를 보존해야 하는가

**가장 잘 재현된 결과는 19 mm 높이의 fold였다.** V43의 point contact 결과만 보면 해당 위치에서 contour 변형 방향을 지우는 편이 유리했다. Tip의 실제 모양을 고려한 사전 예측은 네 도구 중 세 개에서 손해를 예상했다. 새 물체를 PyBullet로 충돌시킨 결과, 세 조건 모두에서 손해가 확인됐다. 반대로 31 mm 높이에서 긴 tip이 projection을 유리하게 만든다는 예측은 점추정에서 맞았지만 개별 조건의 동시 95% 구간은 0을 포함했다.

**네 도구를 같은 비중으로 쓰면 fold와 ripple 모두 contour 방향을 보존하는 편이 유리했다.** 이전 실험에서 fold의 정확한 위치와 어긋난 위치 사이에 관찰한 평균 손익 반전은 새 solver에서 재현되지 않았다. Ripple에서 위치가 어긋날 때의 손해는 재현됐다. 이 결과를 읽을 때는 각 접촉 위치의 손익과 여러 위치를 묶은 평균을 구분해야 한다.

![왼쪽은 도구 평균 projection 손익과 동시 95% 구간, 오른쪽은 사전 예측 대비 독립 solver가 산출한 손익이다.](contact_patch_v44.png)

그림 A의 양수는 projection이 손해라는 뜻이다. B의 마름모는 도구를 고려했을 때 v43에서 예상한 손익 방향이 바뀐 다섯 조건이다. 점 48개는 같은 물체와 catalog를 공유한다.

## 1. 어떤 예측을 언제 고정했나

물체의 7점 query profile에는 기준 형상과 국소 변형이 함께 보인다. Original decoder는 전체 profile을 사용하고, privileged projection은 실제 변형 방향을 알고 whitened profile에서 그 성분을 지운다. 지운 방향이 센서 모양만 바꾸면 예측이 개선될 수 있다. 같은 방향이 실제 tip이 닿는 위치와 접선까지 바꾸면 stick을 구별하는 단서도 함께 사라진다. 이를 같은 물체에서 **D = Brier(projected) − Brier(original)**로 측정했다. Brier는 예측한 stick 확률과 binary outcome의 제곱오차다. D는 두 방법의 차이이므로 양수이면 projection이 손해다.

- **사전 예측**은 완료된 v43 물체와 예측 확률만 사용했다. 각 도구가 과거 contour의 어디에 처음 닿는지 정적으로 구하고, 그 접선·마찰을 v43의 impulse law에 넣어 대리 outcome을 구했다. 각 물체에서 대리 outcome과 두 frozen 확률로 예상 D를 계산했다. 설계자는 family·tip·offset을 알고, 배포 decoder에는 이 정보를 주지 않았다. 비교 기준은 도구와 관계없이 v43의 같은 조건 D를 복사하는 carryover와 모든 조건에서 손해를 예측하는 always-harm이다. [사전 예측 48개와 식](FORECAST_KO.md)
- **실행 순서**는 2026년 11월 16일 한국 시각 18:34:52에 forecast 파일을 서버에서 동결하고 18:35:50에 primary 분석을 동결한 뒤 18:37:37에 별도 solver의 target 충돌 실행을 허용하는 순서였다. 새 prediction 파일에는 독립 solver label이 없다. Neutral flat fixture에서 속도 보존을 먼저 확인했다. [Forecast checksum](forecast44.sha256) · [분석 계획](PLAN_KO.md) · [실행 승인 시각](SOLVER_GO)
- **측정 단위**는 신규 catalog 12개다. 각 catalog의 4개 환경에서 미리 지정한 첫 8개 물체를 골랐다. 384회 추출에서 고유 prototype은 383개였다. 각 대상에 2 shape(fold·ripple), 2 surface, 2 contact law, 3 contour variant, 2 명목 높이(19·31 mm), 4개 도구, 3개 상대 위치를 교차했다. Primary impact는 221,184회였다. 같은 물체의 counterfactual 충돌과 4개 sensor noise 반복은 대응시켰다. Source bank·decoder는 v43에서 동결한 것을 유지했다. V43에서 사용한 family 내부의 새 catalog를 검사한 설계다. [GPU 입력과 표본 감사](GPU_KO.md)
- **외부 검증 범위**는 미래 query dynamics다. 기존 장치로 얻었다고 가정한 factual evidence, paid query view 또는 deployment probe와 calibration 예산은 두 decoder에서 같았다. 새 engine에서 evidence를 다시 관측하는 end-to-end 실험은 후속 질문이다. Circle 단면의 sphere와 수평 capsule을 사용해 tip radius 0.5·2 mm와 곧은 sole 길이 0·2 mm를 독립적으로 바꿨다. PyBullet의 contact manifold와 sequential impulse가 충돌 후 속도를 결정한다. Surface field와 material parameter의 정의에는 v43과 공유하는 부분이 있다. [충돌 모형과 neutral control](SOLVER_KO.md)

## 2. 사전 예측은 어디서 맞았나

아래 D는 네 도구, 두 높이와 세 contour variant를 균등 평균한 값이다. Nominal은 offset 0, shifted는 −0.5와 +0.5 mm를 같은 비중으로 평균했다. 구간은 12개 catalog의 paired t interval에 네 비교를 위한 Bonferroni 보정을 적용했다. 표의 legacy 열은 **같은 v44 물체**를 과거 point solver로 다시 채점한 값이다. 사전 예측에는 과거 v43 물체만 썼다. [Primary 계산과 원자료 요약](stats44.json)

| Shape | Contact 위치 | 사전 예측 D | 같은 물체의 legacy D | PyBullet held-out D [동시 95% CI] |
|---|---|---:|---:|---:|
| Fold | nominal | +0.001739 | −0.000837 | **+0.002496** [+0.000951, +0.004041] |
| Fold | shifted | +0.004697 | +0.006439 | **+0.005250** [+0.003668, +0.006831] |
| Ripple | nominal | +0.003738 | +0.000338 | **+0.004756** [+0.002927, +0.006585] |
| Ripple | shifted | +0.005061 | +0.003911 | **+0.005324** [+0.003345, +0.007302] |

Fold nominal에서 legacy와 PyBullet 간 paired D 차이는 +0.003333 [동시 95% CI +0.001361, +0.005306]이었다. 같은 물체와 같은 확률에서 부호가 바뀌므로 물체 추출의 차이만으로 설명하기는 어렵다. 이 대조에는 도구 모양, friction 처리, contact dynamics, event 판정 방식이 함께 바뀐다. 각각의 기여를 분리하려면 동일 solver 안에서 요인을 따로 교차해야 한다. [동일 물체 대조](stats44.json)

| 사전 지정 평가 | Tool-aware forecast | V43 carryover | Always-harm |
|---|---:|---:|---:|
| 높이·signed offset을 나눈 전체 48개 cell, 점추정 부호 | **43/48** | 38/48 | 34/48 |
| 48개에 대한 동시 95% CI가 0을 제외한 31개 | **31/31** | 28/31 | 25/31 |
| Hollow·bulge만 포함한 48개, 점추정 부호 | **45/48** | 40/48 | 36/48 |
| 높이와 offset을 묶은 16개, 점추정 부호 | 14/16 | 11/16 | **15/16** |

Tool-aware forecast의 D 예측 MAE는 0.002899, carryover는 0.004738이었다. 차이 −0.001839의 paired catalog bootstrap 95% CI는 [−0.001977, −0.001415]였다. 두 forecast의 부호가 갈린 다섯 cell은 모두 fold였고 모두 실측 부호가 새 예측과 같았다. 그 다섯 중 세 곳에서만 동시 구간이 0을 제외했다. 다섯 번의 오예측은 모두 동시 구간으로 부호를 구분하기 어려운 cell이었다. 새 family를 다시 뽑았을 때의 적중률은 이 표로 추정할 수 없다. [16개 집계와 변경된 다섯 예측](FORECAST_RESULT_KO.md) · [모든 cell 및 구간](forecast_eval44.json)

**두 위치에서 관찰한 서로 다른 효과.** Fold 19 mm, offset 0에서 R=0.5/L=2, R=2/L=0, R=2/L=2의 D는 각각 +0.014009, +0.010180, +0.017681이었다. 과거 carryover는 세 경우 모두 −0.001312였다. 이 세 cell이 도구의 형상을 고려해 잘못된 예측을 뒤집은 가장 명확한 결과다. Fold 31 mm, offset +0.5 mm에서는 R=0.5의 L=0 → 2에서 +0.004048 → −0.003625, R=2에서는 +0.006675 → −0.002175였다. 두 긴 sole 각각의 D 구간은 0을 포함한다. 같은 물체에서 긴 sole와 둥근 tip 간 차이를 직접 짝지은 후속 분석에서는 D 차이가 R=0.5에서 −0.007674 [−0.011315, −0.004032], R=2에서 −0.008850 [−0.012126, −0.005575]였다. 즉 sole를 늘릴 때 손익이 이동하는 방향은 분명하지만, 특정 긴 sole에서 projection이 절대적으로 유리하다는 증거는 더 약하다. 두 반지름의 구간을 보정했으며 이 paired 대조는 exploratory 분석이다. [사전 지정 cell](forecast_eval44.json) · [후속 paired contrast](sole_contrast44.json)

## 3. Contact patch와 target label의 물리적 의미

접선축을 따라 본 **realized footprint**는 한 time step에서 동시에 양의 normal force를 받은 두 점 사이의 최대 거리다. 충돌 중 최댓값을 먼저 구한 뒤 물체별로 평균했다. Stick label은 마지막 양의 하중에서 얻은 force-weighted normal로 접선을 정하고, 0.40 ms 시점의 중심 접선 속도 절댓값이 0.020 m/s 이하인지로 판정했다. 물체의 회전은 고정했다. 이 정의는 분리 후의 잔류 속도도 포함하므로 접촉면 내부의 partial slip을 측정할 수 없다. [Solver 명세](SOLVER_KO.md)

| Radius / sole 길이 (mm) | 평균 realized footprint (mm) | 두 점 이상 동시 하중 | Stick 비율 | 작은 둥근 tip과 outcome이 다른 비율 |
|---|---:|---:|---:|---:|
| 0.5 / 0 | 0.0210 | 53.8% | 47.7% | 기준 |
| 0.5 / 2 | 0.0351 | 51.7% | 47.3% | 9.0% |
| 2 / 0 | 0.0045 | 10.0% | 34.9% | 13.8% |
| 2 / 2 | 0.0142 | 10.7% | 38.3% | 15.5% |

2 mm sole가 전체 길이로 압력을 지지한다는 가정은 위 측정과 맞지 않는다. Fold 31 mm, offset +0.5 mm에서 sole를 늘리면 초기 geometric support 위치가 평균 0.984·0.991 mm 움직였다(R=0.5·2 순). 두 효과가 함께 있으므로 관찰된 손익을 넓어진 pressure area 한 가지로 해석할 수 없다. Shifted의 양 형상 평균에서 radius를 늘리면 D가 +0.004710 [동시 95% CI +0.003144, +0.006276] 커졌고, sole를 늘리면 −0.001009 [−0.001528, −0.000490] 변했다. 두 변화의 interaction은 −0.000109 [−0.000550, +0.000333]였다. [접촉 및 Brier contrast](stats44.json) · [초기 support 이동](sole_contrast44.json)

## 4. 절대 held-out Brier loss와 GPU p90

각 decoder가 동일 observation과 target을 받았다. Brier는 물리 평가 대상으로 사전 선택한 물체·도구·형상을 균등 평균한 값이다. GPU latency는 별도로 한 GPU에 512개 물체 전체를 한 번에 투입해 측정했다. Cold start가 끝난 뒤 각 방법을 panel당 세 번, 총 36회 측정했다. 순서는 무작위로 섞었다. [Brier 원자료](stats44.json) · [Timing 원자료와 bootstrap](audit_gpu44.json)

| Decoder | Correction | Nominal Brier ↓ | Shifted Brier ↓ | Warm GPU p90, ms [95% CI] |
|---|---|---:|---:|---:|
| Original | 0% | **0.149187** | 0.156526 | 487.25 [484.91, 488.33] |
| Original | v43 frozen temperature | 0.152246 | **0.155782** | 488.04 [484.55, 489.96] |
| Privileged projection | 0% | 0.152813 | 0.161813 | 486.36 [483.37, 488.39] |
| Privileged projection | v43 frozen temperature | 0.155244 | 0.159412 | 484.70 [483.55, 486.34] |

0% 두 arm의 p90 비율(projected/original)은 0.9982 [paired 95% CI 0.9933, 1.0029]였다. 예측 파일 생성은 빈 RTX PRO 6000 두 장에 나눴고 latency 비교는 같은 장치에서 했다. PyBullet DIRECT의 충돌 계산은 할당된 compute node의 CPU에서 독립적으로 실행했다. p90에는 K256 observation 선택, sensor routing, posterior, 해당 correction, 데이터 전송과 GPU 동기화를 포함한다. Offline target 생성, 실제 probe와 reset에 걸리는 시간, 변형 방향을 실물에서 알아내는 비용은 별도다. [GPU 측정 범위](GPU_KO.md)

## 5. 수치 설정을 바꾼 전체 재검증

Primary에서는 221,184회 모두 접촉이 검출됐다. 사전에 정한 소표본으로 time step과 mesh를 점검한 뒤, 결과의 부호가 설정에 의존하는지 알아보기 위해 같은 두 검사를 전체 물체로 확장했다. 아래 두 재실행은 사후 민감도 분석이다. Target label만 새로 생성했고 사전 예측과 두 decoder 확률은 유지했다. [전체 수치 분석](NUMERICAL_FORECAST_KO.md) · [원속도와 contact manifold 검산](NUMERICAL_KO.md)

| Solver 설정 | Primary 대비 개별 event 변화 | 예측 적중 수 / 48 (tool-aware · carryover · always-harm) | Fold D nominal / shifted | Ripple D nominal / shifted |
|---|---:|---:|---:|---:|
| 20 µs · 0.10 mm mesh | 기준 | 43 · 38 · 34 | +0.002496 / +0.005250 | +0.004756 / +0.005324 |
| 10 µs · 0.10 mm mesh | 12,320회 (5.57%) | 47 · 42 · 38 | +0.003327 / +0.005975 | +0.006565 / +0.007136 |
| 20 µs · 0.05 mm mesh | 4,149회 (1.88%) | 43 · 38 · 34 | +0.002455 / +0.005411 | +0.004062 / +0.005087 |

Primary에서 동시 95% 구간이 0을 벗어났던 31개 cell의 부호는 두 재실행에서도 모두 같은 방향으로 구별됐다. 전체 재실행에서 time step 때문에 바뀐 네 cell은 전부 작은 반지름의 ripple였고, primary의 구간은 0을 포함했다. 세 수치 설정 전체에서 v43 대비 바꾼 다섯 예측은 모두 점추정 부호가 맞았다. 평균의 부호 안정성과 개별 접촉 event의 안정성에는 이처럼 차이가 있었다. 접선 속도 threshold를 0.01 또는 0.04 m/s로 바꿔도 네 primary 평균 D는 모두 양수였다. Primary에서 0.4 ms에도 하중이 남은 충돌은 1.68%였다. 사전 선택한 소표본을 0.8 ms까지 연장한 후속 검사에서는 13,824회 중 label 변화가 없었다. [해상도별 48개 cell](numerical_forecast44.json) · [종료 시점 확인](horizon44.json) · [속도 threshold 분석](stats44.json)

## 6. 논문으로 어느 주장까지 가능한가

**직접 뒷받침되는 기여**는 이미 훈련된 physics decoder에서 contour 방향을 지우는 선택의 손익이 *tip이 도달하는 실제 위치*에 따라 바뀔 수 있고, 그중 일부를 새 solver의 결과를 보기 전에 찾아냈다는 것이다. V43의 변형 방향 자체는 정확히 아는 privileged 조건에서도 이런 손실이 생겼다. Fold 19 mm의 세 조건은 새로운 물체, 세 수치 설정, 강한 historical carryover를 모두 견뎠다. 독립 solver에서 달라질 결과를 미리 명시했다는 점은 반복적인 simulator tuning에 대한 대안 설명을 줄인다. 위치 불확실성이 충돌 법선을 바꾼다는 기초 원리 자체는 기존 연구에도 있다. [사전 예측](FORECAST_KO.md) · [독립 solver 검증](FORECAST_RESULT_KO.md) · [Meißenhelter et al.](https://cgvr.cs.uni-bremen.de/papers/icra40_24/ICRA40_2024_UncertainPhysics.pdf)

**실제로 decoder 선택을 바꾸면 얼마인가.** 모든 높이·signed offset·도구를 균등 가중하고 예상 D가 음수인 11/48개 cell에서만 projection을 적용해 보았다. 같은 물체에서 original을 항상 쓴 경우보다 평균 Brier가 0.001320 낮았고 [paired 차이 −0.001839, −0.000802], carryover로 decoder를 고른 경우보다 0.000993 낮았다 [−0.001263, −0.000724]. 두 구간은 두 비교에 대한 동시 95% 구간이다. 이 분석은 사후에 계산했다. 선택 규칙은 사전 예측 부호를 그대로 사용했지만, 실제 signed offset과 shape를 알고 있다는 조건이 붙는다. 실제 접촉점을 모르는 배포 환경에서 기대할 이득의 크기는 아직 열려 있다. [동일 cell 재채점](diagnostic_routing44.json) · [계산 코드](diagnostic_routing44.py)

**논문에서 제한해야 할 주장**도 세 가지다. 첫째, 높이와 signed offset을 묶으면 단순한 always-harm의 15/16이 tool-aware의 14/16보다 높다. 따라서 장비별 평균 손익만 필요한 사용자에게 이 forecast가 우월하다는 근거는 없다. 둘째, 새 solver의 충돌 결과를 맞혔지만 같은 family·material field·기존 factual sensor를 재사용했다. 독립적인 실물·family·sensor에 대한 외부 타당성은 미검증이다. 셋째, 2 mm sole에서도 양의 하중점은 매우 짧은 구간에 모였다. 실물의 접촉 압력이나 stick/slip을 측정했다는 주장은 과도하다. 공개된 실물 impact 연구에서도 solver·parameter에 따라 slide와 bounce 재현성이 달라졌다. [조건별 검산](forecast_eval44.json) · [Acosta et al.](https://arxiv.org/abs/2110.00541) · [Yuan et al.](https://people.csail.mit.edu/yuan_wz/GelSight1/ICRA15_2740_FI.pdf)

**판단과 다음 확인 실험.** 현 단계의 설득력은 geometry representation에 관한 기전 검증에서 가장 높다. 범용 active acquisition이나 실물에 적용할 decoder 선택법의 독립 논문으로는 증거가 부족하다. 다음 한 번의 실험에서는 초기 contact 위치와 pressure area의 영향을 분리하는 것이 우선이다. 같은 최초 접선·위치를 갖되 compliance가 다른 두 pad를 만들고, 별도로 동일 경도에서 최초 위치만 이동시킨다. 새 contour family의 양·음 손익 cell을 각각 사전에 지정하고 query와 factual sensor를 모두 바꿔 측정한다. 실제 offset을 아는 진단, 영상에서 offset을 추정하는 선택, always-original을 동일 관측 비용으로 비교한다. 접촉점 이동만으로 사전 예측과 Brier 이득이 유지되면 support 위치 설명이 강해진다. 위치를 맞춘 뒤에도 pressure area에 따라 손익이 움직이면 분포를 표현하는 contact model이 필요하다는 근거가 된다. 실물 장비가 없는 단계에서는 pressure를 적분하는 solver로 이 대조를 먼저 시험할 수 있다. [Elandt et al.](https://arxiv.org/pdf/1904.11433)

## 문헌 비교와 현재 증거의 범위

이번 검토는 접촉 위치의 불확실성, 면접촉의 계산 방식, 실제 slip 측정이라는 세 쟁점을 다룬 원저를 골랐다. 아래 논문은 서로 다른 outcome을 평가하므로 오차를 합산하거나 논문 간 성능 순위를 매기지 않았다.

| 연구 질문·원저 | 방법과 표본 단위 | 결과와 결론 | v44 해석의 한계 |
|---|---|---|---|
| 같은 실물 impact를 여러 simulator가 얼마나 설명하는가? Acosta et al. | 실제 cube toss와 Cassie robot의 22회 jump 자료에서 Drake·MuJoCo·Bullet의 contact parameter를 식별하고 궤적을 비교했다. | Cube의 궤적을 대체로 재현했으나 bounce와 slide가 바뀌어 최종 자세 오차가 커지는 사례가 있었다. | 한 engine에서 다른 engine으로 옮겨도 실물에서 어떤 endpoint가 맞는지 알 수 없다. 저자들은 실측 궤적을 사용했다. [원문 §§III–V](https://arxiv.org/pdf/2110.00541) |
| 단일 contact point가 넓은 접촉에서 필요한 wrench를 표현할 수 있는가? Elandt et al. | 내부 pressure field로 contact surface와 net force·moment를 근사했다. Gripper로 pencil을 잡고 회전시키는 예제 등을 시연했다. | 접촉 분포에 따라 달라지는 회전 구속을 설명할 수 있었다. | Bullet의 소수 활성 contact point 사이 거리와 연속 압력장의 면적은 다른 측정량이다. [원문, Fig. 4](https://arxiv.org/pdf/1904.11433) |
| Slip–stick transition을 포함한 compliant dynamics를 안정적으로 적분할 수 있는가? Castro et al. | Regularized friction용 TAMSI를 개발하고, 흔들리는 parallel-jaw gripper가 mug를 잡는 경우 등에서 안정성과 비용을 평가했다. | Transition 부근의 stiff dynamics를 고려하는 적분이 수치적 어려움을 줄였다. | 적분기·friction regularization이 다르다. v44의 half-timestep 대조는 개별 label과 평균 Brier에 대한 자체 검산이 필요하다. [원문 §VI](https://arxiv.org/pdf/1909.05700) |
| 실물에서 contact patch 전체가 한 번에 미끄러지는가? Yuan et al. | GelSight elastomer의 영상 sequence에서 접촉 영역 내부의 shear·torsional deformation과 slip 전파를 측정했다. | 주변부에서 slip이 시작돼도 중앙부에 static contact가 남는 incipient slip을 관찰했다. | Rigid tip의 충돌 후 tangential speed로 정의한 binary event는 국소 incipient slip 자체를 측정하지 않는다. [원문](https://people.csail.mit.edu/yuan_wz/GelSight1/ICRA15_2740_FI.pdf) |
| 위치 오차로 변하는 normal이 충돌 후 상태에 어떻게 전파되는가? Meißenhelter et al. | Armadillo·Bunny·Cup mesh의 충돌에서 sphere hierarchy 및 분포 근사를 sampling·linearization과 비교했다. | 이 시험에서 제안한 근사가 대량 sampling보다 빨랐고 단순 선형화보다 결과 분포에 가까웠다. | 위치 → normal의 중요성에는 선례가 있다. Frozen decoder에서 어떤 contour 정보를 지울지의 Brier 예측은 해당 실험의 outcome이 아니다. [원문 §III](https://cgvr.cs.uni-bremen.de/papers/icra40_24/ICRA40_2024_UncertainPhysics.pdf) |
| Rigid-body impact parameter가 실물 물성으로 해석되는가? Fazeli et al. | 2,000회 planar drop 중 1,718회 유효 관측을 사용해 여섯 contact model의 impulse와 파라미터를 분석했다. | 실제 impulse를 표현할 수 있는 범위가 제한되고 맞춘 parameter의 물리적 해석이 어려운 사례가 있었다. | 독립 solver와 같은 결과를 얻어도 실제 friction coefficient나 실제 contact area를 식별했다는 증거가 되지는 않는다. [원문](https://arxiv.org/pdf/1710.04979) |

**연구 지식 지도.** 위치오차와 tip 외곽 → 실제 접촉 위치와 법선 → contact impulse·접선 운동 → stick 가능성이라는 물리 경로가 있고, 7점 profile → geometry likelihood → stick 확률이라는 추정 경로가 있다. 위치 불확실성이 법선에 전파되고 접촉 분포가 wrench에 영향을 준다는 근거는 독립적으로 축적되었다. 두 경로를 동일 물체에 짝지어 **어느 방향의 정보를 지우면 어떤 조건에서 Brier가 오르내리는가**를 사전 예측하는 부분이 이 연구의 좁은 질문이다. [Meißenhelter et al.](https://cgvr.cs.uni-bremen.de/papers/icra40_24/ICRA40_2024_UncertainPhysics.pdf) · [Elandt et al.](https://arxiv.org/pdf/1904.11433) · [v43 결과](../blind_decoder_v43/REPORT_KO.md)

**엇갈리는 근거의 해석.** 단순한 접촉 모형이 cube의 전체 궤적은 잘 맞히면서도 접촉 직후 bounce와 slide를 다르게 낼 수 있다. 실물 궤적의 평균 오차, 공간적인 partial slip, 미래 물체의 binary Brier는 측정 단위가 다르다. Acosta et al.의 궤적 재현과 Fazeli et al.의 impulse 한계, Yuan et al.의 부분 미끄러짐은 이 차이를 감안하면 양립한다. v44에서 실험적으로 개입한 radius·sole span의 효과는 테스트한 engine과 outcome 정의에 한해 해석한다. [Acosta et al.](https://arxiv.org/pdf/2110.00541) · [Fazeli et al.](https://arxiv.org/pdf/1710.04979) · [Yuan et al.](https://people.csail.mit.edu/yuan_wz/GelSight1/ICRA15_2740_FI.pdf)

### 남은 연구 질문 열 가지

| 질문 | 근거와 중요성 | 가능한 검증 |
|---|---|---|
| 1. 실제 압력 중심이 geometric support보다 나은 predictor인가? | 분포를 적분한 wrench와 한 접촉점은 다른 정보를 보존한다. [Elandt et al.](https://arxiv.org/pdf/1904.11433) | Pressure-weighted normal, 첫 normal, 마지막 normal로 별도 forecast를 고정해 새 손익 부호를 비교한다. |
| 2. Radius와 compliance는 언제 같은 효과를 내는가? | 넓은 rigid sole와 compliant pad의 접촉 형성이 다르다. [Elandt et al.](https://arxiv.org/pdf/1904.11433) | 같은 외곽의 경도만 바꾸어 contact area·stick 확률·Brier를 교차 측정한다. |
| 3. Micro-slip과 gross slip 중 어느 label을 예측해 왔는가? | 실제 patch에서는 가장자리부터 미끄러질 수 있다. [Yuan et al.](https://people.csail.mit.edu/yuan_wz/GelSight1/ICRA15_2740_FI.pdf) | 촉각 영상에서 두 event를 각각 표기하고 속도 기반 label과 비교한다. |
| 4. 회전 자유도를 주면 fold/ripple 패턴이 달라지는가? | 접촉 moment는 회전 제어 과제에서 중요했다. [Elandt et al.](https://arxiv.org/pdf/1904.11433) | 같은 질량·자세에서 angular inertia만 바꾸고 net moment와 부호를 예측한다. |
| 5. 어느 solver 해상도에서 부호가 안정되는가? | 마찰 전이의 수치 강성이 적분을 어렵게 한다. [Castro et al.](https://arxiv.org/pdf/1909.05700) | Time step·mesh spacing·solver iterations를 독립적으로 변화시키고 같은 물체의 event flip 및 paired Brier를 비교한다. |
| 6. 실제 drift 분포를 써도 평균 predictor가 바뀌는가? | 위치 분포에 따라 normal과 충돌 후 분포가 달라진다. [Meißenhelter et al.](https://cgvr.cs.uni-bremen.de/papers/icra40_24/ICRA40_2024_UncertainPhysics.pdf) | 장비의 signed placement error를 센서로 먼저 추정한 뒤 해당 빈도로 cell을 가중한다. |
| 7. 관측과 query 충돌을 모두 다른 engine으로 옮겨도 되는가? | 실제로 solver 간 이동을 제한하는 원인은 과제별로 다르다. [Acosta et al.](https://arxiv.org/pdf/2110.00541) | Factual/probe engine × query engine을 교차해 representation shift와 label shift를 분리한다. |
| 8. 실제 마찰 계수와 모형의 유효 계수가 달라도 예측하는가? | 실측 impulse를 맞춘 coefficient는 실제 물성으로 단정하기 어렵다. [Fazeli et al.](https://arxiv.org/pdf/1710.04979) | 독립 마찰 측정과 trajectory fit을 분리하고 새 표면 session에 대해 검증한다. |
| 9. 같은 도구를 가정한 shape prior만으로도 예측이 충분한가? | V43에서는 단순히 항상 projection이 손해라고 해도 여러 조건에서 방향이 맞았다. [v43 검산](../blind_decoder_v43/PHYSICS_RESULTS_KO.md) | Carryover·always-harm·tool-aware forecast 간 불일치 cell을 다른 형상에서 미리 지정해 평가한다. |
| 10. 고밀도 geometry 관측 한 번을 구매할 가치가 있는가? | 실제 접촉점의 불확실성은 기하 입력의 가치를 제한한다. [Meißenhelter et al.](https://cgvr.cs.uni-bremen.de/papers/icra40_24/ICRA40_2024_UncertainPhysics.pdf) | 같은 배포 시간에 고밀도 profile과 접촉점 camera를 대조하고 acquisition·reset을 포함한 Brier와 작업 실패 비용을 보고한다. |

## 재현 자료

서버 작업 경로는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/contact_patch_v44`다. GPU 예측, 독립 solver, 사전 forecast를 나누어 실행했다. 서버에서 생성한 primary 결과와 두 해상도 대조, 종료시점 검산을 포함한 label archive 3,456개의 checksum을 로컬 사본과 대조했다. 총 705,024회는 같은 물체의 수치 재실행을 포함한 계산량이다. 추정의 독립 단위는 12개 catalog다. [충돌 archive manifest](solver_outputs44.sha256) · [GPU 종료 기록](gpu_release44.log)

| 재현 단계 | 보관 자료 |
|---|---|
| 사전 예측과 분석 기준 | [Python](forecast44.py) · [동결된 48개 예측 및 deformed-only 분석](forecast44.json) · [분석 동결 checksum](analysis_freeze44.sha256) |
| Source model과 GPU 경로 | [V43 frozen artifact](../blind_decoder_v43/frozen_model43.json) · [GPU runner](runner44.py) · [GPU 독립 검산](audit_gpu44.json) |
| 독립 충돌과 검산 | [PyBullet 구현](solver44.py) · [Neutral fixture](neutral_controls44.json) · [수치 정밀도와 종료시각](NUMERICAL_KO.md) |
| Primary 평가 및 추가 대조 | [동일 물체 Brier와 paired contrast](stats44.py) · [사전 예측 채점](forecast_eval44.py) · [정밀도별 forecast 재채점](numerical_forecast44.py) · [그림 재생성](display44.py) |
