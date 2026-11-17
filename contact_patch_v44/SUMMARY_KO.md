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
