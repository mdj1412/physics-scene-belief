### 같은 모델을 nominal label로만 학습한 경우

Source에서 선택한 robust decoder의 개선이 MLP 추가만으로 설명되는지 점검했다. Hidden width 8, feature 28개, optimizer·seed·45 epochs·학습 물체·관측을 모두 고정하고, 학습 중 label만 nominal outcome으로 바꾼 ERM을 재생성했다. 이 ERM은 target 평가 전에 source 보조 대조로 이미 명세하고 학습한 모델이다. 여기서 추가한 분석은 그 모델의 target 재채점이다. 같은 원자료에 대한 사후 보조 비교로, 새로운 독립 holdout이나 primary hypothesis를 추가한 것은 아니다. 모든 방법은 0% 보정을 사용했다.

| Target held-out Brier | Original physics | Nominal-only ERM | Offset을 포함해 학습한 robust | Robust−ERM, 두 조건 동시 paired 95% CI | Robust가 ERM을 이긴 panel |
|---|---:|---:|---:|---|---:|
| Nominal | **0.130837** | 0.137090 | 0.134838 | −0.002252 [−0.003400, −0.001104] | 11/12 |
| Nonzero offset 균등 평균 | 0.147784 | 0.155147 | **0.143347** | −0.011800 [−0.012977, −0.010624] | 12/12 |

동일 residual architecture를 nominal outcome으로만 학습하면 원래 physics decoder보다 nominal에서 0.006254, shifted에서 0.007363 높은 Brier가 나왔다. 동일 source 물체의 offset별 outcome을 섞어 학습하면 이 ERM보다 shifted Brier가 0.011800 낮아져 original보다도 0.004438 낮았다. Nominal에서는 ERM보다 개선돼도 original보다 0.004002 높았다. 따라서 현재 OOD 구성에서 shifted 개선을 모델 용량 증가만으로 설명하기는 어렵다. 이 대조가 분리하는 요인은 source에서 어떤 위치의 충돌 결과를 학습했는가이다. 실제 접촉 높이나 target law를 식별했다는 증거는 제공하지 않는다.

Source-only 재학습 후 source 선택 panel에서 기존에 기록한 robust−ERM 차이(+0.004200487, −0.018811252)를 두 값 모두 그대로 복원했다. Target마다 구매한 sensor만 사용하고 576개 block에서 기존 robust 확률도 비교했다. 최대 절대 차이는 0이었다. 구간은 12개 panel의 paired 차이와 두 strata에 대한 Bonferroni–t 방식으로 계산했다. 이번 보조군의 GPU p90은 따로 측정하지 않았다. [재현 코드](erm_control43.py) · [Panel별 통계와 모델 hash](erm_control43.json) · [Source-only ERM 계수](nominal_erm43.json)
