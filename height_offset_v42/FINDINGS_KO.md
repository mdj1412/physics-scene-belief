## 물리적 원인과 예측 가능성

**높이가 달라져서 생기는 변화와, 높이가 달라진 뒤 contour 자체가 만드는 변화를 분리했다.** 같은 높이의 base와 hollow·bulge를 비교하면, 실제 stick/slip이 달라지는 비율은 명목 위치에서 0%, −0.125 mm에서 5.11%, +0.125 mm에서 11.96%였다. 물체의 예측 확률을 고정했기 때문에 projection 손익의 높이별 변화는 정확히 −2 E[(p_est−p_orig)(Yδ−Y₀)]다. 같은 위치의 base label로 대체해 채점하면 contour 고유의 기여를 분리할 수 있다. +0.125 mm에서 projection 손익은 −0.00375에서 +0.00890으로 변했다. 약 +0.01264의 변화는 거의 전부 같은 위치에서 변형된 외곽이 실제 event를 바꾼 경우로 설명됐다. 이 simulator에서 마찰 한곗값을 base의 값에 고정해도, local slope가 바꾼 impulse demand만으로 실제 contour flip의 98.30%를 재현했다. [독립 방정식과 물리 검산](PHYSICS_AUDIT_KO.md)

**손해 방향은 이동 거리만으로 예측할 수 없다.** Flare의 31 mm 지점에서 −0.5 mm 이동하면 projection이 Brier를 0.01146 낮췄고, +0.5 mm 이동하면 0.01105 높였다. 음의 방향에서는 profile 전체에 적합한 apparent slope가 실제 접촉점의 slope 변화를 거꾸로 가리켜, 부정확한 단서를 지우는 편이 나았다. 보조 분석에서는 두 slope를 이용해 hollow·bulge의 손익 부호를 사전 예측했다. 32개 조건 중 30개가 맞았고, 항상 손해로 예측하면 27개가 맞았다. 실제 offset과 생성한 외곽을 알아야 계산되는 진단이다. 미래의 미지 환경에서 사용할 수 있는 배포 성능으로 환산할 수 없다. [사전 예측과 실제 결과](PHYSICS_AUDIT_KO.md) · [형상별 slope 그림](geometry42.png)

이 설계에서 배포 접촉의 일부는 명목 높이에 정확히 맞고, 나머지는 여덟 비영 offset 중 균등하게 발생한다고 가정할 수 있다. 이때 잘못 배치되는 접촉 비율이 **23.8%** [panel bootstrap 95% CI 22.3%, 25.6%]를 넘으면, 항상 0%를 적용할 때 original의 기대 Brier가 estimated projection보다 낮아진다. 실제 장비에서 측정한 오차 분포를 넣으면 경계도 달라진다. [배포 비율 민감도](deployment_mix42.json)

## Selector는 무엇을 알아냈고 무엇을 놓쳤는가

세 보정의 선택 비율은 0%가 68.8%, fixed가 16.5%, optimizer가 14.6%였다. 선택 비율을 그대로 유지하면서 환경과의 대응만 없앤 비교보다 selector의 Brier는 0.000266 낮았다. 환경별 선별에 신호가 있었지만, 항상 0%라는 더 강한 기준에 비해 줄인 loss는 약 0.000005에 그쳤다. Saturating law에서는 selector−0%가 −0.000272, crossover law에서는 +0.000262였다. 두 law의 효과가 서로 상쇄됐다. 주 대조의 구간은 사전 지정한 실용적 개선 0.003을 배제한다. 반면 original decoder에 같은 selector를 사용한 보조 대조에서는 −0.000279 [−0.000452, −0.000105]의 작은 개선이 있었다. Original+selector 조합의 GPU p90은 측정하지 않았다. [선택 빈도와 환경별 반사실 대조](SELECTOR_AUDIT_KO.md)

**어떤 보정이 좋은지 알 기회 자체는 남아 있었다.** 같은 환경 안에서 약 64개 별도 물체 각각에 무작위 비영 offset 한 개를 적용해 충돌 결과를 하나씩 얻고, 그 결과로 세 보정 중 하나를 고른 뒤 나머지 물체에서 채점했다. 실제 offset을 알려주지 않아도 항상 0%보다 Brier가 **0.003784** 낮았다 [95% CI 0.003387, 0.004181 감소]. 따라서 숨은 높이 하나만으로 주 selector의 실패를 설명하기 어렵다. 이 보조 실험은 같은 환경에서 추가 target 충돌 결과를 약 64개 구매한 설정이다. 네 번의 calibration 예산에서 해당 개선을 달성했다는 근거는 없다. [물체 분리와 획득 비용](SELECTOR_AUDIT_KO.md)

같은 물체가 여덟 숨은 접촉 높이에 걸쳐 서로 다른 event를 낸 비율은 25.7%였다. 물체와 법칙을 완전히 알더라도 어느 높이에서 닿는지를 모르면 이 균등 혼합에서 최저 가능한 Brier는 0.04833 [0.04776, 0.04890]이다. 실제 높이를 decoder의 마지막 readout에만 넘긴 보조 실험에서도 +1 mm에서는 original의 Brier가 0.19941에서 0.20047로 오히려 소폭 증가했다. 높이를 측정한 뒤 그 위치의 기울기를 표현할 수 있도록 shape hypothesis까지 바꾸는 대조가 필요하다. 변형 축의 정보를 일부 남기는 finite-variance likelihood는 같은 +1 mm에서 hard projection의 0.22163을 0.21459까지 낮췄으나 original의 0.19941에는 미치지 못했다. [숨은 높이의 loss 하한](HIDDEN_FLOOR_KO.md) · [Readout 대조](reveal42.json) · [Finite-variance 대조](soft42.json)

## 논문으로 발전시킬 판단

가장 설득력 있는 결과는 **주변 외곽을 지워도 되는 조건을 물리적으로 정의하고, 그 조건이 깨질 때 확률 예측의 손익까지 뒤집히는 과정을 같은 물체에서 확인한 점**이다. 외곽을 바꾸고 실제 접촉 위치를 고정한 대조, 위치만 바꾸고 센서를 고정한 대조, 같은 likelihood에서 보정 정책만 바꾼 대조가 서로 다른 원인을 구별한다. 이미 알려진 invariance의 위험을 넘어서는 부분은 signed placement error, local tangent, event flip과 Brier의 관계를 수치로 검증했다는 점이다. 현재 근거는 합성 benchmark의 mechanism study를 지지한다. 독립 catalog는 기존 두 shape family와 두 analytic contact law에서 추출했다. 가장 좁은 lobe의 폭은 4 mm이고 point contact를 사용했다. 실물의 허용 오차, 새로운 material에서의 재현성, 임의 shape에 대한 무감독 policy 개선은 미검증이다.

다음 실험에서 먼저 바꿀 것은 selector의 복잡도보다 **접촉 위치의 측정과 국소 기울기의 표현**이다. 실제 높이 측정 유무와 quadratic·local spline contour bank를 교차하고, tip radius를 달리해 point contact에서 보인 손익 반전이 finite contact patch에서도 남는지 본다. 실물에서는 접촉부 영상으로 위치와 stick/slip을 따로 기록하고, 새 물체 단위로 분할해 signed offset별 paired Brier를 평가한다. Selector를 계속 개발할 근거는 환경당 추가 충돌을 0·4·16·64회로 늘렸을 때 어느 비용부터 항상 0%를 일관되게 이기는지에 있다. 두 실험 모두 원래 decoder와 0%를 유지하고, 추가 센서·충돌 비용을 포함한 작업 시간도 기록해야 한다.
