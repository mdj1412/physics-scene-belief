# Contact law를 바꾸면 optimizer와 fixed의 Brier 차이가 왜 바뀌는가

이 표에서 양수는 saturating에서 crossover로 바꿀 때 optimizer가 고정값에 비해 불리해졌다는 뜻이다. 같은 물체·같은 noise에서 두 law로 계산한 세 값, 즉 base probability, signed cue와 보정량 (z,w), 실제 stick label을 하나씩 바꿨다. 세 값을 교체하는 여섯 순서에서 각 값의 marginal change를 평균했다. 따라서 세 기여의 합이 law 간 Δ 변화와 일치한다. 두 law에서 관측된 값을 교차 조합한 offline 진단으로 개별 항의 물리적 인과효과까지 식별할 수는 없다. [보조 계획](FORECAST_DIAGNOSTICS_PLAN_KO.md) · [원자료 계산](law_decomposition40.py)

| Contour | Crossover − saturating | Base probability | Signed cue와 보정량 | Stick label |
|---|---:|---:|---:|---:|
| 전체 평균 | +0.001393 | −0.000034 | +0.000075 | +0.001352 |
| shelf / base | +0.001854 | +0.000013 | +0.000141 | +0.001701 |
| shelf / hollow | +0.002008 | +0.000016 | +0.000127 | +0.001865 |
| shelf / bulge | +0.001674 | +0.000011 | +0.000152 | +0.001511 |
| ridge / base | +0.000941 | −0.000083 | +0.000013 | +0.001011 |
| ridge / hollow | +0.000825 | −0.000091 | +0.000015 | +0.000900 |
| ridge / bulge | +0.001055 | −0.000071 | +0.000003 | +0.001123 |

전체 12개 독립 panel 단위 pointwise 95% interval은 law 간 총 차이 [+0.001166, +0.001620], base probability [−0.000111, +0.000043], signed cue와 보정량 [−0.000035, +0.000185], label [+0.001144, +0.001559]이다. 같은 panel에 속한 여섯 contour와 두 block은 panel 내에서 먼저 평균했다. [수치와 구간](law_decomposition40.json)

Saturating 측의 두 prediction을 유지하고 orientation만 뒤집은 saturating label을 붙이면 Δ가 +0.000380 변했다. 실제 crossover label을 붙였을 때 변화는 +0.001286였다. Crossover label이 같은 방향 saturating label과 달랐던 비율은 4.67%이고, 반대 방향 saturating label과 달랐던 비율은 2.74%였다. 이 수치는 law 변화를 방향 하나로 근사할 수 있는 범위를 보여 준다. 두 law 사이의 비교이므로 v39에서 v40으로 옮겨 온 예측오차 자체를 완전히 분해한 값은 아니다.

Target score를 보기 전 v39 내에서 한 law를 가린 검산도 수행했다. 다른 law에서 얻은 Δ를 그대로 복사했을 때 네 조건의 MAE는 0.000329, nominal signed friction amplitude로 두 방향의 기존 관측 label을 섞었을 때는 0.000227였다. 두 조합은 계수가 1로 제한되어 두 예측이 같았다. 과거 두 law가 모두 같은 방향의 friction을 주므로 이 검산으로 crossover에서 방향이 뒤집히는 경우를 검증할 수는 없었다. [과거 law 교차 검산](source_law_check40.json)
