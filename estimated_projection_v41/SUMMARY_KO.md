# v41 · 새로운 물체와 접촉 높이에서 contour 변형을 제거한 뒤에도 보정 선택이 도움이 되는가

**두 v38 보정 정책 중에서는 환경별 선택 이득이 남았다. 그러나 같은 decoder에서 보정을 완전히 꺼 두는 편이 평균적으로 더 정확했다.** 처음 보는 flare·roll 물체와 19·31 mm contact height에서, 접촉 결과를 알려주지 않은 별도 물체 32쌍으로 contour 변형 방향을 추정했다. 12개 독립 catalog panel의 576개 block을 끝까지 평가했다. 방향 추정과 이후 보정의 가치를 따로 보면 결론이 달라진다.

| 질문 | 결과 | 판정 |
|---|---|---|
| Target contact label 없이 유효한 방향을 찾았는가? | 형상·높이별 평균 principal angle 10.63–17.01°. Fixed 보정의 held-out Brier가 original 0.125320 → estimated 0.121716, oracle 0.121816. | Estimated−original은 −0.003604 [−0.003875, −0.003332]. Estimated−oracle의 동시 95% 상한 +0.000080은 사전 허용값 +0.000300 이내다. |
| 이 방향을 없앤 뒤에도 보정 정책을 고를 이유가 있는가? | Estimated에서 selector 0.120552, fixed 0.121716, 항상 optimizer 0.121731. | Selector−fixed −0.001164, selector−optimizer −0.001179. 두 동시 95% 구간 모두 0 아래다. 선택 자체의 이득은 사전 실용 기준 0.003에 미달했다. |
| 보정 자체를 써야 하는가? | 같은 estimated decoder에서 보정을 끄면 0.120129. Selector는 그보다 +0.000423 [개별 95% CI +0.000219, +0.000628] 높았고 12개 panel 중 1개에서만 이겼다. | 이 catalog 배합에서 우선 사용할 방법은 **estimated projection + 보정 없음**이다. 두 보정 사이의 선택 성공을 보정 자체의 필요성으로 확대할 수 없다. |
| Latency 허용 기준을 만족하는가? | 1,024개 물체의 warm GPU p90은 estimated + selector 1,445.4 ms, estimated + fixed 1,442.4 ms. Ratio 1.002 [0.995, 1.233]. | 점추정 차이는 +0.21%지만 상한이 사전 허용한 +2%를 넘었다. 같은 latency라는 판정은 유보한다. 64회 passive readout의 실물 획득 시간은 별도다. |

보정 없는 방법까지 같은 GPU에서 직접 맞대결한 추가 측정에서, estimated + 보정 없음의 warm p90은 **1,423.7 ms**, estimated + selector는 **1,440.6 ms**였다. 같은 관측을 처리할 때 보정을 끄면 loss와 latency가 함께 줄었다. [동일 GPU crossover](latency_estimated_zero41.json)

가장 큰 변화는 **같은 물체의 외곽만 바꿨을 때 예측 확률이 흔들리는 정도**였다. Query-view 가지의 평균 절대 변화가 0.05679에서 0.00857로 줄었다. 반대로 변형이 없는 base contour만 평가하면 projection이 Brier를 0.001420 높였다. 실물에 전달할 주장은 주변 contour를 항상 지우자는 것이 아니라, **접촉 결과를 보존하는 변형이 얼마나 자주 일어나며, 그 방향이 물리 정보와 얼마나 겹치는지 함께 측정해야 한다**는 것이다. [주 결과](stats41.json) · [Matched contour 분해](contour_breakdown41.json) · [보정 없는 대조](no_correction41.json)

![방향 복원, 형상·높이별 projection의 효과, 같은 decoder 내 정책 선택 효과](v41.png)

그림 A는 12개 panel의 평균이다. B의 구간은 네 shape·height 대조를 함께 보정했다. C의 원·사각형은 각 decoder 안에서 두 사전 대조를 함께 보정한 95% 구간이고, 삼각형은 보정 없음과 사후 비교한 개별 95% 구간이다. B와 C에서 왼쪽이 첫 번째 방법의 Brier가 낮은 방향이다.
