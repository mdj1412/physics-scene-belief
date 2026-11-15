## 1. 결론

**Acquisition error와 action gap에 따라 components를 늘리는 adaptive compression을 구현했지만, 이번 실험에서는 ranking 안정성과 compute 효율을 함께 개선하지 못했다.** Decision loss는 nested512와 가까웠고 사전 5% noninferiority 조건도 충족했지만, GPU에서는 nested512보다 느렸으며 더 저렴한 fixed-size mixtures와 strong rules 대비 이득도 입증하지 못했다.

Fresh 512 worlds × 16 actual observation repeats의 matched condition은 다음과 같다. Latency는 동일한 optimized timing pass에서 얻은 두 event tasks의 aggregated median이며, 낮을수록 좋다.

| Method | Contact Brier loss | Stick Brier loss | GPU online ms |
|---|---:|---:|---:|
| **Adaptive region** | **0.052373** | **0.078740** | **14.581** |
| Adaptive global | 0.052302 | 0.079426 | 8.723 |
| Fixed region K32 | 0.052373 | 0.077076 | 5.475 |
| Fixed region K128 | 0.052373 | 0.078740 | 5.949 |
| Fixed global K128 budget | 0.052372 | 0.079331 | 6.245 |
| Optimized LMMSE16 | 0.052296 | 0.083311 | 3.784 |
| Optimized LMMSE64 | 0.049676 | 0.080689 | 4.406 |
| Nested128 | 0.052298 | 0.078737 | 4.727 |
| Nested512 | 0.052372 | 0.079387 | 10.322 |
| Always-query | 0.052295 | 0.098184 | 1.647 |
| Always-probe | 0.120361 | 0.077625 | 1.625 |

Adaptive region의 stick-event loss는 LMMSE16보다 평균 약 5.49% 낮지만 adjusted CI가 0을 포함한다. Difference는 **−0.004571 [−0.013684,+0.003870]**이다. V16에서 관찰한 LMMSE 대비 개선이 이번 새 worlds에서도 통계적으로 명확하게 재현됐다고 말할 수 없다. Adaptive global 및 fixed global 대비 우위도 확인되지 않았다.

Nested512 대비 mean-loss ratio의 95% upper는 contact **1.000045**, stick **1.003879**로 사전 1.05 기준 이하다. 하지만 GPU latency는 **14.58 ms 대 10.32 ms**로 오히려 약 41% 높다. 따라서 accuracy와 2× speedup을 함께 요구한 approximation screen은 실패했다. Fixed128과 mean loss가 사실상 같은데 비용은 약 2.45배다.

Ranking 안정성도 개선되지 않았다. Acquisition seed를 바꾸었을 때 stick-event의 Top-1 agreement는 adaptive region **77.54%**, fixed32 **79.49%**, fixed128 **77.73%**, adaptive global **88.48%**, nested512 **76.56%**였다. Adaptive minus fixed32는 **−1.95 percentage points**, secondary 95% CI **[−6.05,+2.34]**다. Nested512도 변동성이 있으므로 이를 exact ranking oracle로 취급하지 않는다.

실패 조건은 구체적이다. Contact에서는 region당 K16에서 381/512 worlds가 끝났지만, stick에서는 385/512 worlds가 K128까지 갔고 **357개가 cap-unresolved**였다. 그중 **278개는 paired sampling-error 항만으로도 action gap 이상**이었다. Fixed outer512 samples에서 components만 늘리는 방식은 이런 불확실성을 해소하지 못한다.

추가 observation 자체는 유용했다. No-observation loss는 contact 0.181039, stick 0.098995였다. 그러나 adaptive selection은 contact에서 always-query를, stick에서 always-probe를 넘는 추가 가치를 보여 주지 못했다. Stable ranking이나 더 많은 components를 쓰는 것 자체가 좋은 decision을 보장하지 않는다.

**현재의 component-only adaptive compression을 독립적인 method paper의 핵심으로 추진하는 것은 권하지 않는다.** 이번 단계의 성과는 error source를 분리해, representation resolution을 늘려야 할 경우와 outer integration budget을 바꿔야 할 경우가 다름을 실험으로 확인한 것이다. 후속 방법은 이 차이를 실제 resource allocation으로 연결해야 하며, 단순히 cap을 더 높이는 접근은 우선순위가 낮다.
