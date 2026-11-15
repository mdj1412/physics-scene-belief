## 1. 결론

**Conditional predictive mixture를 유지하면 v15의 false-information counterexample을 해결한다. Fresh-world 실험에서도 pair selector의 실패를 상당히 줄였다. 다만 region별 compression의 독자적 이득과 nested baseline을 대체할 만큼 안정적인 accuracy–compute trade-off는 아직 입증하지 못했다.**

새로운 512 iid worlds와 world당 16 observation noise repeats를 사용했다. Primary matched condition의 결과는 다음과 같다. Event loss는 Brier loss이며 낮을수록 좋다. GPU latency는 factual fit부터 selected observation의 posterior update까지 포함한 5개 tasks의 aggregated median이다.

| Method | Contact-event loss | Stick-event loss | GPU online ms |
|---|---:|---:|---:|
| **Region32** | **0.044491** | **0.066922** | **4.265** |
| Equal-size global mixture | 0.044537 | 0.067871 | 3.771 |
| Optimized LMMSE16 | 0.044758 | 0.073383 | 3.392 |
| Optimized LMMSE64 | 0.042943 | 0.070734 | 4.438 |
| Nested128 | 0.044291 | 0.066739 | 4.767 |
| Nested512 | 0.044319 | 0.065929 | 10.068 |
| V15 pair selector | 0.055675 | 0.072749 | 3.204 |
| Always-query | 0.044473 | 0.077896 | 1.584 |
| Always-probe | 0.130390 | 0.068684 | 1.542 |

Region32는 pair 대비 contact-event loss를 **20.1%**, stick-event loss를 **8.0%** 줄였고 두 차이의 primary adjusted CI가 모두 0 아래였다. LMMSE16 대비 stick-event 평균 개선은 **8.8%**이며 difference는 −0.006462, adjusted 99.58333% CI [−0.013470, −0.000015]였다. CI upper가 0에 가까우므로 강한 보편적 우위로 해석하지 않는다. 별도 bootstrap Monte Carlo audit에서도 방향은 유지됐지만 fixed QMC banks의 uncertainty를 해소한 검증은 아니다.

Nested512 대비 GPU online speedup은 **2.36×**다. 그러나 사전에 정한 approximation screen은 **실패**했다. Contact-event mean-loss ratio의 95% upper는 1.01596으로 1.05 이하였지만 stick-event는 **1.05773**이었다. 따라서 “nested 수준의 loss를 5% 이내로 보존하면서 2배 이상 빠르다”는 결론을 내릴 수 없다. 평균 차이가 작거나 superiority CI가 0을 포함한다는 이유로 equivalence를 주장하지 않는다.

Global mixture 대비 개선은 두 events 모두 adjusted CI에서 확인되지 않았다. Global mixture도 두 counterexamples를 해결했다. 따라서 이번 진전의 근거는 **pairwise separation 대신 conditional mixtures를 합한 뒤 decision risk를 계산한 것**에 있으며, **region stratification만의 필요성**까지 입증한 것은 아니다.

추가 observation 자체는 유용하다. Region32의 contact/stick losses는 no-observation의 0.192616/0.088885보다 낮았다. 하지만 adaptive selection의 추가 가치는 별도 문제다. Contact에서는 always-query와 사실상 같은 loss이며 always-query가 훨씬 저렴하다. Stick에서는 always-probe 대비 평균 2.57% 낮지만 adjusted CI가 0을 포함한다. Strong fixed rule을 넘는 근거는 충분하지 않다.

현재 판단은 **유용한 acquisition correction과 후속 연구용 baseline은 확보했지만, 독립적인 method paper의 핵심으로 확정할 단계는 아니다**이다. 특히 compact decoder까지 사용하면 matched stick loss가 0.076230으로 악화된다. 이번 결과는 full factual posterior를 대체한 compact belief의 성공이 아니라, QMC factual posterior 위에서 acquisition을 압축한 결과다.
