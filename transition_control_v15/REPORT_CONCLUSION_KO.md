## 1. 결론

**Expanded prior와 일치하는 control에서도 현재 disagreement selector는 strong baselines보다 나빴다. V14의 실패를 model mismatch만으로 설명할 수 없다.** Contact/no-contact 및 slip/stick transition을 넣었지만, 사전에 정한 “LMMSE failure”와 “disagreement superiority” screening은 모두 통과하지 못했다.

Matched control의 contact-event Brier loss는 pair **0.068424**, LMMSE **0.053824**, nested512 **0.053849**였다. Pair−LMMSE difference는 **+0.014600**, adjusted99.5% CI **[+0.007747,+0.022694]**로 약27.1%의 악화를 지지했다. Stick-event에서도 pair는0.077388로 nested512의0.065355보다 나빴고, difference **+0.012033 [ +0.004646,+0.020379 ]**였다. Always-probe 대비 stick-event 악화도 adjusted CI에서 확인됐다.

Nested512의 stick-event mean loss는 LMMSE보다 약7.05% 낮았다. 그러나 adjusted CI가0을 포함했고, 사전에 요구한 최소10% improvement도 충족하지 못했다. 따라서 “transition을 넣자 LMMSE의 명확한 실패 조건을 찾았다”고 결론 내리지 않는다.

Model mismatch의 영향은 별도로 확인했다. 같은 shifted data에서 correct-law pipeline을 쓰면 smooth-velocity 및 stick-event loss가 줄었다. Pair의 contact-event에서는 선택 경로의 차이가 컸고, smooth-velocity에서는 decoder 차이가 더 컸다. 하지만 correct-law model을 제공해도 pair가 강한 baseline을 이기지는 못했다.

Compute도 강한 implementation과 비교했다. Observation moments caching과 efficient matrix-vector contractions를 적용한 뒤 GPU aggregated median은 **LMMSE 약3.18 ms, pair3.27 ms, nested5129.68 ms**였다. Nested 대비 절약은 있지만 LMMSE 대비 speed advantage는 없다. Always-query/always-probe는 더 저렴하면서 해당 event task에서 pair보다 loss도 낮았다. 최초 비최적화 baseline의 큰 latency를 algorithmic superiority로 사용하지 않는다.

**현재 pair score를 독립적인 method-paper의 핵심으로 계속 추진하는 것은 권하지 않는다.** 이번 결과는 model mismatch, numerical approximation, acquisition objective의 문제를 나누어 확인한 diagnostic evidence다. 후속 연구를 한다면 cutoffs를 반복 조정하기보다, 개별 hypothesis pair의 구별과 decision-region 전체의 구별이 다르다는 문제부터 해결해야 한다.
