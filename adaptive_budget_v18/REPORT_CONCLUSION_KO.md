## 1. 핵심 결론

**Compression error와 outer integration error를 따로 추정해 K와 M을 늘리는 adaptive budget allocation을 구현하고, 서버에서 512개의 새로운 worlds × 16 observation noise repeats로 검증했다. Component-only의 ranking 불안정성은 줄었지만, strong fixed mixture를 넘어서는 decision-loss–compute 우위는 입증하지 못했다.**

Matched control의 stick-event에서 independent acquisition-seed Top-1 agreement는 component_only 76.37%에서 adaptive 82.62%로 6.25 percentage points 증가했다. Paired secondary95% CI는 [+2.34,+10.16] percentage points다. Outer samples를 고정하고 components만 늘리는 방법에 남던 문제가 실제로 줄어든 것이다. 다만 이 CI가 +5 percentage points를 모두 넘는 것은 아니며, 사전 screen의 +5 기준은 point estimate 기준이다.

Adaptive의 stick Brier loss는 0.071078, fixed128/M4096은 0.071307, global128/M4096은 0.070957, nested2048은 0.071242였다. Primary adjusted CI에서는 이 strong baselines 대비 loss 감소가 확인되지 않았다. Global mixture는 Top-1 agreement도 90.82%로 더 높았다. 따라서 region preservation과 두 error proxies의 사용 자체가 우수한 acquisition을 보장한다는 주장은 지지되지 않는다.

GPU online median을 두 tasks에 걸쳐 모으면 adaptive 7.87ms, fixed128/M4096 6.03ms, global128/M4096 5.87ms, nested2048 27.86ms다. Nested 대비 3.54× speedup과 사전 loss-ratio tolerance를 만족하여 approximation screen은 통과했다. 그러나 adaptive는 fixed128/M4096보다 30.5% 느려 adaptivity screen은 실패했다. **어려운 stick task만 보면 adaptive 18.52ms 대 nested2048 28.28ms로 speedup은 1.53×이며, fixed128/M4096 6.14ms보다 약 3.02× 느리다.** Aggregate median만으로 어려운 cases의 효율을 설명하면 과장된다.

CPU에서는 early stopping이 도움이 됐다. Contact median은 adaptive 4.08ms 대 fixed128/M4096 50.09ms, stick은 43.64ms 대 50.81ms였다. 그러나 작은 fixed128/M512는 stick 6.04ms이며, 그 baseline 대비 adaptive의 loss 감소도 primary CI에서 확인되지 않았다. CPU 결과도 strong fixed-budget frontier를 넘었다는 충분한 증거는 아니다.

이번 결과의 가장 분명한 가치는 **compression refinement만으로 해결되지 않는 outer-integration 불확실성을 드러내고, 두 축의 refinement가 어디까지 도움을 주는지와 실제 latency가 왜 따라오지 않는지 검증한 것**이다. 현재 버전을 독립적인 새 method의 완성으로 보기는 어렵다. 아래에 positive evidence, counterevidence, 구현·검증 자료와 선행연구 대비 주장 가능한 범위를 모두 남겼다.
