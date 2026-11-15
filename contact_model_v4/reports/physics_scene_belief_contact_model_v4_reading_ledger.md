# Contact-Model Ambiguity: 원문 검토 기록

검토일: 2026-11-05. 설명은 한국어, 논문의 technical term은 English로 유지했다.

이 목록은 이번 연구 설계에 직접 사용한 원문의 **확인 범위**다. ‘Method 확인’은 모든 appendix와 proof까지 독해했다는 의미가 아니다. 기존 v4의 2017–2026년 대표 논문 목록은 연구 흐름 분석에 승계했으며, 이번에 그 모든 원문을 다시 읽었다고 주장하지 않는다. 새 preprint에는 확인하지 않은 conference acceptance를 붙이지 않는다.

## 1. 네 연구자의 핵심 논문

| 연구자 / 논문 | 이번에 읽은 범위 | 실험에 반영한 질문 |
|---|---|---|
| Yale Song — [CausalCity](https://arxiv.org/html/2106.13364) | Introduction, Related Work, benchmark 구성, §5–6 | Factual과 counterfactual이 공유하는 confounder는 무엇이며, perception을 우회한 부분을 공개했는가? |
| Yale Song — [Neural-Sim](https://arxiv.org/html/2207.11368) | §2–3 | Downstream task에 유용한 data selection과 실제 새 measurement를 구분했는가? |
| Seon Joo Kim — [STM](https://arxiv.org/html/1904.00607) | §2–3 | Predicted mask를 memory에 다시 넣는 것이 독립적인 evidence를 추가하는 것은 아니다. |
| Seon Joo Kim — [Omni-RGPT](https://arxiv.org/html/2501.08326) | Introduction, §2–3 | Region reference와 시간에 걸친 object binding이 유지되는가? |
| Seon Joo Kim — [RelationVGGT](https://relationvggt.github.io/) | 저자 project page의 Method; 논문 전문 독해로 표시하지 않음 | Multi-view semantic relation과 mechanical contact constraint를 구분한다. |
| Seon Joo Kim — [HuRo](https://arxiv.org/html/2609.10706v5) | §2–3의 camera/hand estimation 및 retargeting 설명 | Observation conversion과 action alignment를 별도 검증한다. |
| Seungryong Kim — [CATs](https://arxiv.org/html/2106.02520) | §2–3 | Semantic matching과 같은 physical instance의 correspondence는 다르다. |
| Seungryong Kim — [Trace4D](https://arxiv.org/html/2611.02759) | §2, §3.1–3.2; geometry-aware training의 세부 전체는 미검토 | Reconstruction-derived 4D reference를 metric GT 또는 causal model로 취급하지 않는다. |
| Seungryong Kim — [DreamX](https://arxiv.org/html/2611.05377) | §2, §3.1–3.2 | Compact representation이 reconstruction뿐 아니라 intervention response를 보존하는가? |
| Lingjie Liu — [UniPixie](https://arxiv.org/html/2606.05399v2) | §2–3.3, Appendix 6.1의 range annotation | Soft/stiff material range와 evidence-conditioned calibrated posterior를 구분한다. |
| Lingjie Liu — [Vid2Sim](https://arxiv.org/html/2506.06440) | §2–4.1 | Material inference, reduced simulation, video fitting 중 어느 단계의 오류인가? |
| Lingjie Liu — [Pixie](https://arxiv.org/html/2508.17437) | Abstract와 project 설명 중심; 상세 Method 재독으로 표시하지 않음 | Plausible physics prior가 실제 물체의 identified physics를 뜻하지 않는다. |

Trace4D와 DreamX는 각각 2026-11-02, 2026-11-03의 공개 원고 날짜를 확인했다. 이 두 논문의 결과를 독립적으로 재현하지 않았으며, 제안된 representation에서 얻은 설계 관점만 사용했다.

## 2. Recursive reading으로 내려간 Background

| 경로 / 논문 | 확인 범위 | 얻은 설계 제약 |
|---|---|---|
| CausalCity → [CLEVRER](https://arxiv.org/html/1910.01442) | Related Work, dataset/task 정의 | Descriptive, predictive, explanatory, counterfactual task를 섞지 않는다. |
| CLEVRER → [CoPhy](https://arxiv.org/html/1909.12000) | Introduction, §3, §4.1의 de-rendering·GCN·latent confounder inference | Paired intervention은 shared confounder와 명시적인 initial-condition intervention을 필요로 한다. |
| CausalCity → [V-CDN](https://arxiv.org/html/2007.00631) | §2의 causal discovery와 dynamics 설명 | Continuous parameter와 discrete relation을 함께 추정한다는 발상 자체는 새롭지 않다. |
| Omni-RGPT → [MotionEpic / Video-of-Thought](https://arxiv.org/html/2501.03230) | Introduction, §2, pipeline 개요 | Spatio-temporal graph를 쓰는 것 자체를 novelty로 삼을 수 없다. |
| MotionEpic → [Action Genome](https://arxiv.org/html/1912.06992) | §3, §4.1 및 scene graph feature bank 개요 | ‘Contact’ semantic label은 force, friction, collision normal의 supervision과 다르다. |
| Trace4D → [Pi3](https://arxiv.org/html/2507.13347) | §2–3.1 | Permutation-equivariant reconstruction이 scale ambiguity까지 없애는 것은 아니다. |
| Pi3 → [VGGT](https://arxiv.org/html/2503.11651) | §2, §3.1–3.2 | Point map, depth, camera, tracking feature는 measurement model을 구성할 재료다. |
| Vid2Sim → [Simplicits](https://arxiv.org/html/2407.09497) | Introduction/Related Work, §3.1–3.3 | Occupancy-based interface와 neural skinning field는 simulation 표현이지 material posterior가 아니다. |
| UniPixie → [PhysDreamer](https://arxiv.org/html/2404.13026) | §2–4.1 | Generated dynamics를 이용한 plausibility와 실제 measurement에 의한 identification을 분리한다. |

화살표는 본문/Related Work에서 확인한 reference를 따라간 경로다. 위 연구를 모두 baseline으로 구현한 것은 아니다. 서로 다른 input, supervision, simulation class를 가진 논문의 발표 수치를 우리 toy experiment와 직접 비교하지 않았다.

## 3. 가장 가까운 inference / design 문헌

| 논문 | 확인 범위 | 현재 주장에 대한 제한 |
|---|---|---|
| [Task-Specific Next-Best-View with Uncertain Geometry](https://arxiv.org/html/2605.05095) | 이전 §3–4 검토를 승계; 이번 §2.2–2.3 재확인 | Geometry uncertainty와 downstream utility 기반 view selection은 이미 존재한다. |
| [IDTD](https://arxiv.org/html/2610.10905) | §III–IV의 compensation 및 Schur-complement FIM | Parameter compensation이라는 현상 자체를 novelty로 부르지 않는다. |
| [Action-BED](https://arxiv.org/html/2606.23662) | 이전 §3–4 검토 승계; 이번 Introduction 재확인 | Expected future task loss 기반 design은 새 원리가 아니다. |
| [Inside-Out SMC²](https://arxiv.org/html/2402.07868) | §2–3 | Naive prior particles의 실패와 strong sequential inference의 한계를 구분한다. |
| [Kennedy–O’Hagan calibration](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/1467-9868.00294) | Publisher abstract / bibliographic record만 확인 | Model discrepancy를 parameter uncertainty와 분리하는 일반 원리는 오래된 연구다. |
| [VideoPhysEdit](https://arxiv.org/html/2609.35134v1) | 이전 deep dive의 원문 검토 승계 | Multiple physical candidates 자체는 기여가 아니다. |
| [PhysTwin](https://arxiv.org/html/2503.17973) | 이전 deep dive의 원문 검토 승계 | Sparse real observations에서 unseen interaction 검증이라는 강한 기준이 이미 있다. |

## 4. 문헌에서 실험으로 연결한 방식

1. **문제 정의:** CoPhy의 shared-confounder 관점을 따라 모든 query가 같은 geometry/material을 공유하도록 했다.
2. **Strong baseline:** Inside-Out SMC²를 그대로 재현한 것은 아니지만, 단순 importance sampling보다 강한 adaptive-tempering SMC와 rejuvenation을 사용했다. SMC 자체의 novelty를 주장하지 않는다.
3. **Task utility:** EVSI와 parameter EIG를 구분하고, validation으로 선택한 query-type 및 query-value rule을 추가했다.
4. **Error taxonomy:** Numerical solver setting, finite-particle error, sensor approximation, collision-proxy misspecification을 나눠 보고했다.
5. **Real evidence:** Deform360의 같은 물체에 대한 세 interaction을 확보했지만, material point correspondence와 initial-state registration 없이 counterfactual GT를 선언하지 않았다.

추가 reading은 새로운 failure가 생길 때 다시 확장한다. 현재는 ‘아직 읽지 않은 reference가 많다’는 이유로 실험을 끝없이 미루기보다, 위 가설 중 실제 결과가 지지한 것과 기각한 것을 명확히 나누는 편이 더 유익하다.
