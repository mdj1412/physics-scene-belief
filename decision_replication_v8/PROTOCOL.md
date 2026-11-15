# Independent corner replication v8

2026-11-11 KST. Outcome를 열기 전에 정한 내부 protocol이며 public preregistration은 아니다.

- 새 independent prior worlds 64개, corner family만 사용한다. v7 worlds 재사용 없음. `seed('truth',case)`는 SHA256(`['replication_v8','truth',case]`)의 첫 32 bits다. Results에 유리한 worlds를 선별하지 않는다.
- 기존 5D physics, factual grazing contact, precise view covariance= native covariance ×0.0625, 6 counterfactual queries, 4 observation actions, equal observation cost를 유지한다.
- Factual observation은 world마다 하나다. 선택 이후 observation noise만 32회 반복하며 common random numbers를 모든 policies에 사용한다.
- Independent outer generator: factual full SMC 16384, H512 hypothetical observations. 모든 methods에서 동일하다. Primary current predictive mean도 이 generator에서 구한다. 이 공통 비용은 compute 비교에서 제외하고 별도 명시한다.
- Bridge: factual SMC 4096, conditional SMC N1024, 12 MH moves/stage, ESS threshold .75, 2 independent runs의 cross-product utility. 모든 actions에 H512 고정, 결과에 따른 H 변경 없음.
- SNIS: 두 독립적인 factual SMC streams A/B. Standard utility는 pooled conditional mean의 squared shift /2, cross-fitted utility는 두 streams의 conditional mean shifts의 inner product /2다. Common finite current mean에 의한 bias 및 self-normalization bias까지 제거한다고 주장하지 않는다. Independent SMC streams 내부에는 genealogy dependence가 남는다.
- Fixed baselines: standard SNIS4096 (A 첫 bank), standard 및 cross-fitted SNIS8192 (A/B 각4096), standard 및 cross-fitted SNIS16384 (A/B 각8192).
- Budget baselines: bridge의 factual bank 생성부터 conditional summaries/utility 계산까지 잰 GPU-synchronized wall time 이내에 가능한 큰 independent A/B bank ensemble을 생성한다. Bank 생성, query forward simulation, action likelihood features, importance accumulation, utility 계산 비용을 포함한다. Disk I/O와 shared outer generator/actual decoder는 제외한다. 고정 baseline 이후 chunk를 추가하며 남은 시간에 맞게 줄인다. 완료된 checkpoints 중 budget 이내의 가장 큰 것을 선택한다. Target utilization ≥95%; 실제 비율을 보고하며 부족하면 정확히 equal-budget라고 부르지 않는다. Time overshoot checkpoint는 primary에서 제외하되 계산 비용은 보고한다. 모든 비교는 같은 GPU에서 수행한다.
- Policy를 case별로 고정하고 hash를 남긴 뒤 actual observations를 연다. Common full SMC decoder N2048, 18 MH, 모든 actions, R32. 어떤 selector도 자신의 decoder를 사용해 이득을 얻지 않는다.
- Primary contrasts: bridge − standard4096, bridge − crossfit16384, bridge − standard_budget, bridge − crossfit_budget. World별 loss는 32 repeats와 6 queries 평균한 velocity squared error /2다. 64 worlds를 bootstrap cluster로 삼고 50000 draws, ordinary95% 및 4 contrasts Bonferroni98.75% CI를 보고한다. Negative가 bridge 우세다.
- Ranking은 H512 bootstrapping, half-sample agreement, practical regret를 분리한다. World contact status 등은 descriptive mechanism analysis이며 primary subgroup claim을 추가하지 않는다.
- Full acquisition reference audit: 미리 정한 cases 0,1,2,3,8,9,10,11에서 H128, N2048, moves18, 2 independent runs. 동일 H128로 methods를 비교한다. Decoder N4096 audit는 cases0–3, 동일 R32다. 이것들은 finite numerical reference다.
- Inference 단계에서는 outcome 기반 조기 종료나 method 변경을 하지 않는다. 긴 allocation에서 완료까지 진행한다. 짧은 두 GPU는 worlds preparation만, timeout 전 종료한다.
- 논문 판단: 독립 replication, utility-estimator control, compute-bounded baseline을 모두 통과해야 강한 algorithmic advantage를 주장한다. Tempered SMC 자체는 새로운 algorithm이 아니다.
