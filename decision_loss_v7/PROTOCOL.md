# New-world repeated-noise decision-loss audit — v7

2026-11-11, v7 outcomes를 생성/열람하기 전에 고정했다. V6의 failure 분석을 알고 설계한 후속 실험이다. Formal preregistration은 아니다.

## 질문과 primary comparison

Tempered bridge의 더 정확한 hypothetical posterior가 factual-bank SNIS보다 **실제로 선택한 observation의 expected prediction loss**를 낮추는가? Primary comparison은 native pixel-only precision, equal observation cost에서 bridge와 SNIS4096의 paired MSE difference다. SNIS16384를 강한 scaling baseline으로 함께 비교한다.

## 새로운 worlds와 independent randomness

- 48개 새로운 prior-drawn worlds: corner/gate/deflection 각각 16. World seed `202611110700 + case`. V5/v6 worlds를 재사용하지 않는다. Geometry/physics uniform prior 및 6 queries는 v6와 같다.
- World당 하나의 새로운 factual Gaussian x-velocity measurement를 생성한다. Factual measurement는 그 world 안에서 고정한다. 새로운 observation의 noise만 32회 독립적으로 반복한다. 이 반복을 32개의 independent worlds로 계산하지 않는다.
- Inference factual bank 16384 particles를 만들고 fixed random permutation의 첫 4096을 bridge/SNIS4096에 사용한다. SNIS16384는 전체 bank를 사용한다.
- Hypothetical observations는 **독립 seed로 생성한 별도 factual bank 8192 particles**에서 발생시킨다. SNIS bank의 self-inclusion을 제거한 primary comparison이다. 모든 methods에 같은 outer observations를 제공한다. Current predictive mean도 이 independent generator bank에서 공통으로 계산한다.
- Seed namespaces: world 202611110700+, inference bank 710000+, generator bank 720000+, permutation 730000+, outer 740000+, repeated observations 750000+, SMC run SHA-256-derived seeds. Seeds와 actual input files를 저장한다.

## 사전에 정한 conditions

1. Native: v5 pixel_only calibrated covariance를 그대로 사용한다.
2. Precise: 같은 feature likelihood의 view standard deviation만 0.25배로 줄인다. 이것은 **synthetic continuous-proxy stress test**이며 더 좋은 실제 camera나 새 raw-image model을 구현했다는 뜻이 아니다.

각 precision에서 비용 profiles는 equal `[0,0,0,0]`, view_cheaper `[0,.002,.002,.002]`, view_costlier `[.002,0,0,0]`다. Action order는 view/left/right/center. 비용 단위는 velocity MSE `(m/s)^2`에 더해지는 utility units이며 실제 장비의 비용을 측정한 값이 아니다. 정확히 하나의 observation을 선택하는 constrained task이다. No-observation은 descriptive baseline으로만 보고한다.

## Inference와 selection

- Bridge target과 resample–move implementation은 v6 그대로 사용한다. N1024,12 MH moves/stage,ESS=.75N,2 independent inner runs.
- Main outer H=256. MSE-scale EVSI는 두 independent posterior-mean shifts의 cross-product를 2로 나눈 값이다. SNIS는 같은 current mean에 대한 squared mean shift/2다. Finite-bank 및 finite-SMC bias는 남는다.
- Candidate utility는 EVSI minus fixed observation cost. Methods별로 독립적으로 action을 선택한다. Outcome이나 truth를 보고 utility/precision/cost를 선택하지 않는다.
- Bridge의 common-outer bootstrap 1000회에서 winner regret upper95%가 `max(5e-5,.05*maximum gross EVSI)`보다 큰 query/precision/cost가 있으면 해당 world를 H512로 확장한다. 최대 budget의 ambiguity는 보존한다. Outcomes를 보지 않고 결정하며 rigorous sequential coverage를 주장하지 않는다.
- Strong full-SMC acquisition audit: 사전 선택한 cases 0,1,16,17,32,33에서 H128,N2048,18 MH,2 independent runs. Main bridge의 동일 앞 128 observations와 비교한다. 전 세계에 대한 convergence proof는 아니다.

## 실제 decision loss의 평가

- 동일 true world에서 각 action/precision의 새로운 observations 32개를 생성한다. Methods는 그 measurements를 보기 전에 action을 고정한다.
- 모든 selection policies는 **공유하는 prior-start full SMC decoder** N2048,18 MH를 사용한다. 선택된 action의 posterior mean과 true counterfactual velocity 사이 component-mean squared error에 fixed cost를 더한다. 이로써 acquisition 효과와 후속 decoder 차이를 분리한다.
- 동일 observations에 대한 SNIS4096/16384 predictions도 저장해 end-to-end decoding effect를 secondary로 분리한다. 이 결과를 selection-only benefit으로 표현하지 않는다.
- Decoder audit: 위 6 cases의 같은 32 observations를 N4096,18 MH, independent seed로 다시 계산한다. 공유 decoder의 approximation이 policy 비교 결론을 바꾸는지 평가한다.
- Repeated-noise empirical action oracle는 32 repetitions의 true losses를 보고 action을 고르는 privileged evaluation-only lower bound다. Optimistic finite-R selection bias가 있으므로 실행 가능한 policy라고 부르지 않는다. 가능하면 16/16 split oracle도 함께 보고한다.

## Rasterization mismatch audit

사전 선택한 12 cases `[0,1,2,3,16,17,18,19,32,33,34,35]`에서 native precision에 한해 synthetic RGB renderer/centroid sensor로 post-observation noise를 8회 반복한다. Factual evidence와 selection policies는 primary experiment와 동일하다. 이는 새로운 observation의 likelihood mismatch audit이며 real video나 완전한 image-to-physics validation이 아니다. Precise condition은 이 audit에 포함하지 않는다.

## 통계와 ambiguity 해석

- 48 worlds를 cluster로 사용한 paired bootstrap. Query와 noise repeat를 independent sample로 세지 않는다. Primary 95% interval, 사전 조건 2 precisions×3 costs×2 SNIS baselines의 12 comparisons에는 Bonferroni-level bootstrap intervals를 보조로 표시한다. 50000 bootstrap draws.
- Stable agreement, stable disagreement, unresolved ranking을 actual outcome를 보지 않고 구분한다. Exact top-action ambiguity와 small-regret near-tie를 구분한다. SNIS ESS, precision, bank size, family를 predeclared descriptive breakdown으로 분석한다. Small subgroups의 발견을 confirmatory claim으로 과장하지 않는다.
- Outcome noise의 반복에 따라 paired loss estimate가 어떻게 바뀌는지 R=1/4/8/16/32 prefixes로 분석한다. 더 유리한 R을 골라 primary result로 바꾸지 않는다.
- 어떤 조건에서도 추가 이득이 없으면 그 결과를 보고한다. Tempering 자체는 기존 방법이다. 논문 가능성은 새로운 worlds에서의 decision-loss 개선과 비용 대비 가치, 실제 measurement mismatch에서의 견고함으로 평가한다.

## 실행

Server: `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/decision_loss_v7`. 현재 idle allocations를 확인하고 GPU당 guarded worker 하나를 실행한다. Grouping은 실행 효율만 바꾸며 대상/budget/seed group metadata를 기록한다. 기존 결과와 사용자 작업은 변경하지 않는다.
