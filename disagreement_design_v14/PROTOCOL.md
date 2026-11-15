# v14 pre-test protocol — 2026-11-12 KST

목적: factual evidence를 비슷하게 설명하는 hypotheses의 query disagreement를 해소하는 observation을 선택하면 strong baselines보다 decision loss를 줄일 수 있는가?

Primary 비교는 같은 expanded QMC16384 posterior/decoder에서 acquisition만 바꾼다. Primary method는 `qmc16_pair`. Compact expanded GH3/4-step에도 같은 selector를 별도로 적용하되 compact는 full posterior로부터 distillation하지 않는다. v13 expanded model equations/prior, one-observation budget, fixed query/visibility를 그대로 사용한다.

Selector: factual posterior에서 deterministic systematic inverse-CDF resampling으로 K256 hypotheses를 얻는다. Factual log likelihood가 committee max에서8 이내인 hypotheses에 대해 pairwise |loglik_i-loglik_j|<=2를 요구한다. Pair query difference D_ij=||T_i-T_j||²/2. Candidate action score는 eligible pairs의 평균 D_ij*(1-exp(-||mu_ai-mu_aj||²_Ra^-1/8)). 후자는 equal-covariance Gaussian observation distributions의 Bhattacharyya overlap complement다. Exact expected Bayes-risk reduction이라는 주장은 하지 않는다. Eligible weighted query difference가 없거나 모든 scores가 수치적으로0이면 동일 belief의 LMMSE로 fallback한다. K와 cutoffs는 test 전에 고정하고 test에서 조정하지 않는다.

Ablations: factual-fit band를 제거한 query-weighted score, query weight를 제거한 model-discrimination score. Baselines: expanded compact 및 QMC16/64 LMMSE, always-probe, always-query-view, no-observation diagnostic. QMC16-pair와 compact-pair의 common decoder 비교로 acquisition/inference 효과를 구분한다.

Strong expected-loss baseline: inner QMC16384(seed20261112)와 independent outer QMC16384(seed20261113)의 factual weights를 이용한 cross-fitted nested expected posterior variance reduction. Hypothetical observation128개를 scrambled Sobol inverse-CDF hypothesis draw + Gaussian noise로 생성한다. 같은 hypothetical randomness를 actions 사이에 공유한다. Actual prediction은 inner bank만 사용한다. Outer bank fitting/cache 및 selection 비용을 timing에 포함한다. 이 baseline도 finite integration이며 exact oracle은 아니다.

Fresh test256 base worlds(seed3100101+137*index) ×6 paired conditions(matched, viscous, saturating half/full, speed-weakening half/full) ×16 actual noise repeats. v13에서 이미 본 law families의 새로운 worlds이며 새 law-family holdout이라고 표현하지 않는다. c0 negative control은 그대로 두고 실제 변경된 c1 subset도 별도 보고한다. Pilot16 matched worlds(seed3000101)에서 implementation checks만 수행하고 hyperparameter 선택은 하지 않는다.

Primary six contrasts: qmc16_pair vs qmc16_native / qmc16_probe / qmc16_nested128 on saturating_full and weakening_full. Independent world 단위30000 paired bootstrap, Bonferroni99.1667% CI. Success screening은 두 full laws에서 세 baseline 모두에 adjusted difference upper<0, matched loss ratio upper95%CI<=1.10. Compute claim은 같은 hardware에서 nested128 대비>=2x speedup 및 matched harm criterion을 함께 요구하되 QMC native/probe latency도 모두 공개한다. 기준은 diagnostic engineering targets이지 보편적 application tolerance가 아니다.

Numerical audit: first16 base worlds×6 conditions에서 K512 pair committee, nested512 및 다른 hypothetical Sobol seed의 nested512를 비교한다. Committee/ranking uncertainty와 resulting actual decision loss 차이를 평가하고 main policy를 audit로 바꾸지 않는다.

Timing: GPU 및 CPU2threads float64 batch1,48 fixed cases×3, warmup, explicit synchronization. Factual fit/selector/selected-observation update+prediction 포함. Setup/shared caches/per-world state/committee memory는 별도. All-action prediction caches는 evaluation에만 쓰고 online benchmark는 선택한 observation만 받는다.

Integrity: two-hypothesis physical examples(identical factual velocity, different query)와 counterexample(geometry ambiguity)의 expected action을 검사한다. Compact policies/evaluations가 reference import 전에 완료된다. All policies freeze 후 actual private outcomes로 평가한다. v13 sealed artifacts는 변경하지 않는다.
