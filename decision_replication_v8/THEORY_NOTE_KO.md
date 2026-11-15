# Cross-fitting과 decision error의 해석

이 문서는 실험 결과를 보기 전에 estimator 해석을 정리한 것이며, 새로운 theorem이나 proved adaptive algorithm을 주장하지 않는다.

## Squared-shift bias

Factual evidence D와 hypothetical observation y를 고정하고 true conditional predictive mean을 m, current mean을 m0라 하자. Estimate를 mhat=m+b+epsilon으로 쓰고 E[epsilon]=0라 하면:

`E[||mhat-m0||²/2] = ||m-m0||²/2 + (m-m0)·b + ||b||²/2 + tr(Cov(epsilon))/2`.

따라서 squared-shift utility에는 variance의 positive term이 생긴다. 하지만 finite-bank SNIS의 모든 error가 positive bias는 아니다. Self-normalization bias b의 방향에 따라 첫 bias term은 양수 또는 음수가 될 수 있다.

Independent A/B estimates의 cross-product는 D,y,m0를 고정했을 때:

`E[(mhatA-m0)·(mhatB-m0)/2] = ((m-m0)+bA)·((m-m0)+bB)/2`.

Independent sampling error의 covariance term은 사라지지만 bA,bB와 common finite m0 error는 남는다. 같은 SMC population을 둘로 split하면 genealogy dependence 때문에 independence 가정이 성립하지 않을 수 있어, 이번 구현은 별도의 seeds로 independent factual SMC streams를 만든다. 각 stream 내부의 particles가 i.i.d.가 되는 것은 아니다.

Pooled SNIS는 각 half의 conditional normalization constants에 비례해 두 means를 결합한다. 따라서 `mpooled=(mA+mB)/2`가 일반적으로 성립하지 않는다. 이번 Standard/Cross-fitted 비교는 같은 total particles를 공유하는 estimator control이지만, exactly equal marginal conditional means에 대한 하나의 algebraic variance subtraction은 아니다.

## Acquisition ranking과 decision loss

True Bayesian acquisition value를 U(a), approximate value를 Uhat(a)라 하자. 만약 모든 actions에 대해 `|Uhat(a)-U(a)|≤epsilon`이면 approximate maximizer ahat의 Bayesian regret는 `U(a*)-U(ahat)≤2epsilon`이다. 이는 maximum의 정의와 triangle inequality로 나온다.

다만 이번 실험은 이 uniform error bound를 certified한 것이 아니다. Outer bootstrap은 bank와 realized inner SMC estimates를 고정하며, ESS도 proposal support error나 missed posterior modes를 완전히 반영하지 않는다. 따라서 bootstrap interval 또는 ESS 하나만으로 위 bound의 전제를 충족했다고 말할 수 없다.

또한 Bayesian expected regret와 특정 hidden world의 realized loss difference는 다르다. Posterior 평균상 near-tied actions라도 한 fixed world에서의 차이는 클 수 있다. Independent worlds를 늘려 population mean loss를 검증해야 하는 이유다.

## 가능한 후속 기여의 조건

Action gap에 비해 estimator error가 큰 queries만 refine하는 정책을 제안하려면 적어도 다음을 보여야 한다.

1. Fixed-bank bootstrap 외에도 independent bank variability, posterior approximation error 또는 proposal mismatch를 반영하는 error indicator.
2. Indicator로 정한 budget이 hindsight actual outcomes에 접근하지 않는다는 data boundary.
3. Equal wall-time ceiling에서 bank expansion, cross-fitted SNIS, always-refine Bridge와 비교한 actual decision loss.
4. Known-family synthetic proxy를 넘어서는 likelihood mismatch에서의 calibration 또는 failure characterization.

이번 v8은 3번의 controlled baselines를 실행하고 1–2번 설계에 필요한 empirical failure를 검토한다. 새로운 selective refinement policy를 이미 구현·검증했다고 표현하지 않는다.
