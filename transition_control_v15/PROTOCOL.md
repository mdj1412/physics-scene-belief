# v15 사전 protocol — 2026-11-12

## 목적과 control

Expanded prior에서 iid worlds를 직접 생성하여 population/model mismatch를 제거한다. 원래 6 mixture components의 masses (1/8,1/8,1/8,1/8,1/4,1/4), uniform s,a,e,b0,b1을 유지하고 unknown boundary offset d~U[-4,4] mm를 truth와 inference에 동일하게 추가한다. Candidate contact law는 base frictionless/Coulomb 및 flexible sigmoid law다.

Matched와 shifted data를 같은 latent parameters/noise로 생성한다. Shifted condition은 c2 component의 tangential fraction만 sigmoid(b0)*[0.2+0.8/(1+(|vt|/scale)^2)], scale=0.1+0.7 sigmoid(b1)로 바꾼다. 이때 nominal inference는 기존 law, privileged correct-law inference는 shifted law를 안다고 가정한다. Cells: matched/nominal, shifted/nominal, shifted/correct. 마지막은 deployable baseline이 아니라 model mismatch를 분리하는 diagnostic이다.

Factual boundary profiles와 velocities, 세 observation actions/query_view/factual_view/probe 및 noise는 v14와 같다. Boundary offset을 profile에 포함한다. At most one observation, equal action costs. No-observation은 별도 diagnostic이다.

## Queries와 losses

모든 public query context는 factual profile의 least-squares straight-line fit에서 정한다. Hidden truth로 transition cases를 골라내지 않는다. Query y=25mm. Contact threshold x_end는 factual profile line을 query y까지 extrapolate한 값이다. Contact iff true boundary x(y)<=x_end. Contact하지 않으면 incoming velocity를 유지한다. 이는 finite-reach contact gate가 붙은 local impulse surrogate이며 full trajectory solver가 아니다.

Slip query의 incoming vy=(r-s_hat)/(1+r*s_hat), r=.22*(1+.625), s_hat=factual-profile estimated slope clipped[-.6,.6]. 따라서 nominal median restitution의 Coulomb transition을 public estimate로 겨냥한다. Exact stick label iff residual tangential velocity absolute<=1e-8. c2 has soft friction, exact stick 거의 없음. True law label을 selector에 제공하지 않는다.

Tasks: smooth_velocity, contact_velocity, contact_event, slip_velocity, slip_event. Velocity loss=half squared Euclidean error; event loss=Brier. Event Bayes decision은 posterior probability이며 0/1 control action 성공률과 다르다. All five targets의 actual predictions는 Bayesian posterior mean이다. LMMSE는 acquisition approximation에만 사용한다.

## Methods와 sampling

QMC16k nominal/correct inner banks independent Sobol seed2026111201, independent outer16k seed2026111202. QMC64k nominal bank seed2026111203. 실제 truth는 NumPy iid (seed4100101+137*i), 512 independent worlds, actual observation repeats16. Pilot8 seed4000101은 correctness만 검증한다.

Same decoder comparisons: LMMSE, pair K256 v14 score/cutoffs8/2, unfiltered pair, agnostic discrimination, cross-fitted nested128/nested512, nested512 posterior-variance diagnostic, always-query, always-factual-view, always-probe, none. QMC64k LMMSE/pair와 fixed actions를 full test에 포함한다. Policy는 private targets/actual observations를 읽기 전에 모두 freeze한다.

Nested acquisition은 outer posterior draws와 hypothetical Gaussian observations를 이용하여 INNER decoder의 actual squared error against OUTER target을 직접 평균한다. v14 variance proxy와 구분한다. Common randomness across actions, Sobol plan seed81531. Nested512 variance proxy도 기록하여 finite-bank discrepancy를 확인한다. Exact oracle이라고 부르지 않는다.

첫32 base worlds에서 nested2048 두 hypothetical seeds81531/81532 및 independent QMC64k inner/outer nested512로 numerical sensitivity를 audit한다. PairK512도 비교한다. Test 결과로 main settings를 조정하지 않는다.

## Statistics / claims

Primary family10 contrasts: matched nominal의 contact_event/slip_event에서 pair minus LMMSE/always-probe/always-query/nested512 각4, nested512 minus LMMSE 각1. 30,000 paired bootstrap, independent world 단위, Bonferroni99.5% CI. LMMSE failure claim은 nested512의 relative mean improvement>=10% AND adjusted difference upper<0일 때만 한다. Disagreement superiority는 각 task에서 네 baseline에 모두 adjusted upper<0일 때만 주장한다. 나머지 tasks/cells/subgroups는 secondary95% CIs.

Matched vs shifted risk difference는 task distribution 자체도 달라지므로 pure mismatch effect로 부르지 않는다. 같은 shifted worlds에서 nominal-vs-correct inference를 비교하고 cross-decoder action swaps로 decoder/selection 영향을 분리한다. 각 비교에 finite-integration error가 남음을 명시한다. Audit reference의 internal expected risk ranking과 held-out actual loss를 혼동하지 않는다.

## Compute / integrity

GPU 및 CPU2threads float64 batch1. Fixed20 matched worlds×5 tasks, repetitions3; factual fit, query-specific target generation, selection, selected-observation update 포함. Shared bank setup/cache는 별도; per-task timing은 five-task batched reuse를 하지 않는다. QMC64 같은 실제 budget sweep 포함, equal time라고 가정하지 않는다. Predictive mean 검증, probability range, branch physical checks, QMC prior weights, same noise pairing 및 private outcome access 분리 검증. V14 manifest 전체 보존.
