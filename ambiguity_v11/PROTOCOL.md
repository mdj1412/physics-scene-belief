# v11: visibility/query를 고정한 geometry–contact ambiguity

작성일 2026-11-12 KST. Test outcome을 보기 전에 작성한 protocol이다.

## 질문과 한계

동일한 visibility, query, sensor noise 아래에서 factual evidence에 따라 적절한 observation이 바뀌는가? Boundary-preserving measurements를 사용하는 updated plug-in, learned rule, factorized posterior보다 joint posterior가 실제 decision loss를 낮추는가?

Unknown은 finite candidate family 안의 uncertainty를 뜻한다. Open-world geometry discovery나 실제 RGB reconstruction을 뜻하지 않는다. 새 acquisition algorithm을 제안하는 실험이 아니라 필요조건을 검증하는 diagnostic이다.

## Model

Boundary x(y)=s*y+a*f_g(y), L=25 mm. g=0: f=L*z²/2, g=1: f=L*(z²/2-0.35*z⁴), z=y/L. Both families have derivative s at factual y=0, but derivatives s+a and s-0.4a at query y=L. s~Uniform[-.4,.4], a~Uniform[-.5,.5], e~Uniform[.35,.9]; g and contact class c are independent Bernoulli(.5) in prior. c=0 frictionless, c=1 Coulomb friction coefficient .22. No rotational inertia, gravity, finite radius or multi-contact; point mass and single impulse.

Factual profile: fixed 9 rows y∈[-10,10] mm, independent Gaussian position noise 1.2 mm. Calibrated synthetic boundary profile; not a new raw RGB experiment. Factual impact v=(1,.35), output noise .07 m/s per component. Query always y=25 mm, incoming v=(1,.6). Every world has the same visibility and same query.

Actions: none, query-view (7 rows 19..31 mm, position noise 1 mm), factual-view (7 rows -8..8 mm, position noise .35 mm), diagnostic-probe (reset impact y=0, v=(1,-.7), velocity noise .05 m/s). Query-view may reveal local slope and family. Factual-view resolves geometry at factual contact. Probe distinguishes contact responses; it does not reveal query outcome. One observation/action per episode, equal action cost for primary analysis.

Independent continuous truth draws: pilot 8, rule training 256, rule validation 128, test 256, misspecification stress test 128. Seeds 51001,61001,71001,81001,91001 respectively; per-world +137*i. Test gets 32 actual noise repeats, shared across policies. No world is filtered using target gap or successful ranking. Sample size is a fixed compute design, not a power calculation.

## Inference and baselines

Tensor Gauss–Legendre quadrature over (s,a,e), exact sum over (g,c); order is selected only by pilot numerical audit. Hypothetical observations are sampled independently from factual posterior; primary outer H=256. Acquisition minimizes posterior expected squared velocity error. CF-SNIS uses two independent factual-posterior banks, 8192 total, and independent outer draws. Cross-product correction does not make self-normalized estimates unbiased.

Joint Bayes prediction, updated hard-family/contact plug-in (conditional posterior parameter means, nonlinear response), and factorized-factual posterior q(geometry)*q(contact,e) are compared. Plug-in is re-estimated after selected observation; a strong plug-in selector minimizes its own posterior expected actual loss, integrating possible hypotheses, not a certainty-only heuristic. This selector is privileged with full uncertainty for fair decoder comparison and is explicitly labeled.

Rules: best fixed action; learned RBF kernel ridge rule and CART rule using only factual profile/velocity. Hyperparameters chosen on validation posterior expected regret, not test truth. A posterior-summary uncertainty heuristic is also allowed if fixed before test. All rules may choose any of the three actions or none. Learned rules are amortized acquisition distilled from Bayes expected-risk labels, a strong baseline whose offline training cost is reported separately.

## Evaluation and decisions

Primary MSE=||predicted velocity−true velocity||²/2. Paired world-level means over 32 noise repeats, 30,000 cluster bootstrap resamples. Primary joint-vs-none, joint-vs-best learned rule, joint-vs-strong updated plug-in, joint-vs-factorized use Bonferroni 98.75% CIs. Other comparisons descriptive 95% CIs. No equivalence margin is specified; a CI containing zero means inconclusive, not equivalent.

Policy/public-data hashes freeze before private test evaluation. Report both acquisition with a common joint decoder and each method's native decoder. Same-action comparison isolates posterior/model-averaging from observation selection. Entropy/correlation and action frequencies are diagnostic, not predeclared causal subgroup claims.

Numerical checks: contact energy nonincrease, frictionless reduction, finite-difference slopes, quadrature order convergence, independent outer repeat ranking, SNIS ESS. If near ties remain, report regret and uncertainty rather than calling a finite-H winner the true optimum. Refine deterministic decoding without changing frozen choices when needed; preserve initial outputs.

Misspecification stress test uses a third boundary f=L*(z²/2-.15*z⁴+.06*z⁶) and friction coefficient Uniform[.1,.3], with inference candidate set unchanged. This is secondary and must not be pooled into in-support primary results.
