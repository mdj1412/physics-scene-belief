## 2. Model mismatch와 acquisition approximation을 어떻게 분리하는가

V14의 `matched` condition은 base contact laws에서만 truth를 생성하고 inference는 flexible laws에도 prior mass를 두었다. 따라서 population prior까지 일치하는 control은 아니었다. V15에서는 truth와 inference가 같은 expanded prior를 사용한다.

- Geometry family 2개 × base frictionless/Coulomb 2개: 각 prior mass 1/8.
- Geometry family 2개 × flexible contact law: 각 prior mass 1/4.
- `s∼U[−0.4,0.4]`, `a∼U[−0.1,0.1]`, `e∼U[0.35,0.9]`, `b₀∼U[−4,2]`, `b₁∼U[−2,2]`.
- Unknown boundary offset `d∼U[−4,4] mm`를 truth와 inference에 동일하게 추가한다.
- Worlds는 iid draws, inference bank는 별도의 scrambled Sobol QMC samples다. True world를 bank에서 뽑지 않는다.

Nominal flexible contact law는 다음과 같다.

\[
j_t=v_t\,\sigma\left(b_0+b_1\log\frac{|v_t|+0.02}{0.5}\right).
\]

Shifted data는 같은 latent parameters와 observation noise를 사용하면서 flexible component의 tangential fraction만 다음으로 바꾼다.

\[
j_t=v_t\,\sigma(b_0)\left[0.2+\frac{0.8}{1+(|v_t|/s_0)^2}\right],
\qquad s_0=0.1+0.7\sigma(b_1).
\]

Base frictionless/Coulomb worlds는 unchanged negative controls다. 다음 세 cells를 비교한다.

| Cell | 실제 data-generating law | Inference law | 목적 |
|---|---|---|---|
| Matched / nominal | Expanded nominal prior와 law | Nominal | Population/model mismatch 제거 |
| Shifted / nominal | Flexible law만 변경 | Nominal | Misspecified inference/design |
| Shifted / correct | 같은 shifted data | 올바른 shifted law | Privileged correct-law diagnostic |

마지막 cell은 실제 배포 가능한 방법이나 새 알고리즘의 성능이 아니다. 어떤 contact law를 써야 하는지 알고 있다는 privileged assumption으로 model mismatch의 영향을 분리한다.

같은 shifted worlds에서 nominal selector의 action을 correct-law decoder로 재평가한다. 그러면 전체 차이를 다음처럼 분해할 수 있다.

\[
L_{\mathrm{nom},a_{\mathrm{nom}}}-L_{\mathrm{correct},a_{\mathrm{correct}}}
=\left(L_{\mathrm{nom},a_{\mathrm{nom}}}-L_{\mathrm{correct},a_{\mathrm{nom}}}\right)
+\left(L_{\mathrm{correct},a_{\mathrm{nom}}}-L_{\mathrm{correct},a_{\mathrm{correct}}}\right).
\]

첫 항은 같은 action에서 decoder 차이, 둘째 항은 correct decoder 아래의 selection 차이다. 두 항은 이 비교 경로에 따른 decomposition이며 보편적으로 독립적인 원인이라는 뜻은 아니다. Finite QMC integration error도 남는다. Matched/shifted 평균 차이만으로 pure mismatch effect를 추정하지 않는다.

## 3. Transition task를 어떻게 만드는가

Factual evidence는 9개 boundary profile samples와 factual outgoing velocity다. Profile noise SD는 1.2 mm, velocity noise SD는 0.07 m/s다. Query position은 `y=25 mm`다.

### Contact/no-contact

Factual profile에 straight-line least-squares fit을 하고 query position까지 extrapolate한 값을 public `x_end`로 사용한다. Candidate boundary가 `x(y)≤x_end`이면 contact, 그렇지 않으면 no-contact로 정의한다. No-contact일 때 outgoing velocity는 incoming velocity `(1,0.9)`를 유지한다.

이는 finite-reach contact gate가 붙은 local impulse surrogate다. 전체 trajectory를 적분하여 continuous collision detection을 수행한 simulator는 아니다. True geometry로 transition 근처 worlds만 사후 선택하지 않으며, public evidence에서 정한 query를 모든 hypotheses가 공유한다.

### Slip/stick

Public profile에서 추정한 slope `ŝ`를 `[-0.6,0.6]`으로 clip하고,

\[
r=0.22(1+0.625),\qquad v_y=\frac{r-\hat s}{1+r\hat s}
\]

로 query incoming velocity를 설정한다. Median restitution의 Coulomb transition을 public estimate로 겨냥하는 방식이다. Exact stick label은 residual tangential velocity의 절댓값이 `1e−8` 이하일 때 1이다. Flexible soft-friction law에는 일반적으로 exact stick이 없으므로, 이 task에는 contact-model identity에 대한 uncertainty도 포함된다.

| Task | Prediction | Loss |
|---|---|---|
| Smooth velocity | Always-contact outgoing velocity | Half squared Euclidean error |
| Contact velocity | Contact gate 이후 velocity | Half squared Euclidean error |
| Contact event | Contact probability | Brier loss |
| Slip velocity | Transition을 겨냥한 query velocity | Half squared Euclidean error |
| Stick event | Exact-stick probability | Brier loss |

Event tasks의 Bayes decision은 probability forecast다. Robot의 discrete control action 성공률을 측정한 것으로 표현하지 않는다. 구현에서는 event target에 `√2`를 곱해 공통 half-squared-error 계산이 Brier loss와 일치하도록 했다.

Observation actions는 한 번의 `query_view`, `factual_view`, `probe`이며, 각각 query 부근 boundary profile, factual 부근 정밀 profile, 다른 incoming velocity의 contact response를 제공한다. 모든 actions의 sensing cost는 같게 두었다. `none`은 별도 no-observation diagnostic이다. 실제 sensing/reset 비용이나 raw RGB processing은 이번 실험에 포함하지 않는다.

## 4. 무엇과 비교하는가

Primary comparison은 같은 expanded QMC16384 posterior/decoder에서 acquisition만 바꾼다.

1. **LMMSE:** Query target과 candidate observation의 covariance를 이용한 linear prediction gain.
2. **Disagreement:** V14의 K256 committee, factual log-likelihood cutoffs 8/2, query difference와 Gaussian observation discriminability의 pairwise score.
3. **Cross-fitted nested128/512:** Independent outer bank가 hypothetical world/observation을 생성하고, inner bank decoder의 squared error를 outer query target에 대해 직접 평균한다.
4. **Fixed actions:** Always-probe, always-query-view, always-factual-view, none.
5. **QMC65536:** LMMSE와 disagreement, fixed actions의 더 큰 bank 비교.
6. **Ablations:** Factual-fit filtering 제거, query weighting 제거, nested posterior-variance proxy.

LMMSE는 최종 predictor가 아니다. 모든 방법의 actual observation 이후 prediction은 같은 posterior mean이다. 따라서 final nonlinear inference를 일부러 약하게 만든 baseline과 비교하지 않는다.

Nested512는 V14의 expected posterior variance만 계산하는 방식과 달리, independent outer hypotheses의 targets를 사용하여 finite inner decoder의 expected loss를 직접 평가한다. 그렇더라도 finite inner/outer banks와 hypothetical samples를 사용하는 approximation이다. Exact oracle로 부르지 않는다.

첫 32 worlds에서는 nested2048의 두 hypothetical seeds, QMC64k inner/outer nested512, pair K512를 audit한다. Numerical audit를 본 뒤 primary K나 sampling settings를 바꾸지 않는다.

