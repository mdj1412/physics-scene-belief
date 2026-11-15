# Expanded-prior control, contact/no-contact와 slip/stick transition — v15 최종 보고서

작성일: 2026-11-13 KST. 기존 protocol과 numerical settings를 유지하여 서버 실험을 완료했다. 이 보고서는 2026-11-12의 진행 보고서를 대체하는 최종 결과다.

## 1. 결론

**Expanded prior와 일치하는 control에서도 현재 disagreement selector는 strong baselines보다 나빴다. V14의 실패를 model mismatch만으로 설명할 수 없다.** Contact/no-contact 및 slip/stick transition을 넣었지만, 사전에 정한 “LMMSE failure”와 “disagreement superiority” screening은 모두 통과하지 못했다.

Matched control의 contact-event Brier loss는 pair **0.068424**, LMMSE **0.053824**, nested512 **0.053849**였다. Pair−LMMSE difference는 **+0.014600**, adjusted99.5% CI **[+0.007747,+0.022694]**로 약27.1%의 악화를 지지했다. Stick-event에서도 pair는0.077388로 nested512의0.065355보다 나빴고, difference **+0.012033 [ +0.004646,+0.020379 ]**였다. Always-probe 대비 stick-event 악화도 adjusted CI에서 확인됐다.

Nested512의 stick-event mean loss는 LMMSE보다 약7.05% 낮았다. 그러나 adjusted CI가0을 포함했고, 사전에 요구한 최소10% improvement도 충족하지 못했다. 따라서 “transition을 넣자 LMMSE의 명확한 실패 조건을 찾았다”고 결론 내리지 않는다.

Model mismatch의 영향은 별도로 확인했다. 같은 shifted data에서 correct-law pipeline을 쓰면 smooth-velocity 및 stick-event loss가 줄었다. Pair의 contact-event에서는 선택 경로의 차이가 컸고, smooth-velocity에서는 decoder 차이가 더 컸다. 하지만 correct-law model을 제공해도 pair가 강한 baseline을 이기지는 못했다.

Compute도 강한 implementation과 비교했다. Observation moments caching과 efficient matrix-vector contractions를 적용한 뒤 GPU aggregated median은 **LMMSE 약3.18 ms, pair3.27 ms, nested5129.68 ms**였다. Nested 대비 절약은 있지만 LMMSE 대비 speed advantage는 없다. Always-query/always-probe는 더 저렴하면서 해당 event task에서 pair보다 loss도 낮았다. 최초 비최적화 baseline의 큰 latency를 algorithmic superiority로 사용하지 않는다.

**현재 pair score를 독립적인 method-paper의 핵심으로 계속 추진하는 것은 권하지 않는다.** 이번 결과는 model mismatch, numerical approximation, acquisition objective의 문제를 나누어 확인한 diagnostic evidence다. 후속 연구를 한다면 cutoffs를 반복 조정하기보다, 개별 hypothesis pair의 구별과 decision-region 전체의 구별이 다르다는 문제부터 해결해야 한다.


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



## 5. Decision loss와 primary comparisons

512 independent worlds × two paired data-generating conditions × five query tasks × actual observation noise 16 repeats를 평가했다. Inference cells는 matched/nominal, shifted/nominal, shifted/correct의 세 개다. World마다 noise repeats를 평균한 뒤 independent world 단위로 paired bootstrap했다. 16 repeats나 두 paired conditions를 independent sample로 세지 않았다.

Velocity tasks는 half squared Euclidean error `(m/s)²`, event tasks는 dimensionless Brier loss다. 서로 다른 task losses를 합쳐 단일 accuracy score로 만들지 않는다. `slip_event`는 exact stick probability를 예측하는 task의 code name이다. `native`는 LMMSE acquisition, `pair`는 disagreement selector, `query`/`probe`는 fixed action이다.


### matched_nominal

| Method | smooth_velocity | contact_velocity | contact_event | slip_velocity | slip_event |
|---|---|---|---|---|---|
| native | 0.014347 | 0.080798 | 0.053824 | 0.005781 | 0.070314 |
| pair | 0.015407 | 0.107641 | 0.068424 | 0.006261 | 0.077388 |
| nested128 | 0.014126 | 0.080859 | 0.053870 | 0.005674 | 0.067470 |
| nested512 | 0.014021 | 0.080887 | 0.053849 | 0.005757 | 0.065355 |
| variance512 | 0.014055 | 0.080887 | 0.053849 | 0.005830 | 0.066965 |
| query | 0.020191 | 0.081123 | 0.053824 | 0.008181 | 0.076109 |
| factual | 0.022504 | 0.165745 | 0.110378 | 0.009385 | 0.085204 |
| probe | 0.014363 | 0.199845 | 0.134435 | 0.005821 | 0.066926 |
| none | 0.022336 | 0.297527 | 0.201007 | 0.009568 | 0.080576 |
| unfiltered | 0.015803 | 0.111195 | 0.069881 | 0.006238 | 0.079446 |
| agnostic | 0.018231 | 0.184089 | 0.121218 | 0.007210 | 0.078676 |
| qmc64_native | 0.014050 | 0.077952 | 0.051285 | 0.005696 | 0.066871 |
| qmc64_pair | 0.014677 | 0.097168 | 0.057803 | 0.006159 | 0.072897 |
| qmc64_query | 0.019405 | 0.078226 | 0.051285 | 0.007993 | 0.074530 |
| qmc64_probe | 0.013970 | 0.195754 | 0.131039 | 0.005762 | 0.066411 |

### shifted_nominal

| Method | smooth_velocity | contact_velocity | contact_event | slip_velocity | slip_event |
|---|---|---|---|---|---|
| native | 0.011506 | 0.076596 | 0.053715 | 0.005934 | 0.065505 |
| pair | 0.012112 | 0.100377 | 0.066056 | 0.006358 | 0.073418 |
| nested128 | 0.011420 | 0.076560 | 0.053710 | 0.005880 | 0.065040 |
| nested512 | 0.011381 | 0.076606 | 0.053728 | 0.005916 | 0.064321 |
| variance512 | 0.011496 | 0.076573 | 0.053728 | 0.005938 | 0.065724 |
| query | 0.013603 | 0.076552 | 0.053704 | 0.007845 | 0.068057 |
| factual | 0.016008 | 0.160191 | 0.109908 | 0.009145 | 0.077250 |
| probe | 0.011693 | 0.182467 | 0.127126 | 0.005971 | 0.063700 |
| none | 0.015973 | 0.289653 | 0.200599 | 0.009330 | 0.074121 |
| unfiltered | 0.012278 | 0.101827 | 0.067787 | 0.006307 | 0.073608 |
| agnostic | 0.013514 | 0.172742 | 0.117807 | 0.007221 | 0.073565 |
| qmc64_native | 0.011138 | 0.073672 | 0.051518 | 0.005832 | 0.063024 |
| qmc64_pair | 0.011688 | 0.088260 | 0.057080 | 0.006236 | 0.067856 |
| qmc64_query | 0.013006 | 0.073627 | 0.051518 | 0.007604 | 0.066918 |
| qmc64_probe | 0.011372 | 0.181133 | 0.124873 | 0.005887 | 0.062374 |

### shifted_correct

| Method | smooth_velocity | contact_velocity | contact_event | slip_velocity | slip_event |
|---|---|---|---|---|---|
| native | 0.008118 | 0.075210 | 0.053948 | 0.005590 | 0.057095 |
| pair | 0.008644 | 0.082659 | 0.059777 | 0.006206 | 0.064158 |
| nested128 | 0.008034 | 0.077140 | 0.055303 | 0.005590 | 0.058577 |
| nested512 | 0.008089 | 0.075246 | 0.053989 | 0.005594 | 0.057815 |
| variance512 | 0.008131 | 0.075212 | 0.053948 | 0.005670 | 0.057314 |
| query | 0.009481 | 0.075228 | 0.053948 | 0.007340 | 0.063267 |
| factual | 0.011782 | 0.158749 | 0.110702 | 0.008634 | 0.073539 |
| probe | 0.008823 | 0.174883 | 0.123741 | 0.005718 | 0.057057 |
| none | 0.012865 | 0.288784 | 0.201459 | 0.008953 | 0.070702 |
| unfiltered | 0.009105 | 0.086474 | 0.060883 | 0.006260 | 0.066081 |
| agnostic | 0.010933 | 0.158192 | 0.111092 | 0.007886 | 0.068709 |

### Primary family와 confidence intervals

Matched contact-event/stick-event에서 pair 대 native/probe/query/nested512, nested512 대 native의 총10 contrasts를 사전에 고정했다. 30,000 paired bootstrap과 Bonferroni-adjusted **99.5% CI**다. Difference가 음수이면 앞에 적힌 방법이 유리하다. CI가 0을 포함하는 것을 equivalence로 해석하지 않는다. 나머지 contrasts는 secondary95% CI로 `results/summary.json`에 저장했다.


| Task / comparison | Loss difference [99.5% CI] | Relative mean change |
|---|---|---|
| contact_event/pair-native | +0.014600 [+0.007747, +0.022694] | +27.13% |
| contact_event/pair-probe | -0.066011 [-0.086624, -0.047031] | -49.10% |
| contact_event/pair-query | +0.014600 [+0.007747, +0.022694] | +27.13% |
| contact_event/pair-nested512 | +0.014576 [+0.007731, +0.022673] | +27.07% |
| contact_event/nested512-native | +0.000025 [-0.000000, +0.000122] | +0.05% |
| slip_event/pair-native | +0.007074 [-0.001959, +0.016462] | +10.06% |
| slip_event/pair-probe | +0.010462 [+0.001876, +0.019658] | +15.63% |
| slip_event/pair-query | +0.001279 [-0.009450, +0.011699] | +1.68% |
| slip_event/pair-nested512 | +0.012033 [+0.004646, +0.020379] | +18.41% |
| slip_event/nested512-native | -0.004959 [-0.012254, +0.001992] | -7.05% |

LMMSE failure screening은 nested512가 mean loss를 최소10% 낮추고 adjusted CI upper<0인 조건이다. 판정: {'contact_event': False, 'slip_event': False}. Pair가 네 baseline 모두에 우위를 확보하는 screening 판정: {'contact_event': False, 'slip_event': False}. 이는 사전 engineering criteria이며 모든 application에 적용되는 보편적 임계값은 아니다.

![Event losses](results/event_losses.png)

Bars는 means이며 uncertainty는 위 paired CI table로 판단한다.

### Observation choices

| matched_nominal | Task | [none, query_view, factual_view, probe] |
|---|---|---|
| native | contact_event | [0, 501, 11, 0] |
| pair | contact_event | [0, 418, 34, 60] |
| nested512 | contact_event | [0, 498, 8, 6] |
| native | slip_event | [0, 202, 13, 297] |
| pair | slip_event | [0, 41, 133, 338] |
| nested512 | slip_event | [0, 168, 30, 314] |

| shifted_nominal | Task | [none, query_view, factual_view, probe] |
|---|---|---|
| native | contact_event | [0, 499, 12, 1] |
| pair | contact_event | [0, 417, 33, 62] |
| nested512 | contact_event | [0, 498, 8, 6] |
| native | slip_event | [0, 197, 11, 304] |
| pair | slip_event | [0, 39, 113, 360] |
| nested512 | slip_event | [0, 169, 34, 309] |

| shifted_correct | Task | [none, query_view, factual_view, probe] |
|---|---|---|
| native | contact_event | [0, 501, 11, 0] |
| pair | contact_event | [0, 447, 40, 25] |
| nested512 | contact_event | [0, 498, 7, 7] |
| native | slip_event | [0, 112, 3, 397] |
| pair | slip_event | [0, 36, 118, 358] |
| nested512 | slip_event | [0, 165, 33, 314] |

## 6. Model mismatch와 decoder/selection decomposition

같은 shifted data에서 nominal pipeline loss minus correct-law pipeline loss를 계산했다. Positive difference는 nominal pipeline이 더 나쁘다는 뜻이다. Correct-law는 privileged diagnostic이다. Nominal choices를 correct decoder에 적용하여 decoder effect와 selection effect를 나누었고, 두 항의 합이 total과 일치하는지 검증했다. Finite-bank error가 남으므로 exact continuous-posterior의 causal decomposition으로 확대 해석하지 않는다.


| native | Total [95% CI] | Decoder [95% CI] | Selection under correct decoder [95% CI] |
|---|---|---|---|
| smooth_velocity | +0.003389 [+0.002191, +0.004704] | +0.003055 [+0.001990, +0.004256] | +0.000333 [-0.000031, +0.000701] |
| contact_velocity | +0.001386 [+0.000157, +0.002642] | +0.001335 [+0.000109, +0.002591] | +0.000052 [-0.000010, +0.000135] |
| contact_event | -0.000234 [-0.001289, +0.000821] | -0.000258 [-0.001315, +0.000797] | +0.000025 [-0.000000, +0.000074] |
| slip_velocity | +0.000344 [+0.000147, +0.000543] | +0.000222 [+0.000046, +0.000405] | +0.000122 [+0.000026, +0.000231] |
| slip_event | +0.008410 [+0.002614, +0.014459] | +0.005573 [+0.001484, +0.010072] | +0.002837 [-0.001304, +0.007086] |

| pair | Total [95% CI] | Decoder [95% CI] | Selection under correct decoder [95% CI] |
|---|---|---|---|
| smooth_velocity | +0.003468 [+0.002157, +0.004961] | +0.003124 [+0.001933, +0.004493] | +0.000344 [-0.000035, +0.000746] |
| contact_velocity | +0.017718 [+0.008821, +0.027792] | +0.004135 [+0.000469, +0.008124] | +0.013583 [+0.005334, +0.022847] |
| contact_event | +0.006279 [+0.001632, +0.011270] | +0.001393 [-0.000960, +0.003786] | +0.004886 [+0.000729, +0.009366] |
| slip_velocity | +0.000152 [-0.000120, +0.000445] | +0.000261 [+0.000090, +0.000439] | -0.000109 [-0.000304, +0.000098] |
| slip_event | +0.009261 [+0.003674, +0.015224] | +0.005673 [+0.001196, +0.010432] | +0.003588 [+0.000430, +0.007114] |

| nested512 | Total [95% CI] | Decoder [95% CI] | Selection under correct decoder [95% CI] |
|---|---|---|---|
| smooth_velocity | +0.003292 [+0.002138, +0.004564] | +0.002852 [+0.001772, +0.004058] | +0.000440 [+0.000074, +0.000800] |
| contact_velocity | +0.001360 [+0.000133, +0.002623] | +0.001344 [+0.000111, +0.002603] | +0.000016 [-0.000120, +0.000131] |
| contact_event | -0.000261 [-0.001330, +0.000798] | -0.000234 [-0.001292, +0.000822] | -0.000027 [-0.000142, +0.000047] |
| slip_velocity | +0.000322 [+0.000120, +0.000528] | +0.000295 [+0.000120, +0.000474] | +0.000027 [-0.000075, +0.000137] |
| slip_event | +0.006506 [+0.000635, +0.012262] | +0.007385 [+0.003255, +0.011769] | -0.000879 [-0.004921, +0.003050] |

실제로 law를 바꾼 flexible-component worlds는 **257**, unchanged base-law controls는 **255**다. Changed-only means는 다음과 같다. 이 subgroup은 전체 population primary와 구분한다.

| Cell / method | smooth_velocity | contact_velocity | contact_event | slip_velocity | slip_event |
|---|---|---|---|---|---|
| matched_nominal/native | 0.016871 | 0.082506 | 0.052023 | 0.006376 | 0.028324 |
| matched_nominal/pair | 0.018109 | 0.111309 | 0.068745 | 0.006735 | 0.034937 |
| matched_nominal/nested512 | 0.016322 | 0.082648 | 0.052023 | 0.006338 | 0.027152 |
| matched_nominal/probe | 0.016762 | 0.222464 | 0.145256 | 0.006532 | 0.029445 |
| shifted_nominal/native | 0.011212 | 0.074134 | 0.051805 | 0.006681 | 0.018744 |
| shifted_nominal/pair | 0.011546 | 0.096837 | 0.064028 | 0.006928 | 0.027029 |
| shifted_nominal/nested512 | 0.011063 | 0.074118 | 0.051783 | 0.006654 | 0.025092 |
| shifted_nominal/probe | 0.011443 | 0.187844 | 0.130694 | 0.006831 | 0.023018 |
| shifted_correct/native | 0.008623 | 0.072368 | 0.051872 | 0.006135 | 0.019292 |
| shifted_correct/pair | 0.008837 | 0.078804 | 0.056522 | 0.006518 | 0.019846 |
| shifted_correct/nested512 | 0.008612 | 0.072340 | 0.051859 | 0.006101 | 0.020264 |
| shifted_correct/probe | 0.009288 | 0.175939 | 0.123832 | 0.006367 | 0.017649 |

## 7. LMMSE blind spot과 numerical audit

LMMSE는 nonlinear dependence 전체가 아니라 covariance를 통한 linear prediction gain을 계산한다. Symmetric hypotheses `h∈{−2,−0.5,0.5,2}`와 target `1{|h|<1}`의 abstract mechanism check에서는 covariance가0이면서 observation이 target을 거의 완전히 구별했다. No-observation Brier0.25, Gaussian noise SD0.1의 고정 finite samples에서 measured Brier 약1.66e−34였다. 이는 abstract illustration이며 physics population 성능을 대신하지 않는다.

Secondary subgroups는 public factual posterior만으로 정했다. Ambiguous는 event probability `[.2,.8]`, low-linear-gain은 event variance>.05이면서 maximum LMMSE gain/variance<.2다. Hidden targets로 유리한 subset을 고르지 않았다. 작은 subgroup의 descriptive CI는 primary evidence와 구분한다.


| matched_nominal | n | Pair − native [95% CI] | Nested512 − native [95% CI] |
|---|---|---|---|
| contact_event/ambiguous | 370 | +0.018193 [+0.011948, +0.025256] | +0.000000 [+0.000000, +0.000000] |
| contact_event/low_linear_gain | 5 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |
| slip_event/ambiguous | 134 | +0.028715 [+0.009608, +0.048804] | -0.011139 [-0.025026, +0.003357] |
| slip_event/low_linear_gain | 205 | +0.009819 [-0.003545, +0.023762] | -0.010651 [-0.022597, +0.001009] |

| shifted_nominal | n | Pair − native [95% CI] | Nested512 − native [95% CI] |
|---|---|---|---|
| contact_event/ambiguous | 368 | +0.015236 [+0.008558, +0.022540] | -0.000015 [-0.000046, +0.000000] |
| contact_event/low_linear_gain | 5 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |
| slip_event/ambiguous | 120 | +0.029732 [+0.011186, +0.048970] | -0.004245 [-0.019818, +0.011268] |
| slip_event/low_linear_gain | 185 | +0.017496 [+0.002387, +0.032345] | -0.000874 [-0.013978, +0.012340] |

### Integration sensitivity: first32 worlds

| matched_nominal | Contact event: agreement / loss change | Stick event: agreement / loss change |
|---|---|---|
| pair512 | 96.88% / +0.001705 | 90.62% / -0.007415 |
| nested2048 | 100.00% / +0.000000 | 84.38% / -0.000024 |
| nested2048_b | 100.00% / +0.000000 | 75.00% / +0.001297 |
| qmc64_nested512 | 100.00% / -0.000635 | 68.75% / -0.001135 |

| shifted_nominal | Contact event: agreement / loss change | Stick event: agreement / loss change |
|---|---|---|
| pair512 | 96.88% / +0.001705 | 87.50% / -0.007448 |
| nested2048 | 100.00% / +0.000000 | 81.25% / -0.000001 |
| nested2048_b | 100.00% / +0.000000 | 75.00% / -0.000001 |
| qmc64_nested512 | 100.00% / -0.001122 | 71.88% / -0.001343 |

| shifted_correct | Contact event: agreement / loss change | Stick event: agreement / loss change |
|---|---|---|
| pair512 | 96.88% / +0.000009 | 96.88% / -0.000962 |
| nested2048 | 100.00% / +0.000000 | 78.12% / -0.000967 |
| nested2048_b | 100.00% / +0.000000 | 75.00% / +0.000002 |

Pair K512는 main pairK256과, nested variants는 main nested512와 비교했다. QMC64 nested는 bank size와 scramble도 바뀌므로 pure outer-sample audit와 다르다. First32 worlds는 전체512에 대한 uniform stability 보증이 아니다. Primary choices는 audit를 보고 변경하지 않았다. Independent-seed nested2048 내부 risk에 대한 regret도 `results/summary.json`에 저장했으며 실제 held-out loss와 구분한다.

### Posterior ESS와 common-action decoder audit

| Cell | Factual ESS median | Factual ESS p05 | Pair eligible fraction | Pair fallback counts (five tasks) |
|---|---|---|---|---|
| matched_nominal | 116.905791 | 41.166811 | 0.773026 | [0, 8, 15, 0, 137] |
| shifted_nominal | 116.388685 | 43.557425 | 0.775929 | [0, 8, 16, 0, 151] |
| shifted_correct | 124.639965 | 43.505473 | 0.776921 | [0, 8, 15, 0, 161] |

| Cell / task | QMC16 pair-selected ESS median | QMC64 same-action ESS median | QMC16 fraction ESS<20 | QMC16 loss − QMC64 decoder loss [95% CI] |
|---|---|---|---|---|
| matched_nominal/contact_event | 26.467484 | 102.174200 | 31.09% | +0.002129 [-0.000823, +0.005168] |
| matched_nominal/slip_event | 20.659254 | 76.689790 | 47.94% | +0.002663 [-0.000459, +0.005838] |
| shifted_nominal/contact_event | 27.005930 | 104.988872 | 29.50% | +0.002759 [-0.000377, +0.005968] |
| shifted_nominal/slip_event | 22.507360 | 86.771731 | 42.36% | +0.003436 [+0.000491, +0.006479] |

## 8. Compute 비용

RTX PRO6000 Blackwell Server Edition, float64, batch1에서 측정했다. CPU는 같은 node의2threads다. Fixed20 matched worlds×5tasks×3repetitions, method당300 measurements이며 warmup 후 factual fit, query-specific target generation, acquisition, selected-observation update/prediction을 포함했다. GPU synchronization을 적용했다. Selector에는 unselected actual observations를 전달하지 않고, 선택 후 해당 observation 하나로 update한다. Online action/prediction은 stored evaluation과 대조했다.


### 수학적으로 동일한 optimized runtime: 최종 compute 비교

최초 timing에서 skinny matrix contraction에 따른 큰 latency 차이가 관찰되어 사후 implementation audit를 추가했다. LMMSE의 observation moments를 cache하고 nested prediction을 output별 matrix-vector products로 계산했다. Direct-risk decision에 사용하지 않는 posterior-variance 계산도 제거했다. Primary policies/settings/losses는 바꾸지 않았다. 동일100 cases×3에서 actions와 predictions, 비교 가능한 candidate scores가 일치하는지 검증했다. 아래 optimized 비용을 최종 trade-off 평가에 사용한다.

| Method | GPU median ms | CPU median ms | GPU contact-event ms | GPU stick-event ms |
|---|---|---|---|---|
| native | 3.176 | 2.028 | 2.612 | 3.131 |
| pair | 3.274 | 2.298 | 2.791 | 3.247 |
| nested128 | 4.681 | 31.109 | 3.801 | 4.736 |
| nested512 | 9.685 | 94.472 | 8.708 | 9.618 |
| qmc64_native | 4.484 | 5.823 | 3.914 | 4.418 |
| qmc64_pair | 4.440 | 4.089 | 3.960 | 4.832 |
| query | 1.582 | 1.065 | 1.095 | 1.598 |
| probe | 1.501 | 0.989 | 1.033 | 1.517 |

| Method | Optimized bank/moment cache bytes | Observed temporary bytes | GPU warm setup ms |
|---|---|---|---|
| native | 24117248 | 1572864 | 125.618 |
| pair | 8650752 | 3629056 | 4.833 |
| nested128 | 17301504 | 51261952 | 13.128 |
| nested512 | 17301504 | 51274240 | 8.945 |
| qmc64_native | 96468992 | 6291456 | 10.065 |
| qmc64_pair | 34603008 | 6291456 | 8.227 |
| query | 8650752 | 1572864 | 5.120 |
| probe | 8650752 | 1572864 | 3.856 |

Optimized aggregated median 기준 nested512/pair는 GPU **2.96배**, CPU **41.12배**다. Pair/native GPU latency ratio는 **1.03**다. Fixed actions와 더 저렴한 LMMSE도 함께 비교해야 한다.

![Optimized loss and latency](results/optimized_tradeoff.png)

### 최초 runtime 측정값: implementation sensitivity 기록

다음 수치는 원래 code의 결과로 보존한다. 이를 optimized baseline 대비 algorithmic speedup으로 사용하지 않는다.

| Method | GPU median ms | GPU p90 ms | CPU median ms | GPU contact-event ms | GPU stick-event ms |
|---|---|---|---|---|---|
| native | 12.180 | 12.836 | 2.241 | 7.591 | 8.168 |
| pair | 3.535 | 4.144 | 2.316 | 3.214 | 3.855 |
| nested128 | 30.252 | 31.263 | 35.306 | 4.581 | 5.544 |
| nested512 | 104.985 | 106.687 | 83.225 | 10.345 | 11.329 |
| qmc64_native | 36.283 | 37.188 | 6.500 | 20.102 | 20.596 |
| qmc64_pair | 4.343 | 5.159 | 4.291 | 3.845 | 4.746 |
| query | 1.890 | 2.185 | 1.062 | 1.327 | 1.947 |
| probe | 1.845 | 2.177 | 1.109 | 1.277 | 1.922 |

GPU aggregated median 기준 nested512/pair latency ratio는 **29.70**다. Pair/native latency ratio는 **0.29**다.

CPU aggregated median 기준 nested512/pair latency ratio는 **35.93**다. Pair/native latency ratio는 **1.03**다.

| Method | Belief weights bytes | Bank cache bytes | Observed GPU temporary peak bytes | Warm setup ms |
|---|---|---|---|---|
| native | 131072 | 8650752 | 2492928 | 4.718 |
| pair | 131072 | 8650752 | 3629056 | 5.103 |
| nested128 | 262144 | 17301504 | 51266048 | 10.590 |
| nested512 | 262144 | 17301504 | 51287552 | 18.077 |
| qmc64_native | 524288 | 34603008 | 9964032 | 5.696 |
| qmc64_pair | 524288 | 34603008 | 6291456 | 7.272 |
| query | 131072 | 8650752 | 1572864 | 4.922 |
| probe | 131072 | 8650752 | 1572864 | 3.946 |

State bytes는 per-world posterior weight arrays만 센 numerical payload다. Bank cache는 theta/observation/whitened arrays이며 Python objects, CUDA context, allocator reserve, Sobol plan 등 전체 process memory가 아니다. Temporary peak는20 smooth-velocity cases에서 관찰한 추가 CUDA allocated memory다. 모든 tasks/inputs의 global peak로 부르지 않는다. Setup은 warm process bank construction이며 cold startup 전체가 아니다. Physical sensing/reset 비용은 측정하지 않았다. 서로 다른 sample budgets를 같은 compute라고 가정하지 않고 실제 latency를 제시했다.

최초 optimized audit에서는 nested128의 no-observation diagnostic score까지 동일하다고 검사하여 assertion이 발생했다. Main nested128은 conditional128 samples와 diagnostic512 samples를 함께 저장하지만 online128은 모두128 samples를 쓴다. 선택 후보는 actions1..3뿐이므로 그 실제 candidate scores를 비교하도록 audit만 수정했다. Main policies, selected actions, predictions 또는 primary results를 변경한 것은 아니다. 실패 로그도 보존했다.

## 9. 결과 해석과 논문 기여 가능성

### Transition이 존재한다고 adaptive selection이 필요한 것은 아니다

Matched contact event에서 LMMSE는512 worlds 중501개에서 query_view를 선택했고 pair는418개만 선택했다. Pair는34개에서 factual_view,60개에서 probe를 선택했다. 이 task에서는 query 근처 geometry를 직접 보는 fixed rule이 매우 강했다. Always-query의 loss는 LMMSE와 수치상 거의 같았다. Event probability가 `[.2,.8]`인 ambiguous370 worlds에서도 LMMSE와 nested512의 choices로 얻은 loss는 동일했고, pair는 평균적으로 더 나빴다.

따라서 contact boundary가 불연속이라는 사실과 LMMSE의 action ranking이 실패한다는 것은 다른 주장이다. 현재 sensor menu에서는 query_view가 필요한 정보를 직접 제공한다. Ambiguity를 넣었다는 이유만으로 더 복잡한 acquisition의 필요성이 생기지 않았다.

Stick event에서는 더 복잡한 pattern이 나타났다. Low-linear-gain205 worlds에서 nested512−LMMSE mean difference는−0.010651이었지만 secondary95% CI는[−0.022597,+0.001009]였다. Nonlinear acquisition이 도움이 될 가능성은 남아 있으나, 이 subgroup만으로 확정할 수 없다. Pair는 이 subset에서도 개선을 보이지 않았다. Ambiguous134 worlds에서는 pair−LMMSE difference가+0.028715였고 secondary CI도 악화를 지지했다.

### Model mismatch와 finite integration이 설명하는 부분

동일한 shifted data에서 pair의 nominal−correct-law smooth-velocity difference는+0.003468이었다. 그중 decoder term이+0.003124, selection term이+0.000344였다. 반대로 contact-event total difference +0.006279 중 selection term은+0.004886, decoder term은+0.001393이었다. Correct-law 정보가 필요한 위치가 task마다 다르다. 이는 선택한 decomposition path의 결과이며 원인이 완전히 독립적이라는 뜻은 아니다.

QMC16의 actual-observation 이후 ESS도 충분히 큰 것은 아니었다. Matched pair choices에서 contact-event median ESS는26.47, stick-event는20.66이었고 각각 약31.1%,47.9%의 world/noise combinations에서 ESS<20이었다. QMC64를 같은 actions의 decoder로 쓰면 median ESS가102.17/76.69로 올라갔다. 따라서 finite integration error를 무시할 수 없다.

그렇지만 QMC64에서도 pair의 contact/stick event losses는0.057803/0.072897로 native의0.051285/0.066871보다 평균적으로 높았다. Numerical resolution을 높이는 것만으로 현재 selector의 우위가 나타난 것은 아니다. 이 큰-bank 비교는 independent scramble도 바뀌므로 pure sample-count effect와 구분한다.

Ranking audit에서 contact-event nested choices는 first32 worlds에서 안정적이었다. Stick-event는 nested512와 다른-seed nested2048의 action agreement가75%, QMC64 nested512와는68.75%였다. 이 차이를 숨기지 않는다. 다만 같은 audit의 independent-seed reference 내부 평균 regret는 nested512가0.000118, pair가0.005667이었다. Action disagreement 비율만으로 decision impact를 판단해서도 안 된다. Reference 자체가 finite approximation이므로 true Bayes regret라고 부르지 않는다.

### Pair score의 구조적 약점: decision-region mixtures의 overlap

현재 score는 query predictions가 다른 **개별 hypotheses pairs**가 observation에서 멀리 떨어지는지 계산한다. 그러나 개별 pairs가 멀어도, 각 decision region의 전체 observation distributions가 동일할 수 있다. 이 경우 실제로는 query label에 대한 information이 없다.

이를 확인하기 위해 primary 결과를 본 뒤 별도의 **abstract explanatory counterexample**을 구성했다. Physics test를 추가로 튜닝한 것이 아니다. Factual likelihood가 모두 같은 네 hypotheses에 prior1/4씩, binary query labels `[0,0,1,1]`을 둔다. Noise SD는1이다.

- Uninformative action의 observation means: `[-4,+4,-4,+4]`. 두 labels의 conditional mixture가 정확히 동일하므로 observation 이후에도 label probability는1/2이고 Bayes Brier loss는0.25다.
- Informative action의 means: `[-0.5,-0.5,+0.5,+0.5]`. 두 labels의 distributions가 달라 observation이 query prediction을 개선한다.

그런데 구현한 pair score는 첫 action에 약0.249916, 두 번째에 약0.058752를 주어 uninformative action을 더 높게 평가한다. Equal factual likelihood라 cutoffs를 바꾸는 것으로 이 반례가 사라지지 않는다. Brier-scaled target에서 첫 action score는 `¼(1−exp(−8))`, 두 번째는 `½(1−exp(−0.125))`로 직접 계산된다. Expected losses는 서버에서 Gauss–Hermite80 quadrature로 확인했다.

이 반례는 현재 pair objective가 exact Bayes-risk reduction이 아님을 구체적으로 보여 준다. Physics worlds의 모든 오류가 이 mechanism으로 발생했다고 입증한 것은 아니다. 하지만 score 자체가 decision-region overlap을 처리하지 못할 수 있다는 구조적 limitation은 있다. Contact/stick test에서 다른 observations로 자주 전환하면서 loss가 높아진 현상과 함께 검토할 근거가 된다.

Factual-fit band ablation의 개선은 두 matched event tasks에서 secondary CI가0을 포함했다. Query weighting은 contact-event에서 agnostic discrimination보다 분명히 나았지만, 그것만으로 LMMSE/fixed rule 대비 contribution이 성립하지는 않는다. Stick-event pair fallback은137/512였으므로, zero committee score에서는 LMMSE로 되돌아가는 경우도 적지 않았다. Fallback이 모두 true certainty라는 뜻은 아니며 finite committee가 rare outcomes를 놓칠 가능성도 남는다.

### 선행연구와 구분해야 할 주장

EC²는 hypotheses와 equivalence classes를 기반으로 information gathering을 다룬다. “서로 다른 decision을 만드는 hypotheses를 구별한다”는 발상은 기존 연구 맥락 안에 있다. 이번 Gaussian overlap pair score는 EC²의 objective를 그대로 구현한 것이 아니며, 그 논문의 adaptive-submodularity guarantees를 가져올 수 없다. [Golovin, Krause, Ray, 2010 원문](https://proceedings.neurips.cc/paper/2010/file/1e6e0a04d20f50967c64dac2d639a577-Paper.pdf)

GoBOED는 amortized posterior와 differentiable convex decision layer를 결합해 goal-driven design을 수행한다. 현재 discrete sensor menu의 contact-event forecasting과는 다르지만, downstream objective에 맞춘다는 점 자체는 novelty가 아니다. 이 연구의 learned method를 직접 재현하여 비교한 것은 아니다. [Go, Qian, Yoon, 2026 preprint](https://arxiv.org/html/2605.26093v1)

GBOED는 Gibbs inference와 Gibbs EIG를 통해 inference/design 양쪽의 robustness를 다루며, outliers와 noise misspecification을 평가한다. Maximin robust Bayesian experimental design은 adversarial perturbation과 robust belief update를 함께 정식화한다. 이번 contact-law mismatch setting과 동일하지 않으며 두 방법을 benchmark로 재현하지 않았지만, model uncertainty에 대응하는 observation selection이라는 주장에도 선행연구와의 구체적 차이가 필요하다. [Barlas, Sloman, Kaski, 2025 preprint](https://arxiv.org/html/2511.07671v1), [Abdulsamad et al., 2026 preprint](https://arxiv.org/html/2603.14094v1)

### 내가 권하는 연구 판단

현재 score의 cutoffs나 K를 더 탐색하여 positive result를 얻는 방향은 멈추는 편이 좋다. Matched control에서의 악화, optimized compute comparison, explicit mixture-aliasing counterexample를 함께 보면 현재 objective를 주된 novelty로 삼을 근거가 약하다. Negative evidence를 정리한 diagnostic 결과로 보존할 가치는 있지만, 이것만으로 major-conference method contribution이 확보됐다고 볼 수 없다.

계속한다면 **decision-region별 conditional predictive mixture를 보존하는 compact acquisition**이 더 타당한 가설이다. 개별 pair distances 대신 `p(y|decision region, factual evidence)`의 overlap과 actual expected decision loss를 근사해야 한다. 적어도 위 counterexample에서는 uninformative action의 utility를0으로 판정해야 한다. 이는 새로운 구현·성능이 확인된 결과가 아니라 후속 설계 기준이다.

그 다음의 go/no-go 조건도 명확해야 한다. 먼저 independent physics worlds에서 optimized LMMSE와 task-specific fixed rule이 실제로 실패하는 regime를 확보하고, conditional-mixture approximation이 cross-fitted nested reference의 decision quality를 더 적은 compute로 유지하는지 검증해야 한다. 적절한 non-inferiority margin은 application loss tolerance에 따라 실험 전에 정한다. Well-specified exact Bayes-optimal selector를 accuracy로 이겨야만 approximation이 가치 있는 것은 아니다. 그러나 이번 pair는 cheap baselines에도 밀렸으므로 그 일반론으로 현재 결과를 성공으로 바꾸지는 않는다.

이번 실험은 raw RGB, realistic trajectory simulation, real robot intervention을 검증한 것이 아니다. Geometry profiles, finite-reach gate, instantaneous contact, equal action costs를 사용한 analytical simulation이다. 실제 논문으로 확대하려면 mixture mechanism의 physical relevance와 realistic sensing/action costs를 별도로 입증해야 한다.


## 10. 실행과 재현성

Server project: `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/transition_control_v15`. 기존 allocation2680491, node56에서 실행했다. 2026-11-12의 세 차례 interruption은 별도 Blender process를 감지한 guard가 자신의 worker만 종료한 것이다. 2026-11-13 GPU가 비워졌다는 사용자 지시 후 process/memory를 확인하고 재개했다. 기존 jobs/allocation은 변경하지 않았다.

`PROTOCOL.md`는 primary settings, `DIAGNOSTICS_SPEC.md`는 secondary subgroup 기준이다. `model.py`와 `acquisition.py`는 candidate forward/acquisition, `generate.py`는 private generator, `run_policy.py`는 public-only policy selection, `evaluate.py`는 policy freeze 이후 actual outcomes 평가를 담당한다. `benchmark.py`는 online timing, `analyze.py`와 `diagnostics.py`는 paired statistics 및 numerical diagnostics다.

Actual outputs는 `results/policies`, `results/audits`, `results/evaluations`, `results/world_losses.npz`, `results/summary.json`, `results/secondary_diagnostics.json`, `results/timing_gpu.json`, `results/timing_cpu.json`에 있다. `source_policy.json`, `policy_freeze.json`, worker logs와 exit records가 실행 evidence다. `INTEGRITY.json`과 `MANIFEST.json`에 source/policy/result 및 이전 v14 artifacts 검증을 남겼다.

실제 simulation/inference/statistics는 모두 allocated compute node에서 수행했다. Local mirror는 준비·검토·전달용이다. 재실행할 때는 sealed outputs를 덮어쓰지 않고 새 directory에 code/data를 복사한다. `run_resume.py` → `benchmark.py --cpu` → `analyze.py` → `diagnostics.py` → `write_report.py` → `seal.py` 순서이며, GPU와 CPU timing을 다른 workload와 겹치지 않게 실행한다.
