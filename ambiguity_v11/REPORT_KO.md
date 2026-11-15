# Geometry–contact ambiguity v11: 동일 visibility/query에서 observation의 가치

작성일: 2026-11-12 KST. 서버에서 실제 구현·실행한 controlled diagnostic이다. 설명은 한글, 전문용어는 English로 작성했다.

## 핵심 판단

**Visibility와 query를 고정해도 factual evidence에 따라 유리한 observation이 달라지는 조건을 만들었다. 하지만 strong rule과 mixture plug-in을 넘어서는 full joint posterior의 추가 이득은 이번 sample에서 확정하지 못했다.**

Main test의 joint selector는 query-view155회, probe101회를 선택했다. 모든 world에서 visibility/query/sensor precision이 같으므로 v10의 visibility + query lookup만으로 이 선택을 설명할 수 없다. 다만 finite outer sampling으로 정한 선택이므로, 실제 conditional risk의 차이가 안정적인지는 아래 independent ranking audit과 함께 봐야 한다.

추가 observation은 유효했다. Joint MSE는 **0.005762328**, no-observation은 **0.011540406**이고 primary99% CI에서도 개선이 확인됐다. Updated hard plug-in은 **0.006350650**으로, joint가 약9.26% 낮았으며 이 차이도 primary99% CI가0을 포함하지 않았다.

반면 **updated mixture plug-in은0.005771137**로 joint에 매우 가까웠다. 두 selector의 선택도256/256 동일했다. Joint−mixture의99% CI는0을 포함하므로 추가 이득을 확정할 수 없다. 이는 equivalence가 증명됐다는 뜻도 아니다.

Strong learned rule의 MSE는 **0.005981087**, LMMSE는 **0.005835918**이었다. Joint가 평균에서는 낮지만, learned rule 대비 primary99% CI와 LMMSE 대비 descriptive95% CI 모두0을 포함했다. **비싼 nested acquisition이 강한 간단한 selector보다 실제로 낫다는 주장은 지지되지 않았다.** Standard와 Cross-fitted SNIS의 actual MSE도 모두0.005741478로, joint selector보다 평균이 약간 낮았다. 이 미세한 차이를 SNIS의 우위로 해석해서도 안 된다.

Factorized factual belief는 **0.006585476**으로 joint보다 평균 loss가 컸지만, 사전에 정한 primary99% CI는0을 포함했다. 따라서 “geometry–contact correlation을 제거하면 반드시 성능이 유의하게 나빠진다”는 결론은 보류한다. Correlation의 존재와 그것을 보존할 때의 statistically resolved decision benefit은 서로 다른 주장이다.

마지막으로, mixture plug-in도 joint inference로 mode weights와 conditional means를 구한다. 이 결과는 **joint inference 전체가 불필요하다는 증거가 아니라**, 현재 조건에서는 각 mode 안의 continuous distribution을 prediction 단계에서 끝까지 적분하는 이득이 작다는 증거다. Compact mixture만 유지하는 별도 inference algorithm의 성능이나 compute 절감은 아직 측정하지 않았다.

## Primary test 결과

Independent test **256 worlds × 32 additional-observation noise repeats**를 사용했다. 모든 world의 visibility와 query는 같다. Joint의 no-observation 대비 평균 MSE 감소율은 **50.07%**다. 이 비율은 pilot에서 의도적으로 만든 ambiguity distribution에 대한 값이다.

| 방법 | Decision MSE (m²/s²) |
| --- | --- |
| 추가 observation 없음 | 0.011540406 |
| Best fixed rule + joint decoder | 0.006408630 |
| Learned rule + joint decoder | 0.005981087 |
| LMMSE + joint decoder | 0.005835918 |
| Updated hard plug-in | 0.006350650 |
| Updated mixture plug-in | 0.005771137 |
| Factorized factual belief | 0.006585476 |
| Standard SNIS + joint decoder | 0.005741478 |
| Cross-fitted SNIS + joint decoder | 0.005741478 |
| Joint selector + joint decoder | 0.005762328 |

![Decision loss](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/ambiguity_v11/results/analysis/decision_loss.png)

Figure의 CI는 방법별 descriptive95% CI다. Primary 판단은 아래 paired comparisons를 사용한다.

### Primary paired comparisons

차이는 Joint − comparator다. 음수이면 Joint가 유리하다. World-cluster bootstrap30,000회와5 comparisons에 대한 Bonferroni99% CI를 사용했다.

| Comparator | 평균 차이 | 99% CI |
| --- | --- | --- |
| 추가 observation 없음 | -0.005778077 | [-0.007864577, -0.003906174] |
| Learned rule + joint decoder | -0.000218758 | [-0.000925061, +0.000262484] |
| Updated hard plug-in | -0.000588321 | [-0.001189305, -0.000096986] |
| Updated mixture plug-in | -0.000008809 | [-0.000043465, +0.000024541] |
| Factorized factual belief | -0.000823147 | [-0.001925486, +0.000070352] |

### Observation 선택과 decoder를 분리한 비교

| 조합 | Decision MSE |
| --- | --- |
| Joint choices + joint decoder | 0.005762328 |
| Joint choices + hard plug-in | 0.006532175 |
| Joint choices + mixture plug-in | 0.005771137 |
| Joint choices + factorized decoder | 0.006613002 |
| Hard plug-in choices + common joint decoder | 0.005681959 |
| Factorized choices + common joint decoder | 0.005739001 |
| Learned rule choices + mixture plug-in | 0.005988075 |

이 표의 same-action comparisons는 representation/decoding 효과를, common-decoder comparisons는 acquisition 효과를 분리한다. 모든 수치의 paired descriptive95% CI는 `results/analysis/summary.json`에 있다.

## 같은 visibility/query에서 선택이 달라졌는가

| Policy | none | query-view | factual-view | probe | Joint와 일치율 |
| --- | --- | --- | --- | --- | --- |
| Joint selector + joint decoder | 0 | 155 | 0 | 101 | 100.00% |
| Updated hard plug-in | 0 | 138 | 0 | 118 | 93.36% |
| Updated mixture plug-in | 0 | 155 | 0 | 101 | 100.00% |
| Factorized factual belief | 0 | 167 | 0 | 89 | 95.31% |
| Standard SNIS + joint decoder | 0 | 154 | 0 | 102 | 97.27% |
| Cross-fitted SNIS + joint decoder | 0 | 154 | 0 | 102 | 97.27% |
| 추가 observation 없음 | 256 | 0 | 0 | 0 | 0.00% |
| rule_rbf | 0 | 142 | 0 | 114 | 85.55% |
| rule_tree | 0 | 155 | 0 | 101 | 85.94% |
| Best fixed rule + joint decoder | 0 | 0 | 0 | 256 | 39.45% |
| Learned rule + joint decoder | 0 | 155 | 0 | 101 | 85.94% |
| LMMSE + joint decoder | 0 | 181 | 0 | 75 | 89.06% |

![Action context](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/ambiguity_v11/results/analysis/action_context.png)

이 figure의 모든 점은 같은 visibility/query를 갖는다. Horizontal axis는 factual profile에서 구한 slope estimate이고 vertical axis는 frictional contact의 posterior probability다. Ground-truth contact class를 selector에 제공하지 않았다.

Factual posterior의 geometry–contact mutual information은 평균 **0.14397 nats**, median **0.08984 nats**였다. |Corr(s,e)| 평균은 **0.22334**다. MI는 geometry=(g,s,a), contact=(c,e) 묶음 사이의 dependence를 측정한다. Correlation의 존재 자체가 decision improvement를 증명하는 것은 아니다.

### Ranking audit

| 항목 | 측정값 |
| --- | --- |
| worlds | 32 |
| H_per_repeat | 1024 |
| order | 61 |
| main_agreement | 1.0 |
| repeat_agreement | 1.0 |
| mean_reference_regret | 0.0 |
| max_reference_regret | 0.0 |
| median_reference_margin | 0.0031647700152314714 |

Order61, independent outer samples1024개씩 두 번으로 first32 test worlds를 다시 평가했다. Main policies를 바꾸지 않았다. 이 reference도 finite outer estimate이며 exact optimum은 아니다. Reference regret는 평균 reference risks 중 선택 action과 가장 작은 값의 차이다.

### 두 가지 구체적인 factual evidence 사례

| 사례 | 선택 | Factual p(frictional) | Query-view expected risk | Probe expected risk |
| --- | --- | --- | --- | --- |
| World 000 | query_view | 0.000130 | 0.004795 | 0.007918 |
| World 023 | probe | 0.555981 | 0.031484 | 0.006452 |

World000에서는 contact class가 거의 결정되어 geometry measurement가 더 유리했다. World023에서는 contact class가 여전히 모호해 probe가 더 유리했다. Visibility와 query는 동일하다. 두 사례는 first32 worlds 안에서 각 선택의 predicted risk margin이 가장 큰 사례이며, actual test loss나 true contact class로 고르지 않았다. 대표 빈도나 평균 효과를 추정하는 표는 아니다.

![Posterior examples](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/ambiguity_v11/results/analysis/posterior_examples.png)

Figure는 geometry family와 amplitude를 marginalize한 slope–restitution quadrature posterior mass다. 각 row는 하나의 world, 각 column은 contact class다.

## Candidate family 밖의 stress test

| 방법 | Misspecification MSE |
| --- | --- |
| 추가 observation 없음 | 0.014756974 |
| Learned rule + joint decoder | 0.010803425 |
| LMMSE + joint decoder | 0.010660299 |
| Updated hard plug-in | 0.014225721 |
| Updated mixture plug-in | 0.011009937 |
| Factorized factual belief | 0.012494595 |
| Cross-fitted SNIS + joint decoder | 0.010863829 |
| Joint selector + joint decoder | 0.010975069 |

별도128 worlds 결과다. In-support test와 합쳐 평균내지 않았다. Third boundary family와 continuous friction을 동시에 바꾼 exploratory stress test이며 real-world validation은 아니다.

## 실험 질문과 v10에서 바꾼 점

v10에서는 visibility와 query를 알면 거의 같은 observation을 선택할 수 있었다. v11은 **모든 world의 visibility, query position, incoming query velocity, sensor noise를 동일하게 고정**했다. 바뀌는 것은 latent scene과 그 scene에서 얻은 factual measurement다.

검증 질문을 세 단계로 나눴다.

1. 동일 visibility/query에서도 factual evidence에 따라 유리한 observation이 달라지는가?
2. Geometry와 contact uncertainty를 함께 유지하면, 하나의 model을 고르는 updated plug-in보다 실제 prediction이 좋아지는가?
3. 그 이득에 full joint posterior와 nested acquisition까지 필요한가, 아니면 mixture plug-in·learned rule·moment-based acquisition으로도 충분한가?

**Unknown boundary는 정해진 두 candidate families 중 어떤 family와 parameter가 맞는지 모르는 상황**이다. 임의의 unknown shape를 새로 발견하는 open-world reconstruction은 아니다. 별도 misspecification stress test에서 candidate family 밖의 geometry와 friction을 넣었지만, model class를 확장하거나 학습하지 않았다.

## Geometry와 contact model

2D point mass가 static boundary에 한 번 충돌한다. Boundary는

\[
x(y)=s y+a f_g(y),\qquad L=25\text{ mm},\quad z=y/L,
\]

\[
f_0(y)=Lz^2/2,\qquad f_1(y)=L(z^2/2-0.35z^4)
\]

로 정의했다. Continuous parameters는 \(s\sim U[-0.4,0.4]\), \(a\sim U[-0.10,0.10]\), \(e\sim U[0.35,0.9]\)다. Geometry family \(g\)와 contact class \(c\)는 각각 prior probability 0.5이며 모든 prior factors는 독립이다.

Factual contact인 \(y=0\)에서 두 families의 slope는 모두 \(s\)다. Query contact인 \(y=L\)에서는 각각 \(s+a\), \(s-0.4a\)가 된다. 따라서 factual contact response만으로는 두 families를 구분할 수 없지만 query response는 달라질 수 있다. Factual profile에서도 family 간 차이가 작도록 central region만 관찰한다.

Contact candidates는 **frictionless**와 **Coulomb friction(\(\mu=0.22\))**이다. Surface normal/tangent로 incoming velocity를 분해하여 normal impulse에 restitution을 적용하고, tangential impulse를 \(\min(|v_t|,\mu(1+e)v_n)\)으로 제한했다. Sticking/sliding transition을 포함하지만, rotation, finite-radius body, gravity, deformation, multi-contact는 포함하지 않는다. 측정 대상은 지정한 contact 직후의 velocity이며 긴 trajectory나 재충돌을 simulation한 결과가 아니다.

## 모든 world에 같은 factual sensor와 query 제공

| 항목 | 고정 설정 |
| --- | --- |
| Factual visibility | y=−10..10 mm의9 boundary rows |
| Factual geometry noise | Row별 독립 Gaussian, standard deviation1.2 mm |
| Factual impact | y=0, incoming velocity=(1,0.35) m/s |
| Factual velocity noise | Component별0.07 m/s |
| Query | y=25 mm, incoming velocity=(1,0.9) m/s |
| Prediction target | Query contact 직후2D velocity |

Geometry input은 calibrated boundary profile이다. Row별 geometry 정보를 유지하지만 **이번 v11에서 raw RGB image를 새로 처리한 것은 아니다.** v10의 image-to-boundary 경로 전체를 다시 검증했다고 해석하면 안 된다. Camera calibration, segmentation failure, photometric noise는 별도 문제로 남는다.

### 추가 observation 후보

| Action | 측정 | Noise standard deviation |
| --- | --- | --- |
| none | 추가 측정 없음 | — |
| query-view | y=19..31 mm의7 boundary rows | 1 mm |
| factual-view | y=−8..8 mm의7 boundary rows | 0.35 mm |
| probe | Reset한 y=0 contact에 incoming velocity=(1,−0.7) 적용 후 velocity | 0.05 m/s |

Probe는 query outcome을 직접 보여주지 않는다. 한 episode에서 action 하나만 수행하며, primary comparison은 action 간 동일 비용을 가정한다. 실제 camera 이동 비용과 physical reset 비용을 측정한 것은 아니다. “View” 역시 calibrated region measurement를 뜻하며 3D camera trajectory optimization을 구현한 것은 아니다.

## 왜 joint posterior가 생기는가

\(\theta=(g,s,a,c,e)\), factual evidence를 \(D=(B_f,v_f')\)라 하면

\[
p(\theta\mid D)\propto p(B_f\mid g,s,a)\,p(v_f'\mid s,c,e)\,p(\theta).
\]

Geometry와 contact가 prior에서는 독립이어도, factual velocity를 설명할 때 **slope, restitution, friction class가 서로 보상**할 수 있다. Geometry measurement로 slope를 더 정확히 알게 되면, 새로운 collision을 보지 않고도 어떤 contact explanation이 맞는지 달라질 수 있다. 이 경로가 factorized belief와 joint belief의 차이다.

Query target을 \(T(\theta)\)라 두고 \(\ell(\hat T,T)=\|\hat T-T\|^2/2\)를 사용했다. Joint decoder는 posterior predictive mean이다. Joint selector는 hypothetical observation 이후의 posterior variance를 평균하여 expected decision loss가 가장 작은 action을 고른다.

\[
a^*(D)\approx\arg\min_a E_{Y_a\mid D}\left[\tfrac12\operatorname{tr}\operatorname{Cov}(T\mid D,Y_a)\right].
\]

Outer expectation은 finite Monte Carlo로 추정한다. 따라서 report에서 선택된 action을 무조건 “수학적으로 증명된 최적 action”이라고 부르지 않고, independent outer repeats로 ranking 안정성을 별도로 확인한다.

## Baselines와 비교의 의미

| 방법 | Observation 선택 | 선택 후 prediction |
| --- | --- | --- |
| Joint | Expected posterior decision risk | Joint posterior predictive mean |
| Best fixed rule | Training 평균 risk가 가장 낮은 action | Joint decoder |
| Learned rule | Validation에서 선택한 RBF/CART/fixed rule | Joint decoder |
| LMMSE | Query–observation covariance 기반 expected linear-estimator gain | Joint decoder |
| Updated hard plug-in | 자신의 decoder에 대한 expected risk | MAP discrete family/contact + 해당 mode의 conditional parameter means |
| Updated mixture plug-in | 자신의 decoder에 대한 expected risk | 각 discrete mode의 point-parameter response를 posterior mode probability로 평균 |
| Factorized factual belief | Factorized belief에서 계산한 expected risk | Factorized factual belief에 새 likelihood를 곱해 갱신 |
| Standard / Cross-fitted SNIS | Independent posterior banks로 acquisition 추정 | 동일 joint decoder |

Hard plug-in도 **새 observation을 받은 뒤 다시 model과 geometry를 추정**한다. 처음 추정값을 고정하는 약한 baseline이 아니다. Acquisition 단계에는 full posterior를 제공해 이 decoder가 유리한 observation을 고를 수 있게 했다. 따라서 computationally cheap plug-in system의 구현이라고 주장하지 않는다.

Mixture plug-in은 discrete uncertainty를 유지하지만 continuous uncertainty를 forward response에서 적분하지 않는다. 각 mode의 conditional mean parameter를 한 번씩 넣는다. 이 baseline이 joint와 비슷하면 “모든 continuous correlation을 보존하는 full posterior가 필수”라는 주장보다 “너무 일찍 하나의 explanation으로 collapse하면 손해”라는 해석이 적절하다.

Factorized baseline은 factual posterior를

\[
q_D(g,s,a,c,e)=p_D(g,s,a)\,p_D(c,e)
\]

로 바꾼다. 두 marginal은 보존하고 두 묶음 간 dependence만 제거한다. 새 observation 이후에는 \(q_D\times\text{likelihood}\)를 정규화하므로 계속 mean-field projection을 하는 방식은 아니다.

Learned rule은 factual image mean이 놓이는 span(y,y²,y⁴)의 orthonormal projection3개와 factual velocity2개를 입력으로 받는다. Equal Gaussian row noise 아래에서는 candidate model에 관한 likelihood 정보를 잃지 않는 sufficient statistics다. RBF kernel ridge의16 configurations와 decision-focused CART의20 configurations를 validation expected risk로 비교한다. 이는 Bayes risk labels를 사용하는 강한 amortized baseline이지만, training set256개로 모든 가능한 learned policy를 대표하는 것은 아니다.

LMMSE는 nested outer sampling 없이

\[
\tfrac12\operatorname{tr}\{\operatorname{Cov}(T,Y_a)[\operatorname{Cov}(Y_a)+R_a]^{-1}\operatorname{Cov}(Y_a,T)\}
\]

를 계산한다. Joint posterior moments를 계산하는 비용은 필요하다. Nonlinear mixture에서는 이 값이 exact Bayes gain과 같지는 않다.

SNIS는 order41 quadrature로 근사한 factual posterior에서 총8192 samples를 뽑는다. Cross-fitting은4096개씩 독립 bank로 나누어 posterior-mean displacement의 cross-product를 사용한다. 이는 continuous exact posterior에서 sampling한 것이 아니며, self-normalized estimator의 bias를 모두 제거하지도 않는다.

SNIS의 current predictive mean도 quadrature reference를 공유한다. 따라서 이 비교는 factual inference부터 모두8192 particles만으로 해결하는 end-to-end algorithm의 비교가 아니라, 주로 hypothetical update/acquisition estimator를 비교한 것이다.

## Splits, policy freeze, uncertainty

최종 실험은 pilot8, training256, validation128, in-support test256, misspecification test128 worlds로 구성했다. Training/validation/test seeds는 분리했다. 각 test world에는 추가 observation noise32회를 생성하고 정책 간 같은 noise outcomes를 공유했다. 한 world의 factual measurement는 하나이며,32회는 추가 observation의 반복이다.

각 test policy는 public factual evidence만 보고 저장했다. Policy와 public files의 SHA256을 고정한 뒤 private target과 actual observations를 열었다. LMMSE는 별도 public-only code와 extra freeze를 사용했다. 결과가 나쁜 worlds를 제거하지 않았다.

Primary MSE는 noise repeats를 world 안에서 먼저 평균한 뒤 worlds 사이에서 평균했다. CI는30,000 paired world-cluster bootstrap으로 계산한다. Primary5 comparisons는 Bonferroni99% CI, 나머지는 descriptive95% CI다. Sample size는 fixed compute design이며 formal power analysis나 equivalence margin은 지정하지 않았다.

### Pilot에서 조건을 바꾼 사실

첫 pilot은 amplitude bound0.5와 query velocity=(1,0.6)을 사용했다. Query-view가8/8에서 우세해서 원하는 질문을 검증할 수 없었다. Test outcome을 평가하기 전에32 pilot worlds에서 bound0.10/0.18/0.28과 query velocity=(1,0.9)를 조사했다. Query-view/probe 빈도는21/11,25/7,32/0이었다. 두 action이 가장 균형 있게 경쟁한0.10을 선택했다.

따라서 이 실험은 ambiguity가 중요한 조건을 의도적으로 만든 diagnostic이다. 현실의 평균 성능이나 이런 조건의 발생 빈도를 추정하지 않는다. 최초 pilot와 변경 기록은 보존했다.

## Misspecification stress test

별도128 worlds는 세 번째 boundary

\[
f_2(y)=L(z^2/2-0.15z^4+0.06z^6)
\]

와 continuous friction coefficient \(\mu\sim U[0.1,0.3]\)를 사용했다. Inference는 여전히 original two boundary families와 frictionless/\(\mu=0.22\)만 사용한다. Geometry와 contact mismatch를 동시에 바꿨으므로, 실패가 나타나도 둘 중 어느 하나의 원인으로 분리해 단정할 수 없다.

## Numerical checks와 남은 근사 오차
Main acquisition은 float64 Gauss–Legendre order41, actual decoder는 order61이다. Independent test의 first32 worlds에서 모든 action과32 noise repeats를 order81과101로 다시 계산했다. Main choices와 main results는 덮어쓰지 않았다.
| Decoder | 최대 velocity component 차이:61→101 | 최대 차이:81→101 |
| --- | --- | --- |
| joint | 5.6360899e-05 m/s | 3.6035233e-05 m/s |
| plugin | 9.6342286e-05 m/s | 4.6430449e-05 m/s |
| mixture | 5.7272116e-05 m/s | 3.6942816e-05 m/s |
| factorized | 3.8829515e-05 m/s | 2.0240891e-05 m/s |

| Policy/native decoder | 선택 action의 평균 MSE 변화:61→101 | 변화:81→101 |
| --- | --- | --- |
| joint | +4.0776644e-08 | +1.97077554e-08 |
| plugin_aware | +1.7453943e-08 | +1.58218851e-08 |
| mixture_aware | +3.92881347e-08 | +1.86491258e-08 |
| factorized | +1.59928832e-08 | +9.37684682e-09 |
| rule_best | +3.46241464e-08 | +1.96745799e-08 |
| lmmse | +4.07986491e-08 | +1.97171118e-08 |

이는 first32 worlds의 audit이며 전체384 test/stress worlds에 대한 uniform numerical error bound는 아니다. Hard plug-in은 discrete model의 argmax가 바뀌면 prediction이 불연속적으로 변할 수 있다. 매우 작은 method 차이에 대해서는 numerical error와 statistical uncertainty를 함께 봐야 한다.
Physics validation에서는 최대 energy excess=-0.047324566, finite-difference slope error=1.367878e-11, factorized marginal preservation error=1.3877788e-17였다.

## 논문 기여 가능성에 대한 판단

**문제 설정은 v10보다 좋아졌지만, “복잡한 joint-posterior acquisition이 강한 baseline을 이긴다”는 method paper의 핵심 주장은 아직 성립하지 않는다.** Visibility/query shortcut을 제거한 것은 진전이다. 하지만 depth3 tree, LMMSE, mixture plug-in이 경쟁력을 유지했다는 사실을 중심에 놓아야 한다.

### 이번 결과로 말할 수 있는 것

- 같은 visibility/query라도 factual evidence가 contact class를 이미 구분했는지에 따라 필요한 observation이 바뀐다. World000과023이 분명한 예다.
- Primary distribution에서는 추가 observation의 실제 이득이 있고, 하나의 discrete explanation을 고르는 updated hard plug-in 대비 joint predictive mean의 개선도 primary99% CI에서 확인된다.
- Discrete model uncertainty를 유지하는 mixture plug-in은 full posterior predictive mean과 매우 가까웠다. Observation 선택도 동일했다. 현재 설정에서 full predictive integration을 추가하는 이득은 작거나 unresolved다.
- Factorized belief의 평균 성능은 낮았지만 primary99% CI는0을 포함했다. 이 결과를 correlation preservation의 확정적 우위라고 포장하면 안 된다.

### 이번 결과로 말할 수 없는 것

- 새로운 Bayes acquisition objective를 제안했다고 말할 수 없다.
- Strong learned rule이나 LMMSE보다 실제 decision loss를 확실히 줄였다고 말할 수 없다.
- Mixture plug-in의 inference가 저렴하다거나 full joint inference를 완전히 대체했다고 말할 수 없다. 이번 mixture baseline도 joint posterior의 conditional moments를 사용한다.
- Natural RGB, arbitrary unseen shape, unknown physical law 전체를 해결했다고 말할 수 없다.
- Current quadrature/SNIS diagnostic을 기존 tempered SMC bridge의 새로운 우위 증거로 사용할 수 없다. 이번에 bridge algorithm을 비교하거나 개선한 것은 아니다.

### 내가 선택할 방향

지금부터 toy world를 더 복잡하게 만들어 full joint selector가 이기는 조건을 계속 찾는 것보다는, **어떤 uncertainty를 유지해야 decision이 유지되는지**를 연구 질문으로 삼는 편이 낫다. 현재로서는 다음 가설이 더 설득력 있다.

> 하나의 explanation으로 너무 일찍 collapse하는 것이 주요 손실 원인일 수 있다. 반면 full predictive integration과 expensive acquisition은, 소수의 model hypotheses와 적절한 posterior moments를 사용하는 방법으로 대체할 여지가 있다.

이 가설은 아직 compact inference algorithm의 성공을 증명한 것은 아니다. 다음 구현 후보는 **geometry/contact model별 작은 mixture와 component 내부의 joint moments**를 유지하고, observation 이후 갱신하는 방법이다. Quadrature에서 계산한 moments를 그대로 받아오는 방식이 아니라 독립적으로 inference/update해야 하며, 학습한 rule·LMMSE·full quadrature와 cost–loss trade-off를 비교해야 한다.

특히 다음 validation은 세 조건을 분리하는 것이 좋다.

1. **Known-family, matched sensor:** Compact mixture가 full reference의 prediction과 acquisition을 어느 정도 보존하는지 확인한다.
2. **Unseen boundary 또는 unseen contact law를 각각 변경:** Mixture가 confident wrong explanation에 고정되는지, 새 observation이 오히려 bias를 강화하는지 확인한다.
3. **Boundary measurements의 calibration shift:** Sensor precision을 정확히 안다는 가정을 풀어, geometry uncertainty가 physics uncertainty로 전파되는 문제가 실제로 남는지 확인한다.

Strong rule 대비 improvement를 다시 주장하려면, 먼저 minimum useful effect와 sample size를 정한 **새 independent replication**이 필요하다. 이번 CI를 본 뒤 같은 test set에 worlds를 계속 추가해 유의해지는 순간 멈추는 방식은 피해야 한다. 현재 CI는 small effect에 대해 충분히 좁지 않고, 한 번 학습한 rule의 training-seed variation도 포함하지 않는다.

### Stress test가 주는 시사점

Candidate family 밖의128 worlds에서 joint MSE는0.010975069, no-observation은0.014756974였다. 추가 observation의 평균 이득은 남았지만, learned rule(0.010803425)과 LMMSE(0.010660299)의 평균은 joint보다 낮았다. Mixture plug-in(0.011009937)은 여기서도 joint에 가까웠다. 따라서 “더 정교하게 posterior를 적분하면 model mismatch에서도 더 잘 된다”는 근거는 없다.

이것은 별도의 shifted distribution이며 geometry와 contact를 동시에 바꿨다. In-support와 stress test의 absolute loss 차이를 순수한 model mismatch의 causal effect로 해석할 수 없다. 또한 이 평균 순서를 통계적으로 확정된 method ranking으로 해석하지 않는다.

현 단계에서는 **controlled diagnostic과 representation ablation으로서의 가치가 있다. 새로운 acquisition method의 논문 기여로는 아직 부족하다.** 다음 투자는 full SMC/bridge의 규모 확대보다 compact multi-model belief의 독립적 구현, mismatch 원인 분리, 실제 geometry measurement에 대한 검증에 두는 편이 합리적이다.


## 선행연구와 논문 기여의 경계

2026-11-12 기준으로 아래 primary sources의 objective와 관련 설명을 확인했다. 이번 실험에서 이 논문들의 전체 system을 재구현하거나 성능을 직접 비교한 것은 아니다.

| 선행연구 | 이미 다룬 부분 | v11에서 주장하면 안 되는 novelty |
| --- | --- | --- |
| [BayesSim, RSS2019](https://www.roboticsproceedings.org/rss15/p29.html) | Simulator parameters의 posterior inference와 Sim2Real domain randomization | Physics parameters를 point estimate 대신 distribution으로 유지한다는 원리 자체 |
| [Amortized Bayesian Experimental Design for Decision-Making, NeurIPS2024](https://proceedings.nips.cc/paper_files/paper/2024/hash/c59f05d7ab3638b138cc61f32e1a7cd1-Abstract-Conference.html) | Downstream decision utility를 고려한 experimental design과 amortized design/prediction | Prediction에 중요한 observation을 고른다는 일반적 formulation |
| [Action-BED, 2026 arXiv preprint](https://arxiv.org/html/2606.23662v1) | Expected future loss를 design policy와 downstream action policy의 joint optimization으로 표현; explicit posterior 없이도 학습 가능한 formulation | Expected decision loss acquisition이나 explicit posterior가 모든 좋은 decision에 필수라는 주장 |
| [Active Learning of Model Discrepancy with Bayesian Experimental Design, 2025 arXiv](https://arxiv.org/html/2502.05372v2) | Model discrepancy가 BED를 해칠 수 있는 문제와 sequential discrepancy correction | Model mismatch를 고려해야 한다는 문제 제기 자체 |

Action-BED와 TNDP를 보면 **최종 prediction에 필요한 정보와 full parameter posterior는 같은 요구사항이 아니다.** 이번 연구에서도 “full posterior를 유지하니까 더 원리적이다”는 설명만으로는 contribution이 성립하지 않는다. 동일한 decision quality를 더 작은 mixture나 amortized predictor로 얻을 수 있다면 그것이 중요한 비교 기준이다. 이는 해당 연구들의 objective를 현재 실험에 적용해 해석한 판단이다. [TNDP 원문](https://arxiv.org/html/2411.02064v1), [Action-BED 원문](https://arxiv.org/html/2606.23662v1).

Model discrepancy 연구는 candidate model set에 true model이 들어 있어야 한다는 가정의 한계를 직접 설명한다. v11의 primary는 바로 이 closed-world 가정을 사용한다. 별도 stress test를 통과하더라도 임의의 unknown physics를 다뤘다는 의미가 아니며, 실패한다면 단순히 particles나 quadrature points를 늘려 해결할 문제인지부터 다시 나눠야 한다. [Model Discrepancy 원문](https://arxiv.org/html/2502.05372v2).

### 이 실험이 기여로 발전하려면 필요한 증거

현재 가능한 contribution 후보는 새로운 Bayes objective보다 **geometry와 contact explanations가 서로 보상하는 상황에서 어떤 belief approximation이 decision에 충분한지 측정하는 diagnostic**이다. 이를 논문으로 키우려면 다음 증거가 필요하다.

1. **Generator와 inference의 분리:** 같은 analytic family에서만 truth와 posterior를 만드는 조건을 넘어, 독립 simulator와 다른 contact laws에서도 관찰된 효과가 남아야 한다.
2. **Observation 비용과 실행 가능성:** Camera movement, calibration error, probe/reset 비용을 포함한 cost-normalized decision loss로 비교해야 한다. Query contact point를 직접 지정할 수 있다는 현재 가정도 검토해야 한다.
3. **강한 compact baseline:** Mixture plug-in, LMMSE, 더 많은 training worlds로 학습한 amortized rule을 비교하고, selector와 decoder를 교차 평가해야 한다. 작은 tree를 이겼다는 것만으로 복잡한 inference가 필수라고 결론낼 수 없다.
4. **Contact에 중요한 geometry 정보의 실제 측정:** Natural images나 실제 boundary measurements에서 slope/curvature의 uncertainty를 calibration하고, geometry error가 intervention prediction으로 전파되는 과정을 검증해야 한다.
5. **Robustness의 원인 분리:** Unseen boundary family와 unseen contact law를 한 번에 바꾸지 않고 각각 바꾸어, 실패가 geometry mismatch인지 contact mismatch인지 규명해야 한다.

이 항목은 이번 turn에서 모두 실행한 결과가 아니라, 현재 diagnostic을 넘어 논문 기여를 판단하기 위한 다음 기준이다. 본 보고서의 numerical 결과만으로 real-world performance나 주요 학회 채택 가능성을 확정하지 않는다.

## 실행 및 재현 기록

서버 경로: `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/ambiguity_v11`. `datacenter3`의 기존 allocation2680491, compute node `node56`, idle NVIDIA RTX PRO 6000 한 개를 사용했다. 기존 jobs와 v10 결과는 변경하지 않았다.

Python 환경은 기존 `../continuous_v2/.venv/bin/python`이다. Main simulation/inference는 GPU, rule fitting과 analysis는 compute node의 CPU에서 수행했다. Login node에서는 numerical experiment를 실행하지 않았다.

LMMSE와 order81/101 numerical audit도 같은 allocation의 CPU에서 CUDA를 비활성화하여 수행했다. GPU primary worker와 충돌하지 않도록 분리했으며, 이 audit은 CPU/GPU execution 차이까지 포함한 cross-check다.

Main acquisition의 test world당 평균 시간은 **0.9121초**다. Joint, hard/mixture plug-in, factorized, Standard/CF-SNIS를 함께 계산한 시간이며 개별 method latency가 아니다. Rule fitting에는 **1.029초**가 들었고 training risk-label 생성 비용은 별도다. Online compute가 동일한 budget이라고 주장하지 않는다.

Validation에서 선택한 learned rule은 **tree**이다. RBF width=2.0, ridge=0.1; CART depth=3, min-leaf=4. SNIS median ESS는 query-view/factual-view/probe 순서로 **[2445.803851309872, 4095.435121151371, 2408.0181442694625]**다.

Code와 result files의 SHA256 manifest를 별도로 저장한다. `source_main.json`, `environment.json`, `pipeline_done.json`, worker logs, 각 split의 `freeze.json`, `extra_freeze.json`, `integrity.json`이 실행과 policy provenance를 기록한다.

주요 files: `model.py`(physics/inference), `run.py`(public-only acquisition·freeze·evaluation), `rules.py`(strong rules), `extra.py`(LMMSE), `numerical.py`(high-order audit), `analyze.py`(paired analysis), `PROTOCOL.md`·`DESIGN_ADDENDUM.md`·`LMMSE_ADDENDUM.md`(설계 기록).
