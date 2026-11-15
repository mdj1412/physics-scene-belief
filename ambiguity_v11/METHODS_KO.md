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
