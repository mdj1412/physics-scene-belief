# Contact evidence v10: Factual ambiguity와 추가 observation의 가치

작성일: 2026-11-11 KST. 서버에서 실제 구현·실행한 결과다. 설명은 한글, technical terms는 English로 작성했다.

## 1. 결과와 판단

**Boundary-preserving geometry와 강한 inference baseline을 사용해도 추가 observation의 이득은 유지됐다. 그러나 복잡한 acquisition algorithm이나 joint posterior만의 이득은 입증되지 않았다.** 이번 설정에서는 visibility와 query를 함께 사용하는 간단한 rule과 plug-in geometry가 거의 같은 결과를 얻었다.

새로운 **128 independent test worlds × 16 observation-noise repeats × 6 queries**에서, Boundary Cross-fitted SNIS32768의 actual decision MSE는 **0.000593319**, no-observation baseline은 **0.011367665**였다. 평균 MSE는 약 **94.78%** 감소했다. 이는 일부러 occlusion을 넣은 controlled diagnostic에서의 결과이며 자연적인 장면의 평균 개선율이 아니다.

Boundary Standard SNIS8192, Cross-fitted SNIS8192/32768, Rao–Blackwellized quadrature selector는 **768개 query episodes에서 모두 같은 action**을 선택했다. Common decoder를 사용했으므로 actual loss도 동일하다. 이 결과는 additional evidence의 가치가 이 실험에서 finite-particle failure의 산물은 아니라는 근거다. 반면 cross-fitting의 특별한 이득을 보이는 결과는 아니다.

Development worlds에서 정한 **visibility + query rule**은 768개 중 **766개**의 선택이 일치했다. 이 rule에 posterior-mean plug-in geometry까지 연결한 supplementary baseline도 MSE **0.000593929**를 얻었다. 정교한 selector와 거의 같은 수치이며, paired descriptive CI도 차이를 확정하지 못했다.

![Actual decision loss](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/contact_evidence_v10/results/analysis/decision_loss.png)

이 figure는 mean을 표시한다. Uncertainty는 아래 paired world-level CI 표를 참고한다.

## 2. 어디서 무엇을 실행했는가

| 항목 | 실행 기록 |
| --- | --- |
| 접속 | `datacenter3` → `yonsei-ai-login3` |
| Compute node | `node56` |
| 기존 사용자 allocation | `2680491` |
| 사용 GPU | Allocation 내부의 NVIDIA RTX PRO 6000 1개 |
| 실행 전 확인 | GPU memory 0 MiB, compute process 없음. 두 차례 확인 후 실행 |
| 실행 경로 | `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/contact_evidence_v10` |
| Python 환경 | 기존 `../continuous_v2/.venv/bin/python` |
| 단계 | Validation → pilot 8 worlds → development 32 worlds → test 128 worlds → numerical refinement → analysis |
| 완료 | Primary worker와 refinement worker 모두 exit code 0; foreign GPU process 탐지 없음 |

기존 allocation과 다른 jobs를 변경하거나 취소하지 않았다. Main numerical work는 할당된 compute node의 GPU에서 수행했고, analysis와 plotting은 같은 node에서 CUDA를 비활성화하여 수행했다. Login node에서 simulation/inference를 실행하지 않았다.

이번 실험은 작은 analytic contact model과 image sensor를 사용한다. 대형 model training이나 dataset 다운로드는 필요하지 않았다. Primary pipeline은 약45초에 완료됐다. 짧은 runtime은 GPU batching과 낮은 dimension 덕분이며, 일반적인 raw-video inverse physics가 이 정도 비용이라는 의미는 아니다.

## 3. v9의 첫 실험을 어떻게 구체화했는가

### 3.1 Scene hypotheses

2D point mass가 수평 속도 **1 m/s**로 static boundary에 충돌한다. Boundary는 다음과 같다.

\[
x(y)=x_0+a_L b_L(y)+a_R b_R(y),
\]

\[
b_j(y)=\tfrac12\{1+\cos[\pi(y-c_j)/w]\},\quad |y-c_j|<w,
\]

support 바깥에서는 \(b_j=0\)이다. Centers는 \(c_L=-25\) mm, \(c_R=25\) mm, support radius는 \(w=18\) mm다. \(a_L,a_R\)는 ±4 mm 범위에서 local boundary의 모양을 바꾼다.

Independent truncated Gaussian prior를 사용했다. \(x_0\)의 standard deviation은1 mm, amplitudes는2.5 mm, restitution은 mean0.65·standard deviation0.12다. Geometry bounds는 ±4 mm, restitution bounds는[0.45,0.85]다. Test worlds는 prior에서 독립적으로 생성했으며 response gap이 큰 world만 남기지 않았다.

Collision은 friction 없는 single instantaneous impulse다. Local slope를 \(s=x'(y)\), restitution을 \(e\)라 하면,

\[
v_x'=1-\frac{1+e}{1+s^2},\qquad
v_y'=\frac{(1+e)s}{1+s^2}.
\]

이 범위에서는 outgoing x velocity가 음수여서 second contact가 없다. Finite-radius sphere, gravity, deformable contact, friction, 여러 번의 collision은 다루지 않았다. 따라서 기존 v8 simulator의 단순한 연장 실행이 아니라 v9의 identifiability 질문을 분리하기 위한 새 diagnostic이다.

### 3.2 Factual evidence가 같게 되는 이유

Factual impact는 **y=0**에서 일어난다. 두 local bumps의 support 밖이므로 \(s=0\), \(v_x'=-e\)다. Factual dynamic observation에는0.06 m/s의 Gaussian noise를 더한다.

Factual image에는 central boundary와 무작위로 선택한 한쪽 lobe만 보인다. 다른 쪽 lobe는 occluded다. Twin hypothesis는 **가려진 lobe의 amplitude 부호만 바꾸고 나머지 값을 유지**한다.

따라서 두 hypotheses는 다음을 공유한다.

- Visible pixels의 noise-free mean과 sensor covariance.
- Factual collision response와 그 observation distribution.
- Object identity, restitution, visible geometry.

즉 이번에는 “대략 비슷함”보다 강한 **동일한 factual likelihood**를 만들었다. 이는 의도적으로 설계한 occlusion의 결과다. 자연 영상에서 이런 ambiguity가 얼마나 자주 발생하는지를 측정한 결과는 아니다.

### 3.3 Intervention에서 실제로 달라졌는가

Query impact y는 **−33, −25, −17, +17, +25, +33 mm**다. 모든 query는 동일한 velocity-prediction type이며 impact position만 다르다.

| 검증 항목 | 결과 |
| --- | --- |
| 128 twin pairs의 visible image mean 최대 차이 | **0** |
| Factual response 차이 | 구성상0: 동일 restitution, central flat contact |
| Hidden lobe의 off-center queries에서 velocity L2 gap 중앙값 | **0.3710 m/s** |
| 같은 subset에서 gap >0.1 m/s 비율 | **79.69%** |
| Lobe center queries(±25 mm)의 최대 gap | **0** |

마지막 행은 중요한 negative control이다. Center에서는 slope가0이므로 모양이 달라도 velocity가 같다. 모든 query에서 ambiguity가 중요하도록 만들지 않았다. 전체6 queries를 합친 gap의 median도0이다. 큰 gap을 보고할 때는 **hidden off-center subset**임을 명시해야 한다.

![사전에 고정한 world 000의 twin 예시](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/contact_evidence_v10/results/analysis/twin_example.png)

World 000은 최대 gap을 찾아 고른 예시가 아니다. Figure에서 gray region은 factual occlusion이고 arrows는 intervention 후 velocity 방향이다.

## 4. Boundary-preserving geometry는 무엇인가

### 4.1 실제 image sensor 경로

Synthetic orthographic strip을 subpixel occupancy로 rasterize하고 **pixel-space Gaussian noise**를 더했다. Factual sensor는81개의 y rows와48개의 x pixels를 갖고, x pixel pitch는0.5 mm다. Visible rows만 전달한다. Per-pixel noise standard deviation은0.15다.

각 row의 unclipped intensity를 적분하면 boundary x position을 추정할 수 있다. 이 extractor의 mean은 실제 boundary position과 일치하며, noise standard deviation은

\[
\sigma_x=\Delta x\sqrt{W}\,\sigma_I
\]

로 계산된다. Factual profile에서는 약0.520 mm다. PNG는 사람이 보기 위해 clip했지만 **inference에는 unclipped floating-point sensor**를 사용했다. Synthetic scalar occupancy image이며 natural RGB, learned segmentation, photometric domain shift의 실험이 아니다.

Boundary-preserving representation은 visible row별 profile 전체를 유지한다. Known compact-bump basis에 fit하여 local slope를 추정할 수 있다. Raw pixels의 모든 정보를 보존하는 statistically optimal extractor라고 주장하지 않는다.

### 4.2 비교한 geometry baseline

| Representation | 입력과 추정 | 비교 목적 |
| --- | --- | --- |
| Boundary-preserving | 동일 pixels에서 추출한 row별 boundary profile | Local boundary 정보를 유지하는 강한 known-family baseline |
| Coarse global-area | 동일 visible rows의 평균 하나 | 의도적인 information-loss control |
| Boundary plug-in | 관측 후 geometry posterior mean을 하나의 boundary로 대입 | Posterior predictive integration이 꼭 필요한지 확인 |

Coarse global-area는 최신 geometry methods 전체를 대표하지 않는다. 이 비교에서 boundary-preserving 방식이 좋았다는 사실만으로 새 representation contribution이 성립하지 않는다.

## 5. 추가 observation과 inference

선택지는 no observation, left boundary view, right boundary view, center restitution probe다.

- **Left/right view:** 지정한 lobe의17 rows를 새로 측정한다. Pixel pitch0.25 mm, width64 pixels, pixel noise0.1이므로 row boundary noise는0.2 mm다.
- **Probe:** Scene을 reset하고 y=0에서 restitution을 다시 측정한다. Noise는0.015 m/s다. Query impact point의 response를 직접 보여 주지 않는다.
- **No observation:** 기존 factual evidence로만 예측한다.

여기서 view는 새로운 independent boundary measurement의 이름이다. 실제3D camera trajectory나 perspective 변화는 모델링하지 않았다. 가려진 region을 다른 aperture에서 볼 수 있다는 controlled sensor assumption이다.

Factual image feature와 restitution likelihood가 linear Gaussian이므로 posterior의 Gaussian conditioning 후 bounded rejection으로 **independent posterior samples**를 생성할 수 있다. 이 단계에 SMC를 사용하지 않았다. **SNIS는 hypothetical observation 이후 update를 근사하는 데 사용**했다.

| Selector | 구현 |
| --- | --- |
| Standard SNIS8192 | Pooled bank의 squared-shift utility |
| Cross-fitted SNIS8192 | Independent streams 각4096의 cross-product utility |
| Cross-fitted SNIS32768 | Independent streams 각16384의 cross-product utility |
| RB quadrature | Nuisance x0를 analytic marginalization하고 amplitude를2D quadrature로 적분; restitution mean도 analytic integration |
| EIG | Finite-bank posterior/prior KL의 expected value; descriptive baseline |
| Query-type rule | Development32 worlds에서 전체 queries에 쓸 하나의 action 선택 |
| Query-value rule | Development에서 impact y별 action 선택 |
| Visibility + query rule | Development에서 visible side와 impact y 조합별 action 선택 |

Boundary selectors는 independent outer generator의 동일한 **H256 hypothetical observations**를 사용했다. Standard와 cross-fitted8192는 같은 particles를 공유한다. Cross-fitting은 self-normalization bias를 없앤 unbiased EVSI가 아니다. RB quadrature도 finite outer Monte Carlo를 사용하므로 true Bayes-optimal policy라고 부르지 않는다.

## 6. 실제 decision loss

모든 test policies를 먼저 freeze하고 SHA256을 저장했다. 그 뒤 private actual observations를 읽었다. 모든 selectors는 **동일한 boundary factual evidence와 선택된 actual observation을 사용하는 common decoder**로 평가했다. Selector 자체의 효과를 비교하기 위한 조건이다. Coarse/native decoder 차이는 다음 절에서 따로 보고한다.

Loss는 \(\|\hat v-v\|^2/2\), 단위는m²/s²다. World 안의16 noise repeats와6 queries를 평균한 뒤128 worlds를 평균했다. Query마다 선택하는 observation은 별도 decision episode이며, 한 world에서 여러 observations를 순서대로 누적한 실험이 아니다.

| Method | Actual MSE, common decoder |
| --- | ---: |
| No observation | 0.011367665 |
| Random3 expected loss | 0.007648965 |
| Query-type rule | 0.006937638 |
| Always left view | 0.005517556 |
| Always probe / plug-in certainty selector | 0.010491700 |
| EIG | 0.001179241 |
| Query-value rule | 0.000809888 |
| Coarse CF8192 selector | 0.000809888 |
| Visibility + query rule | **0.000593931** |
| Boundary Standard SNIS8192 | **0.000593319** |
| Boundary Cross-fitted SNIS8192 | **0.000593319** |
| Boundary Cross-fitted SNIS32768 | **0.000593319** |
| Boundary RB quadrature selector | **0.000593319** |

Random3는 세 추가 actions를 uniform하게 고를 때의 expected loss를 계산했다. 한 번 뽑은 random seed의 우연한 선택 결과가 아니다. Query-type rule은 development에서 right view를 선택했기 때문에 test에서 always-left보다 나빴다. Test 결과를 본 뒤 rule을 바꾸지 않았다.

### 6.1 사전에 정한 primary contrasts

아래 difference는 **Boundary CF32768 − comparator**다. Negative이면 CF가 더 낮은 loss를 얻었다. World-cluster paired bootstrap30000 draws를 사용했고,4 primary comparisons에 Bonferroni98.75% intervals를 적용했다.

| Comparator | Mean difference | Adjusted98.75% CI |
| --- | ---: | --- |
| No observation | −0.010774346 | [−0.013282698, −0.008427544] |
| Query-type rule | −0.006344319 | [−0.008736011, −0.004225807] |
| Query-value rule | −0.000216569 | [−0.000342401, −0.000111018] |
| RB quadrature selector | 0 | [0, 0] |

No observation 대비 약94.78%, query-value rule 대비 약26.74%의 mean MSE 감소다. Quadrature와의[0,0]은 frozen choices가 전부 같아서 empirical paired differences가 모두0인 결과다. 모든 미래 worlds에서 두 방법이 equivalent하다는 검정 결과가 아니다.

128/128 worlds에서 CF의 noise/query-averaged loss가 no observation보다 낮았다. 이 count는 별도 significance test가 아니며, 반복 noise를 독립 worlds처럼 세지 않았다.

## 7. 가장 강한 간단한 baseline이 결론을 어떻게 바꾸는가

### 7.1 같은 query에서도 observation이 달라지는가

**그렇다.** 예를 들어 query y=−33 mm는 동일하지만,

- Negative-y lobe가 보이는66 worlds에서는65개가 probe,1개가 left view를 선택했다.
- Negative-y lobe가 가려진62 worlds에서는62개 모두 left view를 선택했다.

Factual visibility에 따라 geometry uncertainty와 restitution uncertainty의 상대적 중요성이 달라진 것이다. Query center±25 mm에서는 geometry slope가0이므로 모두 probe를 선택했다.

![Visibility와 query에 따른 선택](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/contact_evidence_v10/results/analysis/selection.png)

하지만 **선택이 달라진다는 사실만으로 sophisticated acquisition이 필요하지는 않았다.** Visibility + query rule은 **99.74% agreement**를 얻었다. CF 대비 rule의 mean loss 차이는 약6.12e−7이며, 반대 방향으로 쓴 CF−rule의 descriptive95% CI는 약[−1.83e−6,0]이다. 차이는 한 world의 두 queries에 집중된다.

### 7.2 Plug-in geometry도 강하게 만들면 어떤가

Geometry를 고정하고 image observation의 가치를0으로 보는 certainty selector는 항상 probe를 선택하므로 좋지 않았다. 그러나 이것만을 plug-in의 대표로 두면 약한 baseline을 고른 셈이다. 그래서 **같은 추가 observation을 얻은 뒤 geometry를 다시 추정하는 plug-in baseline**을 supplementary control로 평가했다.

| End-to-end condition | MSE |
| --- | ---: |
| Boundary plug-in, factual only | 0.011567394 |
| Coarse representation, factual only | 0.018548689 |
| Coarse CF8192 + coarse native decoder | 0.003684163 |
| Boundary plug-in + visibility/query rule | **0.000593929** |
| Boundary plug-in + CF32768 선택 | **0.000593326** |
| Boundary posterior predictive mean + CF32768 선택 | **0.000593319** |

Plug-in + visibility/query rule과 CF/common decoder의 difference는6.10e−7, descriptive95% CI는[−4.06e−7,2.11e−6]이다. 같은 CF 선택에 plug-in decoder만 적용한 difference는6.48e−9, CI는[−5.44e−7,5.60e−7]이다. 추가 이득을 확정할 수 없다.

따라서 **새 evidence를 얻는 것은 중요하지만, 그 이후 full posterior predictive integration이 이 조건에서 필수라는 주장은 지지되지 않는다.** 또한 factual impact를 flat region에 놓았기 때문에 restitution과 local geometry가 분리된다. 이 실험은 correlated geometry–restitution posterior의 필요성을 시험하지 않았다.

## 8. Numerical validation과 compute

### 8.1 구현 검증

| 검사 | 결과 |
| --- | ---: |
| Noise-free image extractor와 analytic boundary의 최대 차이 | 2.60e−18 m |
| Analytic slope와 finite difference의 최대 차이 | 2.17e−11 |
| 테스트한 collisions의 kinetic energy excess | 0 |
| 테스트한 outgoing vx의 최대값 | −0.3256 m/s |
| Initial validation: quadrature64 vs96 prediction 차이 | 2.88e−10 m/s |
| Initial validation: independent MC131072 mean과 quadrature 차이 | 0.000281 m/s |

### 8.2 실제 평가에서 발견한 quadrature 문제와 수정

Initial validation만으로 충분하다고 보지 않고16 test worlds의 actual observations에서도 order64/96을 비교했다. 최대 prediction 차이가 **0.000142 m/s**여서 더 높은 order로 재평가했다.

모든128 worlds와 모든 actual actions/noise repeats를 **order128 common decoder**로 다시 계산했고 **order192**와 비교했다. 최대 prediction 차이는 **9.48e−13 m/s**였다. 최종 표는 order128 결과다. 기존 order64 결과는 보존했으며 **acquisition choices는 변경하지 않았다.** Initial64에서128로 바뀐 single action/repeat/query loss의 최대 차이는8.06e−6이었다. 평균 CF MSE는0.000593322에서0.000593319로 바뀌었다.

이 refinement와 강한 plug-in end-to-end control은 primary results 확인 이후 추가한 audit라는 점을 명시한다. 자세한 기록은 [Numerical addendum](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/contact_evidence_v10/NUMERICAL_ADDENDUM.md)에 있다.

### 8.3 Acquisition ranking 안정성

Boundary Standard8192, CF8192, CF32768, quadrature64의 top1 agreement는100%였다. 사전에 정한 첫16 worlds의 H512 independent outer resampling audit도 H256 CF 선택과100% 일치했다. 같은 outer samples에서 quadrature64/96 utility의 최대 차이는4.31e−7이었다.

이는 이번 action ranking이 안정적이라는 근거이며, infinite-compute utility를 정확히 계산했다는 보장은 아니다. Ambiguity 자체가 쉬운 visibility rule로 거의 설명되는 구조라는 점이 더 중요하다.

### 8.4 측정한 acquisition compute

| 작업 | World당 평균 |
| --- | ---: |
| Factual posterior32768 draw | 1.67 ms |
| Standard + CF8192 utility 공동 계산 | 18.20 ms |
| Standard + CF32768 utility 공동 계산 | 61.79 ms |
| RB quadrature64 utility | 10.98 ms |
| Coarse8192 utility | 17.83 ms |

같은 GPU의 synchronized wall time이다. Outer generator, common decoder, disk I/O와 actual sensing cost는 제외했다. Standard/CF는 같은 routine에서 함께 계산했으므로 각각의 최적화된 단독 runtime이 아니다. Exact compute-matched algorithm benchmark로 해석하지 않는다. 이 known low-dimensional family에서는 model-aware quadrature가 이미 빠르고 강하다.

## 9. Observation cost가 있다면

Primary 비교는 physical prediction loss다. 별도 descriptive sensitivity에서는 추가 action의 cost를 모두 같은 λ로 두고, estimated utility−λ가 no-observation보다 클 때만 관찰하도록 했다. λ의 단위는MSE와 같게 환산한 가정이며 실제 camera/robot 비용을 측정한 것이 아니다.

| λ | 관찰하는 query 비율 | Actual MSE | MSE + 평균 cost |
| --- | ---: | ---: | ---: |
| 0 | 100% | 0.000593319 | 0.000593319 |
| 0.0001 | 100% | 0.000593319 | 0.000693319 |
| 0.001 | 89.06% | 0.000689482 | 0.001580107 |
| 0.01 | 33.33% | 0.001222286 | 0.004555619 |

Cost가 크면 hidden off-center geometry를 확인하는 observation만 주로 남는다. 이 결과를 “현실에서도 camera 이동보다 probe가 싸다”는 식으로 해석할 수는 없다.

## 10. v9 가설에 대한 판정

| 가설 | 이번 판정 |
| --- | --- |
| Factual evidence와 양립하지만 intervention response가 다른 hypotheses가 존재한다. | **확인.** Exact likelihood-equivalent twins를 구성하고 response gap을 측정했다. |
| Boundary-preserving geometry를 사용해도 추가 observation이 도움이 된다. | **확인.** Known-family synthetic sensor에서94.78% mean MSE 감소가 남았다. |
| 같은 query type·value에서도 필요한 observation이 달라질 수 있다. | **확인.** Factual visibility가 선택을 바꿨다. |
| Cross-fitted SNIS 같은 강한 baseline을 사용해도 observation gain이 남는다. | **확인.** Standard, cross-fitted, quadrature 모두 같은 선택과 gain을 얻었다. |
| 복잡한 acquisition이 강한 간단한 rule보다 실질적으로 필요하다. | **현재 근거 약함.** Visibility + query rule이99.74% 일치했다. |
| Joint geometry/material posterior가 plug-in보다 필수다. | **미입증.** Strong updated plug-in baseline과 차이를 확인하지 못했다. |
| Unknown model family와 real video에서도 성립한다. | **미검증.** Correctly specified basis와 calibrated synthetic sensor만 사용했다. |

이번 결과는 v8과 모순되지 않는다. v8은 numerical approximation과 utility estimator가 효과를 설명할 수 있음을 보였고, v10은 그런 어려움을 줄인 뒤에도 **관측되지 않은 geometry 때문에 정보가 부족할 수 있음**을 보였다. 동시에 그 부족함을 해결하는 선택이 간단한 rule로 충분할 수 있음을 확인했다.

## 11. 논문 기여 가능성과 다음 결정

현재 결과만으로는 새로운 acquisition method 논문을 밀어붙이지 않겠다. “가려진 부분을 보면 좋다”는 결과는 예상 가능한 positive control이다. Boundary profile을 보존하는 것, known-family posterior를 적분하는 것, 관찰 가치를 task loss로 계산하는 것 자체도 새 원리가 아니다.

이번 구현의 가치는 **동일 factual likelihood, query-sensitive response, 실제 pixel observation, strong inference control, policy freeze, numerical convergence**를 한 protocol 안에서 확인할 수 있는 실험 기반에 있다. 특히 weak plug-in을 이긴 뒤 끝내지 않고, updated plug-in + visibility rule이 거의 같은 성능임을 확인한 것이 연구 판단에 중요하다.

후속 연구를 진행한다면 우선 **visibility flag로 거의 결정되는 현재 shortcut을 제거했을 때도 의미 있는 문제인지** 확인해야 한다. 단순히 noise를 키워 difficult benchmark를 만드는 것으로 충분하지 않다. 같은 visibility와 비슷한 factual fit을 갖는 장면에서 local contact topology, unknown boundary, measurement bias가 실제로 서로 다른 evidence를 요구하는지 검증해야 한다.

그 다음에도 간단한 boundary fitting + rule이 충분하면, acquisition algorithm 개발을 중단하고 perception 또는 measurement calibration으로 방향을 좁히는 편이 타당하다. Unknown family를 추가한 경우에는 모든 방법에 같은 candidate support를 주고 비교해야 한다. 이번 결과로 model revision을 해결했다고 주장하지 않는다.

실제 영상과 robot 실험은 이번에 수행하지 않았다. Synthetic aperture view를 real camera acquisition으로, point collision을 실제 rigid-body contact로 일반화하려면 추가 검증이 필요하다.

## 12. 코드·데이터·재현

- [Model와 image sensor](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/contact_evidence_v10/model.py)
- [Generation, acquisition, evaluation pipeline](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/contact_evidence_v10/run.py)
- [실행 전 protocol](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/contact_evidence_v10/PROTOCOL.md)
- [최종 수치 JSON](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/contact_evidence_v10/results/analysis/summary.json)
- [Policy freeze](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/contact_evidence_v10/results/test/freeze.json)
- [Numerical refinement 기록](/Users/mindongjun/Documents/Codex/2026-10-30/new-chat/work/physics-scene-belief/contact_evidence_v10/results/refinement.json)

재실행은 새 output directory에서 해야 한다. 원 pipeline은 allocation 내부에서 `python guard.py pipeline`, 추가 numerical audit는 `python guard.py refine.py`, CPU analysis는 `CUDA_VISIBLE_DEVICES= python analyze.py`로 실행했다. 기존 결과를 덮어쓰는 재실행은 피하고, 먼저 allocation과 GPU process를 확인해야 한다.

World별 public factual features와 private truth/actual observations, frozen choices, original/refined decoder outputs를 모두 보존했다. 기존 v8 sealed 결과는 수정하지 않았다.
