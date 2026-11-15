# Contact evidence v10 — 결과 확인 전 protocol

작성: 2026-11-11. v9의 첫 단계 A–C에 해당하는 synthetic diagnostic이다.

## 문제

2D point mass가 static boundary x=f(y)에 수평 속도 1 m/s로 충돌한다. Gravity와 friction은 없고 single instantaneous restitution impulse를 사용한다. Boundary는 x0 + aL*bL(y) + aR*bR(y), b는 중심 ±25 mm, 반경 18 mm의 compact cosine bump다. Amplitude bounds는 ±4 mm, x0 bounds는 ±4 mm다. Restitution bounds는 [0.45,0.85]. 최대 slope는 0.35로 outgoing x velocity가 음수여서 second contact가 없다. Finite-radius sphere와 deformable contact로 일반화하지 않는다.

Independent truncated Gaussian prior: x0 sd 1 mm, aL/aR sd 2.5 mm, restitution mean .65 sd .12. Factual impact는 y=0이므로 local bump와 무관하게 vx=-e다. Factual sensor sd=.06 m/s.

Factual image는 calibrated orthographic boundary strip이며 central region과 무작위 한쪽 lobe만 보인다. 다른 lobe는 occluded다. 선형 floating-point pixel sensor에 iid Gaussian noise를 더한다. Per-row soft occupancy integral을 추출하면 local profile을 보존하고 그 noise distribution은 정확한 Gaussian이다. PNG는 시각화용으로만 clip하며 extraction에는 unclipped sensor를 사용한다. 이는 natural RGB perception이나 learned segmentation의 검증이 아니다.

Twin hypothesis는 hidden amplitude 부호만 바꾼다. 따라서 factual image distribution과 factual velocity가 정확히 같다. Query y는 [-33,-25,-17,17,25,33] mm다. Twin이 항상 크게 다르도록 test worlds를 선별하지 않으며 gap distribution을 보고한다.

## 관측과 비교

추가 action은 no observation, left boundary view, right boundary view, center restitution probe다. View는 동일 geometry의 독립 pixel sensor measurement이며 probe는 reset된 독립 velocity measurement(sd=.015)다. Query impact point와 일치하는 probe는 없다. Cost는 primary physical loss에 더하지 않고 별도 sensitivity에서 다룬다.

Boundary-preserving representation은 모든 visible row의 profile을 사용한다. Coarse global-area representation은 visible rows의 평균 하나로 축약한다. 후자는 의도적 information-loss control이며 모든 일반 geometry methods의 대표가 아니다.

Primary inference/control:
- Boundary Cross-fitted SNIS8192: independent A/B 각4096 samples.
- Boundary Cross-fitted SNIS32768: independent A/B 각16384 samples.
- Boundary Standard SNIS8192: 같은 A/B pooled samples.
- Rao–Blackwellized quadrature acquisition: x0를 analytic marginalization, aL/aR 2D quadrature, restitution mean analytic integration. Known-family model-aware strong control이다.
- Coarse Cross-fitted SNIS8192, plug-in certainty policy, fixed actions, validation-tuned query-type/value rules.

Factual posterior는 linear Gaussian conditioning 후 bounded rejection으로 independent samples를 생성한다. SMC를 쓴다고 부르지 않는다. SNIS는 hypothetical observation update에서 사용한다. Outer generator는 candidate bank와 independent이며 모든 boundary selectors에 같은 H256 hypothetical observations를 준다. Reference quadrature는 order64, numerical audit order96이다. Self-normalization bias 때문에 cross-fitting을 unbiased EVSI라고 부르지 않는다.

## 평가

Development 32 worlds와 test 128 worlds는 seed·world를 분리한다. Test world당 16 actual observation noise repeats, 6 queries. Test 수는 compute 내 고정한 diagnostic sample size이며 power-guaranteed confirmatory design이라고 주장하지 않는다. Pilot는 별도 8 worlds. Pilot에서는 code correctness, runtime과 numerical convergence만 확인하며 효과 방향으로 test configuration을 선택하지 않는다.

모든 test policies를 먼저 freeze하고 SHA256을 저장한 뒤 private actual observations를 decode한다. Common decoder는 boundary factual evidence와 selected observation을 사용하는 quadrature posterior mean이다. Geometry input 차이를 숨기지 않기 위해 coarse native decoder와 boundary plug-in decoder의 end-to-end 결과도 별도 보고한다.

Primary contrasts: Boundary CF32768 minus (1) no observation, (2) validation query-type rule, (3) validation query-value rule, (4) boundary quadrature selector. World-level paired bootstrap, 4 contrasts Bonferroni 98.75% intervals. 다른 비교는 descriptive 95% CI. MSE=||vhat-v||²/2, 단위 m²/s². Noise repeats/queries는 independent worlds로 세지 않는다.

Primary minimum useful difference와 equivalence margin은 설정하지 않았으므로 CI가0을 포함하면 equivalence로 해석하지 않는다. Cheap rule 또는 quadrature와 비슷하면 새 acquisition algorithm의 novelty를 주장하지 않는다. Twin pair의 exact equality는 occlusion을 의도적으로 설계한 결과이며 자연적인 발생률 추정이 아니다.
