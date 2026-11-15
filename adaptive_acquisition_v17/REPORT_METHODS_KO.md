## 2. 구현한 adaptive compression

Factual evidence를 본 후 QMC16384 inner posterior를 얻고, event label 0/1의 두 decision regions로 나눈다. 각 region의 posterior mass를 유지하면서 region당 K개의 systematic representatives를 추출한다. Representative마다 원래 Gaussian sensor likelihood를 유지한다. K는 전체 components 수가 아니다. 두 regions가 모두 존재하면 replica당 2K개이며, 이번 방법은 두 replicas를 사용한다.

별도의 seed로 생성한 QMC16384 outer posterior에서 512개의 hypothetical world–observation pairs를 만든다. 이 outer sample은 simulator ground truth나 실제 미래 observation이 아니다. Public factual evidence에서 추론한 posterior predictive samples다. 모든 candidate actions와 expansion stages가 같은 outer samples를 공유한다.

각 compact replica가 hypothetical observation 이후 예측한 event probability와 outer sample의 event label 사이의 Brier loss를 계산한다. 두 replicas의 risks를 평균한다. 두 prediction을 평균한 ensemble decoder의 risk와는 다르다. 실제 observation을 받은 뒤에는 공통 QMC16384 full decoder를 사용하므로 primary 비교는 acquisition 차이를 측정한다.

현재 best action을 b, 다른 action을 a, 두 action의 estimated risk 차이를 gₐ라고 두면 stopping score는 다음과 같다.

\[
g_a(K)=\hat R_a(K)-\hat R_b(K),
\]
\[
e_a(K)=2\widehat{SE}_{\rm paired}(\ell_a-\ell_b)
+|g_a(K)-g_a(K_{\rm prev})|
+\tfrac12|g_a^{(1)}(K)-g_a^{(2)}(K)|.
\]

K를 **8→16→32→64→128**로 늘린다. K≥16에서 모든 competitor에 대해 gₐ>eₐ이면 멈춘다. 현재 best가 이전 단계와 달라도 이전 risk vector에서 현재 b와의 차이를 다시 계산한다. 모든 action risks의 range가 10⁻¹² 이하면 numerical tie로 기록하고 query view를 선택한다. K128에서도 gap이 분리되지 않으면 `cap-unresolved`로 기록하고 현재 estimated risk가 가장 작은 action을 선택한다.

첫 error 항은 outer sampling variation, 두 번째는 successive-resolution drift, 세 번째는 compression replicas 사이의 variation을 겨냥한다. **이 합은 true acquisition error의 보장된 upper bound가 아니다.** QMC rows는 iid가 아니며, 두 replicas만으로 shared bias를 탐지하지 못할 수 있고, successive stopping에 대한 coverage 보정도 없다. 따라서 `gap separated`는 내부 heuristic을 만족했다는 뜻이며 certified optimal action이라는 뜻이 아니다.

이번 구현은 outer sample count를 512로 고정한다. Sampling error가 action gap보다 크면 components만 늘려서는 해결되지 않을 수 있다. 이 상황의 빈도를 별도로 기록했다. 각 K에서는 같은 systematic phases를 쓰되 quantile locations가 달라져 representatives를 다시 구성한다. 기존 components를 그대로 유지하며 일부만 추가하는 nested coreset implementation은 아니다. Timing에는 이 재계산과 stopping checks를 모두 포함했다.

Posterior mass≤10⁻¹⁰인 region만 pruning하고 renormalize한다. Omitted mass를 기록한다. Inner/outer banks, full decoder와 cached forward predictions를 계속 사용하므로 bank-free compact belief라고 부르지 않는다.

| Method | 역할 |
|---|---|
| adaptive | Region별 compression과 위 stopping rule |
| adaptive_global | 같은 최대 total components, outer samples, replicas, stopping rule의 global resampling |
| fixed32 / fixed128 | 같은 cross-fitted risk objective에서 region당 K32/K128을 고정; stopping의 필요성 확인 |
| global128 | Fixed128과 같은 total components의 global resampling |
| region32 / global32 | V16 coherent mixture risk, representative당 hypothetical noise Q4 |
| native / qmc64_native | Optimized LMMSE16 / LMMSE64 |
| nested128 / nested512 | Independent inner/outer QMC16을 사용하는 optimized nested expected-loss baseline |
| query / probe / factual / none | Fixed rules 및 no-observation diagnostic |

Adaptive methods의 stopping rule을 켠 비교와 끈 비교는 같은 cross-fitted objective를 사용한다. V16 coherent methods와의 차이는 adaptivity뿐 아니라 risk estimator까지 포함하므로 pure adaptivity effect라고 해석하지 않는다.

## 3. 실험 설계와 해석 범위

새로운 **512 iid worlds**, world당 **16 actual observation noise repeats**를 사용했다. Seed는 6100101+137×world index이며 correctness pilot 8개는 별도 seed6000101이다. V16의 worlds를 primary test로 재사용하지 않았다. Contact-event와 stick-event 두 tasks에 집중했고, continuous velocity task의 성능을 이번 결과로 주장하지 않는다.

Model은 V16의 unknown boundary geometry, restitution과 contact-law uncertainty를 유지한다. Boundary는 x(y)=d+s y+a f_g(y), 두 geometry families와 frictionless/Coulomb/flexible laws를 사용한다. Prior는 geometry×두 base laws의 네 components에 각각 1/8, geometry×flexible law의 두 components에 각각 1/4이다. d∼U[−4,4] mm, s∼U[−0.4,0.4], a∼U[−0.1,0.1], e∼U[0.35,0.9], b₀∼U[−4,2], b₁∼U[−2,2]다.

L=25 mm, z=y/L일 때 f₀=Lz²/2, f₁=L(z²/2−0.35z⁴)다. Nominal flexible tangential impulse는 vₜ sigmoid[b₀+b₁ log((|vₜ|+0.02)/0.5)], shifted law는 vₜ sigmoid(b₀)[0.2+0.8/(1+(|vₜ|/s₀)²)]이고 s₀=0.1+0.7 sigmoid(b₁)다. Base laws는 바꾸지 않는다.

Factual evidence는 y=0에서 incoming velocity (1,0.35)의 outgoing velocity와 −10…10 mm의 boundary profiles 9개다. Noise SD는 각각 0.07 m/s와 1.2 mm다. Query y=25 mm에서 public factual profiles를 직선으로 extrapolate한 값이 contact/no-contact threshold다. Stick query의 incoming direction 역시 public slope estimate로 정하고 residual tangential velocity absolute≤10⁻⁸을 stick label로 쓴다. Hidden truth로 favorable worlds를 선택하지 않는다.

Candidate observations는 query view(19…31 mm의 profiles 7개, SD1 mm), factual view(−8…8 mm의 profiles 7개, SD0.35 mm), probe(y=0, incoming velocity(1,−0.7), outgoing velocity SD0.05 m/s)다. Selectors는 셋 중 하나를 선택한다. Sensing cost는 같다고 가정하며 no-observation은 별도 diagnostic이다.

Matched nominal, shifted data/nominal inference, shifted data/correct-law inference의 세 cells를 평가했다. Correct-law는 privileged diagnostic이다. Physics는 finite-reach contact gate와 instantaneous local impulse model이며 full trajectory solver, raw RGB reconstruction, real robot validation은 아니다.

Primary inference banks는 inner seed2026111201, outer seed2026111202, LMMSE64 seed2026111203이다. Primary acquisition seed81531을 고정하고 actual outcomes를 보기 전에 policies를 freeze했다. 512 matched worlds 모두에서 seed81532로 adaptive/fixed cross-fitted mixtures와 nested512를 반복했다. 여기서는 hypothetical samples와 compression phases가 함께 달라진다. 두 원인을 따로 분리한 factorial audit은 아니다.

각 cell의 첫 64 worlds에는 seed81533의 nested2048 reference를 사용했다. 첫 64 matched worlds에서는 inner/outer bank seeds2026111301/2026111302로 policy를 다시 계산했다. Bank audit의 actual loss는 공통 primary decoder로 평가하여 acquisition 차이를 분리했다. 모든 posterior seeds에 대한 일반화를 증명하는 충분한 multi-seed study는 아니다.

Ranking 안정성은 주로 **Top-1 action agreement**로 측정한다. Full ordering의 모든 permutation stability를 측정한 것은 아니다. LMMSE와 fixed rules는 같은 bank에서 deterministic하므로 acquisition seed를 바꾸어도 action이 같다. 이것이 정확한 ranking임을 뜻하지 않으므로 independent-bank sensitivity와 reference agreement를 함께 본다. Nested2048도 exact Bayes oracle은 아니다.

Primary 12 contrasts는 두 tasks에서 adaptive 대 native/nested512/adaptive_global/fixed128/global128/task-specific fixed rule(contact의 query, stick의 probe)다. World별로 16 repeats를 평균한 뒤 30,000 paired bootstrap과 Bonferroni 99.58333% CI를 사용했다. 나머지는 secondary 95% CI다. Fixed banks와 acquisition plans에 조건부인 CI이므로 모든 numerical uncertainty를 포함하지 않는다.

사전 approximation screen은 두 tasks 모두에서 adaptive/nested512 mean-loss ratio의 95% upper≤1.05 및 GPU aggregated median speedup≥2다. Adaptivity screen은 fixed32 대비 Top-1 repeatability 10 percentage points 이상 증가, fixed128 대비 GPU median 20% 이상 절약, 각 task의 fixed128 대비 loss-ratio upper≤1.05다. 분석 코드는 repeatability 조건을 두 tasks 모두에 적용했다. Contact에서 baseline agreement가 이미 높은 경우 이 조건에는 ceiling limitation이 있으므로 screen 실패만으로 모든 가치를 부정하지 않고 task별 수치와 cost를 함께 해석한다. Threshold를 outcomes에 맞춰 바꾸지 않는다.
