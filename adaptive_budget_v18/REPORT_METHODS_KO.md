## 2. Adaptive budget allocation의 구현

K는 label region당 representative 수, M=4m은 전체 hypothetical sample 수다. K는 16→32→64→128, m은 128→256→512→1024로 늘어난다. K/2 comparison을 위해 K8 prefix도 유지한다. 두 label regions가 모두 존재하면 replica당 components는 2K개이며, compression replicas는 2개다.

각 region의 factual posterior mass를 유지하면서 nested scrambled-Sobol quantiles로 representatives를 얻는다. K를 늘릴 때 기존 representatives가 prefix에 남는다. Original sensor Gaussian kernels와 event labels를 그대로 사용한다. Posterior mass≤10⁻¹⁰인 region만 pruning하고 renormalize하며 omitted mass를 기록한다.

Independent outer QMC16384 posterior에서 4개의 독립적인 randomized QMC scrambles로 hypothetical world–observation pairs를 만든다. 하나의 scramble 안에서는 m개의 samples가 서로 독립인 것은 아니다. Error estimate에는 각 scramble의 평균을 사용한다. Hypothetical labels는 candidate posterior에서 나온 값이며 실제 hidden world의 target이 아니다.

현재 risk가 가장 작은 action을 b라 하고 competitor a와의 gap을 gₐ라고 하면 다음 error proxies를 계산한다.

\[
g_a=\hat R_a-\hat R_b,
\qquad e_{C,a}=|g_a(K,M)-g_a(K/2,M)|+\tfrac12|g_a^{(1)}-g_a^{(2)}|,
\]
\[
e_{O,a}=2\,\mathrm{SD}\{g_a^{[1]},\ldots,g_a^{[4]}\}/\sqrt4.
\]

Superscript (1)/(2)는 compression replica, [1]…[4]는 outer scramble이다. Risk는 두 compact decoders가 independent outer targets에 대해 내는 Brier loss의 평균이다. 두 predictions를 평균한 ensemble decoder의 risk와는 다르다. K/2와 K의 비교는 같은 outer samples를 사용한다.

모든 competitor에서 gₐ>eC,ₐ+eO,ₐ이면 `gap-separated`로 멈춘다. 그렇지 않으면 gₐ−eC,ₐ−eO,ₐ가 가장 작은 competitor를 기준으로 더 큰 error proxy의 축을 두 배로 늘린다. 그 축이 cap이면 다른 축을 늘리고, 두 caps에 도달하면 `cap-unresolved`로 기록한다. 모든 action risks의 range가 10⁻¹² 이하면 `risk-tie`로 멈추고 query view를 선택한다.

**이 error proxies는 formal confidence bounds가 아니다.** Compression drift와 두 replicas가 shared bias를 놓칠 수 있고, 4개 outer randomizations는 variance estimate에 충분히 큰 표본이라고 보장할 수 없다. Factor2가 adaptive stopping의 95% coverage를 뜻하지도 않는다. Model mismatch와 finite factual-bank error는 별도로 남는다.

Risk-tie도 물리적으로 모든 observations가 무용하다는 증명은 아니다. 특히 global resampling은 minority label support를 놓쳐 constant prediction을 만들 수 있다. Primary region mixtures는 label mass를 보존하지만, 그 구조적 성질만으로 실제 decision loss 우위가 보장되지는 않는다.

### Cached implementation과 strong fixed baseline

Likelihoods의 class-wise log sums를 cache한다. K가 늘면 새 components와 기존 hypothetical samples의 조합만, M이 늘면 새 hypothetical samples와 현재 components의 조합만 평가한다. 따라서 한 실행에서 평가한 component–observation combinations는 최종 K×M rectangle과 일치한다. 중복 계산을 매번 새로 시작하지 않는다.

Adaptive engine은 최대 K의 representatives와 최대 M의 hypothetical coordinates를 미리 준비한다. Adaptive K/M은 실제 likelihood evaluation prefix를 정한다. 따라서 sample preparation과 representation memory가 최종 선택된 작은 budget에 비례해서 모두 줄었다고 주장하지 않는다. 이 preparation, cache allocation과 stopping checks를 online timing에 포함한다.

Original kernel은 logcumsumexp를, equivalent optimized kernel은 block logsumexp와 logaddexp를 사용한다. Fixed mixtures는 불필요한 adaptive prefix/error checks 없이 direct risk kernel을 사용한다. 전체 primary policies에 대해 action/risk equivalence와 adaptive stopping histories를 검증한 뒤 같은 optimized timing pass에서 비교한다. Original GPU timing도 보존한다.

## 3. Comparators와 protocol

| Method | 변경하는 budget / 역할 |
|---|---|
| adaptive | Error proxy에 따라 K 또는 M 선택 |
| balanced | 불확실하면 가능한 두 축을 함께 증가 |
| component_only | M512 고정, K만 증가 |
| outer_only | K128 고정, M만 증가 |
| fixed32_512 / fixed128_512 | Small outer budget의 strong fixed mixtures |
| fixed32_4096 / fixed128_4096 | Large outer budget의 strong fixed mixtures |
| global32_512 / global128_4096 | 같은 total components의 global resampling |
| native / qmc64_native | Optimized LMMSE16 / LMMSE64 |
| nested512 / nested2048 / nested4096 | 같은 4 outer scrambles를 사용하는 full-inner QMC16384 risk |
| query / probe / factual / none | Fixed rules와 no-observation diagnostic |

새로운 **512 iid worlds**(seed7100101+137×index), world당 **16 actual observation noise repeats**를 사용한다. Correctness pilot8은 seed7000101이다. Primary tasks는 contact-event와 stick-event다. 저장된 task 이름 `slip_event`는 slip/stick transition의 stick label을 평가한다. V17과 worlds, representative sequence, outer estimator가 달라졌으므로 V17의 평균과 직접 비교해 개선율을 만들지 않는다. Ablations는 V18 안에서 같은 estimator를 사용한다.

Physical model은 V17과 같은 local impulse surrogate다. Boundary x(y)=d+s y+a f_g(y), L=25mm, f₀=L(y/L)²/2, f₁=L[(y/L)²/2−0.35(y/L)⁴]를 쓴다. Geometry2 × base contact laws2의 네 components에 prior mass 각각1/8, geometry2 × flexible law의 두 components에 각각1/4를 둔다. Parameters는 d∈[−4,4]mm, s∈[−0.4,0.4], a∈[−0.1,0.1], restitution e∈[0.35,0.9], b₀∈[−4,2], b₁∈[−2,2]의 uniform priors다.

Factual evidence는 y=0, incoming velocity(1,0.35)의 outgoing velocity와 −10…10mm boundary profiles9개다. Noise SD는 각각0.07m/s와1.2mm다. Public factual profile extrapolation으로 y=25mm의 contact threshold를 만들며, public slope estimate로 slip/stick query direction을 정한다. Stick label은 residual tangent absolute≤10⁻⁸이다.

Candidate observations는 query view(19…31mm, profiles7, SD1mm), factual view(−8…8mm, profiles7, SD0.35mm), probe(y=0, incoming(1,−0.7), outgoing velocity SD0.05m/s)다. Selector는 하나를 선택하며 sensing cost는 같다고 가정한다. 실제 observation 이후의 decoder는 공통 full QMC16384이다. LMMSE64만 QMC65536 decoder를 사용한다.

Matched nominal, shifted data/nominal inference, shifted data/correct-law inference를 평가한다. Shift는 flexible law의 speed dependence를 바꾸며 base laws는 그대로다. Correct-law는 privileged diagnostic이다. Nominal tangential impulse는 vₜ sigmoid[b₀+b₁ log((|vₜ|+0.02)/0.5)], shifted impulse는 vₜ sigmoid(b₀)[0.2+0.8/(1+(|vₜ|/s₀)²)], s₀=0.1+0.7 sigmoid(b₁)다.

Primary banks는 inner2026111201, outer2026111202, QMC64 2026111203이다. Primary master seed91531을 고정했다. Main policies를 freeze한 뒤 private outcomes를 평가했다. Actual target, hidden parameter, unselected actual observation은 policy 입력으로 사용하지 않는다.

Primary12 contrasts는 adaptive 대 fixed128_512/fixed128_4096/global128_4096/nested2048/balanced/task-specific fixed rule을 두 tasks에서 비교한다. World별 repeats를 먼저 평균한 뒤 30,000 paired bootstrap과 Bonferroni99.58333% CI를 사용한다. 나머지는 secondary95% CI다.

Approximation screen은 두 task 모두에서 nested2048 대비 mean-loss ratio의 95% upper≤1.05와 GPU median speedup≥2다. Adaptivity screen은 stick repeatability가 component_only보다 5 percentage points 이상 높고, 두 tasks의 fixed128_4096 대비 ratio upper≤1.05이며 GPU median이20% 이상 낮은지를 본다. Contact에는 ceiling 때문에 별도 agreement-gain threshold를 요구하지 않는다. 5%/20%는 engineering thresholds이며 application-derived costs가 아니다.

### Independent audit의 수정과 error-source 검증

Code review에서 original alternative master92531의 outer stream이 primary representative seed92531과 겹치는 것을 발견했다. 원래 audit는 보존했고, 결과를 살펴보기 전에 disjoint master101531로 secondary audit를 교체하기로 기록했다. Primary policy, data와 thresholds는 바꾸지 않았다. 자세한 기록은 `AUDIT_CORRECTION.md`다.

최종 repeatability는 corrected master101531의 512 matched worlds와 primary를 비교한다. Supplemental script는 public evidence와 primary stopping budgets만 읽고, private outcomes를 읽지 않으며 결과를 별도로 freeze한다. 두 runs에서는 representatives와 outer samples가 함께 달라진다.

각 cell의 첫64 worlds에는 별도 master93531의 nested8192 reference를 사용한다. 이는 exact Bayes oracle이 아니다. 같은 hypothetical samples에서의 full-inner risk와 compact risk 차이를 통해 compression proxy를 점검한다. 첫64 matched worlds에서는 primary representatives를 고정하고 corrected outer scrambles만 바꾸어 outer proxy를 점검한다.

저장된 worst-gap 값으로 competitor를 유일하게 식별할 수 없는 near-tie cases는 proxy audit에서 제외하고 그 수를 기록한다. Outer audit는 두 independent estimates의 차이를 보는 것으로, true integration error의 직접 측정이나 nominal coverage test가 아니다. Ranking은 주로 Top-1 agreement이며 full ordering stability 전체를 의미하지 않는다.

이 실험은 raw RGB, full collision trajectories 또는 real robot을 검증하지 않는다. Factual banks와 cached candidate forward predictions도 계속 필요하다. 따라서 bank-free belief나 end-to-end visual physics inference의 성능으로 확대 해석하지 않는다.
