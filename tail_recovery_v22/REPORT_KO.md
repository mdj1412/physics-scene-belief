# v22 · Top-K가 제외한 scene hypotheses의 posterior mass 회복과 observation 선택

**약 7.15 ms의 GPU p90에서 tail 보존 방법과 factual top-K는 7,680개 bank–world–law 조합 전부에서 같은 observation을 골랐다.** 두 방법의 held-out Brier loss 차이는 정확히 0이었다. Fixed global mixture 대비 보존 방법의 Brier 차이는 nominal에서 +0.000251, 두 unseen laws 평균에서 +0.000549였고, 두 adjusted confidence intervals 모두 0을 포함했다. 양수는 보존 방법의 loss가 더 높다는 뜻이다. 비교 대상은 별도 calibration에서 latency만으로 정한 top K3328, 보존 K2816, fixed global K128이다. 세 방법의 K는 서로 다른 support 구성을 나타낸다.

가설이 되살아나는 현상은 **K512로 support를 제한하고, factual evidence로 예상하기 어려운 observation이 들어올 때** 분명했다. 세 law와 세 action에 걸쳐 collective recovery는 실제 simulator draws의 0.45–1.61%였다. 같은 law·action 안에서 surprise 하위 quartile에서는 한 번도 없었고, 상위 quartile에서는 1.59–6.43%였다. Nominal에서 가장 자주 회복을 일으킨 action은 factual view(1.41%)였고, 두 unseen laws에서는 probe(1.44%, 1.61%)였다. Latency에 맞춘 K3328에서는 pre-specified collective 및 individual recovery가 각각 368,640회 중 0회였다. 이 횟수에는 같은 world와 noise를 공유하는 paired observations가 포함된다.

**K512에서 tail 보존은 full-bank reference의 선택을 더 자주 재현했다.** Nominal의 선택 불일치는 13.79%에서 2.11%로 줄었지만, 보존−top Brier 차이는 −0.000081 [95% CI −0.000373, +0.000103]이었다. Top과 full의 선택이 다르던 353건 중 282건에서 top은 세 actions에 대한 risk를 사전 tie tolerance 안에서 구별하지 못했다. 반면 full이 선택한 최선과 차선의 median gap도 이 353건에서 0.000000755에 불과했다. 순위 복원과 실현 loss 개선을 분리해야 하는 이유다.

**논문 기여 가능성은 진단 결과에 있다.** 동일한 hypotheses에서 factual truncation, observation likelihood, posterior label error, action 선택, paired Brier를 끝까지 연결했고, 작은 support에서 관찰한 ranking 개선이 latency를 맞추면 사라지는 경계를 측정했다. 반면 보존 방법의 단독 성능 우위와 label stratification 고유의 효용을 뒷받침하는 증거는 약하다. 후속 검증의 우선순위는 action의 utility gap이 실질적으로 큰 조건을 독립적으로 설계한 뒤, observation 전에 알 수 있는 factual signals만으로 tail budget을 배정해 같은 p90에서 loss를 낮출 수 있는지 보는 것이다.


## 무엇을 ‘회복’으로 측정했는가

Factual boundary profile과 충돌 직후 속도를 본 뒤 생긴 16,384개 external hypotheses의 가중치를 w라 하자. 높은 w의 K개를 H, 나머지를 T로 고정했다. 추가 observation Y를 본 뒤에도 집합 H와 T의 구성은 그대로 두고, full external bank의 posterior weights를 likelihood로 다시 계산했다. 각 action에서 factual tail mass와 updated tail mass를 대응시켰다.

**Collective recovery**는 T의 factual mass가 5% 이하이면서 Y 이후 10% 이상이고 동시에 5배 이상 증가한 경우다. **Individual recovery**는 T에 속했던 개별 hypothesis가 Y 이후 mass 1% 이상이고 자신의 factual weight 대비 10배 이상이 된 경우다. 두 정의는 다른 현상을 센다. 여러 작은 hypotheses가 함께 회복되거나, tail 전체의 mass는 작아도 그 안의 한 hypothesis가 두드러질 수 있다. 사건의 빈도는 action별 5 bank pairs × 512 worlds × 16 noise repeats를 분모로 쓴 descriptive rate이며, 이 40,960개가 독립 물체라는 가정은 하지 않았다.

Stick probability의 변화에는 간단한 항등식이 있다. T의 updated mass를 m, updated head와 tail 내부의 stick probabilities를 각각 pH, pT라 하면 **p(full)−p(top) = m × (pT−pH)**이다. 모든 진단 updates에서 이 등식을 float64 오차 10⁻⁹ 이내로 검산했다. 큰 mass가 돌아와도 두 집단이 같은 label을 예측하면 top-K의 label probability는 거의 바뀌지 않는다. 반대로 m과 label 분리가 함께 크면 작은 factual weight가 중요해진다. 모든 mass와 label probability는 지정된 finite nominal bank의 값이다.

## 보존 방식과 실험 설계

총 K slots 중 상위 3K/4개는 원래 weight로 보존한다. 나머지 hypotheses를 각 candidate scene에서 계산한 query stick/non-stick label로 나눈다. 각 그룹에서 K/8개를 factual weights에 비례하는 inverse-CDF stratified resampling으로 뽑고, 각 slot에 그 그룹의 factual mass를 K/8로 나눈 값을 부여한다. 복원 추출이므로 일부 slots는 같은 hypothesis를 나타낸다. 전체 factual mass와 두 label masses를 보존하며, 그룹 내부의 observation likelihood 분포는 표본으로 근사한다. Pilot에서 전체 mass 및 label별 mass를 10⁻¹⁰ 이내로 검산했다. Tail sample plan은 고정했으며 실제 Y나 숨겨진 world label은 selector에 제공하지 않았다.

평가에는 새 worlds 512개, 다른 seed의 independent inner/outer bank pairs 다섯 쌍, actual noise repeats 16개를 사용했다. Nominal, speed threshold, directional damping은 V21과 같은 세 law다. 모든 조건에서 nominal model로 inference했다. Geometry와 noise를 같은 world에서 공유하고 flexible contact class에만 stress를 적용했다. 변경하지 않은 base classes에서 선택과 실측 loss가 같은지 확인했다. 개발용 nominal worlds 32개와 별도 calibration bank는 test에서 사용하지 않았다.

**Mechanism comparison**은 K512/M512에서 top, 보존 방법, full external label evaluator를 비교했다. 세 방법 모두 동일한 hypothetical observations, compact q, conditional Brier score를 쓰고 label posterior support만 바꾼다. Full evaluator도 M512와 finite bank를 쓰는 numerical reference다. 선택 후에는 모든 방법이 같은 full inner decoder로 해당 actual observation을 받아 stick probability를 예측한다. 따라서 아래의 decision loss는 approximation이 action을 바꾸면서 생긴 실현 Brier 차이다. Truncated decoder를 실제 배포한 loss와는 다르다.

**Deployment comparison**은 top과 보존 방법의 M을 2,048로 고정하고 K를 각각 latency로 선택했다. Fixed global은 원래 K128/M4096 direct kernel이다. Pilot 16 worlds에서 넓은 K grid를 측정하고, 다음 16 worlds에서 가까운 후보를 다시 측정해 세 방법의 p90이 서로 가깝도록 선택했다. 이때 Brier loss를 사용하지 않았다. Test에서는 bank·law별 첫 24 worlds × 2회, 방법 순서를 섞어 GPU를 동기화했다. Factual fit, support 구성, acquisition, 선택된 observation update를 포함한 batch-1 float64 latency다. Plan·bank의 static setup과 실제 sensing/reset 시간은 제외했다. 허용한 근접 범위는 각 비교 쌍의 p90 ratio 0.90–1.10이며, 실제 test 비율을 아래에 제시한다.

개발에서 확정한 top은 top3328m2048, 보존 방법은 keep2816m2048다. 두 번째 pilot의 p90은 각각 7.118, 7.093, fixed global 7.107 ms였다.

## 어느 observation에서 mass가 돌아왔는가

아래 집계는 모든 action을 실제 simulator에서 짝지어 측정한 counterfactual evaluation이다. Surprise는 factual predictive distribution 아래에서 actual Y의 negative log marginal density를 whitened coordinates로 계산했다. 차원이 다른 action끼리 절대값을 비교하지 않고 각 law·action 안의 하위/상위 quartile을 비교했다. ‘회복 시 label error’는 해당 조건에서 full external probability와의 mean absolute difference다. 두 방법의 실현 Brier와는 별개의 진단이다.

### 진단용 top K512

| Law · action | 회복 건수 / 40960 | Collective % | Individual % | Surprise Q1 / Q4 회복 % | 회복 시 mean mass | 회복 시 top / 보존 label error | 보존 support에 포함된 회복 mass % |
|---|---|---|---|---|---|---|---|
| nominal · Query view | 232 | 0.57 | 0.74 | 0.00 / 2.11 | 0.366 | 0.0474 / 0.0548 | 14.66 |
| nominal · Factual view | 576 | 1.41 | 2.50 | 0.00 / 5.17 | 0.285 | 0.0226 / 0.0283 | 14.45 |
| nominal · Probe | 384 | 0.94 | 1.87 | 0.00 / 3.75 | 0.261 | 0.0154 / 0.0087 | 11.49 |
| speed_threshold · Query view | 186 | 0.45 | 0.53 | 0.00 / 1.59 | 0.255 | 0.0229 / 0.0298 | 15.90 |
| speed_threshold · Factual view | 524 | 1.28 | 2.60 | 0.00 / 4.53 | 0.282 | 0.0208 / 0.0329 | 15.51 |
| speed_threshold · Probe | 588 | 1.44 | 2.89 | 0.00 / 5.74 | 0.236 | 0.0172 / 0.0153 | 14.90 |
| directional_damping · Query view | 239 | 0.58 | 0.74 | 0.00 / 2.17 | 0.361 | 0.0462 / 0.0331 | 13.92 |
| directional_damping · Factual view | 550 | 1.34 | 2.65 | 0.00 / 4.73 | 0.290 | 0.0234 / 0.0299 | 14.66 |
| directional_damping · Probe | 658 | 1.61 | 2.48 | 0.00 / 6.43 | 0.265 | 0.0067 / 0.0038 | 9.55 |

### latency에 맞춘 top K

| Law · action | 회복 건수 / 40960 | Collective % | Individual % | Surprise Q1 / Q4 회복 % | 회복 시 mean mass | 회복 시 top / 보존 label error | 보존 support에 포함된 회복 mass % |
|---|---|---|---|---|---|---|---|
| nominal · Query view | 0 | 0.00 | 0.00 | 0.00 / 0.00 | — | — / — | — |
| nominal · Factual view | 0 | 0.00 | 0.00 | 0.00 / 0.00 | — | — / — | — |
| nominal · Probe | 0 | 0.00 | 0.00 | 0.00 / 0.00 | — | — / — | — |
| speed_threshold · Query view | 0 | 0.00 | 0.00 | 0.00 / 0.00 | — | — / — | — |
| speed_threshold · Factual view | 0 | 0.00 | 0.00 | 0.00 / 0.00 | — | — / — | — |
| speed_threshold · Probe | 0 | 0.00 | 0.00 | 0.00 / 0.00 | — | — / — | — |
| directional_damping · Query view | 0 | 0.00 | 0.00 | 0.00 / 0.00 | — | — / — | — |
| directional_damping · Factual view | 0 | 0.00 | 0.00 | 0.00 / 0.00 | — | — / — | — |
| directional_damping · Probe | 0 | 0.00 | 0.00 | 0.00 / 0.00 | — | — / — | — |

마지막 열은 full posterior의 실제 tail mass 중 보존 방법이 한 번 이상 뽑은 hypotheses에 속한 비율이다. Resampled weight의 합과는 정의가 다르다. 표의 — 는 해당 event가 없어 조건부 평균을 계산할 수 없다는 뜻이다.

| Law · K512 | Base / flexible 회복 % (query) | Base / flexible 회복 % (factual) | Base / flexible 회복 % (probe) | 실제 top 선택의 회복 % | 실제 보존 선택의 회복 % |
|---|---|---|---|---|---|
| nominal | 0.20 / 0.92 | 1.21 / 1.60 | 0.72 / 1.15 | 0.73 | 0.75 |
| speed_threshold | 0.20 / 0.70 | 1.21 / 1.35 | 0.72 / 2.13 | 0.93 | 1.18 |
| directional_damping | 0.20 / 0.96 | 1.21 / 1.47 | 0.72 / 2.47 | 0.94 | 0.95 |

| Law | K512 sensitivity (≥5%,2× / ≥10%,5× / ≥20%,10×) % | K512 회복 중 post label error≥0.05 % (query / factual / probe) |
|---|---|---|
| nominal | 2.61 / 0.97 / 0.36 | 34.48 / 12.67 / 12.50 |
| speed_threshold | 3.25 / 1.06 / 0.35 | 9.68 / 11.26 / 8.84 |
| directional_damping | 2.98 / 1.18 / 0.46 | 29.71 / 14.18 / 4.86 |

### 큰 회복의 실제 예시

| 조건 · bank/world/repeat (0-based) | Action | Tail factual → updated | 제외된 최대 atom의 factual rank | 그 atom의 post mass / 증가배수 | Full / top / 보존 stick probability | 선택 (top/보존/full) |
|---|---|---|---|---|---|---|
| nominal · 3/128/1 | Query view | 0.0238 → 0.9949 | 673 | 0.2305 / 4208× | 0.1752 / 0.4068 / 0.0205 | Probe / Probe / Probe |
| speed_threshold · 3/165/8 | Factual view | 0.0422 → 0.8954 | 816 | 0.0756 / 1689× | 0.2036 / 0.0802 / 0.2711 | Probe / Probe / Probe |
| directional_damping · 2/128/1 | Query view | 0.0223 → 0.9855 | 722 | 0.1216 / 3274× | 0.2168 / 0.0063 / 0.0225 | Probe / Probe / Probe |

각 law에서 K512 material recovery 중 updated mass가 가장 큰 사례를 사후에 골랐다. 빈도와 평균은 위의 전체 표로 판단해야 한다.

| Law | 회복 atom의 rank | Atom factual → updated | 실제 Y에 대한 whitened RMS residual · 회복 atom / factual MAP | Tail 대 head의 평균 likelihood ratio |
|---|---|---|---|---|
| nominal | 673 | 0.00548% → 23.05% | 0.50 / 5.39 | 7,990× |
| speed_threshold | 816 | 0.00448% → 7.56% | 1.18 / 4.49 | 194× |
| directional_damping | 722 | 0.00371% → 12.16% | 0.53 / 5.36 | 2,972× |

Physical coordinates에서 세 사례의 likelihood를 독립적으로 재계산했다. Nominal query view에서 factual rank 673의 non-stick hypothesis는 actual Y와 0.50 noise standard deviations의 RMS 차이를 보였고, factual MAP의 차이는 5.39였다. 이때 제외 집합 전체가 2.38%에서 99.49%로 커졌다. 그러나 이 world에서 사전에 고른 action은 세 selector 모두 probe였다. 따라서 이 query view의 큰 posterior 변화는 세 정책 사이의 실현 loss 차이를 만들지 않았다.

## 회복이 실제 선택과 loss에 미친 규모

Full-vs-top 비교에서는 같은 hypothetical Y에서 score를 계산했으므로, 이 numerical reference 대비 flip은 해당 support substitution에 귀속할 수 있다. 실제 Y는 action을 고른 뒤에 얻어진다. 따라서 아래의 회복 subgroup과 flip의 연관은 선택 전에 그 Y를 볼 수 있다는 의미가 아니다.

각 bank-world에서 첫 8개 repeats의 세 potential actions 중 하나라도 material recovery가 있으면 회복 집단으로 분류했다. 나머지 8개 repeats로만 집단별 actual Brier를 평가했다. 이 분리는 동일 noise draw로 집단을 정의하고 loss를 측정하는 문제를 줄인다. 동일 world를 공유하므로 개별 observation에서의 인과적 매개효과를 추정하는 설계는 아니다. 집단별 interval은 bank와 world를 재추출한 conditional mean의 descriptive 95% bootstrap이다.

| 조건 · subgroup | bank-world 수 | Top/full flip % | 보존/full flip % | Top/보존 flip % | Top−full Brier | 보존−top Brier [95% CI] |
|---|---|---|---|---|---|---|
| nominal · recovery | 179 | 11.17 | 3.91 | 11.17 | 0.001221 | -0.000750 [-0.004426, +0.000181] |
| nominal · other | 2381 | 13.99 | 1.97 | 14.53 | -0.000006 | +0.000012 [-0.000126, +0.000126] |
| speed_threshold · recovery | 212 | 10.85 | 2.83 | 11.79 | 0.000942 | +0.000159 [-0.000012, +0.000571] |
| speed_threshold · other | 2348 | 10.82 | 1.62 | 10.95 | -0.000017 | +0.000033 [-0.000016, +0.000125] |
| directional_damping · recovery | 202 | 6.93 | 1.98 | 6.44 | 0.001079 | +0.000027 [-0.000035, +0.000189] |
| directional_damping · other | 2358 | 12.21 | 1.87 | 12.21 | -0.000018 | +0.000026 [-0.000015, +0.000104] |

| Law · 전체 worlds, K512 | Top/full flip % | 보존/full flip % | Top → 보존 risk error RMS | Top−full Brier [95% CI] | 보존−full Brier [95% CI] | 보존−top Brier [95% CI] |
|---|---|---|---|---|---|---|
| nominal | 13.79 | 2.11 | 0.000190 → 0.000195 | +0.000122 [-0.000066, +0.000547] | +0.000042 [-0.000250, +0.000440] | -0.000081 [-0.000373, +0.000103] |
| speed_threshold | 10.82 | 1.72 | 0.000215 → 0.000225 | +0.000098 [-0.000084, +0.000533] | +0.000105 [-0.000007, +0.000472] | +0.000007 [-0.000212, +0.000154] |
| directional_damping | 11.80 | 1.88 | 0.000202 → 0.000215 | +0.000103 [-0.000079, +0.000540] | +0.000089 [-0.000017, +0.000461] | -0.000014 [-0.000232, +0.000116] |

Risk error는 action-common offset을 뺀 RMS다. Full과 선택이 동일하면 공통 actual decoder가 받는 observation도 같아 두 방법의 world loss 차이는 0이다. Flip 횟수가 많아도 거의 동률인 action 사이에서 주로 일어나면 평균 Brier는 크게 변하지 않을 수 있다.

| Law | Top/full flip에서 full action gap median | 일치에서 gap median | 다른 action 선택 시 top−full Brier · 마지막 8회 [95% CI] | 다른 action 선택 시 보존−top Brier · 마지막 8회 [95% CI] |
|---|---|---|---|---|
| nominal | 0.000001 | 0.003475 | +0.000576 [-0.000448, +0.003357] | -0.000289 [-0.002050, +0.000654] |
| speed_threshold | 0.000001 | 0.006707 | +0.000577 [-0.000693, +0.004182] | +0.000396 [-0.000082, +0.001225] |
| directional_damping | 0.000001 | 0.005153 | +0.000581 [-0.000578, +0.003855] | +0.000218 [-0.000110, +0.000843] |

### Tie 처리와 tail sampling 대조 실험

사전 규칙은 세 risk의 최대−최소가 10⁻¹² 이하면 query view를 고른다. 아래 sensitivity는 저장한 세 risk를 그대로 두고 해당 tie에서 probe를 고르도록 바꾼 사후 분석이다. Full-bank reference에도 같은 규칙을 적용했다. 마지막 열은 원래 배포 규칙에서 보존과 fixed global이 서로 다른 action을 선택한 횟수다. Law당 분모는 2,560 bank–world 조합이다.

| Law | K512 top/full 불일치 건수 | 그중 top이 three-way tie인 건수 | Tie→probe일 때 top/full 불일치 % | Tie→probe일 때 보존/full 불일치 % | 배포 보존/fixed 불일치 건수 |
|---|---|---|---|---|---|
| nominal | 353 | 282 | 8.52 | 2.11 | 733 |
| speed_threshold | 277 | 228 | 6.60 | 1.72 | 617 |
| directional_damping | 302 | 234 | 8.01 | 1.88 | 672 |

또한 같은 GPU에서 K512/M512, 동일한 exact head와 tail slots, 같은 resampling jitter로 tail을 label별 분할 없이 하나의 weighted distribution에서 뽑았다. 동일한 세 potential actions의 held-out loss로 평가했다. 이 exploratory ablation은 primary experiment가 끝난 뒤 추가했다. 이 대조의 조건은 동일한 K와 M이며 별도 latency calibration은 수행하지 않았다.

| Law | Pooled-tail Brier | Top−pooled Brier [95% CI] | Label별 보존−pooled Brier [95% CI] | Pooled와 label별 보존의 선택 불일치 % |
|---|---|---|---|---|
| nominal | 0.059187 | +0.000111 [-0.000204, +0.000568] | +0.000031 [-0.000344, +0.000463] | 15.20 |
| speed_threshold | 0.111207 | +0.000062 [-0.000249, +0.000522] | +0.000069 [-0.000194, +0.000457] | 11.45 |
| directional_damping | 0.066627 | +0.000083 [-0.000223, +0.000517] | +0.000070 [-0.000202, +0.000456] | 12.62 |

Pooled-tail의 Brier 점추정이 세 law에서 모두 label별 보존보다 낮았고, 세 paired intervals 모두 0을 포함했다. 두 방식은 선택의 약 11–15%에서 서로 달랐다. Label별 보존이 Brier를 줄이는 고유한 원인이라고 판정할 수 없다.

## Held-out Brier loss와 GPU 비용

| Policy | Nominal | Speed threshold | Directional damping |
|---|---|---|---|
| top512m512 | 0.059298 | 0.111269 | 0.066710 |
| keep512m512 | 0.059218 | 0.111276 | 0.066697 |
| E-E-mean | 0.059176 | 0.111171 | 0.066608 |
| top3328m2048 | 0.059449 | 0.111890 | 0.066868 |
| keep2816m2048 | 0.059449 | 0.111890 | 0.066868 |
| fixed | 0.059198 | 0.111284 | 0.066375 |
| probe | 0.061495 | 0.114242 | 0.067783 |

| Primary paired contrast | Crossed 98.75% CI | World-only 98.75% CI |
|---|---|---|
| 보존−top · nominal | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |
| 보존−fixed · nominal | +0.000251 [-0.000812, +0.001763] | +0.000251 [-0.000443, +0.001045] |
| 보존−top · unseen 평균 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |
| 보존−fixed · unseen 평균 | +0.000549 [-0.000760, +0.002328] | +0.000549 [-0.000241, +0.001476] |

Primary family의 네 contrasts에 Bonferroni를 적용했다. 20,000 bootstrap draws에서 bank pair와 world를 독립적으로 resample하고 law·method·sensor noise의 pairing을 유지했다. Unseen은 V21에서 정의한 두 stress laws의 동일가중 평균이다. Stress laws 자체를 모집단에서 resample한 interval은 아니다.

| 조건 | Top p50 / p90 ms | 보존 p50 / p90 ms | Fixed p50 / p90 ms | Probe p50 / p90 ms | 보존/top p90 | 보존/fixed p90 | top/fixed p90 |
|---|---|---|---|---|---|---|---|
| nominal | 7.047 / 7.155 | 7.033 / 7.131 | 7.049 / 7.162 | 1.694 / 1.730 | 0.997 | 0.996 | 0.999 |
| unseen | 7.059 / 7.156 | 7.039 / 7.144 | 7.035 / 7.155 | 1.690 / 1.730 | 0.998 | 0.999 | 1.000 |
| speed_threshold | 7.047 / 7.137 | 7.050 / 7.140 | 7.035 / 7.155 | 1.688 / 1.730 | 1.000 | 0.998 | 0.998 |
| directional_damping | 7.069 / 7.166 | 7.028 / 7.144 | 7.035 / 7.153 | 1.692 / 1.731 | 0.997 | 0.999 | 1.002 |

| Bank | 보존−top · nominal / threshold / directional | 보존−fixed · 같은 순서 |
|---|---|---|
| 1 | +0.000000 / +0.000000 / +0.000000 | +0.000958 / +0.001587 / +0.001635 |
| 2 | +0.000000 / +0.000000 / +0.000000 | -0.000181 / +0.000510 / -0.000374 |
| 3 | +0.000000 / +0.000000 / +0.000000 | -0.000047 / -0.000707 / -0.000373 |
| 4 | +0.000000 / +0.000000 / +0.000000 | +0.000050 / +0.000848 / +0.000567 |
| 5 | +0.000000 / +0.000000 / +0.000000 | +0.000475 / +0.000792 / +0.001006 |

![Recovery by action and held-out loss](results/v22.png)

## 검증 자료와 범위

원래 simulator는 local impulse, 하나의 추가 observation, 동일 sensing cost를 사용한다. 복원한 mass는 full external posterior가 부여한 값이며 실제 hidden physics가 정답이라는 뜻은 아니다. Synthetic stress에서 nominal posterior가 크게 확신하는 잘못된 hypotheses가 나타날 수도 있다. Real imagery, 개입 비용, trajectory reward로 일반화하려면 별도 검증이 필요하다.

서버 경로는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/tail_recovery_v22`이다. `PROTOCOL.md`, `policy22.py`, `run22.py`, `stats22.py`에 설계·구현·primary analysis를 기록했다. `subgroup22.py`의 threshold sensitivity와 conditional CIs는 추가 진단이다. `choices_*.npz`는 public-only 선택과 risks, `evaluation_*.npz`는 paired losses와 실제 intervention별 mass, `columns.json`은 그 열의 정의다. 각 단계의 hash는 results 아래 plan.json, choices_freeze.json, sources.json에 있다. 재현할 때는 이전 v18–v21 dependencies의 상대 경로를 유지하고 새로운 실험 폴더에서 v18 guard를 통해 run22.py를 실행한 뒤 compute-node CPU에서 stats22.py, subgroup22.py, example22.py를 실행한다. 추가 대조는 같은 GPU에서 control22.py를 실행한 뒤 CPU에서 control_stats.py를 실행하고, 마지막에 report22.py로 표와 그림을 생성한다.

## 문헌 검토와 project review

V21에서 사용한 numerical integration·experimental design 원문 네 편에, 이번 구현과 직접 연결되는 particle resampling 및 coreset 원문 두 편을 추가했다. 아래의 표와 비교는 이 여섯 편을 대상으로 한다. V21·V22의 수치는 프로젝트 자체 실험이다.

### 1. 논문별 비교와 2. 주요 주제

| 연구 | Research question 및 methodology | Sample · data source | Key findings와 conclusion | 적용 한계 |
|---|---|---|---|---|
| Douc, Cappé & Moulines, *Comparison of Resampling Schemes for Particle Filtering* | 동일한 weighted particles를 resampling할 때 방법에 따라 conditional variance가 어떻게 바뀌는가? Multinomial, residual, stratified, systematic 방식을 수학적으로 비교했다. | Weighted empirical measures, 이론적 variance 비교, systematic 방식의 counterexample. 임상·로봇 표본은 해당되지 않는다. | Residual·stratified 방식은 논문 조건에서 multinomial 대비 variance 개선 성질을 갖는다. Systematic 방식에는 일반화되지 않는 예가 있다. | 이는 resampling 단계의 결과다. Label별 tail을 뽑은 뒤 nonlinear update·discrete action을 거치는 V22 Brier gain은 직접 평가해야 한다. [원문](https://arxiv.org/pdf/cs/0507025) |
| Campbell & Broderick, *Automated Scalable Bayesian Inference via Hilbert Coresets* | 큰 dataset의 likelihood를 weighted subset으로 근사할 수 있는가? Log-likelihood functions의 Hilbert-space approximation을 구성했다. | Real/synthetic datasets의 Gaussian inference, logistic 및 Poisson regression, von Mises–Fisher mixture. | Benchmark에서 작은 weighted data subset으로 posterior를 근사하고 inference 비용을 줄일 수 있었다. | 선택되는 단위는 observations다. V22는 hypotheses를 줄이고 아직 보지 않은 observation에 대한 label probability를 계산한다. [원문](https://jmlr.org/papers/v20/17-613.html) |
| Bartuska et al., *Multilevel randomized quasi-Monte Carlo estimator for nested integration* | Inner approximation과 outer integration 비용을 tolerance에 맞추는가? Multilevel randomized QMC와 bias/variance 분석. | Exact/inexact EIG examples 두 개, 첫 pilot 일부에 1,000 randomizations. | 조건을 만족하는 nested integrals에서 work–accuracy 관계를 분석하고 예제로 검증했다. | Discontinuous contact event의 acquisition regret에는 해당 regularity 조건을 별도 검증해야 한다. [원문 §§3–5](https://arxiv.org/html/2412.07723v1#S5) |
| Sorokin et al., *Fast Bayesian Multilevel Quasi-Monte Carlo* | 다음 sample doubling의 uncertainty 감소를 예측해 계산을 배분할 수 있는가? GP Bayesian cubature와 cost-aware allocation. | Problem·budget별 250 trials, option pricing·elliptic PDE·Darcy flow 등. | 실험한 조건에서 single-sequence Bayesian 방식이 replicated MLQMC보다 효율적인 사례를 보고했다. | Hyperparameters와 kernel에 의존하는 scalar error와 one-step observation 이후 Brier는 다른 endpoint다. [원문 §3](https://arxiv.org/html/2510.24604v1#S3) |
| Dinkel et al., *Efficient Bayesian Optimal Experimental Design for Expensive Computational Models over Finite Design Sets* | 적은 forward calls로 높은 EIG의 design을 찾는가? Common random numbers, conditional averaging, adaptive design elimination. | Synthetic, pharmacokinetic, lung, viscoplastic 사례 네 개, 일부 selection curves에 1,000 runs. | 유망한 후보에 계산을 배분해 높은 선택 정확도에 필요한 model evaluations를 줄였다. | Forward calls와 GPU batch-1 latency의 관계, EIG ranking과 Brier 선택은 비용·측정 단위가 다르다. [원문 §§2–3](https://arxiv.org/html/2607.16933v1#S3) |
| Zhu & Hainy, *Sequential Bayesian Experimental Design for Prediction in Physical Experiments Informed by Computer Models* | 다음 physical observation의 위치가 future prediction을 개선하는가? KOH model, predictive MI, local complexity, Gaussian mixture compression. | Toy study 및 JAK–STAT5 실제 사례. 19개 field time points 중 4개로 초기화해 10 rounds를 평가했다. | 실제 사례에서 MSE와 CRPS에 따라 유리한 criterion이 달랐다. | 보간한 pseudo ground truth와 연속 response의 결과를 stick event로 옮기는 데에는 별도 자료가 필요하다. [원문 §§4–5](https://arxiv.org/html/2603.16756v1#S5.SS4) |

**3. Research knowledge map.** Douc 등의 weighted resampling과 Campbell·Broderick의 weighted approximation은 저장할 support를 고르는 두 접근이다. Bartuska와 Sorokin은 근사 이후의 integration error와 계산 예산을 연결한다. Dinkel은 이 error가 action ranking에 미치는 영향을, Zhu·Hainy는 observation 선택이 후속 prediction에 미치는 영향을 다룬다. V22가 확인하는 연결은 factual support 제거 → 실제 Y의 likelihood reweighting → label probability 차이 → acquisition 순위 변화 → 공통 decoder의 Brier 차이다. 첫 두 단계의 수학적 근거와 마지막 단계의 실제 효과에는 서로 다른 검증이 필요하다. [Douc et al.](https://arxiv.org/pdf/cs/0507025) · [Campbell & Broderick](https://jmlr.org/papers/v20/17-613.html) · [Bartuska et al.](https://arxiv.org/html/2412.07723v1) · [Sorokin et al.](https://arxiv.org/html/2510.24604v1) · [Dinkel et al.](https://arxiv.org/html/2607.16933v1) · [Zhu & Hainy](https://arxiv.org/html/2603.16756v1)

**4. Gap과 5. 상충하는 결과의 비교.** 이 corpus의 직접 근거는 resampling variance, posterior approximation, EIG selection, continuous predictive design에 집중돼 있다. 동일한 contact world에서 factual rank가 낮은 hypothesis의 사후 회복과 최종 Brier를 연결한 직접 비교는 위 논문들에서 확인되지 않았다. Zhu·Hainy에서 평가 metric이 바뀌면 criterion 순위가 달라진다는 결과는 V22에서도 recovered mass, label error, loss를 따로 측정할 이유다. 같은 endpoint에서 이 여섯 논문이 서로 반대 결론을 냈다는 근거는 없다. [Douc et al.](https://arxiv.org/pdf/cs/0507025) · [Dinkel et al.](https://arxiv.org/html/2607.16933v1#S3) · [Zhu & Hainy](https://arxiv.org/html/2603.16756v1#S5.SS4)

**6. Methodology와 7. Evidence synthesis.** Conditional variance의 ordering에는 가정을 명시한 이론이 있다. Numerical examples의 효율에는 반복 trials에 근거한 경험적 증거가 있다. Predictive gain은 사용한 data source와 scoring rule에 의존한다. Counterexample, 250 algorithm trials, 19 physical time points는 같은 모집단의 표본들이 아니므로 pooled effect size를 계산하지 않았다. V22에서는 같은 public evidence에서 approximation만 교체한 paired comparison과 실제 observation 이후에 정의한 descriptive subgroup을 구분한다. [Douc et al.](https://arxiv.org/pdf/cs/0507025) · [Sorokin et al.](https://arxiv.org/html/2510.24604v1#S3) · [Zhu & Hainy](https://arxiv.org/html/2603.16756v1#S5.SS4)

### 8. 근거에서 도출한 연구 질문 10개

| 질문 | 중요한 gap | Variables와 검증 방법 |
|---|---|---|
| 1. 실제로 사후에 회복된 mass가 label을 바꾸는 조건은? | Factual truncation error는 실제 observation likelihood에 의존한다. [V21](../external_label_v21/REPORT_KO.md) | Full update로 tail mass와 tail/head label 분리를 함께 측정. V22에서 직접 실행했다. |
| 2. Tail 보존이 단순한 resampling보다 우수한가? | Stratified resampling 자체에는 선례가 있다. [Douc et al.](https://arxiv.org/pdf/cs/0507025) | 동일 head 비율에서 label 분할 유무, multinomial/stratified를 교차하고 같은 GPU p90 비교. |
| 3. Tail slots의 적절한 비율은 얼마인가? | 계산을 늘릴 때의 이득은 현재 mass만으로 정해지지 않는다. [Sorokin et al.](https://arxiv.org/html/2510.24604v1#S2) | Head fraction과 M을 독립적으로 조절하고 separate pilot과 held-out loss로 검증. |
| 4. Factual evidence만으로 회복 위험을 예측할 수 있는가? | 사후 subgroup은 배포 가능한 신호로 쓰기 어렵다. [V20](../factual_switch_v20/REPORT_KO.md) | Train에서 posterior-predictive risk를 추정하고, disjoint worlds에서 AUROC·calibration·gate utility 비교. |
| 5. 회복한 가설을 label 대신 sensor response로 stratify하면 좋은가? | Weighted approximation의 목적에 따라 필요한 support가 달라진다. [Campbell & Broderick](https://jmlr.org/papers/v20/17-613.html) | Action별 whitened centers clustering, label×center stratification, GPU 생성 비용 비교. |
| 6. Probe의 surprising velocity가 회복을 가장 강하게 드러내는가? | 서로 다른 design이 다른 정보를 드러낼 수 있다. [Dinkel et al.](https://arxiv.org/html/2607.16933v1#S2) | 동일 world에 세 actions를 적용해 action 안의 surprise percentile과 recovery를 paired 비교. |
| 7. Action gap이 작을수록 같은 label 오차가 더 자주 순위를 바꾸는가? | Utility의 상대 차이가 design 선택을 결정한다. [Dinkel et al.](https://arxiv.org/html/2607.16933v1#S2.SS3) | Finite-bank full risk gap strata에서 flip과 downstream regret를 independently 평가. |
| 8. Recovery가 반복 intervention에도 지속되는가? | 이번 one-step update와 sequential experimental design은 다른 horizon이다. [Zhu & Hainy](https://arxiv.org/html/2603.16756v1#S3) | Intervention sequence, posterior support overlap, trajectory-level score를 조절한다. |
| 9. Bank와 tail sampling seed의 어느 쪽이 더 큰 변동원인가? | Integration randomization과 posterior discretization이 별개다. [Bartuska et al.](https://arxiv.org/html/2412.07723v1#S5) | Bank·acquisition·tail draw seed를 완전 교차하고 world 수준 효과를 추정. |
| 10. 실제 물체·sensor에서 같은 현상이 나타나는가? | 실측과 simulator가 다른 physical law를 가질 수 있다. [Zhu & Hainy](https://arxiv.org/html/2603.16756v1#S2) | 같은 물체에 여러 interventions를 가하고 measurement noise·reset costs와 calibration을 함께 추정. |

**9. Research assistant review.** 중복을 허용한 inverse-CDF stratified resampling과 label mass 보존만으로 novelty를 주장하기 어렵다. 실제 recovery 빈도를 셀 때는 bank와 noise를 별도의 독립 물체처럼 세어서는 안 된다. Held-out Brier 차이, interleaved latency, material-recovery 정의에 대한 민감도, full reference의 finite-bank error가 모두 논문 심사의 쟁점이다. 이 여섯 편만으로 분야 전체의 novelty를 판정할 수 없다. [Douc et al.](https://arxiv.org/pdf/cs/0507025) · [Campbell & Broderick](https://jmlr.org/papers/v20/17-613.html) · [V21](../external_label_v21/REPORT_KO.md)

