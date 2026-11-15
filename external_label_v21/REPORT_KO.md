# v21 · Independent bank seeds와 unseen contact laws에서 external label posterior의 가치

**V20에서 관찰한 external label-source의 Brier 이득은 V21에서 재현되지 않았다.** 새 512 worlds, world당 actual noise 16회, 다섯 independent bank pairs로 반복했다. Full external label posterior를 쓴 여덟-cell factorial contrast의 E−C 차이는 nominal +0.000330 [−0.001188, +0.001938], unseen 두 laws 평균 −0.000198 [−0.001539, +0.001150]이었다. 값이 작을수록 external이 유리하다. V20의 nominal point estimate −0.001513과 부호가 달라졌으나 두 버전 간 차이에 대한 별도 검정을 한 것은 아니다.

**GPU 비용을 줄이는 데에는 성공했다.** External bank의 16,384 hypotheses 중 factual weight가 큰 512개만 hypothetical label update에 사용하면, 같은 M512에서 nominal p90이 full external의 8.96 ms에서 4.94 ms로 약 45% 감소했다. Top-K와 full external이 같은 action을 고른 비율은 nominal 87.7%, speed threshold 90.2%, directional damping 89.2%였다. 선택이 완전히 같지 않아도 두 정책의 평균 Brier 차이는 각 law에서 절대 0.000057 이하로 관측됐다. 이 수치로 loss equivalence를 선언할 tolerance는 사전에 설정하지 않았다.

**Fixed global mixture와 GPU latency를 맞춰도 held-out Brier superiority는 확인되지 않았다.** 별도의 fine calibration에서 고정한 K3072/M2048은 nominal p90 6.90 ms 대 fixed 7.09 ms, unseen p90 6.89 ms 대 fixed 7.10 ms로 사전 정의한 90–100% 범위에 들었다. Nominal에서 Brier는 0.061701 대 0.062039, 차이 −0.000338 [−0.001880, +0.000379]이었다. Unseen에서는 0.096533 대 0.096334, 차이 +0.000199 [−0.000587, +0.001160]이었다. CI는 두 supplemental endpoints에 대해 보정한 97.5%다. 추가 계산으로 근접 latency를 확보했지만 두 조건에서 모두 loss를 낮춘다는 기준은 충족하지 못했다.

두 unseen laws를 따로 보면 이유가 더 잘 드러난다. Directional damping에서는 full E−C가 다섯 seeds 모두 음수였으나 같은 worlds의 불확실성을 반영한 95% CI가 0을 포함했다. Speed threshold는 flexible class에 새로운 sticking 영역을 만들어 이 집단의 stick rate가 nominal 0%에서 22.7%가 됐다. Nominal inference를 유지한 채 동일한 budgeting으로 이를 해결할 수 있다는 근거는 얻지 못했다. Nominal에서 full E−C가 유리했던 bank pair는 1/5였다.

**논문 기여 판단.** 이 데이터로 지지되는 주장은 factual weight truncation이 이 simulator에서 external label evaluation을 빠르게 근사한다는 것이다. 독립적인 방법 논문의 핵심으로 삼을 “같은 compute에서 fixed global보다 정확하다”는 주장은 보류하는 편이 타당하다. 연구를 이어간다면 우선 top-K가 버린 hypotheses가 다음 observation에서 큰 posterior mass를 얻는 사례에 집중하고, generic weighted resampling 및 label-stratified reduction과 같은 p90·같은 M에서 비교할 가치가 있다. 현재 결과는 bank·law·latency를 바꿨을 때 단일-seed의 positive result가 어떻게 달라지는지 보이는 replication 및 failure analysis로 활용할 수 있다.


## 연구 질문과 비교의 단위

Factual boundary profile과 collision velocity를 관측한 뒤 query view, 더 정밀한 factual view, 반대 방향의 probe 중 하나를 고른다. 관심 대상은 query 위치에서 물체가 stick할 확률이다. 모든 방법은 선택한 observation을 받아 동일한 full inner bank로 이를 예측한다. Brier loss는 predicted probability와 실제 binary stick label 차이의 제곱이며 낮을수록 좋다. 여기서 ‘실제’는 hidden simulator world에서 실현한 observation과 event를 가리킨다.

V20의 주된 단서는 hypothetical Y를 받았을 때 label probability를 어느 posterior가 제공하느냐에 있었다. Compact posterior C는 full inner bank에서 resample한 256 hypotheses로 구성하고 두 replicas를 쓴다. External posterior E는 별도의 16,384 hypotheses에 같은 factual evidence를 적용한다. (1) Y를 C/E 중 어디서 생성하는지, (2) Y에 대한 label probability p를 C/E 중 어디서 계산하는지, (3) Bernoulli label을 한 번 뽑는지 또는 정확히 평균하는지를 교차한 여덟 cells를 V20과 같은 M512에서 재실행했다. 모든 cells에서 채점하는 예측 q는 compact decoder가 낸다. Conditional mean loss는 p(1−q)²+(1−p)q²다.

**Label-source effect**는 같은 Y law와 label mode에서 E−C로 바꾼 뒤의 *선택 후 held-out Brier loss 차이*를 계산하고, 네 조합에 같은 가중치를 준 값이다. **Deployment contrast**는 고정한 저비용 policy의 loss에서 fixed global mixture(K128/M4096)의 loss를 뺀 값이다. 두 효과는 서로 다른 질문에 답한다.

## Worlds, contact laws와 독립성

개발용 nominal worlds 128개와 held-out worlds 512개를 분리했다. 각 test world에는 공통 actual observation noise를 16회 적용했다. 하나의 world에서 nominal과 두 stress laws를 모두 생성해 geometry·material draws와 noise가 같은 paired comparison을 만들었다. Inference bank와 decoder는 모두 nominal model을 사용했다. 각 bank pair에는 서로 다른 scrambled Sobol inner·outer banks를 만들었고, 다섯 pairs 간에도 다른 seeds를 사용했다. QMC acquisition plans와 representative quantiles는 고정해 bank가 바뀔 때의 변동을 측정했다.

| Data-generating law | Flexible class의 tangential impulse | 이 stress가 바꾸는 것 |
|---|---|---|
| nominal | vₜ sigmoid[b₀+b₁ log((abs(vₜ)+0.02)/0.5)] | Inference와 일치하는 control |
| speed_threshold | sign(vₜ) min(abs(vₜ), μ max(jₙ,0)), μ=0.05+0.4 sigmoid(b₀+1)/(1+abs(vₜ)/(0.15+0.5 sigmoid(b₁))) | 속도가 증가하면 friction coefficient가 감소하고 flexible class에도 exact sticking이 생긴다. |
| directional_damping | vₜ sigmoid[b₀+1.2 tanh(3vₜ)+b₁vₙ/(1+abs(vₙ))] | 진행 방향의 부호와 normal velocity가 감쇠 비율을 바꾼다. |

두 stress laws는 pilot과 model fitting에서 제외했다. Base frictionless/Coulomb classes는 같은 physics를 유지한다. ‘Unseen 평균’은 두 지정 stress에 같은 가중치를 준 값이다. 두 결과가 모든 contact laws를 대표한다고 해석할 수는 없다.

512 worlds 중 flexible class는 247개다. 세 law 순서로 전체 stick rate는 0.121 / 0.230 / 0.121, flexible class만의 stick rate는 0.000 / 0.227 / 0.000였다. Base classes에서는 세 조건의 hidden label, action choices, actual losses가 같음을 검증했다.

## External posterior의 비용을 줄인 방법

각 world에서 full external bank의 factual weights를 한 번 계산한다. 그중 weight가 큰 K개 hypotheses를 GPU topk로 취하고, query/factual/probe 각각의 hypothetical observation에 대해 선택된 hypotheses에서 원래 prior weight와 likelihood를 곱해 다시 정규화한다. 얻은 p로 conditional Brier risk를 계산한다. Hypothetical Y는 계속 full external bank에서 생성하므로 label posterior의 support만 줄인다. Factual weight가 작더라도 특정 Y에는 likelihood가 클 수 있어 omitted mass와 action agreement를 따로 측정했다. K=N으로 둔 validation world에서 full implementation과 비교했다.

초기 protocol은 K={128,512,2048}, M={512,2048} 후보 중 fixed global p90 이하이면서 개발 loss가 가장 작은 구성을 선택했다. Pilot timing은 별도 bank pair와 24개 개발 worlds, warmup 후 방법별 3회씩 측정했다. 개발 loss는 128 worlds에서 평가했다. 이 개발 평균으로 최종 방법을 정한 후 test outcomes를 열었다.

| 개발 후보 | 개발 mean Brier | Pilot GPU p90 ms |
|---|---|---|
| top128m512 | 0.0677659 | 5.29 |
| top128m2048 | 0.0649046 | 5.52 |
| top512m512 | 0.0628673 | 5.36 |
| top512m2048 | 0.0636445 | 5.70 |
| top2048m512 | 0.0628673 | 5.68 |
| top2048m2048 | 0.0636149 | 6.80 |
| fixed | 0.0640405 | 7.65 |

선택된 구성은 top512m512이다. 전체 bank를 유지했을 때 full risk와의 최대 차이는 2.776e-17였다.

## Label-source replication과 primary endpoints

World와 bank pair를 독립적으로 재추출하는 crossed paired bootstrap을 20,000회 수행했다. 각 draw 안에서 law·method·noise pairing을 유지하며 noise repeats를 독립 worlds로 세지 않았다. 사전 정의한 네 contrasts의 CI는 Bonferroni 98.75%이다. Bank 수가 다섯이므로 seed 모집단에 대한 bootstrap tail 추정은 거칠다. 옆에 bank를 고정한 world-only interval과 seed별 값을 함께 제공한다.

| Primary contrast | Crossed 98.75% CI | World-only 98.75% CI |
|---|---|---|
| Label E−C · nominal | +0.000330 [-0.001188, +0.001938] | +0.000330 [-0.000480, +0.001154] |
| Label E−C · unseen 평균 | -0.000198 [-0.001539, +0.001150] | -0.000198 [-0.001101, +0.000648] |
| Cheap−fixed · nominal | -0.000309 [-0.002145, +0.000867] | -0.000309 [-0.001209, +0.000385] |
| Cheap−fixed · unseen 평균 | +0.000455 [-0.001311, +0.002387] | +0.000455 [-0.000197, +0.001183] |

| Bank inner / outer | Label E−C · nominal / threshold / directional | Cheap−fixed · 같은 순서 |
|---|---|---|
| 3301001 / 4401007 | 0.000321 / -0.000064 / -0.000243 | -0.001026 / -0.000524 / -0.002303 |
| 3311010 / 4413014 | -0.000809 / 0.000349 / -0.000458 | 0.000593 / 0.003806 / 0.001029 |
| 3321019 / 4425021 | 0.001093 / -0.001101 / -0.000236 | -0.001218 / 0.001406 / 0.000763 |
| 3331028 / 4437028 | 0.000830 / 0.001129 / -0.000026 | 0.000633 / 0.000513 / 0.000195 |
| 3341037 / 4449035 | 0.000216 / -0.000657 / -0.000670 | -0.000527 / -0.000538 / 0.000208 |

| Method | Nominal Brier ↓ | Speed threshold Brier ↓ | Directional damping Brier ↓ |
|---|---|---|---|
| C-C-sample | 0.062285 | 0.118956 | 0.076437 |
| C-C-mean | 0.061400 | 0.117198 | 0.075602 |
| C-E-sample | 0.062869 | 0.118805 | 0.076327 |
| C-E-mean | 0.061975 | 0.118025 | 0.075285 |
| E-C-sample | 0.062825 | 0.119744 | 0.076652 |
| E-C-mean | 0.061423 | 0.117927 | 0.075731 |
| E-E-sample | 0.062689 | 0.118428 | 0.076228 |
| E-E-mean | 0.061719 | 0.118292 | 0.075275 |
| fixed | 0.062039 | 0.117315 | 0.075353 |
| top512m512 | 0.061730 | 0.118248 | 0.075331 |
| probe | 0.061406 | 0.124521 | 0.074495 |

| Secondary contrast | Nominal · 95% CI | Speed threshold · 95% CI | Directional damping · 95% CI |
|---|---|---|---|
| Label E−C | +0.000330 [-0.000801, +0.001499] | -0.000069 [-0.001541, +0.001493] | -0.000327 [-0.001482, +0.000849] |
| Cheap−fixed | -0.000309 [-0.001531, +0.000622] | +0.000932 [-0.000646, +0.003104] | -0.000022 [-0.001435, +0.001067] |
| Top-K−full E · E-Y/mean M512 | +0.000011 [-0.000010, +0.000037] | -0.000044 [-0.000163, +0.000012] | +0.000057 [-0.000071, +0.000404] |
| Top-K−compact C · E-Y/mean M512 | +0.000307 [-0.001058, +0.001686] | +0.000321 [-0.001510, +0.002423] | -0.000399 [-0.001974, +0.001111] |

같은 M512에서 top-K와 full E의 Top-1 agreement는 세 조건에서 87.66 / 90.23 / 89.18%였다. Action별 risk에서 공통 offset을 제거한 RMS 차이는 0.000206 / 0.000229 / 0.000215였다. Full-vs-compact pair는 같은 Y, q, mean mode에서 label source만 다르다.

선택 K의 omitted factual mass median은 0.00540 / 0.00598 / 0.00522, p90은 0.03275 / 0.03387 / 0.03247, 최대는 0.05148 / 0.05304 / 0.05236였다. Flexible class만 본 E−C는 0.000919 / 0.000092 / -0.000443, cheap−fixed는 -0.000738 / 0.001836 / -0.000142였다. 이 subgroup은 mechanistic diagnostic이다.

![Bank sensitivity and deployment contrast](results/v21.png)

## Online compute와 latency 근접 대조군

RTX PRO 6000 Blackwell 한 장에서 batch 1, float64로 각 방법을 섞어 실행했다. 각 bank·law에서 held-out 첫 16 worlds×2회 측정했다. 매번 factual fit, 필요한 bank update, top-K와 risk 계산, observation 선택 및 선택된 actual observation update까지 포함했다. Static bank construction, sensing 및 reset 비용은 제외했다. Timing을 측정한 모든 action은 미리 저장한 policy choices와 일치했다. Unseen p90은 두 law의 측정치를 합쳐 산출했다.

| 조건 | 방법 | GPU p50 ms | GPU p90 ms |
|---|---|---|---|
| nominal | top512m512 | 4.83 | 4.94 |
| nominal | fixed | 7.01 | 7.15 |
| nominal | E-E-mean | 8.71 | 8.96 |
| nominal | probe | 1.68 | 1.72 |
| unseen | top512m512 | 4.82 | 4.93 |
| unseen | fixed | 7.02 | 7.15 |
| unseen | E-E-mean | 8.69 | 9.02 |
| unseen | probe | 1.68 | 1.72 |
| speed_threshold | top512m512 | 4.80 | 4.94 |
| speed_threshold | fixed | 7.03 | 7.16 |
| speed_threshold | E-E-mean | 8.70 | 9.02 |
| speed_threshold | probe | 1.68 | 1.73 |
| directional_damping | top512m512 | 4.82 | 4.92 |
| directional_damping | fixed | 7.01 | 7.15 |
| directional_damping | E-E-mean | 8.67 | 9.02 |
| directional_damping | probe | 1.68 | 1.71 |

첫 development 결과의 clock time이 cap에 여유가 있어, held-out losses를 확인하기 전에 별도 latency protocol을 고정했다. 다른 calibration bank에서 M을 4,096까지, K를 4,096까지 넓혀 실제 p90이 fixed global 이하이며 가장 가까운 구성을 택했다. Test에서는 두 방법을 같은 timing pass에서 재측정했다. 사전에 정한 근접 구간은 p90 ratio 0.90–1.00이다. 이 추가 분석은 같은 test worlds와 같은 다섯 banks를 사용한다.

| 추가 calibration 후보 | GPU p90 ms |
|---|---|
| top512m1024 | 4.98 |
| top512m2048 | 5.21 |
| top512m4096 | 5.61 |
| top2048m1024 | 5.54 |
| top2048m2048 | 6.31 |
| top2048m4096 | 7.93 |
| top4096m1024 | 6.18 |
| top4096m2048 | 7.59 |
| top4096m4096 | 10.41 |
| fixed | 7.12 |

추가 선택은 top2048m2048였다. Nominal/두 unseen 순서로 이 방법의 Brier loss는 0.061701 / 0.117715 / 0.075350, fixed global은 0.062039 / 0.117315 / 0.075353였다.

| Held-out 조건 | 후보 p50 / p90 ms | Fixed p50 / p90 ms | p90 ratio | Brier 차이와 CI |
|---|---|---|---|---|
| nominal | 6.20 / 6.33 | 7.03 / 7.13 | 0.887 | -0.000338 [-0.001880, +0.000379] |
| unseen | 6.21 / 6.31 | 7.02 / 7.13 | 0.885 | +0.000199 [-0.000587, +0.001160] |
| speed_threshold | 6.21 / 6.31 | 7.02 / 7.15 | 0.882 | +0.000400 [-0.000661, +0.001687] |
| directional_damping | 6.21 / 6.32 | 7.02 / 7.12 | 0.887 | -0.000003 [-0.000685, +0.000644] |

추가 비교의 nominal·unseen 평균은 두 endpoint에 대해 보정한 97.5% CI, 각 stress law는 descriptive 95% CI다. Pilot의 근접 latency가 새 worlds에서 같은지와 Brier 우위가 있는지를 각각 확인해야 한다.

| Bank | 추가 후보−fixed Brier · nominal / threshold / directional | 주 후보 GPU p90 ms · 같은 순서 | 주 비교 fixed GPU p90 ms |
|---|---|---|---|
| 1 | -0.000473 / 0.000001 / -0.000926 | 4.92 / 4.87 / 4.88 | 7.06 / 7.10 / 7.08 |
| 2 | 0.000161 / 0.001782 / 0.000498 | 4.87 / 4.91 / 4.91 | 7.03 / 7.10 / 7.11 |
| 3 | -0.001586 / 0.000153 / -0.000225 | 4.97 / 4.96 / 4.97 | 7.16 / 7.23 / 7.16 |
| 4 | 0.000091 / 0.000731 / 0.000185 | 5.00 / 4.95 / 4.92 | 7.20 / 7.16 / 7.15 |
| 5 | 0.000118 / -0.000667 / 0.000454 | 4.90 / 4.88 / 4.89 | 7.09 / 7.08 / 7.09 |

각 bank를 한 번씩 제외한 E−C nominal 평균의 범위는 0.000140 / 0.000615였고, 두 unseen의 동일가중 E−C 평균 범위는 -0.000385 / -0.000080였다.

## Fine calibration으로 실제 latency 근접 여부 재확인

위의 coarse calibration은 p90 ratio 0.885로 사전 근접 범위에 도달하지 못했다. 어느 held-out loss도 확인하기 전에 마지막 latency-only grid를 기록했다. 또 다른 별도 calibration bank와 32개 nominal development worlds에서 M2048, K2048부터 4352까지 256 간격을 평가하고, cap 이하인 후보 중 0.97에 가장 가까운 비율을 우선했다. 이후에는 test timing에 따라 다시 조정하지 않았다. 이 추가 비교는 같은 held-out 사례를 쓰는 exploratory 분석이다.

선택된 후보 top3072m2048의 pilot ratio는 0.981였다. Held-out에서는 각 조건의 새 interleaved fixed baseline과 다시 비교했다.

| 조건 | Brier 후보 / fixed | 후보 / fixed p90 ms | p90 ratio | 후보−fixed Brier CI |
|---|---|---|---|---|
| nominal | 0.061701 / 0.062039 | 6.90 / 7.09 | 0.973 | -0.000338 [-0.001880, +0.000379] |
| unseen | 0.096533 / 0.096334 | 6.89 / 7.10 | 0.970 | +0.000199 [-0.000587, +0.001160] |
| speed_threshold | 0.117715 / 0.117315 | 6.89 / 7.08 | 0.973 | +0.000400 [-0.000661, +0.001687] |
| directional_damping | 0.075350 / 0.075353 | 6.89 / 7.10 | 0.970 | -0.000003 [-0.000685, +0.000644] |

Nominal 및 unseen 평균의 interval은 이 추가 비교 두 개에 대한 97.5% CI이며, coarse/fine과 주 실험 전체를 묶은 simultaneous CI는 아니다. Calibration 및 test timing은 별도 측정으로 GPU clock, scheduling, world 구성에 따른 변동을 포함한다. p90 budget과 근접 구간은 이번 연구를 위해 정한 engineering 기준이다. Fine 결과는 results/fine의 design.json과 summary.json에 저장했다.

## 실행 및 자료

서버 경로는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/external_label_v21`이다. `PROTOCOL.md`와 `LATENCY_PROTOCOL.md`에 두 비교의 설정을 기록했다. 코드 `v21_policy.py`가 posterior truncation을, `data_v21.py`가 private world 생성을, `run_v21.py`와 `parity.py`가 public policy selection 및 평가를 담당한다. `results/summary.json`과 `results/parity/summary.json`에 두 비교의 수치가 있고 `policy_*.npz`, `eval_*.npz`, `paired_losses.npz`에 재분석용 데이터를 보관했다.

재현 시 같은 상대 경로에 이전 버전의 dependencies를 두고 새 directory에서 Slurm allocation 안의 idle GPU로 v18 guard를 통해 run_v21.py를 실행한다. Compute-node CPU에서 analyze_v21.py를 실행한 뒤 같은 GPU의 별도 guard directory에서 parity.py를 실행하고 CPU에서 parity_stats.py, report_v21.py 순서로 진행한다. 실험은 synthetic local impulse model의 단일 observation 선택이며 raw RGB, physical robot trajectory 또는 실측 contact law를 포함하지 않는다.

## 문헌 비교와 연구 판단

V20과 같은 네 편을 corpus로 사용했다. 논문들의 공통 질문은 제한된 계산으로 얼마나 정확히 적분하거나 다음 실험을 선택할 수 있느냐는 것이다. V21의 endpoint는 선택한 observation을 받은 뒤의 stick probability에 대한 Brier loss다. 아래에서 문헌의 직접 결과와 이번 연구자가 제안한 검증을 구분한다.

| 논문 | Research question과 methodology | Sample 및 data source | 주요 결과와 결론 | 이번 비교에 적용되는 한계 |
|---|---|---|---|---|
| Bartuska et al., *Multilevel randomized quasi-Monte Carlo estimator for nested integration* | Inner integration·discretization의 해상도를 조절하면서 nested integral의 tolerance를 달성하는 work를 분석했다. Scrambled QMC와 multilevel differences를 사용했다. | Exact/inexact sampling을 다룬 두 EIG examples. 첫 예제에서 variance를 추정하는 pilot에 1,000 randomizations를 썼다. | 조건부 error bound와 두 예제의 수치 결과가 work allocation의 근거다. | Regularity와 noise truncation 가정이 있다. Brier risk의 argmin과 unseen contact transition까지 같은 수렴률을 적용하려면 별도 분석이 필요하다. [§§3–5](https://arxiv.org/html/2412.07723v1#S5) |
| Sorokin et al., *Fast Bayesian Multilevel Quasi-Monte Carlo* | GP posterior variance로 다음 sample doubling의 이득을 예상하고 level별 비용을 배분할 수 있는지 평가했다. | Problem·budget 조합당 250 trials. Single-level functions, option pricing, elliptic PDE, Darcy flow. | 시험한 benchmark에서 single low-discrepancy sequence를 활용하는 Bayesian approach가 replicated MLQMC보다 효율적인 사례를 보고했다. | Numerical integration error와 posterior/kernel assumptions가 중심이다. Ranking 이후의 실측 decision loss와 per-world GPU latency는 다른 측정값이다. [§§2–4](https://arxiv.org/html/2510.24604v1#S3) |
| Dinkel et al., *Efficient Bayesian Optimal Experimental Design for Expensive Computational Models over Finite Design Sets* | Expensive forward model에서 높은 EIG의 design을 적은 호출로 찾기 위해 sample reuse, common random numbers, Rao–Blackwellization, adaptive elimination을 비교했다. | Synthetic, pharmacokinetic, lung, viscoplastic models. 일부 accuracy curves의 반복 횟수는 1,000 runs다. | 후보별 계산량을 조절해 높은 선택 정확도에 도달할 때 필요한 forward evaluations가 감소했다. | 여기서 forward call 수의 감소와 V21의 cached model GPU p90이 다른 결과를 보여도 직접적인 반증이 되지 않는다. [§§2.3–3.4](https://arxiv.org/html/2607.16933v1#S3) |
| Zhu & Hainy, *Sequential Bayesian Experimental Design for Prediction in Physical Experiments Informed by Computer Models* | KOH model에서 미래 prediction을 개선하는 observation 위치를 predictive MI로 고른다. Gaussian mixture compression과 local complexity를 사용했다. | Toy simulation과 JAK–STAT5 실제 사례. 19개 field time points 중 4개로 시작하여 10 sequential rounds를 평가했다. | 실측 사례에서 MSE와 CRPS 기준으로 유리한 criterion이 달랐다. Mixture compression은 설계 계산을 줄이는 구성요소였다. | 희소한 실측을 보간한 pseudo ground truth로 평가했다. 본 실험의 stick event, one-step horizon, latency cap과 조건이 다르다. [§§4–5.4](https://arxiv.org/html/2603.16756v1#S5.SS4) |

**연구 흐름과 knowledge map.** Bartuska의 inner/outer error 분석과 Sorokin의 uncertainty forecast는 적분의 계산 예산을 정하는 흐름이다. Dinkel은 paired design comparison을 통해 계산 오차가 어느 action을 고르는지와 연결했고, Zhu와 Hainy는 observation 선택을 이후 prediction error와 연결했다. V21은 이 두 연결을 따라 external posterior를 줄일 때 추정된 risk의 순위, 실제 선택, observation 이후의 loss를 동일한 scene에서 비교한다. [Bartuska](https://arxiv.org/html/2412.07723v1) · [Sorokin](https://arxiv.org/html/2510.24604v1) · [Dinkel](https://arxiv.org/html/2607.16933v1) · [Zhu & Hainy](https://arxiv.org/html/2603.16756v1)

**근거와 불일치.** 조건을 명시한 numerical integration 이론에는 강한 근거가 있고, 각 논문에서 시험한 문제에 대한 계산 효율은 실험적 근거가 있다. 반면 비용을 forward calls, integration error 또는 GPU wall time으로 정의하면 방법의 순위가 달라질 수 있다. 같은 실측 사례에서도 MSE와 CRPS가 서로 다른 design criterion을 선호한 것은 이 구분의 실제 예다. 네 논문에 동일한 contact test와 predictor를 놓고 결론이 충돌한 결과는 확인되지 않았다. Human sample size, simulator runs, bank seeds를 공통 실험 단위로 합산할 근거도 없다. [Sorokin §3](https://arxiv.org/html/2510.24604v1#S3) · [Dinkel §3](https://arxiv.org/html/2607.16933v1#S3) · [Zhu & Hainy §5.4](https://arxiv.org/html/2603.16756v1#S5.SS4)

**연구 gap과 근거 수준.** 이 corpus는 cost-aware computation과 predictive experimental design의 선례를 제공한다. 네 편에서 V20의 E−C label-source effect가 independent banks와 unseen laws에서 반복되는지에 대한 직접 근거는 확인되지 않았다. 또한 factual posterior에서 버린 낮은 가중치의 hypotheses가 hypothetical observation을 본 뒤 중요해질 가능성은 남는다. V21에서 확인하는 것은 truncation의 평균 효과와 실제 배포 비용이고, discarded factual mass를 uniform error bound로 해석하려면 추가 가정이 필요하다. [Bartuska §3](https://arxiv.org/html/2412.07723v1#S3) · [Zhu & Hainy §4.1](https://arxiv.org/html/2603.16756v1#S4.SS1) · [V20 실험](../factual_switch_v20/REPORT_KO.md)

### 후속 연구 질문 10개

| 질문 | 근거가 되는 gap과 의미 | 변수와 적합한 검증 |
|---|---|---|
| 1. Bank를 바꾸면 label-source effect의 부호가 유지되는가? | V20은 bank pair 하나에 조건부였다. [V20](../factual_switch_v20/REPORT_KO.md) | V21에서 seed별 paired differences와 crossed bootstrap을 보고했다. 후속 검증은 더 많은 bank pairs로 tail risk를 추정한다. |
| 2. Unseen law가 새로운 sticking region을 만들면 selector가 따라가는가? | Predictive design에서 simulator discrepancy가 실제 예측에 관여할 수 있다. [Zhu & Hainy §2](https://arxiv.org/html/2603.16756v1#S2) | V21의 두 stress laws를 출발점으로 threshold와 friction range를 교차하고 law 단위 holdout을 둔다. |
| 3. Low-weight hypotheses가 observation 후 큰 posterior mass를 얻는 경우는 언제인가? | Compression은 computational approximation을 추가한다. [Zhu & Hainy §4](https://arxiv.org/html/2603.16756v1#S4) | Full bank의 post-observation discarded mass와 paired action regret를 같은 hypothetical samples에서 측정한다. |
| 4. Label-conditioned truncation이 rare stick gain을 보존하는가? | V19에서 compressed label support가 사라진 cases가 있었다. [V19](../conditional_risk_v19/REPORT_KO.md) | 같은 K·GPU p90에서 top-K, probability-weighted resampling, label-stratified selection을 비교한다. |
| 5. Risk가 비슷한 actions에만 external labels를 사용하면 좋은가? | Pairwise design comparison은 후보별 계산의 역할을 보여준다. [Dinkel §2.4](https://arxiv.org/html/2607.16933v1#S2.SS4) | Disjoint validation에서 action-gap threshold를 고정하고 forced probes, fixed global과 비교한다. |
| 6. 동일한 predictor로 K와 M의 marginal value를 추정할 수 있는가? | Forecasted uncertainty reduction과 marginal compute를 함께 비교하는 선례가 있다. [Sorokin §2.3](https://arxiv.org/html/2510.24604v1#S2.SS3) | K·M·bank seed를 crossed design으로 바꾸고 held-out loss 감소/ms를 평가한다. |
| 7. Acquisition plan의 seed도 바꾸면 같은 결론인가? | Bank를 바꾸는 것과 integration points를 바꾸는 것은 서로 다른 replication이다. [Bartuska §5](https://arxiv.org/html/2412.07723v1#S5) | Bank와 acquisition scrambles를 완전히 교차해 variance components를 추정한다. |
| 8. 어떤 sampling law에서 compute와 loss가 함께 개선되는가? | V20에서 law와 label source 효과가 달랐다. [V20](../factual_switch_v20/REPORT_KO.md) | 동일한 probabilities·observation noise로 sampling marginal만 바꾸고 subgroup Brier 및 latency를 비교한다. |
| 9. Expensive sensing이 들어가면 fixed global과의 순위가 변하는가? | Forward-call cost와 online latency의 차이가 있다. [Dinkel §3](https://arxiv.org/html/2607.16933v1#S3) | View capture·probe·reset cost를 개별 측정해 총 decision utility를 사전 정의한다. |
| 10. 실제 contact 영상에서 예측이 개선되는가? | 실물 physical response를 대상으로 한 predictive design 사례는 있지만 contact counterfactual의 실측 검증은 별도 문제다. [Zhu & Hainy §5](https://arxiv.org/html/2603.16756v1#S5) | 동일 물체의 반복 interventions로 calibration과 Brier를 평가하고 simulator-only conclusions와 비교한다. |

**Research assistant review.** Top-K truncation, randomized integration, predictive selection 자체를 새 방법이라고 주장하기 어렵다. 방법 논문에는 generic posterior reduction과의 공정한 비교, contact law 수준의 독립 표본 확대, real sensing cost가 더 필요하다. 이 네 논문만으로 active perception이나 Bayesian experimental design 분야 전체의 novelty를 판정할 수 없다. Statistical significance의 유무는 해당 비교의 CI와 함께 읽어야 하며, seed 수와 stress law 수가 다른 불확실성을 나타낸다는 점이 중요하다. [Dinkel §2](https://arxiv.org/html/2607.16933v1#S2) · [Zhu & Hainy §4](https://arxiv.org/html/2603.16756v1#S4)

