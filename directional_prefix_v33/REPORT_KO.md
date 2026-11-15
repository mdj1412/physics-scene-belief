# v33 · 새 물체와 표면에서 어떤 접촉 이력에 probe를 더 쓸 가치가 있는가?

**탐색에서 고른 집단이 별도 cohort에서도 양의 절대 gain과 나머지 집단 대비 우위를 함께 갖는다는 가설은 확인되지 않았다. 고정 배분보다 안정적으로 낮은 loss를 보인 learned policy는 없었다.** 동일한 training compute로 학습한 2×2와 1×1의 전체 held-out Brier 차이(2×2 − 1×1)는 +0.00013 [-0.00034, +0.00060]였다. 작은 값이 좋은 지표다. 아래 prefix 분석에서는 반대로 gain이 클수록 좋다.

**직접 검증한 prefix:** plus 방향, 2→4회 단계에서 관측한 접선 방향 probe velocity의 spread가 같은 block 여덟 환경 중 하위 네 곳인 prefix. 독립 cohort B의 absolute gain은 -0.00022 [-0.00167, +0.00124], 반대 절반 대비 gain 차이는 -0.00040 [-0.00200, +0.00120]였다. 한 환경의 한 번의 충돌이 아니라, 그 환경에서 새로 만난 물체 128개에 대한 Brier를 얼마나 줄였는지가 질문이다.

## 동일 비용에서 전체 정책은 어땠는가?

| 정책 | 세 학습 쌍 평균 Brier ↓ | 전체 GPU p90, ms [95%] ↓ | Deployment-only p90, ms |
|---|---|---|---|
| 1×1 | 0.125424 | 1,478.8 [1,469.5, 1,490.7] | 962.8 |
| 2×2 | 0.125553 | 1,480.2 [1,468.3, 1,483.5] | 964.2 |
| Fixed 4 | 0.125697 | 1,416.0 [1,404.5, 1,423.7] | 963.0 |
| Uncertainty-only | 0.125192 | 1,477.0 [1,466.9, 1,490.2] | 963.1 |

모든 정책이 block당 8개 환경에 calibration probe 32회, 이어지는 1,024개 물체에 deployment probe 104회를 썼다. Fixed 4는 모두 4회씩 배정한다. 다른 세 정책은 모두 2회로 시작해 4개 환경만 4회, 그 중 2개만 8회까지 계속한다. 1×1과 2×2는 같은 ridge architecture를 쓰고 학습용 gain을 측정하는 방식이 다르다. Uncertainty-only는 각 시점의 friction coupling posterior SD로 다음 환경을 고른다. 128개 물체 중 13개에만 probe를 쓸 수 있고 나머지는 query view를 본다. 뒤이어 별도 collision에서 stick할 확률을 공통 decoder로 채점한다.

| Brier 차이 | 고정 세 모델의 평균 [simultaneous 95%] | training seed까지 재표집 [simultaneous 95%] |
|---|---|---|
| 2×2 − 1×1 | +0.00013 [-0.00034, +0.00060] | +0.00013 [-0.00063, +0.00089] |
| 1×1 − fixed 4 | -0.00027 [-0.00098, +0.00043] | -0.00027 [-0.00106, +0.00052] |
| 2×2 − fixed 4 | -0.00014 [-0.00087, +0.00058] | -0.00014 [-0.00098, +0.00069] |
| 1×1 − uncertainty-only | +0.00023 [-0.00028, +0.00074] | +0.00023 [-0.00038, +0.00084] |
| 2×2 − uncertainty-only | +0.00036 [-0.00016, +0.00088] | +0.00036 [-0.00029, +0.00101] |

Primary interval은 네 object–surface 조합 안에서 8개씩의 독립 block을 재표집한다. 양쪽 방향, 같은 물체의 32 sensor noise draws, 각 training seed를 독립 데이터처럼 더하지 않았다. 두 번째 구간은 독립 training pairs가 세 개뿐인 sensitivity다. 다섯 대비에 미리 정한 기준은 absolute Brier 0.003이다. [통계 계획](POLICY_PLAN_KO.md) · [원수치·latency ratio](policy_stats33.json)

## 어떤 paid prefix에서 추가 접촉이 유리했는가?

처음 지불한 접촉 2회 또는 4회가 paid prefix다. 같은 prefix에서 그대로 멈출 때와 추가로 2회 또는 4회를 살 때를 비교했다. Gain = Brier(stop) − Brier(continue). 발견 집단 A와 검증 집단 B는 각각 네 물체·표면 조합에서 독립 block 4개씩, 모두 16개를 사용했다. 각 block은 여덟 환경의 같은 물체를 양쪽 방향으로 시험한다. 한 stage와 방향에서 8개 환경 중 고른 절반은 cohort마다 64개이며, 독립 분석 단위는 16개 blocks다. 두 독립 calibration futures와 두 독립 deployment batches를 교차해 얻은 네 값을 평균했다. Stop과 continue는 똑같은 deployment 물체·정답·sensor noise를 공유한다. 평가는 모델의 예상치가 아닌 simulator의 실제 stick outcome으로 한다.

Cohort A에서만 관측 가능한 3가지 feature × 상·하위 절반 × 2 stages × 2 방향, 합계 24개 후보를 비교했다. 대상의 absolute gain과 나머지 절반 대비 gain 차이 중 더 작은 값이 가장 큰 규칙을 골랐다. 그 규칙은 **plus 방향, 2→4회 단계에서 관측한 접선 방향 probe velocity의 spread가 같은 block 여덟 환경 중 하위 네 곳인 prefix**였다. 선정 파일과 A의 지문을 기록한 뒤 새 물체·새 접촉·새 noise로 cohort B를 생성했다. 추가로 방향이나 stage를 고르지 않는 사전 가설인 높은 posterior SD도 검증했다.

| 대상 / 검증량 | 탐색 A | 독립 B [simultaneous 95%] | 별도 policy worlds의 1×1 측정 [simultaneous 95%] |
|---|---|---|---|
| High posterior SD: absolute gain | +0.00073 | +0.00032 [-0.00019, +0.00082] | +0.00105 [+0.00035, +0.00175] |
| High posterior SD: high − low | +0.00017 | +0.00021 [-0.00050, +0.00093] | +0.00117 [-0.00013, +0.00248] |
| 선택 집단: absolute gain | +0.00186 | -0.00022 [-0.00167, +0.00124] | -0.00027 [-0.00207, +0.00154] |
| 선택 집단 − 나머지 | +0.00236 | -0.00040 [-0.00200, +0.00120] | -0.00080 [-0.00311, +0.00151] |

높은 posterior SD 가설의 두 하한이 모두 0을 넘었는가: **아니오**. A가 선택한 규칙은 **아니오**. 두 하한이 모두 0.003을 넘었는가는 각각 **아니오 / 아니오**다. 이 실용 기준은 실제 한 번의 접촉 가격을 환산한 값이 아니다.

B에서 선택 집단의 반대 절반의 absolute gain은 +0.00019였다. Probe 추가 전후에 deployment observation이 바뀐 비율은 선택 집단 10.0%, 나머지 9.7%였다. 구매 결정을 바꿔도 새 관측이 더 정확하지 않으면 gain은 증가하지 않는다. 

마지막 열은 A와 B 어느 쪽에서도 쓰지 않은 full-policy worlds에 같은 규칙을 적용했다. 같은 합성 표면 조건이지만 한 prefix에서 continuation·deployment batch가 각각 하나이므로 B의 2×2 측정보다 시끄럽다. Third panel로 A의 후보를 다시 선택하지 않았다. 모든 환경의 4회 prefix를 평가하는 것과 adaptive policy에서 4회까지 살아남은 환경만 평가하는 것도 다른 질문이다. 어떤 집단에 더 많이 쓸지 정할 때에는 그 stage에 실제 도달할 확률까지 필요하다. [후보 24개와 선정 지문](results_prefix/discovery_lock.json) · [독립 B 전체 결과](results_prefix/prefix_gain33.json) · [stage별 결과와 대각 반복](PREFIX_RESULTS_KO.md) · [제3 집단 기준](TRANSFER_PLAN_KO.md) · [제3 집단 결과](transfer33.json)

같은 규칙을 비교할 다른 환경 없이 사용할 수 있는지도 확인했다. A에서만 정한 절대 threshold 0.0873699를 B로 옮겼을 때 대상 71개, 나머지 57개였다. 대상 gain은 -0.00051 [-0.00170, +0.00078], 반대 집단 대비 차이는 -0.00149 [-0.00313, +0.00029]였다. 이 보조 분석의 구간은 pointwise이며 primary 판정을 대체하지 않는다. [전체 threshold 검증](results_prefix/prefix_gain33.json) 두 측정에서 이 population SD는 |v₁−v₂|/2다. 따라서 threshold에 해당하는 두 tangential velocity의 간격은 0.17474이다. 단위는 simulator의 velocity 좌표계다.

선택 집단에서 관측한 2×2 평균 gain은 35/64개 환경에서 양수였고, median +0.00075 (IQR -0.00388~+0.00372)였다. 반대 절반에서 관측한 2×2 평균 gain은 32/64개 환경에서 양수였고, median -0.00014 (IQR -0.00310~+0.00248)였다. 이는 두 futures·두 batches의 관측 통계이며, 각 환경의 true expected gain이 양수일 확률의 추정치가 아니다.

**발견된 규칙으로 실제 예산을 배정하면?** 실험자가 plus 방향임을 알고 있을 때, 선택된 한 단계에서만 해당 규칙을 적용하고 나머지는 uncertainty-only로 배정했다. 같은 32-probe budget에서 이 정책의 Brier는 0.125418, fixed 4 대비 -0.00028 [-0.00101, +0.00045], uncertainty-only 대비 +0.00023 [-0.00021, +0.00067]였다. 여덟 환경 중 같은 횟수의 접촉을 받을 곳이 바뀐 비율은 uncertainty-only 대비 29.3%였다. 이 진단 정책은 표면 방향이라는 추가 정보를 받으므로 네 primary policies의 공정한 대체 baseline은 아니다. 별도 GPU p90은 측정하지 않았다. [규칙을 임의로 바꾸지 않기 위한 사전 설계](SHADOW_PLAN_KO.md) · [실제 trajectory의 재계산](shadow33.json)

![전체 policy와 독립 gain 검증](v33.png)

## Friction 방향을 바꾸면 v32의 결과가 유지되는가?

V32 directional law에서 2×2 − 1×1은 세 학습 쌍 평균 +0.00088이었다. 당시 첫 seed의 부호를 본 후 세분화한 결과였다. V33은 그 결과를 직접 시험하기 위해 두 object profile과 surface response 두 가지에 대해 방향을 양쪽으로 미리 지정했다. 같은 물체와 noise를 쓰되 반대 orientation에서 물리 법칙이 내는 probe response와 stick label을 다시 생성했다. [v32 결과](../compute_matched_stopping_v32/REPORT_KO.md)

| 방향·대비 | 2×2 − 1×1 Brier [세 대비 simultaneous 95%] |
|---|---|
| Plus | +0.00029 [-0.00027, +0.00085] |
| Minus | -0.00003 [-0.00070, +0.00064] |
| Plus − minus | +0.00032 [-0.00057, +0.00120] |

Plus 방향에서 세 모델 쌍 각각의 2×2 − 1×1은 +0.00055, +0.00043, -0.00012였다. V32에서와 달리 세 seed가 같은 방향을 가리키지 않았다.

| 합성 물체 / 표면 / 방향 | 1×1 | 2×2 | Fixed 4 | Uncertainty-only | 2×2 − 1×1 |
|---|---|---|---|---|---|
| rounded / fine / plus | 0.14090 | 0.14059 | 0.14122 | 0.14039 | -0.00031 |
| rounded / fine / minus | 0.12143 | 0.12182 | 0.12224 | 0.12132 | +0.00039 |
| rounded / coarse / plus | 0.15978 | 0.16043 | 0.16036 | 0.15967 | +0.00065 |
| rounded / coarse / minus | 0.11274 | 0.11192 | 0.11269 | 0.11302 | -0.00083 |
| ridged / fine / plus | 0.11714 | 0.11742 | 0.11734 | 0.11753 | +0.00027 |
| ridged / fine / minus | 0.10324 | 0.10372 | 0.10235 | 0.10234 | +0.00048 |
| ridged / coarse / plus | 0.13701 | 0.13754 | 0.13755 | 0.13647 | +0.00054 |
| ridged / coarse / minus | 0.11115 | 0.11099 | 0.11183 | 0.11080 | -0.00016 |

방향을 반전하면 같은 물체의 query stick label이 14.2%에서 바뀌었다. 두 learned policy는 방향 전체에서 환경별 calibration 횟수를 59.0% 다르게 배정했다. 하지만 최종 128개 물체에서 실제로 서로 다른 sensor를 고른 비율은 6.0%였다. 횟수 배정과 최종 관측 선택 사이에서 효과가 얼마나 작아지는지 보여주는 수치다.

두 방향 사이의 policy 차이를 같은 횟수에서의 downstream risk와 횟수 배정 변화로 분해했다. 배정 변화의 기여는 -0.00010 [-0.00047, +0.00029], downstream risk 변화의 기여는 +0.00042 [-0.00016, +0.00099]이었다. 두 점추정의 합이 위의 plus − minus 차이와 일치한다. 후자에는 direction에 따른 probe 응답·query label·관측 선택 변화가 함께 들어 있다. [보조 분석 계획](MECHANISM_PLAN_KO.md) · [계산식과 결과](mechanism33.json)

전체 정책 결과를 본 후 탐색적으로, 동일한 32개 budget에서 420개 배분을 모두 대조했다. 정답을 먼저 알고 같은 block에서 고르는 사후 Brier는 0.12114였지만, 같은 object·surface·orientation의 다른 일곱 blocks에서 배분을 정한 뒤 옮기면 0.12548이었다. 균등 무작위 배분의 기댓값은 0.12562였다. Hindsight가 확보한 0.00456의 여지는 새 prefix에서 달성 가능한 이득의 증거가 아니다. Block 간 이전에는 실제 population coupling을 알려 준 보조 기준을 사용했다. [사후 계산과 조건](headroom33.json)

## Training compute와 재현 범위

| Training pair | 1×1 학습 환경 / GPU wall-s | 2×2 학습 환경 / GPU wall-s | 1×1 target Brier | 2×2 target Brier |
|---|---|---|---|---|
| 1 | 597 / 539.34 | 255 / 537.57 | 0.125244 | 0.125816 |
| 2 | 595 / 538.71 | 254 / 537.83 | 0.125639 | 0.125464 |
| 3 | 601 / 539.20 | 256 / 537.40 | 0.125390 | 0.125380 |

Source Coulomb에서 만들어 둔 v32의 세 model pairs를 사용한다. 각 pair는 같은 15개 features와 stage별 ridge penalty 3, 같은 training allowance 540초를 썼다. 세 pair 각각의 1×1 및 2×2 source 데이터는 독립이다. v33의 target에서 추가 학습 비용은 0초다. 첫 pair의 네 policy는 서버 GPU가 전체 경로를 실행하고 실제로 고른 관측으로 Brier를 산출한다. 나머지 두 pair는 같은 새 환경에서 stage별로 갈 수 있는 2·4·8회 결과를 저장한 뒤, 해당 scorer의 순차 결정으로 고른 결과를 재생한다. 이 때 두 번째 stage에는 첫 stage에서 통과한 네 환경의 score만 들어간다.

물체의 표면 윤곽은 offset + slope × y + curvature × y²/(2h) + d × y³/(3h²)이며 h=0.025다. Rounded와 ridged는 원래 curvature의 0.90·1.10배와 d=−0.060·+0.060을 각각 사용한다. 이름은 이 두 개의 합성 곡선 계열을 가리킨다. 곡면상의 contact normal은 같은 윤곽의 미분으로 구한다. 새 표면의 유효 마찰계수는 μ_eff = aμ[1 + Dη tanh(vₜ/τ)]다. Fine에서는 (a,η,τ)=(0.98,0.17,0.34), coarse에서는 (1.04,0.22,0.50)이다. D는 시작 전에 +1 또는 −1로 정했다. 운동 방향에 따라 마찰이 대칭인 표면의 기준은 같은 geometry와 aμ를 유지한 상태에서 η=0으로 구한다. factual intervention에서 접촉이 sticking인 조건, calibration probe에서 sliding인 조건은 매 생성마다 검사했다. 측정 전 모델의 가설 공간에는 cubic contour도 방향 효과도 들어 있지 않다.

하나의 object/surface block은 독립 calibration pool 여덟 개와 calibration과 겹치지 않는 deployment objects 1,024개를 포함한다. 같은 block의 +/− 방향에 동일한 물체, latent variables, noise, posterior bank, 순서가 쓰인다. Random generator lane을 나눠 calibration, deployment, bank를 구분한다. +/−에서 probe response와 intervention의 stick label은 물리에 따라 달라질 수 있다. 네 조합마다 8개 blocks를 따로 뽑고 2개 방향을 각각 평가한다. 총 독립 block 수는 32개다. 모든 policy에서 calibration은 block당 32회이고, 별도 deployment probe는 128개 물체마다 13회, block당 104회다.

GPU-synchronized full-call p90 측정은 각 조합의 첫 2개 block에서 두 방향 모두 실시한다. 각 평가 block에서 한 번 warm-up 후 세 번씩 방법 순서를 무작위로 섞으므로 policy당 48개의 시간 측정과 8개의 독립 timing block이 생긴다. 한 call에서 factual evidence와 사용할 수 있는 probe 값부터 각 policy의 posterior, 순차 allocation, selector, 1,024개 선택 관측의 최종 probability를 다시 구한다. 공통 detector와 privileged source decoder 초기화, 표면 전환, reset, sensor wait, simulation, 파일 기록은 제외한다. 전용 GPU guard가 시작 때 물리 장치가 비어 있는지 확인하고 외부 사용자 작업이 나타나면 자신의 child process만 정지시킨다.

독립 구현은 별도 seeds의 32,768개 합성 물체에서 유한 차분으로 contour normal을 구하고 Newton impulse와 tangential impulse를 NumPy로 다시 계산했다. Torch와의 최대 outgoing velocity 오차는 1.90×10⁻¹¹이고 query stick label은 모두 일치했다. 별도 검사에서 +/−의 factual evidence·query view·isotropic label이 정확히 일치했다. [물리 검증 원자료](physics_audit33.json)

평가는 NVIDIA RTX PRO 6000 Blackwell Server Edition, allocation 2744468에서 실행했다. 원확률에서 재계산한 stage Brier와 기록값의 최대 절대 차이는 0.00e+00다. Training seed와 target streams의 중복을 검사했고, mirror pair에만 같은 target streams를 허용했다. 독립 physics audit에서 32,768개 물체를 별도로 생성해 contact normal과 stick 분기를 검증했다. [GPU 완료 기록](results_eval/completion.json) · [평가 코드](evaluate33.py) · [통계·quota 독립 검산](policy_stats33.py) · [물리식 검산](physics_audit33.json) · [독립 prefix GPU 완료 기록](results_prefix/completion.json)

## 실제 표면에서 확인할 수 있는 부분

Tramsen et al.의 [공개 S1 측정표, Tables 1–2](https://doi.org/10.6084/m9.figshare.c.3950773.v2)에 실린 20개 조합을 전부 읽어 pulling 방향 두 가지의 fitted friction coefficient를 대조했다. 저자가 보고한 Δμ = μ_against − μ_along은 9개 조합에서 양수, 11개에서 음수였고 범위는 −0.378~+0.273이었다. 다만 저자가 제시한 Δμ의 standard error는 약 0.05로, |Δμ| > 0.10인 조합은 10/20개였다. 20개 모두에서 부호를 통계적으로 확인했다는 뜻은 아니다. 예를 들어 같은 flat soft substrate에서 stiff sample은 +0.025, soft sample은 −0.248이었다. 첫 값은 측정 오차에 비해 작다. [원문](https://pmc.ncbi.nlm.nih.gov/articles/PMC5805965/) · [계산 코드](physical33.py) · [20개 조합의 재계산 결과](physical33.json)

양쪽 coefficient 차이를 합으로 나눈 signed amplitude (μ_against−μ_along)/(μ_against+μ_along)의 실제 범위는 −0.170~+0.122였다. v33에서 설정한 0.17/0.22는 속도 함수가 포화할 때의 최대 amplitude다. 특히 0.22는 이 측정표의 최대 절댓값보다 크다. 이 표가 뒷받침하는 것은 물체·표면에 따라 강한 마찰 방향이 달라질 수 있다는 점이다. Quasistatic pulling 측정은 v33의 impact law를 fit한 자료도, 합성 surface 이름에 대응하는 재질도 아니다. 이 표에는 v33의 query view·probe·별도 충돌의 stick label 조합이 없으므로 실제 장비에서의 Brier gain은 추정할 수 없다. [실제 contact에서 부분적으로 공통인 또 다른 데이터의 측정 항목](https://web.mit.edu/mcube/push-dataset/)

서버 데이터 파일: `/scratch2/mdj1412/datasets/tramsen2018_friction_supplement.pdf`. 원저자의 MD5 `92fa510e1a75c46ab09372e8000ff8e3`와 대조했다. 서버 CPU와 로컬에서 같은 표를 독립 파싱하고, 모든 방향별 계수 차이를 저자 표의 반올림 값과 0.001 이내에서 대조했다.

## 논문에서 주장할 수 있는 부분과 다음 판단

**연구 판단.** 이 설정에서는 비용을 우선하면 가장 빠른 fixed 4를 선택하는 것이 합리적이다. 평균 Brier만 보면 uncertainty-only가 가장 낮았지만 fixed 4 대비 차이는 −0.00050(pointwise 95% −0.00115~+0.00018)으로, 어느 쪽이 예측에서도 더 낫다고 확정할 수 없다. [두 baseline의 직접 대조](headroom33.json)  v32에서 보였던 plus 방향에서의 1×1 우위는 이번 세 대비를 함께 보정한 기준으로 재현되지 않았다. 방향 간 차이 자체의 구간도 0을 포함했다. 다섯 primary policy 차이의 simultaneous interval이 모두 ±0.003 안에 있었다. 독립 cohort에서 후보 집단의 absolute gain과 나머지 대비 gain 차이가 동시에 양수임을 확인하지 못했다. A에서 가장 큰 숫자만 골라 선택 가능한 prefix의 증거로 제시해서는 안 된다. 두 비교 모두 새 simulation data에서 측정했지만, 여전히 두 유형의 물체와 두 유형의 표면·고정된 여덟 coupling 및 source-law common decoder에 대한 결론이다. 이후 데이터에서도 항상 동일하다는 뜻은 아니다.

**논문 기여 가능성.** 방어 가능한 기여는 같은 물체에서 방향만 바꾼 paired test에, 실제로 지불한 prefix의 gain을 독립 outcome으로 재측정하고, 같은 예산의 전체 stopping loss까지 추적한 검증 절차다. 마찰 방향에 따른 차이, uncertainty 기반 물리 탐색, 학습된 error reduction 자체에는 직접 선례가 있다. 독립 B와 전체 policy의 근거 없이 새로운 acquisition 방법이 우월하다고 주장하기는 어렵다. [실물 active exploration의 직접 선례](https://arxiv.org/html/2411.09020v1) · [group별 방향이 반전된 pulling 실험](https://pmc.ncbi.nlm.nih.gov/articles/PMC5805965/)

**다음 실험에서 먼저 판단할 것.** 실제 표면 session을 통째로 discovery와 confirmation으로 나누고 pulling 양방향을 따로 측정해 sign을 고정한다. 그 뒤 calibration에 쓴 물체와 별도 deployment 물체에서 probe·query view·stick label을 수집한다. 날짜·reset·입사 속도를 함께 기록한다. 현재처럼 factual과 calibration에 한쪽 속도만 줄 때와 양쪽 속도를 번갈아 줄 때를 같은 접촉 횟수로 비교하면, signed evidence 자체가 부족한지와 scorer가 기존 evidence를 쓰지 못하는지를 분리할 수 있다. 실제 contact 시간으로 값을 매긴 실용 기준을 데이터 공개 전에 정하고, 새 session에서도 absolute gain과 excess gain의 두 하한이 기준을 넘으며 전체 policy가 fixed 4보다 나을 때에만 방법 우위를 주장하는 것이 적절하다.

## 문헌에서 본 friction 방향과 추가 probe의 가치

물체가 어느 방향으로 더 잘 미끄러지는지는 두 방향에서 힘을 측정해 확인할 수 있다. 그 방향을 알아내려고 접촉을 한 번 더 살 가치가 있는지를 알려면, 이미 관측한 이력과 그 접촉으로 달라지는 다음 행동, 행동의 성과를 측정할 독립적인 물체까지 필요하다. 아래 6편은 이 경로의 서로 다른 부분을 직접 측정하거나 분석했다. 표의 표본 수는 각 논문이 비교한 단위다. [Tramsen et al.](https://findresearcher.sdu.dk/ws/portalfiles/portal/151776983/rsif.2017.0629.pdf) · [Konyushkova et al.](https://proceedings.neurips.cc/paper_files/paper/2017/file/8ca8da41fe1ebc8d3ca31dc14f5fc56c-Paper.pdf)

### 논문별 질문, 방법과 결론

| 연구와 질문 | 방법 및 표본 | 주요 결과와 저자의 결론 | v33에 적용할 수 있는 범위 |
|---|---|---|---|
| Yu et al. (2016). Planar pushing에서 일정한 friction을 가정할 수 있는가? | Stainless steel 물체 11개, 표면 4종에서 pose와 force를 250 Hz로 기록했다. 별도 sliding scan으로 방향별 dynamic friction을 비교했다. | ABS, Delrin, plywood는 대체로 isotropic이었고 polyurethane에서는 방향별 최대·최소 friction 비율이 약 1.5였다. 방향 의존성은 재료마다 달랐다. | 여러 각도를 훑어 구한 최대·최소 비율이다. 같은 축의 정방향과 역방향을 모델링하려면 지정한 두 방향의 측정이 필요하다. [원문 §§IV–V, Fig. 6](https://arxiv.org/pdf/1604.04038) |
| Ma & Rodriguez (2018). 같은 push를 반복해도 물체가 다르게 회전하는 이유는 무엇인가? | Yu 등의 네 표면에서 같은 push를 각각 2,000회 연속 수행한 기록을 재분석하고 isotropic 및 anisotropic friction model로 수집 과정을 모사했다. | 초기 orientation과 reset이 다음 실험의 출발 조건에 영향을 주었고, anisotropy와 공간적 불균일성이 겉보기의 불규칙한 변동 일부를 설명했다. | 연속 trial에는 순서 의존성이 있다. Yu와 공유하는 원자료를 독립적인 두 번의 물리 실험으로 계산하면 증거가 중복된다. [원문 §§III–VI, Figs. 7–8](https://arxiv.org/pdf/1802.10089) |
| Tramsen et al. (2018). Sawtooth의 모양만으로 friction이 큰 방향을 정할 수 있는가? | Silicone과 epoxy, 세 aspect ratios, 네 substrate roughnesses 등을 조합한 96개 조건에서 along·against 방향 계수를 측정했다. | 대부분 against 방향의 friction이 컸지만 일부 조건에서는 부호가 반전됐다. 저자들은 stiffness와 접촉 형태에 따라 interlocking과 adhesion의 상대적 기여가 달라질 수 있다고 해석했다. | 같은 축에서 부호가 바뀌는 실험적 근거다. Sliding coefficient를 충돌 시 stick probability로 옮기려면 별도 검증이 필요하다. [원문 §2, Figs. 3–5](https://findresearcher.sdu.dk/ws/portalfiles/portal/151776983/rsif.2017.0629.pdf) |
| Dutta, Burdet & Kaboli (2025). Robot이 다음 관측을 예상해 exploratory action을 고를 수 있는가? | Real robot의 push·pull, visual·tactile sensing, dual differentiable filter, N-step information gain을 결합했다. 구성 가능한 물체 120개를 80/10/10%로 나눴다. | Active GNN은 uniform·random action보다 적은 training trajectories로 수렴했다. 제어 예에서는 물체 8개에 조건당 3회씩 push를 실행해 추정한 parameter가 최종 pose에 주는 효과를 평가했다. | 같은 물체의 동작을 고르는 실물 선례다. 한 환경의 calibration이 이후 다른 물체들의 sensor 선택에 주는 효용은 별도의 평가 단위다. [원문 §IV-B, Figs. 10·16](https://arxiv.org/html/2411.09020v1) |
| Konyushkova, Sznitman & Fua (2017). 현재 정보로 label을 살 때의 error reduction을 예측할 수 있는가? | Classifier와 후보의 features에 대해 label 추가 전후의 test error 차이를 regression했다. Balanced·imbalanced Gaussian 예에서 조건별 10,000회의 결과를 평균했다. | Balanced 예에서는 uncertainty가 가장 큰 곳의 이득이 컸지만 class 비율을 2대 1로 바꾸자 최대 이득 위치가 이동했다. 저자들은 실제 분류 자료에서도 학습한 acquisition을 평가했다. | Error reduction regression의 직접적인 선례다. 이 논문의 label은 현재 classifier를 갱신한다. v33 probe의 효용은 후속 물체에서 선택할 sensor에 의존한다. [원문 §§3–5, Fig. 1](https://proceedings.neurips.cc/paper_files/paper/2017/file/8ca8da41fe1ebc8d3ca31dc14f5fc56c-Paper.pdf) |
| Alfonsi, Lapeyre & Lelong (2023). 같은 계산 시간에 conditional expectation을 학습할 때 한 입력을 몇 번 반복해야 하는가? | Least-squares Monte Carlo의 asymptotic excess risk를 분석하고 Gaussian, stochastic differential equation, 보험 위험의 세 수치 예를 검토했다. | 최적 반복 수는 outer·inner sample 비용과 regression 근사에 의존했다. 보험 예의 MSE는 8–9회 부근에서 최소였고 5–20회도 비슷했다. | 이론은 입력별 conditional draws와 regression loss를 다룬다. v33의 crossed samples는 같은 future 또는 batch를 공유하고, 여러 환경의 순위가 다음 접촉을 결정한다. [원문 Theorem 3.2, §§5–6](https://arxiv.org/pdf/2209.04153) |

### 증거의 연결과 공백

| 연결되는 개념 | 증거 수준과 연구 사이의 차이 | 아직 확인해야 할 관계 |
|---|---|---|
| Material pair → sliding direction → friction coefficient | **직접 실측.** Yu의 angular scan과 Tramsen의 along·against 측정은 재료에 따라 friction 방향 효과가 달라진다는 데 일치한다. 두 논문은 방향 정의와 접촉 재료가 달라 특정 속도 함수나 계수 비율에 대한 공통 추정치를 주기 어렵다. [Yu](https://arxiv.org/pdf/1604.04038) · [Tramsen](https://findresearcher.sdu.dk/ws/portalfiles/portal/151776983/rsif.2017.0629.pdf) | Signed tangential velocity와 충돌 직후 상태의 관계를 동일 object–surface pair에서 측정해야 한다. |
| Reset → 다음 initial pose → 관측된 variability | **같은 데이터에 대한 보완적 설명.** Yu가 기술한 반복 간 변동의 일부를 Ma & Rodriguez는 순서와 초기 자세로 설명한다. 두 분석의 차이는 reset까지 모델에 포함했는지에서 생긴다. [Yu](https://arxiv.org/pdf/1604.04038) · [Ma & Rodriguez](https://arxiv.org/pdf/1802.10089) | 같은 표면에서 prefix를 다시 만들 때 reset, 마모, 측정 순서의 영향을 분리할 수 있는가? |
| 관측 이력 → 예상되는 update → 행동의 가치 | **각 과제에서 직접 비교.** Dutta et al.은 물체 parameter에 대한 정보를 활용해 다음 동작을 골랐고, Konyushkova et al.은 test error 감소를 학습했다. 두 연구가 좋은 선택이라고 정의하는 목적이 다르다. [Dutta et al.](https://arxiv.org/html/2411.09020v1) · [Konyushkova et al.](https://proceedings.neurips.cc/paper_files/paper/2017/file/8ca8da41fe1ebc8d3ca31dc14f5fc56c-Paper.pdf) | 여러 물체 사이에 sensor quota를 나눠 줄 때 어느 paid prefix에서 정보가 실제 선택과 Brier를 개선하는지는 두 연구만으로 결정할 수 없다. |
| 반복 횟수 → gain label precision → 순위 → 전체 stopping policy | **비용 배분에는 이론과 수치 예가 있다.** Alfonsi et al.의 비용 분석과 Konyushkova et al.의 순위 실험은 반복 횟수의 효과를 이해할 때 training 비용과 최종 결정 손실을 모두 측정할 이유를 준다. [Alfonsi et al.](https://arxiv.org/pdf/2209.04153) · [Konyushkova et al.](https://proceedings.neurips.cc/paper_files/paper/2017/file/8ca8da41fe1ebc8d3ca31dc14f5fc56c-Paper.pdf) | 같은 계산 시간에서 더 많은 이력 또는 더 정밀한 label이 directional shift 이후 순위를 얼마나 잘 보존하는지는 독립 policy 평가가 필요하다. |

하나의 paid prefix에서 같은 deployment objects와 sensor noise로 멈춤과 계속을 비교하면 그 branch에서 발생한 Brier 차이를 얻는다. 같은 prefix를 유지하고 future와 deployment batch를 다시 뽑으면 그 prefix의 conditional mean gain을 추정할 수 있다. 특정 관측 feature가 그 이득 차이를 일으킨다는 결론에는 해당 feature 또는 그 원인에 대한 조작이 필요하다. 이는 conditional simulation과 realized error reduction의 정의에서 v33 평가에 대해 도출한 해석이다. [Alfonsi et al.](https://arxiv.org/pdf/2209.04153) · [Konyushkova et al.](https://proceedings.neurips.cc/paper_files/paper/2017/file/8ca8da41fe1ebc8d3ca31dc14f5fc56c-Paper.pdf)

### 후속 연구 질문 10개

각 행은 원문에서 확인한 한계를 바탕으로 제안한 가설이며, 마지막 열에 대응하는 독립 실험을 적었다.

| 연구 질문 | 왜 중요한가 | 변수와 검증 방법 |
|---|---|---|
| 1. 접선 속력과 방향을 독립적으로 바꾸면 어느 쪽이 stick 예측을 더 크게 바꾸는가? | Friction은 재료에 따라 속력과 방향 모두에 의존한다. [Yu, §V](https://arxiv.org/pdf/1604.04038) | 같은 물체·표면에서 속력 수준과 ±방향을 교차하고 pre/post velocity와 stick rate를 paired 비교한다. |
| 2. Axis anisotropy와 forward/reverse asymmetry는 서로 다른 prefix를 유리하게 만드는가? | 두 물리 연구에서 방향을 측정한 방식이 다르다. [Yu, Fig. 6](https://arxiv.org/pdf/1604.04038) · [Tramsen, Eq. 2.1](https://findresearcher.sdu.dk/ws/portalfiles/portal/151776983/rsif.2017.0629.pdf) | 네 방향에서 0°/180° 차이와 0°/90° 차이를 각각 추정하고, 독립 물체에서 Brier gain을 검증한다. |
| 3. Soft·stiff 또는 rough·smooth 조합에서 friction 방향이 바뀌면 유리한 prefix의 순위도 바뀌는가? | 재료 조합에 따라 along·against 계수 차이의 부호가 반전됐다. [Tramsen, Figs. 3–5](https://findresearcher.sdu.dk/ws/portalfiles/portal/151776983/rsif.2017.0629.pdf) | 조합별 sign을 별도 friction test로 확정하고, 독립 calibration·deployment sessions에서 같은 ranking contrast를 측정한다. |
| 4. 표면을 회전하면 유리한 prefix가 표면 좌표계에 따라 움직이는가? | Object frame과 실험실 frame을 구분하자 자동 reset의 편향이 드러났다. [Ma & Rodriguez, Figs. 3–4](https://arxiv.org/pdf/1802.10089) | 표면과 물체의 회전을 독립 무작위화하고 surface frame, object frame별 signed effect를 비교한다. |
| 5. 관측 순서를 섞고 표면을 교체해도 gain의 재현성이 유지되는가? | 연속 push에서는 직전 결과가 다음 initial pose를 바꾼다. [Ma & Rodriguez, §§III–IV](https://arxiv.org/pdf/1802.10089) | 같은 방향·속력에서 reset 방식과 시행 순서를 바꾸고, 날짜·표면 session 단위로 분할해 재현성을 측정한다. |
| 6. 다음 probe 방향까지 선택하면 같은 접촉 횟수에 Brier가 더 낮아지는가? | 실물 실험에서는 action 종류에 따라 탐색의 효과가 달랐다. [Dutta et al., §IV-C](https://arxiv.org/html/2411.09020v1) | 고정 방향·선택 가능 방향 policy를 동일 action cost로 학습하고, 횟수·방향을 분리한 ablation을 수행한다. |
| 7. 추가 probe가 sensor 선택을 바꿀 후보에서 gain이 커지는가? | Uncertainty와 실제 error reduction 순위는 label 분포가 달라지면 어긋났다. [Konyushkova et al., Fig. 1](https://proceedings.neurips.cc/paper_files/paper/2017/file/8ca8da41fe1ebc8d3ca31dc14f5fc56c-Paper.pdf) | Prefix에서 예측한 query/probe score 차이로 그룹을 먼저 정하고, 독립 물체에서 action 전환율과 전환당 Brier 차이를 함께 측정한다. |
| 8. 반복 자원을 두 번째 future와 두 번째 deployment batch 중 어디에 쓸 때 새 환경의 순위가 안정되는가? | Conditional repetition의 최적 수는 비용·목표에 따라 달랐다. [Alfonsi et al., Theorem 3.2](https://arxiv.org/pdf/2209.04153) | 같은 GPU 시간에서 1×2, 2×1, 2×2를 비교하고 crossed cells의 행·열 의존성을 보존해 interval을 구한다. |
| 9. 첫 stage에서 다음 stage의 선택 기회까지 반영하면 전체 policy가 개선되는가? | Dutta et al.은 여러 step의 정보를 예상하고, Konyushkova et al.은 후보의 다음 error 감소를 학습한다. [Dutta et al., §III](https://arxiv.org/html/2411.09020v1) · [Konyushkova et al., §4](https://proceedings.neurips.cc/paper_files/paper/2017/file/8ca8da41fe1ebc8d3ca31dc14f5fc56c-Paper.pdf) | 같은 prefix·예산에서 one-step gain과 rollout gain을 label로 학습하고 독립 물체 집단의 전체 stopping loss를 비교한다. |
| 10. Simulator에서 정한 subgroup의 이득이 실제 sensor 비용을 지불할 만한가? | 실물에서는 여러 번의 동작·초기화가 필요하고 각 trial은 순서 영향을 받을 수 있다. [Dutta et al., §IV](https://arxiv.org/html/2411.09020v1) · [Ma & Rodriguez, §III](https://arxiv.org/pdf/1802.10089) | 새 표면 session을 단위로 비용과 Brier의 허용 차이를 사전 고정한 뒤 fixed 4와 adaptive 정책을 무작위배정해 접촉 시간을 포함한 latency 및 Brier를 측정한다. |

### 심사 관점에서 필요한 증거

**기여의 후보는 어느 접촉 이력에서 다음 probe에 지불할 가치가 있는지 재현 가능한 방식으로 측정하는 데 있다.** Directional asymmetry, 실물에서의 active object exploration, error reduction regression, conditional simulation의 compute 배분에는 각각 직접적인 선례가 있다. 이들을 결합한 방법에 대해서는 사전 지정한 prefix 조건의 독립 gain, 그 조건을 구분하는 데 든 비용, 실제로 바뀐 배정, 새 환경의 policy loss까지 제시해야 효용을 판단할 수 있다. [Tramsen et al.](https://findresearcher.sdu.dk/ws/portalfiles/portal/151776983/rsif.2017.0629.pdf) · [Dutta et al.](https://arxiv.org/html/2411.09020v1) · [Konyushkova et al.](https://proceedings.neurips.cc/paper_files/paper/2017/file/8ca8da41fe1ebc8d3ca31dc14f5fc56c-Paper.pdf) · [Alfonsi et al.](https://arxiv.org/pdf/2209.04153)

**남은 실물 증거**는 같은 물체군과 표면에서 prefix, 추가 calibration response, 나중 물체의 ground truth를 함께 기록한 독립 session이다. 힘·속도 자료는 simulator의 방향 효과를 점검하는 데 쓰이고, 이러한 session에서는 downstream Brier와 sensor 비용을 측정할 수 있다. 현재 실험이 추정기에 제공하는 환경 정보 가운데 실제 배포에서 알 수 있는 양을 명시하고, 나머지를 관측 기반 추정값으로 대체했을 때 어떤 결과가 유지되는지 확인할 필요가 있다. [Yu et al., §§IV–V](https://arxiv.org/pdf/1604.04038) · [Ma & Rodriguez, §§III–VI](https://arxiv.org/pdf/1802.10089) · [Dutta et al., §IV](https://arxiv.org/html/2411.09020v1)

## 재현 파일

서버 프로젝트: `datacenter3:/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/directional_prefix_v33`. [재현 순서](REPRODUCE_KO.md) · [고정 simulator](target33.py) · [평가 설계](design_eval.json) · [prefix 설계](PREFIX_PLAN_KO.md) · [paired gain 측정](prefix33.py) · [confirmatory 분석](prefix33_stats.py) · [측정 데이터 출처](physical33.json).
