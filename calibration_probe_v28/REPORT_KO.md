# v28 · Calibration probe가 velocity–friction shift와 다음 관측 선택에 주는 정보

**서로 다른 과거 world를 한 번씩 probe하면, factual evidence만으로 보이지 않던 velocity–friction shift의 방향을 식별할 수 있었다.** 64개 target cohort에서 0회는 방향을 정할 수 없었다. 1·4·16회의 정답은 각각 48/64, 55/64, 61/64였다. 16회에서 50/64개 cohort는 한 방향에 95% 넘는 posterior probability를 부여했고 이 중 방향 오류는 0건이었다. 이는 시험한 shift 크기와 sensor에서의 관측값이다.

**최종 예측의 Brier와 관측 선택의 Brier는 다른 결론을 주었다.** 0회 대비 native Brier 개선을 처음 검출한 budget은 4회였다. Neutral K256 대비 개선을 처음 검출한 budget은 16회였고, 그 차이는 −0.00542 [−0.01050, −0.00035]였다. 그러나 16회와 neutral이 고른 관측을 같은 known-prior decoder로 채점하면 차이는 −0.00118 [−0.00538, +0.00303]이었다. 16회에서 선택한 action은 0회와 8.1%의 worlds에서 달랐으나, 사전 판정 기준에서 관측 선택 자체가 더 유용해졌다고 확인한 budget은 없었다.

**접촉이 비싸다면 4회와 16회의 차이가 중요하다.** 4회의 native Brier는 0.10937, 16회는 0.10855였다. 0회→16회의 평균 개선 중 약 86%가 첫 4회에서 나왔다. 동일한 환경에서 128개 test worlds를 처리할 때 두 방법의 총 probe 횟수는 test world당 0.126과 0.214였고, warm GPU p90은 모두 약 3.26 ms였다. 16회가 모든 배포 조건에서 경제적인지는 실제 접촉·reset 비용과 환경이 유지되는 기간에 달려 있다.


## 1. 무엇을 검증했나

v27에서는 모든 factual collision이 stick했다. 경계의 local slope가 s, restitution이 e일 때 outgoing velocity는 (−e/(1+s²), es/(1+s²))다. Friction coefficient는 관측식에서 빠지고, 두 friction spread 집단에서 restitution 중심을 서로 바꿔도 population의 factual evidence 분포는 같다. 그 때문에 factual worlds를 늘리는 것만으로는 velocity–friction 관계의 방향을 구별할 수 없었다. V28은 calibration 단계에서 다른 입사속도의 충돌을 일으켜 slip을 만들고, 그 반응이 **다른 미래 world**의 관측 선택에도 유용한지 묻는다. [v27 결과](../population_shift_v27/REPORT_KO.md) · [contact model](../within_calibration_v25/physics25.py)

αc는 profile offset과 curvature spread 사이의 정렬, αf는 restitution과 friction spread 사이의 정렬이다. α=1은 원래의 관계이고, 0은 독립, −1은 완전 반전이다. 넓고 좁은 두 집단의 조건부 중심을 반대 집단 값으로 바꿀 확률이 (1−α)/2다. 이번 실험은 αf의 부호와 크기가 다른 여덟 target laws에서 수행했다. 네 쌍은 αc를 고정하고 αf의 부호만 반전한다. 추가 controls는 (0,0)과 (1,1)이다. 같은 simulator와 sensor calibration을 사용했고 모든 physical worlds·calibration contexts·numerical bank seeds를 새로 뽑았다.

각 환경의 독립 batch 8개마다 factual worlds 1,024개를 먼저 관측했다. 미리 무작위로 뽑은 서로 다른 16개 world를 한 줄로 정렬하고 앞에서부터 0·1·4·16개까지만 probe response를 공개했다. 관측 하나는 두 성분의 noisy outgoing velocity다. 같은 물체를 여러 번 측정하지 않았다. 평가 worlds는 batch당 별도 128개다. 정책은 각 평가 world에서 정밀 query view와 probe 중 하나를 고르고, 그 관측을 사용해 다른 충돌의 stick probability를 예측한다. Brier는 이 확률과 binary outcome의 squared difference다. Target은 64개 독립 calibration cohort, 평가 8,192 worlds이며, controls를 포함하면 80개 cohort와 10,240개 평가 worlds다. 각 action의 observation noise를 32번 반복하고 모든 정책에서 같은 potential observations를 공유했다. [사전 protocol](PROTOCOL.md) · [실험 설정](design.json)

## 2. 적은 수의 probe를 어떻게 posterior에 반영했나

Factual-only world에는 p(x | αc)를, probe를 본 calibration world에는 같은 물체의 joint likelihood p(x,y_probe | αc,αf)를 한 번씩 적용했다. 두 가지 geometry 정렬 × 두 가지 friction 정렬로 얻은 4개 component의 likelihood를 사용했다. 8,192개의 별도 simulator 후보로 적분하고, 균일한 독립 hyperprior에서 401 × 201개 grid 지점으로 두 환경 계수의 joint posterior를 구했다. Probe가 0회이면 v27의 factual estimator를 그대로 호출한다. Friction 방향의 posterior probability는 이때 정확히 0.5다.

Calibration evidence로 두 계수가 종속될 수 있으므로 미래 world의 각 component 확률은 posterior에서 (1±αc)(1±αf)/4를 평균해 구했다. K256은 그 네 확률로 만든 inner/outer banks에서 query view와 probe의 risk를 비교한다. 모든 정책에서 같은 acquisition budget과 numerical seeds를 사용했다. Neutral K256은 (0,0)을, known-prior K256은 실제 환경의 두 α를 입력받는다. Known-prior도 개별 world의 숨은 상태나 참 label을 볼 수 없으며 finite-bank comparator다. 두 baseline은 사전 calibration probe를 쓰지 않는다. 각 cohort의 평가 action을 먼저 저장하고 hash를 계산한 뒤, 평가 label과 추가 관측을 생성했다. [Estimator](estimator28.py) · [GPU runner](runner28.py)

## 3. Direction은 몇 번부터 보이는가

| Calibration probes | 방향 정확도 | 동시 95% 범위¹ | 정답 방향에 준 평균 posterior probability | 95% 이상 확신한 비율 | 고확신 오답 수 / 64 | Friction MAE |
|---|---|---|---|---|---|---|
| 0 | 50.0% | 동률, 결정 보류 | 50.0% | 0.0% | 0 | 0.693 |
| 1 | 75.0% | 61.3%–88.7% | 62.8% | 0.0% | 0 | 0.521 |
| 4 | 85.9% | 79.7%–92.2% | 78.3% | 34.4% | 0 | 0.344 |
| 16 | 95.3% | 90.3%–100.0% | 93.3% | 78.1% | 0 | 0.150 |

¹ 0회에서는 방향을 선택하지 않아 동률에 0.5점을 부여했다. 나머지는 posterior probability가 0.5를 넘은 쪽으로 판정했다. 8개 numerical bank/replicate blocks를 10,000회 재추출하고, 사전 지정한 1·4·16회의 정확도에 대해 simultaneous interval을 구했다. 64개 cohort마다 direction 예측은 하나다. 평가 world 수나 observation noise 반복 수를 direction 검정의 표본수로 세지 않았다.

독립 power 실험에서는 αc를 알려 주고 4개 ±αf 쌍과 null에 각 256개씩 새 batch를 생성했다. 총 2,304개 batch이며 주 실험의 Brier 집계에 넣지 않았다. 각 칸은 **paired factual+probe / probe velocity만 / 잠재 정렬 bit를 완벽하게 읽는 oracle**의 방향 정확도다. 각 부호에 같은 가중치를 줬다. Oracle은 부호 후보의 크기를 안다고 가정한 기대 상한이다. 유한 Monte Carlo 관측값은 이론적 기댓값 주위에서 변동한다.

| αf 절댓값 | 0회 | 1회 | 4회 | 16회 |
|---|---|---|---|---|
| 0.35 | 50.0% / 50.0% / 50.0% | 68.4% / 68.8% / 67.5% | 76.2% / 77.0% / 75.2% | 90.2% / 89.8% / 92.3% |
| 0.65 | 50.0% / 50.0% / 50.0% | 77.7% / 77.5% / 82.5% | 91.4% / 91.4% / 91.9% | 100.0% / 99.8% / 99.8% |
| 0.85 | 50.0% / 50.0% / 50.0% | 86.9% / 87.1% / 92.5% | 97.7% / 97.9% / 98.4% | 100.0% / 100.0% / 100.0% |
| 0.92 | 50.0% / 50.0% / 50.0% | 86.9% / 88.3% / 96.0% | 98.6% / 98.2% / 99.5% | 100.0% / 100.0% / 100.0% |

실제 αf=0일 때 95%보다 높은 확률로 방향을 선언한 비율도 측정했다.

| Null에서 발생한 방향 경보 | 0회 | 1회 | 4회 | 16회 |
|---|---|---|---|---|
| 주 실험 (8 batches) | 0.0% | 0.0% | 0.0% | 0.0% |
| 독립 검산 (256 batches) | 0.0% | 0.0% | 8.6% | 9.4% |

구체적인 비교 대상은 1,024개 factual evidence만 있는 환경, noisy paired probe가 추가된 환경, probe 그 자체만 보는 환경이다. Probe에는 normal과 tangential response가 모두 들어 있어 단독으로도 정보를 가질 수 있다. [각 law별 Wilson interval 및 sign power](results/power28.json)

## 4. Held-out Brier와 observation selection

| 정책 | Native Brier ↓ | 공통 known decoder Brier ↓ | Test에서 probe 선택 | 0회와 다른 선택 | Known과 다른 선택 |
|---|---|---|---|---|---|
| 0회 | 0.11440 | 0.11065 | 9.0% | 0.0% | 13.3% |
| 1회 | 0.11232 | 0.10972 | 9.5% | 6.1% | 11.9% |
| 4회 | 0.10937 | 0.10760 | 9.5% | 7.4% | 10.5% |
| 16회 | 0.10855 | 0.10831 | 8.9% | 8.1% | 9.2% |
| Neutral K256 | 0.11397 | 0.10948 | 0.0% | 9.0% | 12.1% |
| Known-prior K256 | 0.10315 | 0.10315 | 12.0% | 13.3% | 0.0% |

Native는 각 방법이 자기 prior로 선택하고 자기 decoder로 예측한 값이다. 공통 decoder는 동일한 known-prior decoder가 각 방법의 선택을 다시 채점한 값이다. 두 정책이 같은 action을 고른 world에서는 공통 decoder loss도 동일하므로, 이 열의 차이는 관측 선택의 효과로 해석할 수 있다. 아래 Δ는 왼쪽 방법의 Brier에서 오른쪽 방법의 Brier를 뺀 값이다. 음수일수록 왼쪽이 유리하다.

| 대비 | Native Δ [simultaneous 95%] | 공통 decoder Δ [simultaneous 95%] |
|---|---|---|
| 0회 − Neutral K256 | +0.00044 [-0.00449, +0.00536] | +0.00117 [-0.00329, +0.00563] |
| 1회 − Neutral K256 | -0.00165 [-0.00681, +0.00350] | +0.00024 [-0.00416, +0.00465] |
| 4회 − Neutral K256 | -0.00460 [-0.01033, +0.00114] | -0.00189 [-0.00700, +0.00323] |
| 16회 − Neutral K256 | -0.00542 [-0.01050, -0.00035] | -0.00118 [-0.00538, +0.00303] |
| 0회 − Known-prior K256 | +0.01126 [+0.00499, +0.01753] | +0.00751 [+0.00155, +0.01347] |
| 1회 − Known-prior K256 | +0.00917 [+0.00368, +0.01467] | +0.00658 [+0.00193, +0.01123] |
| 4회 − Known-prior K256 | +0.00623 [+0.00038, +0.01208] | +0.00445 [-0.00106, +0.00996] |
| 16회 − Known-prior K256 | +0.00540 [+0.00081, +0.01000] | +0.00516 [+0.00088, +0.00944] |
| 1회 − 0회 | -0.00209 [-0.00716, +0.00299] | -0.00093 [-0.00536, +0.00350] |
| 4회 − 0회 | -0.00503 [-0.00900, -0.00106] | -0.00306 [-0.00705, +0.00094] |
| 16회 − 0회 | -0.00586 [-0.00992, -0.00180] | -0.00235 [-0.00632, +0.00162] |

사전 기준에 따라 0회와 neutral 양쪽보다 upper bound가 0 아래로 간 최초 budget은 native에서 **16회**, 공통 decoder에서 **검출 없음**다. Grid 사이를 보간하지 않았다. 유의성은 실질적 차이와 구분한다. 기존 연구용 margin 0.003만큼의 실질 개선은 해당 대비의 upper bound가 −0.003보다 작은지를 기준으로 읽을 수 있다.

동일 replicate block에 속하는 여덟 환경을 함께 재추출하고, 그 안의 test worlds를 방법 간 짝을 유지해 재추출했다. 10,000회 draw, cohort별 4,096개 bootstrap world-mean을 재사용했다. 사전 지정한 22개 native/common 평균 차이에서 표준오차로 나눈 최대 절댓값을 기준으로 구간을 만들었다. 수치는 여덟 target laws에 같은 가중치를 준 평균이며 새로운 shift family의 모집단에 대해 무작위 표집한 구간은 아니다. [원 통계](results/summary.json) · [통계 코드](analyze28.py)

### 선택과 decoder가 각각 설명하는 부분

각 world에서 두 정책이 고른 action을 두 정책의 decoder에 각각 넣었다. Action과 decoder를 바꾸는 두 순서에 대해 변화량을 평균한 Shapley decomposition이다. 성분의 합은 native Brier 차이와 정확히 같다. 다음 구간은 12개 탐색적 성분에 대한 동시 95% 범위다.

| 대비 | Action 성분 | Decoder 성분 |
|---|---|---|
| 1회 − 0회 | -0.00068 [-0.00512, +0.00375] | -0.00140 [-0.00286, +0.00005] |
| 4회 − 0회 | -0.00240 [-0.00628, +0.00149] | -0.00264 [-0.00403, -0.00124] |
| 16회 − 0회 | -0.00205 [-0.00602, +0.00193] | -0.00381 [-0.00540, -0.00222] |
| 1회 − Neutral K256 | +0.00028 [-0.00372, +0.00428] | -0.00193 [-0.00362, -0.00024] |
| 4회 − Neutral K256 | -0.00130 [-0.00595, +0.00335] | -0.00330 [-0.00498, -0.00162] |
| 16회 − Neutral K256 | -0.00084 [-0.00465, +0.00297] | -0.00459 [-0.00650, -0.00267] |

16회−neutral의 native 차이 −0.00542 중 action 성분은 −0.00084, decoder 성분은 −0.00459였다. Decoder 성분의 구간은 [−0.00650, −0.00267]이었다. 두 정책이 같은 결정을 내려도 calibration이 예측 확률을 개선할 수 있음을 보여준다. 8개 replicate 중 하나씩 제외한 16회−0회 native 차이도 −0.00645에서 −0.00493이었다. [분해 수치](results/decomposition.json) · [계산 코드](decomposition28.py)

## 5. 어느 환경에서 결정이 달라졌나

| (αc, αf) | 용도 | 0회 | 1회 | 4회 | 16회 | Neutral | Known | 공통 decoder 16회−0회 |
|---|---|---|---|---|---|---|---|---|
| (-0.65, -0.85) | target | 0.10620 | 0.10375 | 0.10152 | 0.09804 | 0.11080 | 0.09389 | -0.00573 |
| (-0.65, 0.85) | target | 0.11284 | 0.10706 | 0.10405 | 0.10331 | 0.11605 | 0.09151 | -0.00608 |
| (0.35, -0.65) | target | 0.11738 | 0.11894 | 0.11958 | 0.11527 | 0.11766 | 0.11320 | +0.00196 |
| (0.35, 0.65) | target | 0.11681 | 0.11738 | 0.11417 | 0.11205 | 0.11490 | 0.10458 | +0.00059 |
| (-0.2, -0.35) | target | 0.12424 | 0.12698 | 0.12383 | 0.12481 | 0.11838 | 0.12062 | +0.00017 |
| (-0.2, 0.35) | target | 0.11598 | 0.11915 | 0.11305 | 0.11486 | 0.11189 | 0.11013 | +0.00023 |
| (0.72, -0.92) | target | 0.11017 | 0.09898 | 0.09712 | 0.09933 | 0.10954 | 0.09190 | -0.00648 |
| (0.72, 0.92) | target | 0.11162 | 0.10631 | 0.10167 | 0.10070 | 0.11253 | 0.09934 | -0.00346 |
| (0, 0) | control | 0.11506 | 0.11745 | 0.11431 | 0.11531 | 0.11120 | 0.11120 | +0.00038 |
| (1, 1) | control | 0.09186 | 0.08606 | 0.07985 | 0.08070 | 0.11725 | 0.07372 | -0.00734 |

보조 실험에서는 첫 세 replicate의 24개 target cohort에서 동일한 16회 calibration과 3,072개 test worlds를 다시 비교했다. Friction-only는 0회에서 얻은 profile marginal을 고정하고 16회의 friction marginal만 적용한다. Deranged pairing은 각 world에 다른 world의 probe response를 매칭해 원래의 paired likelihood를 적용한다. 후자는 marginal을 정확히 모델링한 probe-only estimator와 목적이 다르다.

| 동일 부분표본 | Native Brier | 공통 decoder Brier | 방향 정확도 |
|---|---|---|---|
| 0회 | 0.11487 | 0.11205 | 50.0% |
| 16회 | 0.10936 | 0.10947 | 91.7% |
| Friction-only | 0.11072 | 0.11044 | 91.7% |
| Deranged pairing | 0.10985 | 0.10979 | 95.8% |
| Neutral K256 | 0.11277 | 0.10937 | — |
| Known-prior K256 | 0.10280 | 0.10280 | — |

| 부분표본 대비 | Native Δ [탐색적 동시 95%] | 공통 decoder Δ [탐색적 동시 95%] |
|---|---|---|
| 16회−0회 | -0.00550 [-0.01015, -0.00085] | -0.00258 [-0.00729, +0.00214] |
| 16회−friction-only | -0.00136 [-0.00482, +0.00210] | -0.00096 [-0.00412, +0.00220] |
| 16회−deranged | -0.00049 [-0.00445, +0.00348] | -0.00031 [-0.00424, +0.00361] |
| Friction-only−0회 | -0.00414 [-0.00911, +0.00082] | -0.00161 [-0.00671, +0.00348] |

부분표본에는 독립 blocks가 3개뿐이다. 작은 차이와 null result의 정밀도가 낮다. [대조군 구현](controls28.py) · [대조 실험 수치](results/controls_summary.json)

### 같은 비용의 더 단순한 estimator

16회 paired estimator와, probe를 받지 않은 1,008개 factual worlds에서 αc를 추정하고 나머지 16개 world의 probe velocity만으로 αf를 추정하는 marginal estimator를 같은 24개 cohort에서 비교했다. 후자는 두 marginal posterior를 곱한다. Paired와 marginal의 방향 정확도는 각각 91.7%, 91.7%였다. Native Brier는 0.10936 대 0.10883, 차이 +0.00054 [-0.00208, +0.00315]였다. 공통 decoder의 차이는 +0.00057 [-0.00195, +0.00309]였다. 두 방법의 action은 4.5%에서 달랐다. 이 결과와 probe-only power를 함께 보면 더 복잡한 paired estimator의 필요성을 별도로 입증해야 한다. 이 대조의 GPU p90은 측정하지 않았다. [Marginal baseline](marginal28.py) · [수치](results/marginal_summary.json)

## 6. GPU latency와 물리적 probe 비용

| 정책 | Warm GPU p90 (ms) | Neutral 대비 ratio [95%] | Calibration fit (ms) | Fit + bank setup (ms) | 128 worlds에 setup 배분한 p90 (ms) | Calibration probes / 128 worlds | 총 probe 횟수 / test world² |
|---|---|---|---|---|---|---|---|
| 0회 | 3.262 | 0.999 [0.997, 1.001] | 23.1 | 68.3 | 3.698 | 0 | 0.090 |
| 1회 | 3.267 | 1.001 [0.998, 1.002] | 14.2 | 55.7 | 3.700 | 1 | 0.103 |
| 4회 | 3.264 | 1.000 [0.998, 1.002] | 17.6 | 59.0 | 3.724 | 4 | 0.126 |
| 16회 | 3.264 | 1.000 [0.998, 1.001] | 29.1 | 70.5 | 3.816 | 16 | 0.214 |
| Neutral K256 | 3.265 | 1.000 [1.000, 1.000] | 0.0 | 40.5 | 3.583 | 0 | 0.000 |
| Known-prior K256 | 3.262 | 0.999 [0.997, 1.001] | 0.0 | 40.4 | 3.579 | 0 | 0.120 |

² 총 probe 비용은 환경당 calibration B회를 H=128개 배포 worlds에 배분한 B/H에, test 시 probe 선택률을 더했다. Query view를 선택하면 추가 접촉은 0건이다. 모든 정책이 world마다 추가 observation 하나를 읽는다. Warm p90은 host input, acquisition, 선택한 observation의 GPU 전송 및 최종 prediction을 포함한다. 각 정책에 12 worlds × 2 repeats × 64 target cohorts = 1,536개의 batch-1 synchronized timings을 사용했다. 실행 순서는 world 안에서 무작위였다. Ratio는 1,500회 paired bootstrap의 pointwise 95% interval이다. Zero-probe와 positive-budget에서 posterior 적분 코드가 다르고 setup은 고정 순서로 측정했으므로, fit 시간의 미세한 순위를 probe당 계산비로 해석하기는 어렵다. 초기 simulator library 구성 1655.8 ms는 1회 비용이다. Fit·bank를 배분한 p90은 실측 warm sample에 cohort setup/H를 더한 계산값이다. 실제 장비의 접촉·이동·reset 시간은 측정하지 않았다.

한 번의 접촉에 Brier 단위 λ를 매긴다면 예상 비용은 native Brier + λ × (B/H + test probe rate)다. 다음은 H=128에서 관측된 평균으로 계산한 손익분기값이다. 양의 추가 접촉과 양의 Brier 개선이 함께 있을 때 λ가 표의 손익분기값보다 작아야 유리하다. 음수이거나 접촉을 줄이는 경우는 해당 열의 부호와 원래 식으로 판단해야 한다. 가격에 대해 실제 사용자 선호나 유의성을 검정한 값은 없다.

| 비교 기준 | Budget | Native Brier 개선 | 추가 calibration 접촉 / world | 총 추가 접촉 / world | Calibration만 가격화한 λ* | 모든 probe를 가격화한 λ* |
|---|---|---|---|---|---|---|
| probe0 | 1 | +0.00209 | 0.0078 | +0.0132 | 0.2670 | 0.1582 |
| probe0 | 4 | +0.00503 | 0.0312 | +0.0366 | 0.1610 | 0.1374 |
| probe0 | 16 | +0.00586 | 0.1250 | +0.1240 | 0.0469 | 0.0472 |
| neutral | 1 | +0.00165 | 0.0078 | +0.1027 | 0.2112 | 0.0161 |
| neutral | 4 | +0.00460 | 0.0312 | +0.1261 | 0.1471 | 0.0364 |
| neutral | 16 | +0.00542 | 0.1250 | +0.2135 | 0.0434 | 0.0254 |

![Direction identification, loss, and physical interactions](results/v28.png)

## 7. 논문 기여 가능성과 다음 검증

**논문에서 가장 잘 지지되는 결과는 calibration interaction의 정보 가치와 그 정보가 쓰이는 단계를 동시에 정량화한 것이다.** Factual stick regime은 friction과 restitution의 관계를 감춘다. 같은 장면에 tangential excitation을 주면 outgoing probe velocity에 두 물리량의 신호가 함께 나타난다. 16회에서 friction MAE는 0.693에서 0.150으로 줄었고, neutral 대비 최종 Brier가 개선됐다. 이 개선의 action 성분과 decoder 성분은 크기와 불확실성이 달랐다. 절대 Brier 0.003을 실용적 개선 기준으로 잡으면 neutral 대비 16회 개선의 simultaneous lower magnitude는 약 0.00035여서 그 크기까지 뒷받침하지는 못한다.

**관측 선택을 개선하는 새 알고리즘으로 주장할 근거는 부족하다.** 고정된 known decoder에서 neutral을 이긴 budget은 없고, 16회에서도 known-prior K256보다 native Brier가 +0.00540 [ +0.00081, +0.01000 ] 높았다. 특히 약한 ±0.35 shift에서는 두 방향 모두 16회 native Brier가 neutral보다 높았다. 반면 ±0.85와 ±0.92에서는 개선이 컸다. 이러한 환경별 차이는 가설을 만드는 descriptive evidence다. αc도 쌍 사이에서 달라지므로 αf 크기 하나만의 인과 효과로 귀속할 수 없다. V28은 한 가지 Coulomb law, 동일 sensor noise, 알려진 component family를 사용했다. 실물 접촉에서 population의 변화가 유지되는지 확인하지 않았다.

**다음 검증은 decision에 정보가 필요한 환경을 먼저 가려내는 것이다.** 독립적인 development environments에서 calibration을 멈출 규칙을 사전에 정한다. 남은 αf uncertainty가 감소해도 query view와 probe의 순위가 변하지 않는 환경에서는 일찍 멈추고, 순위가 바뀔 수 있고 loss 차이가 큰 환경에서만 다음 probe를 구매한다. 독립 test에서는 fixed 4·16회, neutral, probe-only likelihood를 같은 총 접촉 budget과 공통 decoder에서 비교한다. Sensor noise, restitution–friction support overlap, contact law를 각각 바꾸고 실제 reset 시간까지 측정해, 비용을 반영한 selection 이득이 재현되는지를 논문 진행 기준으로 삼을 수 있다. 현재 결과는 물리적 식별 한계와 probe 예산별 trade-off를 다룬 실험 논문에 가깝다. 방법 자체의 신규성은 별도 비교가 필요하다.


## 선행연구와 논문 기여의 범위

### 1. 논문별 비교

검토 범위는 아래 원문 다섯 편이다. 표면에서 측정한 friction, 표본으로 추정한 population proportion, 다음 관측이 갖는 decision utility는 서로 다른 측정 대상이다.

| 연구 | Research question · methodology | Sample · data | 결과 · 저자의 결론 | v28에 적용할 때의 한계 |
|---|---|---|---|---|
| Sundaralingam & Hermans, *In-Hand Object-Dynamics Inference using Tactile Fingertips* | Fingertip으로 힘을 가해 slip을 유발하고 접촉력과 surface normal로 friction coefficient를 추정했다. 독립적인 tilting-platform 측정과 비교했다. | YCB objects 8개, 물체당 10 trials. 두 용기는 plastic과 label 표면에 각각 접촉했다. | 전체 median friction error는 0.0056이었다. 힘을 받을 때 표면이 휘어진 물체에서는 오차가 커졌다. | 같은 표면의 local coefficient를 추정했다. 별도 물체들에서 population coupling을 추정하거나 미래 world의 decision을 평가하지 않았다. [원문 §§IV–V, Fig. 6](https://arxiv.org/pdf/2003.13165#page=6) |
| Fernandes Vaz, Izbicki & Stern, *Quantification Under Prior Probability Shift* | 소수의 labeled target samples와 unlabeled target samples를 함께 써서 prevalence를 추정할 수 있는가? 두 추정량의 MSE로 convex combination의 가중치를 정했다. | §3.1의 simulation에서 target prevalence는 0.3, 관측한 target labels는 10·20·30·40·50개였다. | 두 추정량의 오차가 비슷할 때 결합 후 오차가 줄었다. 하나가 이미 매우 정확한 경우에는 결합 결과가 그 방법과 비슷했다. | Label은 직접 관측된다. v28의 calibration probe는 friction group을 간접적으로 알려주는 noisy velocity다. [원문 Theorem 15, Fig. 5](https://jmlr.org/papers/volume20/18-456/18-456.pdf#page=14) |
| Ahmad & Yu, *Active Sensing as Bayes-Optimal Sequential Decision-Making* | 오답, 추가 관측 시간, sensor 위치 변경 비용을 함께 줄이기 위해 C-DAC의 stopping·sensing 정책을 dynamic programming으로 구했다. | 세 target 위치의 simulated visual search, peripheral vision을 넣은 확장 과제. 기본 belief grid는 두 축 각각 201 bins였다. | 기본 과제에서 switch cost가 0이면 Infomax와 행동이 비슷했다. 비용을 추가하면 같은 정확도에서 C-DAC가 위치 변경을 줄였다. | 한 instance 안에서 sensing을 배분한다. 여러 미래 worlds에 걸쳐 calibration 비용이 상쇄되는 조건은 따로 계산해야 한다. [원문 §§2–3, Figs. 2–3](https://www.auai.org/uai2013/prints/papers/85.pdf) |
| Huang et al., *Amortized Bayesian Experimental Design for Decision-Making* | 다음 design과 최종 decision을 함께 내는 Transformer Neural Decision Process를 downstream utility로 학습했다. | Retrosynthesis planning 사례에서 training molecules 1,500개 × 70 routes, 평가 molecules 50개. T=10, top-3 utility였다. | 이 사례에서 random search보다 높은 route utility를 보고했다. Decision을 기준으로 acquisition을 평가한 선례다. | Acquisition과 최종 inference가 함께 바뀐다. 관측 선택만 검증하려면 decoder를 고정한 비교가 필요하다. [원문 §3, Appendix G.3, Fig. A2](https://proceedings.neurips.cc/paper_files/paper/2024/file/c59f05d7ab3638b138cc61f32e1a7cd1-Paper-Conference.pdf#page=20) |
| Zhao et al., *Amortized Bayesian Experimental Design with In-Context Knowledge Conditioning* | 이전 instance의 관측과 명시적 prior를 Transformer에 제공하고 reliability filter로 현재 acquisition에 반영했다. | 네 과제. CES Table 11은 이전 instances 5개, 10 rounds, 2,000 rollouts다. | Gaussian noise에서 relevant instances의 sPCE는 13.93, 이전 instance가 없으면 13.46, irrelevant instances에서는 13.14였다. 관측의 관련성에 따라 효용이 달라졌다. | Relevant instance의 parameter가 현재 parameter 근방에 생성된다. v28의 calibration world와 test world는 population law를 공유하는 별개의 iid draws다. [2026-11-08 공개 원문 §§3–4, Appendix E.6](https://arxiv.org/html/2611.11512v1) |

### 2–3. Literature review와 knowledge map

실물에서 slip을 유발하면 같은 surface의 friction에 관한 신호가 생긴다. Population estimation에서는 일부 target sample에서 더 풍부한 measurement를 얻는 것이 도움이 될 수 있다. 다른 instance에서 얻은 정보가 현재 acquisition에 쓰일 수 있다는 점에도 선례가 있다. 이 연구들을 v28에 적용하려면 **calibration world의 factual velocity와 같은 world의 probe response → 환경의 velocity–friction coupling에 대한 posterior → 별도 test world의 query/probe 선택 → 선택한 관측으로 stick probability 갱신 → held-out Brier**를 각각 검증해야 한다. 이 연결에서 probe를 받은 과거 물체의 반응은 미래 물체의 정답을 직접 알려주지 않는다. [Sundaralingam & Hermans](https://arxiv.org/pdf/2003.13165#page=7) · [Vaz et al.](https://jmlr.org/papers/volume20/18-456/18-456.pdf#page=14) · [Zhao et al.](https://arxiv.org/html/2611.11512v1)

C-DAC는 추가 관측과 sensor 이동, downstream error에 각각 비용을 부여했고 TNDP는 downstream utility를 직접 정의했다. V28에서 calibration probe B회를 같은 환경의 미래 H개 worlds에 재사용하고 test에서 probe를 고른 비율을 q라고 하면, world당 기대 총 contact probe 수는 B/H + q다. 두 단계의 probe 단가가 다르고 query view에도 비용이 있으면 B·c_cal/H + q·c_probe + (1−q)·c_view로 가중한다. GPU 시간은 별도 단위로 계측해야 한다. [Ahmad & Yu, Eq. 3](https://www.auai.org/uai2013/prints/papers/85.pdf#page=3) · [Huang et al., §3](https://proceedings.neurips.cc/paper_files/paper/2024/file/c59f05d7ab3638b138cc61f32e1a7cd1-Paper-Conference.pdf)

### 4–7. Research gaps, 상충하는 결과, methodology와 evidence synthesis

| 관찰된 패턴 | 비교에 필요한 가정 · 남은 gap | v28의 판정 |
|---|---|---|
| 소수 target labels가 prevalence 추정에 도움이 된 반면, 이전 instances의 관련성이 낮은 조건에서는 추가 정보가 acquisition score를 낮췄다. [Vaz et al., Fig. 5](https://jmlr.org/papers/volume20/18-456/18-456.pdf#page=14) · [Zhao et al., Table 11](https://arxiv.org/html/2611.11512v1) | 전자는 같은 target 모집단의 실제 label, 후자는 관련성이 달라지는 다른 instance의 history다. 측정값도 prevalence error와 sPCE로 다르다. | Nested probes에서 posterior sign accuracy와 held-out Brier를 따로 추적한다. 동일 배치 가정이 깨질 때의 transfer는 별도 실험으로 남는다. |
| Slip을 일으킨 표면에서 friction 추정이 가능했고, deformable surface에서는 정확도가 달라졌다. [Sundaralingam & Hermans, Fig. 6](https://arxiv.org/pdf/2003.13165#page=7) | Simulation의 sliding Coulomb contact와 fingertip force/normal 측정은 noise 및 excitation 조건이 다르다. 실제 물체에서 필요한 B는 이 결과로 정해지지 않는다. | Factual-only null과 유료 관측의 sign accuracy를 대조한다. 실물 검증에서는 표면·물체 단위 분할과 독립적인 friction reference가 필요하다. |
| Switching이 무료인 기본 과제에서는 Infomax와 C-DAC가 비슷했지만, 유료이면 비용을 반영한 정책이 행동을 바꿨다. TNDP는 다른 과제에서 decision utility를 직접 평가했다. [Ahmad & Yu, Figs. 2–3](https://www.auai.org/uai2013/prints/papers/85.pdf) · [Huang et al., Fig. A2](https://proceedings.neurips.cc/paper_files/paper/2024/file/c59f05d7ab3638b138cc61f32e1a7cd1-Paper-Conference.pdf#page=20) | Brier가 내려가도 미래 worlds가 적으면 setup 비용이 상쇄되지 않을 수 있다. 정보량, latency, Brier를 한 단위로 단순 합산할 근거는 없다. | Neutral과 B=0 대비 위험 차이, warm GPU p90, one-time fit, calibration와 deployment의 contact 수를 제시한다. 최적 B라는 결론에는 특정 단가와 H가 필요하다. |

직접적인 실물 근거는 slip excitation과 surface별 friction 추정에 있다. Population estimation 개선에는 가정을 특정한 simulation 및 통계적 근거가 있고, 이전 관측을 재사용하는 근거는 별도의 design 과제에서 나왔다. 이 다섯 편의 근거만으로는 calibration probe 1·4·16회 가운데 어느 것이 미래 Coulomb world의 Brier를 가장 낮추는지, 어느 횟수가 비용을 상쇄하는지 정해지지 않는다. 위 표에서 사용한 friction error, proportion MSE, visual-search switches, sPCE, route utility는 합산하지 않았다. [Sundaralingam & Hermans](https://arxiv.org/pdf/2003.13165) · [Vaz et al.](https://jmlr.org/papers/volume20/18-456/18-456.pdf) · [Ahmad & Yu](https://www.auai.org/uai2013/prints/papers/85.pdf) · [Huang et al.](https://proceedings.neurips.cc/paper_files/paper/2024/file/c59f05d7ab3638b138cc61f32e1a7cd1-Paper-Conference.pdf) · [Zhao et al.](https://arxiv.org/html/2611.11512v1)

### 8. 후속 연구 질문 열 가지

| 질문 · 중요한 이유 | 변수와 검증 방법 | 근거 |
|---|---|---|
| 1. 어느 budget부터 population 방향이 재현되는가? 약한 효과에서는 likelihood가 겹칠 수 있다. | 절대 coupling strength와 B를 교차하고 sign accuracy, posterior probability, interval을 독립 batches에서 평가한다. | [Vaz et al., §3.1](https://jmlr.org/papers/volume20/18-456/18-456.pdf#page=14) |
| 2. Factual–probe pairing 자체의 가치는 얼마인가? 단일 채널로 얻은 이득도 있을 수 있다. | 같은 world의 paired likelihood, 별도 world의 probe marginal, factual-only를 비교한다. | [Sundaralingam & Hermans, §IV](https://arxiv.org/pdf/2003.13165#page=6) |
| 3. 한 표면을 반복 측정할지 여러 표면을 한 번씩 잴지에 따라 추정이 다른가? Sensor noise와 population sampling noise의 비용이 다르다. | 총 접촉 수를 고정하고 독립 worlds 수와 동일-world noise repeats를 바꾼다. | [Sundaralingam & Hermans, §IV](https://arxiv.org/pdf/2003.13165#page=6) · [Vaz et al., §3.1](https://jmlr.org/papers/volume20/18-456/18-456.pdf#page=14) |
| 4. Sign을 맞힌 환경에서 Q/P 순위도 바뀌는가? 방향 판별은 discrete action 변화를 보장하지 않는다. | Sign posterior, predicted risk gap, common-decoder regret를 같은 test worlds에서 연결한다. | [Huang et al., §3](https://proceedings.neurips.cc/paper_files/paper/2024/file/c59f05d7ab3638b138cc61f32e1a7cd1-Paper-Conference.pdf) |
| 5. 어려운 배치에 probe를 집중하면 어떤가? 쉬운 환경에서는 16회가 낭비일 수 있다. | H와 단가를 고정하고 고정 B와 value-of-information stopping의 평균·최대 접촉 수와 Brier를 비교한다. | [Ahmad & Yu, Eq. 7](https://www.auai.org/uai2013/prints/papers/85.pdf#page=3) |
| 6. 같은 B에서 어느 calibration world를 골라야 하는가? 무작위 배정은 최적 선택을 보장하지 않는다. | Random world와 factual-only expected utility 선택을 새 배치에서 같은 cost로 비교한다. | [Huang et al., §3](https://proceedings.neurips.cc/paper_files/paper/2024/file/c59f05d7ab3638b138cc61f32e1a7cd1-Paper-Conference.pdf) |
| 7. Calibration이 얼마나 오래 유효한가? 환경이 바뀌면 과거 probe가 오도할 수 있다. | Regime 변화 시점과 혼입률을 조작하고 reset·sliding window·누적 posterior의 prequential Brier를 비교한다. | [Zhao et al., Appendix E.6](https://arxiv.org/html/2611.11512v1) |
| 8. Calibration과 deployment의 sensor noise가 다르면 과신하는가? | 두 단계의 noise를 독립적으로 조작하고 credible interval coverage와 wrong-sign confidence를 조사한다. | [Zhao et al., Table 11](https://arxiv.org/html/2611.11512v1) |
| 9. 관측된 이득에 실제로 지불할 만한가? 물체 reset이 GPU 계산보다 오래 걸릴 수 있다. | 실측 c_cal, c_probe, c_view와 H로 각 budget의 Brier–cost frontier를 작성한다. | [Ahmad & Yu, Eq. 3](https://www.auai.org/uai2013/prints/papers/85.pdf#page=3) |
| 10. 다른 contact law와 실제 표면에서도 같은 신호를 얻는가? 표면 휨이나 자세 오차가 friction으로 잘못 설명될 수 있다. | Object·surface-disjoint evaluation에서 독립 tilt reference를 수집하고 contact law 및 likelihood misspecification을 교차한다. | [Sundaralingam & Hermans, §V](https://arxiv.org/pdf/2003.13165#page=7) |

### 9. Research assistant review

**가장 방어 가능한 기여 후보는 calibration probe가 factual evidence에서 숨겨져 있던 population 관계를 어느 budget에서 드러내고, 그 정보 중 얼마가 별도의 held-out world에서 관측 선택에 쓰였는지를 측정하는 것이다.** Slip으로 friction을 측정하는 것, 소수 target measurements로 prior를 갱신하는 것, 여러 instance의 관측으로 acquisition을 바꾸는 것에는 선례가 있다. Paired likelihood가 같은 world의 기존 factual observation을 두 번 세지 않는지 검산하고, B=0 대비 효과와 neutral 대비 효과를 각각 확인해야 한다. Native Brier와 common-decoder Brier가 다르면 selection 개선과 final decoding 개선을 분리해야 한다. [Sundaralingam & Hermans](https://arxiv.org/pdf/2003.13165) · [Vaz et al.](https://jmlr.org/papers/volume20/18-456/18-456.pdf#page=14) · [Huang et al.](https://proceedings.neurips.cc/paper_files/paper/2024/file/c59f05d7ab3638b138cc61f32e1a7cd1-Paper-Conference.pdf) · [Zhao et al.](https://arxiv.org/html/2611.11512v1)

**현재 simulator에서 직접 유도한 sign 판별 상한.** Factual stick에서는 두 velocity 성분으로 slope s와 restitution e를 구할 수 있다. Probe가 slide할 때 같은 slope의 접선·법선 방향으로 impulse를 분해하면 그 크기의 비는 friction coefficient μ다. e의 두 중심 주변 분포가 겹치지 않고, μ의 narrow/wide 조건부 support도 겹치지 않는다. 따라서 noiseless한 한 쌍에서는 해당 world가 원래 restitution–friction 정렬을 따랐는지 나타내는 binary indicator를 복구할 수 있다. Hidden indicator는 Bernoulli((1+αf)/2)를 따른다. αc가 같고 절댓값 a가 알려진 두 환경 +a, −a를 같은 확률로 비교하는 경우, 완전한 indicator를 B개 읽은 최선의 sign 정확도는 P(K>B/2)+0.5P(K=B/2), K∼Binomial(B,(1+a)/2)다. 예를 들어 a=0.35에서 B=0·1·4·16의 값은 50.0%, 67.5%, 75.2%, 92.3%다. 이는 random iid calibration worlds의 이론적 **기대 상한**이다. 연속 α의 복구 보장이나 유한 반복에서 측정한 sign 정확도의 hard cap은 아니다. 작은 B에서는 sensor noise를 없애도 population sampling 자체의 오차가 남는다. [물리 모델 및 support](../within_calibration_v25/physics25.py) · [Coulomb impact](../factual_tail_v23/physics23.py) · [coupling flip 생성](../proxy_shift_v26/physics26.py)


## 재현 자료

서버 경로는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/calibration_probe_v28`이다. 할당된 GPU 세 장의 process·메모리·사용률을 확인했고 두 장에는 다른 작업이 있어 node55의 RTX PRO 6000 Blackwell Server Edition 한 장만 사용했다. Primary, independent power, mechanism controls를 같은 GPU에서 순차 실행했다. 각 guard는 외부 프로세스 출현 여부와 worker return code를 기록했다. CPU 통계는 할당된 compute node에서 실행했다. [Primary 종료](gpu_worker/results/worker_exit.json) · [Power 종료](power_worker/results/worker_exit.json) · [Controls 종료](controls_worker/results/worker_exit.json) · [Power 코드](power28.py) · [대조군 통계](controls_stats28.py) · [대표 cohort의 동결·원자료 hashes](results/strong_neg_r0/metrics.json)
