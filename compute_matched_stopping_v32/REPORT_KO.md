# v32 · 같은 학습 시간에서 접촉 이력의 수와 gain label의 정밀도 비교

**같은 학습 시간에서 2×2 평균 label로 학습해도 전체 stopping policy의 Brier는 실용적으로 개선되지 않았다. 1×1, fixed 4, uncertainty-only와의 비교 모두에서 0.003 개선은 simultaneous interval 밖이었다.**

첫 seed에서 scalar gain을 한 번 측정한 1×1은 597개 환경, 2개 futures와 2개 batches를 교차한 2×2는 255개 환경을 봤다. 실제 label 생성·fit 시간은 539.34초와 537.57초였다. 두 정책, fixed 4, uncertainty-only의 held-out Brier는 각각 0.13868, 0.13891, 0.13877, 0.13887였다. 2×2 − 1×1 차이는 +0.00023 [-0.00034, +0.00080]였다.

같은 prefix에서 2×2 평균으로 label을 만들면 conditional noise RMS는 2→4 stage에서 1×1의 61.6%, 4→8 stage에서 60.9%가 됐다. 별도 source data로 처음부터 세 번씩 학습했을 때 repeated − broad Brier는 +0.00023, -0.00023, +0.00034였고, 평균은 +0.00011였다.

여덟 환경의 전체 stopping·acquisition·decoding을 모두 실행한 GPU p90은 1×1 1491 ms, 2×2 1490 ms, fixed 4 1418 ms, uncertainty-only 1489 ms였다. 네 방법 모두 한 block에서 calibration 32회와 deployment probe 104회를 지불했다.

## 실험 설계와 비용

예측 대상은 하나의 환경에서 이미 관측한 접촉 이력 H다. 현재 과정에서 멈추거나 다음 2회 또는 4회의 접촉을 구매하면, 이후 **다른 128개 물체**에서 어떤 관측을 선택할지 달라진다. 각 물체에서 정확한 boundary profile인 query view 또는 다른 입사 속도의 충돌 뒤 2D outgoing velocity인 probe를 얻고, 별도 충돌에서 stick할 확률을 예측한다. 128개 중 13개에만 probe를 쓸 수 있다. 각 training label은 같은 128개 물체·정답·관측 noise로 구한 **Brier(멈춤) − Brier(계속)**의 평균이다. 양수이면 더 접촉한 덕에 이후 예측이 나아졌다. 최종 policy 비교에는 Brier loss 자체를 쓰므로 작을수록 좋다.

- **여러 이력에 한 번씩 측정하는 1×1**에서는 매 prefix에서 독립적인 추가 접촉 경로 한 개와 독립적인 배포 집단 한 개로 gain을 구한다. **이력당 2×2**에서는 같은 prefix에서 접촉 경로 두 개와 128개씩의 배포 집단 두 개를 새로 뽑아 조합 네 개의 gain을 평균한다. 한 행은 같은 접촉 경로를, 한 열은 같은 배포 집단을 공유한다. 서로 다른 두 arm의 이력과 bank seed는 독립적으로 뽑는다.
- 모든 calibration 물체 1,024개의 noisy factual evidence를 먼저 본다. 그중 이미 지불한 2회 또는 4회의 접촉이 prefix다. 추가 접촉은 같은 pool의 아직 구매하지 않은 물체에서 무복원 추출한다. 각 stage의 continuation은 독립적으로 표집한다. 각 후보는 자신의 prefix에 포함된 정보로 score를 내며, 평가 simulator만 hidden geometry·friction·restitution·target label을 볼 수 있다.
- 두 scorer 모두 Coulomb law에서만 학습한 stage별 15-feature ridge regression이다. 입력은 geometry·velocity proxy coupling의 posterior mean/SD와 그 함수, 이미 지불한 접촉 응답의 통계다. v30 feature 정의를 재사용하고 λ=3, train-only normalization, unpenalized intercept를 공통으로 고정했다. 두 training corpora는 label 생성과 fit에 같은 540 GPU wall-seconds를 배정한다. 약 0.5초를 fit에 남기고 새 환경을 추가할 수 없을 때 멈춘다. 종료 결정은 경과 시간과 최근 처리 속도만 본다. 두 방식의 실제 label 수와 소비 시간을 결과에서 확인한다.
- Held-out에서는 같은 여덟 환경에 대해 전체 stopping policy를 적용한다. Adaptive methods는 모두 두 번씩 접촉한 뒤 score 상위 네 곳에 두 번씩 더 주고, 이 네 곳의 **업데이트된 score** 상위 두 곳에 네 번씩 더 준다. 결과는 환경당 2·2·2·2·4·4·8·8회다. Fixed 4는 모두 네 번씩, uncertainty-only는 각 stage에서 friction coupling의 posterior SD가 큰 순서로 같은 방식으로 배정한다. 모든 policy가 calibration 32회, deployment 104회를 써서 총 136회다. 각 block에서 어느 환경이 8회까지 갈지는 해당 policy가 직접 정한다.
- 세 평가 법칙에 한 번도 학습하지 않았다. Static–kinetic split은 static/kinetic friction을 분리하고, speed weakening은 접선 속력에 따라 마찰계수가 변한다. 두 법칙은 이전 버전의 연구자가 이미 분석했다. 이번에 새로 정한 directional law는 접선 속도 부호에 따라 μ_eff = μ[1+0.23 tanh(vₜ/0.28)]로 바뀐다. 이 설정에서 factual collision은 stick을 유지하되 calibration probe는 slide하고 별도 query collision의 label이 달라질 수 있다. 세 법칙에서 모두 acquisition과 common decoder는 source Coulomb을 가정한다. Common decoder는 환경의 실제 population coupling을 받는 privileged 기준이다. 실제 deployment에서 이 계수를 정확히 안다고 가정할 근거는 따로 필요하다.
- 법칙마다 독립적인 16개 block, block마다 여덟 coupling 환경, 환경마다 128개 deployment worlds를 생성한다. 총 384개 환경, 49,152개 물체다. Noise 32회와 같은 물체의 여러 potential observations는 독립 표본 수에 더하지 않는다. 정책 할당 및 observation choices를 파일에 저장한 다음 ground truth와 sensor observations를 평가한다. 법칙별 16개 block을 함께 재표집하고 세 법칙에 같은 가중치를 준다.
- GPU p90은 여덟 환경 전체(1,024개 future worlds)의 wall latency다. 각 법칙의 처음 6개 block에서 네 정책의 순서를 무작위로 섞고 warm-up 한 회 뒤 세 회씩 측정한다. Full 경로는 이미 도착한 factual 및 지불한 probe response에서 시작해 매 정책마다 posterior, stopping decision, selector bank, acquisition, 선택한 관측 decoding을 실제로 다시 실행한다. Warm 경로는 같은 run 안에서 bank 준비가 끝난 시점부터 확률이 host에 도착할 때까지다. 두 경로 모두 CUDA를 동기화하고, 모든 action과 1,024개 확률을 사전 고정한 결과와 대조한다. 실제 sensor/reset 대기, 평가 데이터 생성, 공통 detector와 privileged common decoder 초기화는 포함하지 않는다.

Directional law는 물리적으로 가능한 shift를 만드는 합성 stress test다. Yu et al.은 네 표면에서 sliding 방향을 바꾸어 마찰을 실측했고 polyurethane에서 가장 큰 마찰과 가장 작은 마찰의 비가 약 1.5였다고 보고했다. 이 관측은 방향을 바꾸어 검증할 동기를 준다. 여기 쓴 signed velocity 함수와 계수를 그 자료에 fit하지 않았고 실제 충돌에서의 Brier를 검증한 것도 아니다. [Yu et al., §V, Fig. 6](https://arxiv.org/pdf/1604.04038)

$$
G_{f,b}(H) = R_b(H) - R_b(H \mathbin{+} C_f),\qquad
\widehat{G}_{2\times2}(H) = \tfrac14\sum_{f=1}^2\sum_{b=1}^2 G_{f,b}(H).
$$

여기서 H는 이미 지불한 접촉 이력, C는 같은 pool에서 구입한 추가 접촉, R은 그 이력에서 결정한 관측으로 얻은 deployment Brier다. 2×2를 평균해도 source simulator, 잘못 지정된 contact law, 공통 decoder 또는 K256 근사가 만드는 체계적 오차는 남을 수 있다.

| Training seed | 학습 방식 | 독립 환경 수 | Stage별 prefix 합계 | Gain cells | Label 생성 (s) | Ridge fit (ms) | 청구 GPU 시간 (s) | 별도 warm-up (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | 1×1 · 더 많은 이력 | 597 | 1194 | 1194 | 539.33 | 8.23 | 539.34 | 2.34 |
| 1 | 2×2 · 반복 평균 | 255 | 510 | 2040 | 537.56 | 2.88 | 537.57 | 2.12 |
| 2 | 1×1 · 더 많은 이력 | 595 | 1190 | 1190 | 538.70 | 9.27 | 538.71 | 1.16 |
| 2 | 2×2 · 반복 평균 | 254 | 508 | 2032 | 537.83 | 2.90 | 537.83 | 2.10 |
| 3 | 1×1 · 더 많은 이력 | 601 | 1202 | 1202 | 539.19 | 9.26 | 539.20 | 1.91 |
| 3 | 2×2 · 반복 평균 | 256 | 512 | 2048 | 537.40 | 3.09 | 537.40 | 2.11 |

한 환경에서 2회·4회 prefix를 각각 하나씩 만든다. 2×2의 gain cell 네 개는 같은 prefix에 대한 한 개 label로 평균한다. 각 cell은 128개 deployment worlds 위에서 계산한 한 개 risk 변화량이다. Cell 수 ×128은 고유 물체 수가 아니다. 표의 시간은 같은 모델 GPU를 단독 사용하며 동기화한 wall time이다. Data generation, CPU posterior 처리, acquisition, Brier 계산, block 저장이 label 시간에 들어간다. GPU kernel time 또는 FLOPs와는 다르다. 초기 공통 detector 구성과 shape별 warm-up 및 audit 시간은 별도다. 두 arm의 전체 wall time은 남은 예산에 새 환경을 하나 더 처리할 수 있는지에 따라 조금 달라진다. [고정한 학습 예산](training_plan.json) · [GPU 학습 코드](train_labels32.py) · [GPU ridge fit](fit32.py)

첫 seed에서 model fit 전에 쓰는 파일 읽기·hash까지 포함하면 1×1 540.115초, 2×2 537.829초다. 공통 posterior detector 초기화 1.515초와 CUDA solver warm-up은 양쪽의 배분 시간 밖이다. 비교는 동일한 540초 allowance와 실제 소비량을 기준으로 하며, 물리 실험에서의 수집 시간은 측정하지 않았다.

## Held-out Brier loss와 실제 GPU p90

| 정책 | 세 law 평균 Brier | Static–kinetic | Speed weakening | Directional | 전체 p90 ms [95%] | Deployment p90 ms [95%] |
|---|---|---|---|---|---|---|
| 1×1 · 더 많은 이력 | 0.13868 | 0.14179 | 0.12996 | 0.14428 | 1491.4 [1485.4, 1496.1] | 951.3 [950.1, 951.9] |
| 2×2 · 반복 평균 | 0.13891 | 0.14165 | 0.13008 | 0.14500 | 1489.5 [1486.5, 1490.9] | 950.3 [949.3, 950.5] |
| Fixed 4 | 0.13877 | 0.14171 | 0.13013 | 0.14448 | 1418.2 [1414.5, 1419.1] | 950.3 [949.3, 951.5] |
| Uncertainty-only | 0.13887 | 0.14177 | 0.13039 | 0.14444 | 1488.5 [1482.9, 1491.1] | 950.5 [949.2, 951.4] |

첫 training seed로 고정한 모델의 결과다. 한 번의 full call은 8개 환경의 1,024개 물체를 처리한다. 법칙당 6개 독립 timing block ×3회, 정책당 54개 실측으로 p90을 구했다. Batch p90을 1,024로 나누면 단일 물체의 streaming p90이 되지 않는다. Warm p90과 full p90은 각각의 시간 분포에서 계산했다.

| Brier 차이 | 전체 [5개 simultaneous 95%] | Static–kinetic [95%] | Speed weakening [95%] | Directional [95%] |
|---|---|---|---|---|
| 2×2 − 1×1 | +0.00023 [-0.00034, +0.00080] | -0.00014 [-0.00095, +0.00063] | +0.00013 [-0.00074, +0.00103] | +0.00072 [+0.00017, +0.00130] |
| 1×1 − Fixed 4 | -0.00010 [-0.00081, +0.00062] | +0.00008 [-0.00071, +0.00096] | -0.00017 [-0.00131, +0.00085] | -0.00020 [-0.00114, +0.00076] |
| 2×2 − Fixed 4 | +0.00014 [-0.00068, +0.00095] | -0.00007 [-0.00108, +0.00099] | -0.00004 [-0.00144, +0.00124] | +0.00052 [-0.00033, +0.00145] |
| 1×1 − Uncertainty-only | -0.00019 [-0.00070, +0.00032] | +0.00002 [-0.00066, +0.00069] | -0.00044 [-0.00119, +0.00036] | -0.00015 [-0.00081, +0.00039] |
| 2×2 − Uncertainty-only | +0.00004 [-0.00056, +0.00064] | -0.00013 [-0.00085, +0.00058] | -0.00031 [-0.00137, +0.00071] | +0.00056 [-0.00004, +0.00115] |

각 차이에서 음수는 앞쪽 정책의 loss가 더 낮다는 뜻이다. 한 law 안의 16개 block을 통째로 재표집하는 paired bootstrap을 10,000회 수행했다. 법칙별 결과는 pointwise interval이다. 다섯 primary 대비에는 같은 resample로 구한 최대 표준화 편차를 써 simultaneous interval을 계산했다. 두 이전 law와 새 directional law의 결과를 구분했다. Reference practical margin은 absolute Brier 0.003이다. [사전 통계 계획](STATISTICAL_PLAN_KO.md) · [전체 효과와 p90 ratio 구간](results_eval/paired_stats32.json)

![Held-out Brier loss contrasts and measured GPU p90](results_eval/v32.png)

세 target 법칙에서 같은 latent world의 Coulomb 결과와 실제 stick label이 달라진 물체의 비율은 각각 11.2%, 5.9%, 11.9%였다. 전체 repeated − broad 차이를 그 비율까지 곱한 Brier 기여로 나누면, label이 유지된 집단에서 +0.00005, 바뀐 집단에서 +0.00018였다. 이 그룹은 물성과 선택 난도가 달라질 수 있어 law mismatch의 인과 효과로 해석하지 않는다. [대조 설계](SHIFT_PLAN_KO.md) · [law별 분해 결과](results_eval/shift32.json)

## 독립적으로 세 번 학습하면 유지되는가?

| Brier 차이 | Seed 1 | Seed 2 | Seed 3 | 평균 [target block simultaneous 95%] | 평균 [training·target crossed simultaneous 95%] |
|---|---|---|---|---|---|
| 2×2 − 1×1 | +0.00023 | -0.00023 | +0.00034 | +0.00011 [-0.00027, +0.00049] | +0.00011 [-0.00054, +0.00077] |
| 1×1 − Fixed 4 | -0.00010 | +0.00026 | -0.00009 | +0.00003 [-0.00061, +0.00066] | +0.00003 [-0.00071, +0.00076] |
| 2×2 − Fixed 4 | +0.00014 | +0.00003 | +0.00024 | +0.00014 [-0.00050, +0.00077] | +0.00014 [-0.00057, +0.00085] |
| 1×1 − Uncertainty-only | -0.00019 | +0.00017 | -0.00018 | -0.00007 [-0.00053, +0.00039] | -0.00007 [-0.00066, +0.00052] |
| 2×2 − Uncertainty-only | +0.00004 | -0.00006 | +0.00015 | +0.00004 [-0.00041, +0.00050] | +0.00004 [-0.00051, +0.00059] |

세 training seed의 절대 Brier 평균은 1×1 0.13880, 2×2 0.13891, fixed 4 0.13877, uncertainty-only 0.13887였다. 같은 target block을 공유한다. Seed 2·3은 독립 source data 생성부터 실제 GPU fit까지 반복하고, 동일한 evaluation archive에서 각 모델이 선택한 4곳과 2곳을 순차적으로 다시 구해 해당 action의 loss를 적용했다. 첫 번째 seed의 GPU 정책과 replay가 같은 counts와 Brier를 냈는지 검산했다. 첫 구간은 세 학습 쌍을 고정하고 target blocks만 바꿨을 때다. 마지막 구간은 training seed pair와 law별 target block을 함께 재표집한 탐색적 구간이다. 학습 반복이 세 쌍이므로 다른 모든 training seeds에 대한 정밀한 보장을 주기 어렵다. [재학습 계획](REPEATED_TRAINING_PLAN_KO.md) · [모델별 계산 결과와 seed 제외 분석](results_eval/full_retrain32.json)

### Training seed와 contact law를 함께 바꾼 경우

Loss 차이는 repeated(2×2)에서 broad(1×1)를 뺀 값이다. 양수는 repeated의 Brier가 더 높음을 뜻한다. Primary seed 1의 directional 결과를 확인한 후 이 분석을 정했고, 나머지 두 seed를 새로 학습했다. 세 seed 모두 같은 48개 target blocks에서 평가했다.

| Training seed | split_static | speed_weak | directional | directional − historical 평균 |
|---|---:|---:|---:|---:|
| 1 | -0.000144 [-0.000942, +0.000633] | +0.000129 [-0.000730, +0.001056] | +0.000716 [+0.000165, +0.001276] | +0.000724 [-0.000090, +0.001559] |
| 2 | -0.000515 [-0.001613, +0.000709] | -0.001299 [-0.002619, +0.000044] | +0.001120 [+0.000045, +0.002335] | +0.002027 [+0.000607, +0.003522] |
| 3 | +0.000236 [-0.000537, +0.000963] | -0.000033 [-0.001138, +0.001092] | +0.000802 [-0.000054, +0.001595] | +0.000701 [-0.000362, +0.001754] |
| 3 seed 평균 | -0.000141 [-0.000910, +0.000605] | -0.000401 [-0.001605, +0.000666] | +0.000880 [+0.000288, +0.001628] | +0.001150 [+0.000203, +0.002355] |

각 seed의 괄호는 같은 law 안에서 16개 target blocks를 복원추출한 pointwise 95% interval이다. 세 seed 평균에는 독립 training seed pair도 같이 재표집했다. 각 draw에서 target block index는 세 seed에 공통으로 적용했다. Historical 평균에는 split_static과 speed_weak의 가중치를 절반씩 주었다. Independent training seed가 3개이고, 개별 law·stage 비교의 multiplicity를 조정하지 않았으므로 crossed intervals는 탐색적으로 해석한다.

## Stage별 ΔBrier 기여

각 policy에 대해 4회까지 진행한 환경에서 R4−R2, 8회까지 진행한 환경에서 R8−R4를 계산하고, 각 합을 전체 환경 8개로 나눈다. 아래에서는 두 policy의 stage별 손실 변화를 뺀다. 4→8 기여에는 앞 stage에서 살아남은 환경의 구성이 반영된다.

| Training seed / law | 2→4 기여 | 4→8 기여 | 전체 ΔBrier |
|---|---:|---:|---:|
| 1 / split_static | -0.000509 | +0.000365 | -0.000144 |
| 1 / speed_weak | +0.000107 | +0.000022 | +0.000129 |
| 1 / directional | +0.000547 | +0.000169 | +0.000716 |
| 1 / 세 law 평균 | +0.000048 | +0.000186 | +0.000234 |
| 2 / split_static | -0.000363 | -0.000152 | -0.000515 |
| 2 / speed_weak | -0.001217 | -0.000082 | -0.001299 |
| 2 / directional | +0.000438 | +0.000682 | +0.001120 |
| 2 / 세 law 평균 | -0.000381 | +0.000150 | -0.000231 |
| 3 / split_static | -0.000245 | +0.000480 | +0.000236 |
| 3 / speed_weak | -0.000057 | +0.000024 | -0.000033 |
| 3 / directional | +0.000355 | +0.000448 | +0.000802 |
| 3 / 세 law 평균 | +0.000018 | +0.000317 | +0.000335 |
| 세 seed·세 law 평균 | -0.000105 [-0.000608, +0.000287] | +0.000217 [-0.000145, +0.000585] | +0.000113 [-0.000426, +0.000588] |

각 block의 실제 ΔBrier와 두 stage의 합 사이 최대 잔차는 4.16e-17였다. 전체 48개 block의 probability로 각 observation의 Brier를 다시 계산하고 stable top-13과 순차적 stopping을 재생했다. 독립 코드가 full-budget 재학습 분석의 policy별 Brier, 다섯 contrasts, 세 seed pooled target-only 및 crossed intervals를 절대 오차 2×10⁻¹⁴ 이내로 재현했다.

[분석 계획](SEED_LAWS_PLAN_KO.md) · [전체 추정치와 stage별 interval](results_eval/seed_laws32.json) · [독립 검증 코드](seed_laws32.py)

## 반복으로 label이 안정돼도 예측과 배분이 좋아지는가?

| Stage | 단일 gain의 조건부 noise RMS | 2×2 평균의 조건부 noise RMS | RMS 비율 [95%] | 독립 diagonal 간 환경 순위 Spearman |
|---|---|---|---|---|
| 2→4 | 0.00738 | 0.00455 | 0.616 [0.591, 0.636] | 0.045 |
| 4→8 | 0.00701 | 0.00427 | 0.609 [0.588, 0.628] | 0.022 |

위 표는 첫 학습 쌍의 진단이다. 아래는 각 training seed에서 새로 생성한 prefix로 같은 추정을 반복한 값이다. 비율이 1보다 작으면 반복 평균이 단일 label보다 정밀하다.

| Training seed | 2→4 RMS 비율 [pointwise 95%] | 4→8 RMS 비율 [pointwise 95%] |
|---|---|---|
| 1 | 0.616 [0.591, 0.636] | 0.609 [0.588, 0.628] |
| 2 | 0.624 [0.599, 0.644] | 0.620 [0.596, 0.639] |
| 3 | 0.633 [0.610, 0.651] | 0.633 [0.612, 0.649] |

같은 repeated-arm source prefix 안에서 future·batch·interaction variance를 각각 추정해 단일 측정과 교차 평균의 noise를 Brier 단위로 구했다. 두 축을 겹치지 않게 뽑은 대각 cell 쌍의 순위는 법칙 전체의 순위와 구분한다. 이는 source training label 자체의 변화다. [정밀도 추정](results_train/label_precision.json)

### 학습한 contact law에서도 추가 probe의 가치를 예측했는가?

**독립적으로 재생한 Coulomb 이력 48개에서 두 scorer 모두 추가 접촉의 평균 이득과 양의 상관을 보이지 않았다.** 이 비교는 source law를 그대로 둔 채 훈련에 등장하지 않은 prefix로 바꾼 것이다. 따라서 target law의 변화만으로 약한 예측력을 모두 설명하기는 어렵다. 이 표본에서 예측할 수 있었던 신호의 크기는 여전히 불확실하다. [사전 분석 계획](SOURCE_REFERENCE_PLAN_KO.md) · [수치와 interval](results_reference/source_reference32.json)

v31의 finite-pool panel에서 서로 독립인 Coulomb blocks 6개를 골랐다. 각 block의 8개 환경은 모두 포함했다. 환경마다 이미 지불한 2개 또는 4개 접촉에서 나온 15개 features로 v32의 두 frozen scorer를 각각 실행했다. 실제 label은 같은 prefix에서 추가 접촉 경로 4개와 배포 집단 4개를 교차해 얻은 16개 Brier gain의 평균이다. 한 cell의 gain은 같은 배포 집단·decoder로 측정한 추가 접촉 전 Brier에서 접촉 후 Brier를 뺀 값이다. v31의 모든 historical labels는 v32 fit에서 배제했다. [v31 측정 설계](../probe_gain_reliability_v31/REPORT_KO.md) · [CPU scoring 코드](source_reference32.py)

| Stage | Scorer | MSE ×10⁶ Brier² | 해당 훈련 평균만 쓴 MSE ×10⁶ | MSE 차이 ×10⁶ [95%] | Pearson r [95%] |
|---|---|---:|---:|---|---|
| 2→4 | Broad 1×1 | 9.013 | 8.649 | +0.365 [−0.379, +1.092] | −0.054 [−0.292, +0.180] |
| 2→4 | Repeated 2×2 | 10.002 | 9.787 | +0.216 [−0.482, +0.795] | −0.118 [−0.373, +0.095] |
| 4→8 | Broad 1×1 | 9.604 | 9.619 | −0.015 [−0.285, +0.242] | −0.033 [−0.235, +0.170] |
| 4→8 | Repeated 2×2 | 10.292 | 10.244 | +0.048 [−0.397, +0.440] | −0.023 [−0.239, +0.197] |

MSE 차이는 각 scorer에서 자기 arm의 intercept-only MSE를 뺀 값이다. 어느 stage에서도 feature를 이용해 훈련 평균 한 개를 쓰는 것보다 MSE가 낮아졌다고 판단할 수 없었다. Repeated−Broad MSE 차이는 2→4에서 +0.989 [−0.361, +2.338], 4→8에서 +0.688 [+0.096, +1.217]이었다(모두 ×10⁻⁶ Brier²). 4→8에서 두 arm의 intercept-only MSE 차이만으로도 +0.625×10⁻⁶이었다. 같은 stage에서 실제 held-out 평균 gain은 −0.000912였고 훈련 평균은 Broad +0.000605, Repeated +0.000799였다. 이 stage의 MSE 우열에는 평균값을 맞힌 정도가 크게 기여한다. 환경 사이 순서를 더 잘 예측했다고 볼 근거는 약하다. Pearson r의 Repeated−Broad 차이도 두 stage에서 각각 −0.064 [−0.248, +0.082], +0.009 [−0.180, +0.277]이었다. [전체 지표와 pointwise intervals](results_reference/source_reference32.json)

**어디까지 해석할 수 있나.** V31의 4×4 평균에도 finite-label noise가 남으며 두 축을 공유하는 16개 cell을 독립 표본으로 셀 수 없다. Pearson correlation이 약하면 예측 가능한 신호가 작을 수도 있고 label이 부정확할 수도 있다. 표의 interval은 6개 blocks를 10,000회 paired bootstrap한 pointwise 구간이다. Scorer를 다시 훈련했을 때의 변동과 임의의 다른 물리 법칙은 포함하지 않는다. 4→8에는 모든 환경의 이미 관측한 4회 prefix가 들어간다. 실제 두 단계 정책에서 첫 선택을 통과한 환경만을 평가한 결과와는 모집단이 다르다. 이 표로 최종 deployment Brier나 전체 stopping trajectory의 기준선을 대체할 수 없다. [v31 반복 실험](../probe_gain_reliability_v31/REPORT_KO.md) · [v32 통계 계획](STATISTICAL_PLAN_KO.md)

**재현 기록.** 서버 node56의 allocation 2744469에서 CUDA를 비활성화한 CPU float64로 재구성했다. 당시 저장한 v30 예측값의 최대 절대 오차는 5.64×10⁻¹⁷, posterior friction SD의 최대 절대 오차는 2.89×10⁻¹⁵였다. 원본 prefix·metric hashes와 훈련 기록의 prefix seed 비중복을 확인했다. 새 모델을 학습하거나 historical labels로 parameter를 조정하지 않았다. [Feature 봉인 및 검산](results_reference/prefix_seal.json) · [원본 지문과 model lock 지문](results_reference/source_reference32.json)

### 같은 training prefix에서 averaging만 바꾼 비교

| 관측 집단 | 평균 label scorer | 4개 단일 cell scorer의 Brier 범위 | 평균 label − 단일 cell 정책 평균 [pointwise 95%] |
|---|---|---|---|
| 세 law 평균 | 0.13891 | 0.13878–0.13941 | -0.00010 [-0.00043, +0.00025] |
| Static–kinetic split | 0.14165 | 0.14159–0.14202 | -0.00016 [-0.00055, +0.00025] |
| Speed weakening | 0.13008 | 0.12946–0.13091 | +0.00008 [-0.00074, +0.00098] |
| Directional friction | 0.14500 | 0.14469–0.14574 | -0.00022 [-0.00063, +0.00021] |

동일한 repeated-arm prefix에서 네 cell 중 하나만 골라 학습한 네 정책과 네 cell의 평균으로 학습한 정책을 비교했다. Ridge의 score는 네 single-cell score의 평균과 계산 오차 안에서 같다. 그러나 top-half 선택은 비선형이므로 평균 score가 선택한 물체와 네 policy가 선택한 결과의 평균은 달라질 수 있다. 최종 접촉 예산이 달랐던 환경 비율은 40.3%였다. 이 보조 비교는 단일-cell 쪽의 절약 시간을 추가 prefix에 쓰지 않는다. [설계](ATTRIBUTION_PLAN_KO.md) · [분석](results_eval/attribution32.json)

### 같은 probe budget을 어디에 썼는가?

| 정책 | 2회에서 멈춰 생긴 Brier 차이 | 8회까지 계속해 생긴 Brier 차이 | 동일 histogram의 무작위 배분 대비 차이 |
|---|---|---|---|
| 1×1 · 더 많은 이력 | +0.00004 [-0.00047, +0.00053] | -0.00013 [-0.00057, +0.00033] | -0.00038 [-0.00076, +0.00000] |
| 2×2 · 반복 평균 | +0.00008 [-0.00050, +0.00071] | +0.00005 [-0.00038, +0.00051] | -0.00014 [-0.00058, +0.00032] |
| Fixed 4 | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | -0.00028 [-0.00075, +0.00018] |
| Uncertainty-only | +0.00014 [-0.00043, +0.00071] | -0.00004 [-0.00043, +0.00034] | -0.00019 [-0.00057, +0.00019] |

앞의 두 열을 더하면 fixed 4 대비 총 Brier 차이가 된다. 마지막 열은 2·4·8회에 각각 4·2·2곳을 균등 무작위로 배정하는 정확한 기대 Brier와 비교한다. Fixed 4도 총 32회지만 환경별 횟수 분포는 다르다. 두 learned policy가 stage 1에서 고른 네 환경의 평균 겹침은 78.1%, 마지막 두 곳의 겹침은 49.0%였고, 최종 접촉 횟수는 환경의 38.5%에서 달랐다. [경로별 분석과 cutoff tie 수](results_eval/paths32.json)

**하나의 실제 결정.** Directional law 첫 block에서 두 scorer의 최종 배분이 다른 첫 번째 환경(slot 3, coupling +0.72·-0.92)을 보자. 이 환경에 구매한 접촉 수는 1×1 8회, 2×2 4회, Fixed 4 4회, Uncertainty-only 4회였다. 같은 환경의 128개 물체를 각각 2·4·8회의 접촉 후 예측했을 때 실제 Brier는 0.12231, 0.12081, 0.12062였다. 첫 2회를 받은 시점에 두 scorer가 예측한 2→4 Brier gain은 각각 +0.00101, +0.00136였다. 실제로 계속할지는 같은 block의 나머지 일곱 환경과 비교한 순위가 정한다. 이 예시는 전체 평균을 대체하지 않는다. [해당 block의 실제 결과](results_eval/directional/block_000/metrics.npz)

### 2→4회의 gain으로 첫 네 곳을 고르면 후속 기회를 놓치는가?

| 평가 | 같은 block 정답으로 단기 순위 | 같은 block 정답으로 420개 배분 탐색 | 다른 blocks로 단기 순위 | 다른 blocks로 전체 배분 결정 | Fixed 4 |
|---|---|---|---|---|---|
| 세 law 평균 | 0.13562 | 0.13517 | 0.13900 | 0.13949 | 0.13877 |
| Static–kinetic split | 0.13848 | 0.13816 | 0.14201 | 0.14246 | 0.14171 |
| Speed weakening | 0.12659 | 0.12591 | 0.12938 | 0.12992 | 0.13013 |
| Directional friction | 0.14180 | 0.14143 | 0.14562 | 0.14608 | 0.14448 |

각 행의 단위는 전체 policy Brier다. 4회까지 계속하는 이득이 작아도 8회까지 갈 기회가 중요할 수 있다. 첫 두 열은 target labels를 미리 아는 사후 계산이다. 다음 두 열은 동일 law의 나머지 15개 blocks에서 동일한 여덟 population coupling별로 stage Brier를 추정하고, 떼어둔 block에 그 배분을 적용한다. 이 참조는 law와 coupling을 안다고 가정하고 개별 prefix evidence는 읽지 않는다. 겹치는 학습 folds에 독립 표본 수를 부여하지 않았다. [보조 가설](HORIZON_PLAN_KO.md) · [결과](results_eval/horizon32.json)

부분 training corpus만 쓴 sensitivity에서는 각 arm의 source blocks를 서로소인 절반 또는 삼분의 일로 나눠 동일 ridge policy를 다시 fit했다. 다섯 계산에서 repeated − broad Brier는 -0.00021에서 +0.00034까지였다. 분모가 다른 분할이 서로 겹치므로 다섯 번의 독립 full-budget 실험으로 세지 않았다. [결과와 각 분할의 실제 비용](results_eval/source_subset_sensitivity32.json)

## 논문 기여 가능성과 다음 결정

**현재 선택은 fixed 4다.** 세 번의 독립 학습을 평균한 Brier에서 2×2 − 1×1은 +0.00011 [-0.00054, +0.00077], 2×2 − fixed 4는 +0.00014 [-0.00057, +0.00085]였다. 다섯 비교 모두에서 세 training seeds와 target blocks를 함께 재표집한 구간도 ±0.003 안에 들었다. 이 simulator와 예산에서 그 크기의 차이는 관측과 맞지 않는다. 같은 probe 비용에서 fixed 4는 중간 순위를 다시 계산할 필요가 없어, 실측한 첫 seed의 전체 GPU p90에서 2×2보다 유리했다(2×2/fixed 4: +5.0%).

**새 directional law에서의 재현성.** 개별 seed의 2×2 − 1×1 Brier는 +0.00072, +0.00112, +0.00080였다. 2×2의 손실이 더 컸던 것은 세 쌍 중 3쌍이었다. Directional에서의 차이에서 이전 두 laws의 평균 차이를 빼면 +0.00115 [+0.00020, +0.00235]였다. 이 탐색적 구간은 0을 포함하지 않았다. 첫 seed를 본 후 제기한 보조 가설이고 training seed가 세 쌍뿐이므로, 새 법칙과 seed에서 독립적인 검증이 필요하다. [개별 law와 stage별 검산](results_eval/seed_laws32.json)

**정밀한 label에서 좋은 결정까지 어디가 막혔는가.** 같은 prefix의 conditional noise는 반복 평균으로 줄었다. 독립 Coulomb reference에서는 두 scorer의 gain 예측이 해당 환경의 평균 이득과 양의 상관을 보인다는 근거가 약했다. 또한 repeated-arm이 본 바로 그 이력에서 한 개 cell만 남긴 보조 실험에서도, 네 single-cell policy 평균 대비 2×2의 Brier 차이는 −0.00010 [−0.00043, +0.00025]였다. 따라서 source label의 정밀도 향상은 확인했지만, 그것이 순위를 바꿔서 실제로 이익이 되는 환경을 알아내는 데 충분하다고 볼 근거가 없다. Stage별로 모든 score에 같은 상수를 더해 MSE를 개선하더라도 top-half 순서는 그대로일 수 있다. [독립 Coulomb 평가](results_reference/source_reference32.json) · [같은 prefix에서 label만 바꾼 분석](results_eval/attribution32.json)

**후속 기회가 stage 1의 target에서 빠져 있다.** 2→4 gain이 작은 환경이라도 8회까지 갈 가치가 있다면 첫 단계에서 보존해야 한다. 동일 held-out block의 결과를 미리 아는 사후 계산에서 420개 배분을 모두 살핀 Brier는 0.13517, 단기 이득 순으로 고른 Brier는 0.13562였다. 그러나 해당 law의 다른 15개 blocks에서 coupling별 평균을 배워 새 block에 적용하면 각각 0.13949와 0.13900이었다. 같은 block에서 발견한 0.00045 차이가 직접 배포 가능한 lookahead 이득이라는 근거는 없다. 반대로 이 실험으로 모든 non-myopic 정책의 가능성을 배제할 수도 없다. 이 진단에서 알려 준 것은 coupling이고 개별 prefix의 evidence는 쓰지 않았다. [420개 배분의 독립 block 대조](results_eval/horizon32.json)

**논문에서 주장할 수 있는 부분.** 같은 GPU 시간에서 prefix 수와 label precision 사이의 trade-off를 구현하고, 독립적인 source 예측과 unseen law의 전체 stopping 결정을 연결해 어느 단계에서 효과가 약해지는지 보였다. Ridge를 써서 관측의 가치를 회귀하거나 conditional sampling에 반복을 쓰는 것 자체는 기존 연구에도 있다. 초점은 *측정값의 noise가 줄어든 만큼 물리적 관측 예산 배분도 개선되는가*라는 검증 가능한 질문에 두는 편이 타당하다. 범위는 여덟 coupling, 두 반복 설계(1×1, 2×2), 9분의 학습 예산, 15개 feature와 고정 decoder다. 자료는 다른 반복 수, 다른 feature, 더 긴 학습 시간에 대한 최적점을 결정하지 않는다. [조건부 simulation의 계산 배분](https://arxiv.org/pdf/2209.04153) · [Error reduction 학습](https://proceedings.neurips.cc/paper_files/paper/2017/file/8ca8da41fe1ebc8d3ca31dc14f5fc56c-Paper.pdf)

**다음 실험의 판단 기준.** 실험을 이어간다면 scorer를 더 복잡하게 만들기 전에, 새 surface의 예상 이득이 실제로 다른 surface보다 크고 그 차이를 prefix에서 식별할 수 있는지를 우선 확인한다. 서로 독립적인 calibration 및 deployment 반복으로 그 차이와 실용적 크기를 측정한 뒤, 같은 32-probe budget에서 정책이 얻을 수 있는 여지가 남은 조건을 정한다. 그 조건에서 예측 대상에 4→8회의 후속 가치를 포함하거나 sensor features를 더해, 학습에 쓰지 않은 law·surface와 새로운 training seed에서 fixed 4 대비 Brier를 낮추는지 평가한다. 실용적 threshold와 모집단, calibration 비용을 후속 데이터를 열기 전에 고정해야 현재 세 법칙에서 작았던 차이에 맞춘 사후 선별을 피할 수 있다. [v31의 독립 gain 재현성](../probe_gain_reliability_v31/REPORT_KO.md)

**실험으로 구분할 수 있는 범위.** 입력은 simulator가 제공하는 noisy boundary profile과 outgoing velocity다. Raw video에서 feature를 추출해 최종 결정을 내리는 end-to-end system까지 검증한 결과는 아니다. Common decoder가 target environment의 population coupling을 알고 있는 조건은 acquisition 차이를 비교하기에는 명확하지만, 현실에서 coupling과 contact law를 모두 모르는 경우에는 성능이 달라질 수 있다. Full latency에서 제외한 실제 sensor wait·object reset과 training에서 제외한 공통 초기화 비용도 배치 규모가 바뀌면 상대적 중요도가 달라진다.

**논문으로 가져갈 때 필요한 외적 검증.** 실물에서 완전히 같은 latent world의 두 potential observations를 동시에 얻기는 어렵다. 같은 표면·물체의 reset 오차와 시간 drift를 기록하면서 calibration 물체와 deployment 물체를 분리하고, 관측 가능 횟수와 downstream decision을 고정한 paired 또는 무작위배정 설계가 필요하다. 여러 표면·속도·방향에서 pose와 force를 함께 기록한 공개 planar pushing 연구는 방향 변화와 반복 측정을 설계하는 근거가 된다. 그 데이터만으로 본 연구의 query view·probe·stick label 세 가지가 모두 재구성된다고 전제할 수 없다. [Yu et al., 원문 §§III–VII](https://arxiv.org/pdf/1604.04038)

## v32 문헌 검토 · 더 많은 접촉 이력과 더 정밀한 gain label 중 어느 쪽에 계산량을 쓸 것인가

이미 같은 표면 집단의 물체에 몇 번 접촉했다. 이때까지 얻은 관측을 **paid prefix**라고 하자. 다음 접촉을 사면 앞으로 만날 물체에서 sensor를 더 잘 고를 수도, 선택을 바꿨다가 오히려 손해를 볼 수도 있다. V32에서 학습하고 싶은 값은 지금 가진 정보로 예상한 **추가 접촉 전후의 Brier loss 차이**, 즉 **expected Brier gain**이다. 새 prefix를 생성하면 학습기가 경험하는 상황이 다양해진다. 기존 prefix를 반복 평가하면 그 한 상황에서 평균적으로 얼마를 얻는지 더 정밀하게 알 수 있다. V31의 paid-only panel에서 같은 prefix로 독립 반복한 환경 간 순서 일치율은 2→4회에서 49.1%, 4→8회에서 50.3%였다. 한 번 측정한 순위의 불안정성과 여러 이력에서 공통된 예측 규칙을 배울 수 있는지는 서로 다른 질문이다. [v30 학습과 평가](../learned_probe_value_v30/REPORT_KO.md) · [v31 반복 실험](../probe_gain_reliability_v31/REPORT_KO.md)

## 1. 어떤 연구를 비교했나

같은 예산에서 훈련 자료를 어떻게 생성하는지, noisy target이 무엇을 추정하는지, 그 예측이 실제 결정에 어떤 차이를 만드는지를 기준으로 원문 6편을 선정했다. 비교표의 수치는 각 연구의 실험 규모다. 분야가 다른 실험의 효과 크기를 합산하지 않았다.

| 논문과 연구 질문 | 방법·데이터·표본 | 관측한 결과와 저자의 결론 | v32로 옮길 때의 한계 |
|---|---|---|---|
| Alfonsi, Lapeyre & Lelong (2023). 같은 계산 예산에서 한 입력당 몇 번 조건부로 다시 뽑아야 하나? | Least-squares Monte Carlo의 asymptotic excess risk를 분석했다. Gaussian toy model, 200 time steps의 stochastic differential equation, 보험 손실 모형의 세 수치 예를 비교했다. | 입력 생성 비용 대비 후속 simulation 비용 및 regression approximation이 최적 반복 수에 영향을 주었다. 보험 예에서는 8–9회 부근이 가장 좋았고 5–20회도 비슷한 MSE였다. | 독립적인 conditional draws와 명시적인 비용 모형을 가정한다. 같은 future나 batch를 공유하는 crossed cells나 다단계 stopping에서의 최적 반복 수를 주지 않는다. [원문 Theorem 3.2 및 §5](https://arxiv.org/pdf/2209.04153) |
| Lehtinen et al. (ICML 2018). 깨끗한 target을 얻기 위해 한 장면을 오래 촬영하는 편이 유리한가? | Image denoising 실험에서 Gaussian noise와 2,000 capture units를 고정했다. 100개 이미지 × 20 noisy captures 및 1,000개 이미지 × 2 captures 등으로 CNN을 학습했다. | 동일한 100개 이미지도 noisy pair를 다양하게 만들면 성능이 올랐다. 1,000개 이미지를 쓰면 PSNR이 한 번 더 올랐다. Conditional mean이 유지될 때 noisy target으로도 squared-loss optimum을 보존할 수 있었다. | 입력·출력인 두 이미지가 같은 장면을 공유한다. 집단별 gain을 정렬해서 제한된 probe를 배정하는 실험은 아니다. [원문 §2, §3.1 및 Fig. 1c](https://proceedings.mlr.press/v80/lehtinen18a/lehtinen18a.pdf) |
| Binois, Huang, Gramacy & Ludkovski (Technometrics 2019). 다음 simulation은 이미 방문한 위치와 새 위치 중 어디에 쓸까? | Input-dependent noise를 가진 Gaussian process로 sequential design을 구성했다. 1차원 비교에서는 simulation 200회 예산으로 1,000개 independent experiments를 수행했다. 재고관리·역학 simulator도 다루었다. | 1차원 예에서 sequential heteroskedastic GP가 평균 함수의 out-of-sample RMSE에 가장 좋았다. 같은 예에서 predictive distribution의 proper score에는 random replication이 더 좋았다. 목적이 바뀌면 선호 설계도 달라졌다. | 저자들은 입력 위치를 능동적으로 정할 수 있고 공간적으로 가까운 위치 간 GP 상관을 이용한다. 임의의 acquisition history에 이 smoothness를 가정할 근거는 별도로 필요하다. [원문 §§5.2–5.4 및 Fig. 5](https://arxiv.org/pdf/1710.03206) |
| Konyushkova, Sznitman & Fua (NeurIPS 2017). Label 하나를 살 때 generalization error가 얼마나 줄어들지 배울 수 있는가? | LAL은 현재 classifier와 후보의 features로 error reduction을 회귀했다. Cold-start에서는 2D Gaussian synthetic data로 학습한 regressor를 7개 classification tasks에 적용했다. | Striatum·MRI·Credit card 세 과제에서 iterative LAL은 uncertainty sampling의 최고 점수(IoU·Dice·AUC)에 더 적은 labels로 도달했다. 새로운 label의 가치를 과거 경험으로 회귀하는 직접적인 선례다. | 구입한 label은 classifier 자체를 갱신한다. V32에서는 한 과거 물체의 응답이 이후 다른 물체들의 sensor 선택에 영향을 주며, 여러 환경이 하나의 구매 예산을 공유한다. [원문 §§3–5 및 Fig. 3](https://proceedings.neurips.cc/paper_files/paper/2017/file/8ca8da41fe1ebc8d3ca31dc14f5fc56c-Paper.pdf) |
| Elmachtoub, Liang & McNellis (ICML 2020). 비용을 정확히 예측하는 모델이 더 좋은 행동을 고르는가? | SPOT와 CART를 동일 데이터에서 비교했다. 24개 edge의 shortest-path 문제에서 조건별 10개 datasets, 훈련 200 또는 10,000개, test 1,000개였고 Yahoo! 뉴스 추천도 평가했다. | 저자들의 작은 예에서 얕은 CART가 예측값을 잘 맞히면서도 입력의 28%에서 잘못된 경로를 골랐다. Decision loss로 split을 정한 나무는 한 번의 split으로 두 경로가 바뀌는 지점을 찾았다. | 알려진 최적화 제약에 따라 한 번 행동을 고른다. 관측 이후 posterior와 다음 후보 집합이 모두 변하는 과정의 성능을 입증하지 않는다. [원문 §3.1, §§5.1–5.3](https://proceedings.mlr.press/v119/elmachtoub20a/elmachtoub20a.pdf) |
| Gulrajani & Lopez-Paz (ICLR 2021). Domain shift 결론은 model selection 방식에 얼마나 의존하는가? | DomainBed에서 7개 multi-domain benchmarks와 3개 model selection 기준을 비교했다. 정확도를 held-out domain에서 측정했다. | 공정한 tuning 아래 ERM이 여러 복잡한 대안과 경쟁했고, model selection 기준에 따라 결과가 달라졌다. Target information의 사용을 비교 조건의 일부로 밝혀야 한다고 주장했다. | 이미지 분류의 자연·합성 domain 변화다. 물리 법칙의 변화가 gain의 조건부 평균을 얼마나 바꾸는지는 평가하지 않았다. [학회 원문](https://openreview.net/pdf?id=lQdXeXDoWtI) · [저자 원고 §5](https://arxiv.org/pdf/2007.01434) |

## 2. 연구 흐름과 근거가 지지하는 범위

**Knowledge map.** Alfonsi et al.의 관측 단위는 하나의 입력과 그 입력에서 갈라진 conditional draws다. Lehtinen et al.은 이 구조에서 고정된 capture budget이 있을 때 서로 다른 입력 수를 늘리는 효과를 보여 준다. Binois et al.은 입력별 noise가 다르면 반복 위치까지 조절한다. 세 연구가 만들어 낸 training labels를 LAL처럼 미래 error reduction의 supervision으로 쓸 수 있다. 그 scorer가 정한 배분을 SPOT처럼 실제 decision loss로 평가하고, DomainBed처럼 target data에 접근한 시점을 명시하면 v32의 실험 질문이 완성된다. 마지막 연결은 이 연구에 적용한 제안이다. [조건부 simulation](https://arxiv.org/pdf/2209.04153) · [capture budget](https://proceedings.mlr.press/v80/lehtinen18a/lehtinen18a.pdf) · [sequential design](https://arxiv.org/pdf/1710.03206) · [LAL](https://proceedings.neurips.cc/paper_files/paper/2017/file/8ca8da41fe1ebc8d3ca31dc14f5fc56c-Paper.pdf) · [SPOT](https://proceedings.mlr.press/v119/elmachtoub20a/elmachtoub20a.pdf) · [DomainBed](https://openreview.net/pdf?id=lQdXeXDoWtI)

**근거가 강한 부분.** H를 현재 prefix, G를 다음 접촉 후 한 번 측정한 gain, μ(H)를 그 conditional mean이라고 하면, squared error의 기댓값은 E[(s(H)−μ(H))²] + E[Var(G|H)]다. 그러므로 같은 H에서 독립적으로 반복한 G의 평균은 같은 학습 대상을 유지하면서 두 번째 항을 줄인다. 이 식은 source에서의 conditional expectation에 관한 것이다. 새 contact law에서 E[G|H] 자체가 바뀌면 source의 label을 더 많이 평균해도 그 차이는 남는다. 또한 한 future를 여러 deployment batches에, 한 batch를 여러 futures에 재사용하면 모든 cell이 독립이라는 가정으로 variance를 나눌 수 없다. [Alfonsi et al.](https://arxiv.org/pdf/2209.04153) · [Lehtinen et al.](https://proceedings.mlr.press/v80/lehtinen18a/lehtinen18a.pdf) · [v31 crossed variance decomposition](../probe_gain_reliability_v31/REPORT_KO.md)

**결과가 갈리는 부분.** Noise2Noise의 Gaussian 실험에서는 더 다양한 장면이 유리했고 Alfonsi et al.의 보험 문제에서는 여러 inner samples가 유리했다. 앞쪽은 새로운 영상에서 pixel prediction을 하는 CNN이고 비용 단위는 노출 횟수다. 뒤쪽은 함수 근사 공간을 정한 conditional regression이며 새 입력을 준비하는 계산 비용도 포함한다. Binois et al.은 같은 data-generating process에서도 어떤 metric을 쓰느냐에 따라 우승 설계가 바뀌는 사례를 제시한다. 따라서 이 논문들에서 공통의 최적 반복 수를 도출할 수 없다. [Noise2Noise Fig. 1c](https://proceedings.mlr.press/v80/lehtinen18a/lehtinen18a.pdf) · [Alfonsi et al. Fig. 9](https://arxiv.org/pdf/2209.04153) · [Binois et al. Fig. 5](https://arxiv.org/pdf/1710.03206)

**아직 부족한 근거.** LAL은 학습한 acquisition rule의 가능성을 보여 주고, SPOT는 prediction error와 실제 선택의 비용이 달라질 수 있음을 보인다. 두 연구에서 v32의 held-out Brier 감소량이나 generalization을 읽어낼 수는 없다. V31의 기존 factual pool에서 측정한 49.1%·50.3%의 순서 일치율은 같은 조건에서 한 번 얻은 gain으로 선별할 때의 직접적인 관측이다. 서로 다른 source histories에 공통된 signal을 모으거나 여러 단계에서 구매를 재배정한 정책의 최종 효과는 별도로 검증해야 한다. [LAL](https://proceedings.neurips.cc/paper_files/paper/2017/file/8ca8da41fe1ebc8d3ca31dc14f5fc56c-Paper.pdf) · [SPOT](https://proceedings.mlr.press/v119/elmachtoub20a/elmachtoub20a.pdf) · [v31 결과](../probe_gain_reliability_v31/REPORT_KO.md)

## 3. 검증할 만한 연구 질문 10개

| 연구 질문 | 남은 공백과 중요성 | 조작할 변수와 검증 방법 |
|---|---|---|
| 1. 새 prefix 한 개를 얻는 비용이 어느 정도일 때 기존 prefix 재평가가 유리한가? | Inner/outer cost ratio의 효과는 이론적으로 알려졌지만 접촉 simulator에서는 bank·decoder 준비 비용을 재사용할 수 있다. | 새 prefix의 setup, 새 future와 batch의 marginal time을 분리하고 같은 wall-clock 예산에서 prefix 수와 반복 수를 교환한다. [Alfonsi et al.](https://arxiv.org/pdf/2209.04153) |
| 2. 다음 한 번의 계산은 새 calibration future와 새 deployment batch 중 어디에 쓰는 편이 나은가? | V31에서는 batch main effect와 interaction의 비중이 컸다. 개별 label의 정밀도가 학습한 정책에 주는 효과는 열려 있다. | 같은 source histories에서 1×4와 4×1 label을 비교하고 반복당 비용을 기록한 뒤 새 batch에서 Brier를 측정한다. [v31](../probe_gain_reliability_v31/REPORT_KO.md) |
| 3. 어떤 source histories에만 반복하면 예상된 계산 절감이 실제로 생기는가? | Input-dependent noise를 이용한 설계는 GP에서 유효했다. 현재 scorer에서 큰 noise와 중요한 배분 결정이 같은 곳에 있는지는 알 수 없다. | Source pilot으로 noise를 추정해 균등 배분, noise 기반 배분, quota 경계 기반 배분을 독립 test에서 비교한다. [Binois et al.](https://arxiv.org/pdf/1710.03206) |
| 4. Gain을 잘 맞히는 정도와 정확히 네 곳을 골라 얻는 이득은 어떤 조건에서 갈라지는가? | 예측 MSE에 작게 기여하는 후보가 quota 경계에서 최종 선택을 뒤집을 수 있다. | 같은 features에 ridge와 decision-focused ranker를 학습하고 독립 label MSE, 선택이 바뀐 빈도, signed Brier 차이를 함께 측정한다. [SPOT](https://proceedings.mlr.press/v119/elmachtoub20a/elmachtoub20a.pdf) |
| 5. 새로운 prefix에 담긴 정보는 contact law 변화 뒤에도 gain을 같은 방향으로 예측하는가? | Source에 유효한 신호가 unseen domain에서 유지되는 범위는 제한적이다. | Source에서 정한 score 구간별로 새 laws의 독립 평균 gain을 비교하고 source-only와 target-aware tuning의 자료 접근 시점을 분리한다. [DomainBed](https://openreview.net/pdf?id=lQdXeXDoWtI) |
| 6. 현재 2→4회의 gain을 학습하는 것이 8회까지 이어지는 전체 경로의 목표와 맞는가? | LAL에서는 한 단계의 error reduction과 history를 생성하는 정책이 모두 중요하다. V30에서 stage 2에 도달하는 환경은 이전 선택에 의존했다. | 동일 training blocks에서 one-step gain 및 독립 rollout return을 label로 써서 전체 policy loss와 survivor 분포를 비교한다. [LAL §§3–4](https://proceedings.neurips.cc/paper_files/paper/2017/file/8ca8da41fe1ebc8d3ca31dc14f5fc56c-Paper.pdf) · [v30](../learned_probe_value_v30/REPORT_KO.md) |
| 7. 오차가 큰 prefix에서 score를 0으로 수축하면 잘못된 접촉 배분을 줄이는가? | 반복은 noise를 줄여도 환경 간 실제 차이가 작을 때 확정적인 순위를 보장하지 않는다. | Source에서 정한 shrinkage strength에 대해 uncertainty-only 및 uniform 정책과 paired evaluation을 하고, 유효한 신호를 약화시킨 비용도 센다. [v31](../probe_gain_reliability_v31/REPORT_KO.md) · [Alfonsi et al.](https://arxiv.org/pdf/2209.04153) |
| 8. 촘촘한 반복 측정은 실제로 다른 관측을 선택할 수 있는 환경에서만 가치가 생기는가? | V30 common decoder에서는 같은 world에서 action이 같으면 그 world의 gain이 0이다. 평균 label에는 변경 빈도와 변경당 이득이 섞인다. | Source에서 관측 가능한 proxy로 action change 가능성을 예측한 뒤 독립 batch에서 action 변화율, 변화당 gain, 전체 gain을 함께 평가한다. [v30](../learned_probe_value_v30/REPORT_KO.md) · [SPOT](https://proceedings.mlr.press/v119/elmachtoub20a/elmachtoub20a.pdf) |
| 9. 배포 시 알 수 있는 population 정보만 decoder에 주면 학습 설계의 순위가 바뀌는가? | V30과 v31의 privileged common decoder는 센서 선택을 공정하게 비교하지만 deployment에서 모르는 population coupling을 사용한다. | 동일한 frozen actions를 factual-estimated population decoder와 현재 decoder로 각각 채점하고 training design과 decoder의 interaction을 구한다. [v30](../learned_probe_value_v30/REPORT_KO.md) · [v31](../probe_gain_reliability_v31/REPORT_KO.md) |
| 10. 서로 다른 학습 설계의 차이를 해석할 수 있을 만큼 이득이 재현되는 positive control을 만들 수 있는가? | SPOT의 알려진 decision boundary는 왜 모델이 성공·실패하는지 확인할 기준을 준다. V31의 작은 신호에서는 구현 실패와 정보 부족을 분리하기 어렵다. | 사전에 알려진 feature-dependent gain을 가진 작은 simulator와 신호가 0인 negative control을 함께 둔다. Source-only scorer가 각 control에서 예상한 방향을 재현한 뒤 실제 물리 조건을 평가한다. [SPOT §3.1](https://proceedings.mlr.press/v119/elmachtoub20a/elmachtoub20a.pdf) · [v31](../probe_gain_reliability_v31/REPORT_KO.md) |

## 4. 논문 심사 관점에서 필요한 증거

**기여의 범위.** Gain regression과 inner/outer 배분에는 직접적인 선례가 있다. 여기서 논문으로 삼을 수 있는 질문은 *관측 이력을 다양하게 만들 때와 한 이력의 label을 더 잘 측정할 때, 어느 쪽이 제한된 접촉 예산을 가진 다단계 정책에서 실질적인 차이를 만드는가*이다. Source 예측의 정확도, 실제로 바뀐 구매, 동일 decoder에서의 held-out Brier를 따라가야 이 세 단계 중 어디에서 효과가 사라졌는지 설명할 수 있다. [LAL](https://proceedings.neurips.cc/paper_files/paper/2017/file/8ca8da41fe1ebc8d3ca31dc14f5fc56c-Paper.pdf) · [Alfonsi et al.](https://arxiv.org/pdf/2209.04153) · [SPOT](https://proceedings.mlr.press/v119/elmachtoub20a/elmachtoub20a.pdf)

**동일 compute 주장을 판단하는 법.** 훈련 데이터 생성에서 같은 수의 gain cells를 맞추더라도 prefix·posterior·deployment 평가를 재사용할 수 있는 정도가 다르다. 각 방법의 독립 prefix 수, future 수, deployment batch 수, cell 수, 실제 GPU 시간과 같은 평가 시점의 inference 비용을 각각 기록해야 한다. 전체 실험의 generator pool을 한 번 공유했다면 실제 공유 실행비와 각 방법을 단독으로 실행할 때의 비용 추정치를 구분한다. 같은 코드로 한 번 훈련한 scorer를 여러 worlds에서 평가해 얻은 interval은 그 학습 표본에 조건부다. Training-set 구성 자체의 평균적 우열을 주장하려면 학습 자료 생성과 fit를 여러 독립 seeds에서 다시 해야 한다. [Alfonsi et al.](https://arxiv.org/pdf/2209.04153) · [Binois et al.](https://arxiv.org/pdf/1710.03206) · [v31](../probe_gain_reliability_v31/REPORT_KO.md)

**일반화와 음성 결과.** 동일 contact law의 새 seed는 그 law에서의 재현성 검증이다. 기존 버전에서 설계자가 이미 결과를 본 법칙의 새 seed는 현재 fit에는 held out이지만 연구 전체에서 처음 본 물리 법칙은 아니다. Target law로 scorer나 평가 기준을 고르면 exploratory 결과로 분리해야 한다. CI가 0을 포함하는 상황에서 방법들이 동등하다고 단정하기도 어렵다. 실질적으로 중요한 차이를 사전에 정했다면 CI가 그 범위를 얼마나 제외하는지 제시할 수 있다. 세부 효과가 재현되고 strong baseline 대비 전체 이득이 작다는 결론도, 해당 simulator·예산·decoder에 한정하면 후속 연구가 같은 비용을 다시 쓰지 않게 하는 증거가 된다. [DomainBed](https://openreview.net/pdf?id=lQdXeXDoWtI) · [v30](../learned_probe_value_v30/REPORT_KO.md) · [v31](../probe_gain_reliability_v31/REPORT_KO.md)

**V32 protocol에 적용한 판정.** 두 설계는 동일한 15개 features와 ridge penalty 3을 쓴다. 최초 primary 비교에서는 각 설계에 한 개의 source corpus로 학습하고, 추가 두 seed 쌍에서도 각 corpus 생성부터 다시 수행한다. 두 설계에 서로 다른 seeds를 쓰므로 점추정의 차이에는 설계 효과와 해당 training draw의 차이가 함께 들어간다. 2×2에서 같은 행과 열을 공유하는 네 cell은 한 prefix의 target 하나를 만든다. 평가용 48개 blocks는 이 두 frozen policies의 차이를 세 law에서 정밀하게 측정한다. 세 law 중 signed velocity-dependent friction은 새 stress test이며, 나머지 둘은 앞 버전에서도 평가했다. 다섯 primary contrasts의 simultaneous interval과 사전 설정한 ±0.003 범위로 이 두 정책의 실질적 차이를 판단할 수 있다. 세 full-budget training pairs에서 결과 방향이 일치하는지도 별도로 확인한다. 다른 simulator나 다른 시간 예산에서의 최적 배분으로 일반화하려면 해당 조건의 재검증이 필요하다. [Source 예산과 label 설계](training_plan.json) · [평가 설계](design_eval.json) · [사전 통계 계획](STATISTICAL_PLAN_KO.md)

## 재현 자료

작업 위치는 `datacenter3:/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/compute_matched_stopping_v32`이다. 독립 재학습 두 쌍의 서버 폴더에는 각각 같은 이름 뒤에 `_rep2`, `_rep3`가 붙는다. 모든 model lock은 source Coulomb data만 읽는 별도 학습 코드로 생성했다. 최초 모델의 paired 결과는 이 lock을 선택하거나 갱신하는 입력이 아니다.

- [Training plan](training_plan.json), [총 세 쌍의 재학습 계획](REPEATED_TRAINING_PLAN_KO.md), [세 학습 경로](retrain_roots.json), [held-out split과 timing 기준](design_eval.json), [평가 코드](evaluate32.py), [새 contact law 구현](target32.py).
- [첫 training seed 모델과 GPU 시간](results_train/model_lock.json), [둘째 모델](../compute_matched_stopping_v32_rep2/results_train/model_lock.json), [셋째 모델](../compute_matched_stopping_v32_rep3/results_train/model_lock.json). 같은 detector 및 numerical posterior 설정, feature 순서, ridge 코드와 규제값을 사용한다. 각 arm의 training 및 target data hash, seed·GPU ledger는 해당 폴더에 있다.
- [Source contact·gain 재현 검사](results_train/training_audit.json), [target allocation·후보 score·Brier 독립 검산](stats32.py), [최종 통계](results_eval/paired_stats32.json), [통합 seed 검산](results_eval/seed_audit32.json), [재학습 모델의 policy replay](full_retrain32.py), [독립 source 진단](results_reference/source_reference32.json). Target truth를 열기 전 scoring 및 allocation과 query/probe 선택을 저장했다. GPU timing을 수행할 때 1,024개 확률과 action을 raw 배열에서 재현한 기준값과 매번 대조했다.
