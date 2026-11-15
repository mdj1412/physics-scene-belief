# v36 · 네 번 중 한 번을 query 부근에서 측정하면 friction 보정이 필요한 표면을 가려낼 수 있는가?

**네 번째 접촉을 query 근처로 옮겼을 때, 표면의 방향 효과를 판독하는 AUROC는 0.500에서 0.524로 변했다** (strong 대 weak/null, 네 측정의 paired 95% interval 0.501~0.546). 실제 보정을 켜는 binary gate를 같은 재료 구분 기준에 대입했을 때 balanced accuracy는 51.7%였다.

**네 번째 record를 보정 판단에 추가한 paired Brier 차이는 -0.00129 [-0.00186, -0.00072]**였다. 이 interval은 사전에 정한 0.003 이상의 실용적 Brier 감소를 지지하지 않는다. 세 local regime을 균등 가중했을 때 네 측정 gate의 held-out Brier는 0.11164, conventional fixed 4에서 원래 physics decoder는 0.11299, v35 correction은 0.11534였다. 이 두 end-to-end 차이에는 네 번째 동작과 방향 reader 변화도 들어간다.

Weak 표면의 Gate4−v35는 -0.00536 [-0.00627, -0.00444], query 근처에서 방향성이 사라진 local null은 -0.00940 [-0.01034, -0.00847], 표면 전체에 방향성이 없는 global null은 -0.00669 [-0.00823, -0.00515]였다. Strong 표면의 Gate4−v35는 +0.00368 [+0.00229, +0.00506]였다. Local null과 global null에서 원래 physics와의 차이는 각각 +0.00009 [-0.00083, +0.00101], -0.00027 [-0.00120, +0.00065]로 두 interval 모두 0을 포함했다. 이 실험은 v35의 geometry·response family에서 새로 생성한 synthetic worlds를 사용했다.

| 방법 | Local 평균 | Strong | Weak | Local null | Global null | GPU p90 (ms) |
|---|---|---|---|---|---|---|
| Conventional physics | 0.11299 | 0.12079 | 0.10955 | 0.10863 | 0.10644 | 1,424.2 |
| v35 correction | 0.11534 | 0.11399 | 0.11390 | 0.11812 | 0.11286 | 1,427.6 |
| Matched physics | 0.11286 | 0.12100 | 0.10930 | 0.10827 | 0.10633 | 1,431.9 |
| Matched always apply | 0.11371 | 0.11356 | 0.11190 | 0.11567 | 0.10730 | 1,436.7 |
| Matched gate · 세 측정 | 0.11293 | 0.12103 | 0.10939 | 0.10838 | 0.10628 | 1,436.1 |
| Matched gate · 네 측정 | 0.11164 | 0.11767 | 0.10854 | 0.10872 | 0.10617 | 1,439.9 |

Local 평균은 strong, weak, local null에 같은 비중을 준 값이다. 네 번째 접촉까지 기존 방식으로 진행한 두 방법을 conventional, 네 번째만 query 부근으로 옮긴 네 방법을 matched로 표시했다. 모든 loss는 독립 deployment 물체에서 측정한 held-out Brier이며 낮을수록 좋다. 각 행의 GPU p90에는 표에 해당하는 보정 계산까지 들어간다.

## 어떤 차이가 네 번째 probe의 효과인가

| 같은 world에서의 비교 | Local 평균 Δ Brier [95%] | Weak Δ [95%] | Local null Δ [95%] | Global null Δ [95%] |
|---|---|---|---|---|
| Gate4 − conventional physics | -0.00135 [-0.00231, -0.00038] | -0.00101 [-0.00203, +0.00001] | +0.00009 [-0.00083, +0.00101] | -0.00027 [-0.00120, +0.00065] |
| Gate4 − v35 correction | -0.00369 [-0.00457, -0.00282] | -0.00536 [-0.00627, -0.00444] | -0.00940 [-0.01034, -0.00847] | -0.00669 [-0.00823, -0.00515] |
| Gate4 − Gate3 | -0.00129 [-0.00186, -0.00072] | -0.00085 [-0.00148, -0.00023] | +0.00034 [-0.00033, +0.00101] | -0.00011 [-0.00071, +0.00048] |
| Gate4 − matched always apply | -0.00207 [-0.00274, -0.00140] | -0.00336 [-0.00410, -0.00262] | -0.00695 [-0.00789, -0.00600] | -0.00113 [-0.00179, -0.00046] |
| Gate4 − matched physics | -0.00121 [-0.00183, -0.00059] | -0.00076 [-0.00145, -0.00007] | +0.00045 [-0.00027, +0.00116] | -0.00016 [-0.00078, +0.00045] |
| Gate3 − matched always apply | -0.00078 [-0.00162, +0.00006] | -0.00251 [-0.00342, -0.00159] | -0.00729 [-0.00826, -0.00632] | -0.00101 [-0.00182, -0.00021] |
| Matched physics − conventional physics | -0.00013 [-0.00083, +0.00057] | -0.00025 [-0.00096, +0.00046] | -0.00036 [-0.00112, +0.00041] | -0.00011 [-0.00100, +0.00079] |

Δ는 왼쪽 방법에서 오른쪽 방법을 뺀 값이다. 각 열 안에서 7개 비교에 동시 95% interval을 사용했다. Primary는 세 local regime의 평균이다. 각 regime의 열은 조건별 보조 비교다. v35와 같은 absolute Brier 0.003을 실용 기준으로 사용했다. 같은 원래 decoder를 쓰는 Gate4−Gate3와 Gate4−matched physics를 함께 보면 추가 관측과 최종 보정의 이득을 구분할 수 있다. [사전 분석 계획](ANALYSIS_PLAN_KO.md) · [수치 및 검산](stats36.json)

## Gate가 방향이 약한 표면을 읽었는가

| 두 표면을 구별하는 문제 | 첫 세 관측 AUROC | 네 번째 관측 포함 AUROC [95%] |
|---|---|---|
| Strong 대 weak/null | 0.500 | 0.524 [0.501, 0.546] |
| Strong 대 weak | 0.500 | 0.519 [0.499, 0.539] |
| Strong 대 local null | 0.500 | 0.528 [0.503, 0.553] |
| Weak 대 local null | 0.500 | 0.510 [0.503, 0.516] |

| 표면 | Gate3 보정 적용률 | Gate4 보정 적용률 | 네 번째 측정 후 결정 변경 | 실제로 보정이 유리했던 환경¹ |
|---|---|---|---|---|
| Local strong | 6.8% | 37.3% | 35.9% | 77.5% |
| Local weak | 6.8% | 35.0% | 33.6% | 43.2% |
| Local null | 6.8% | 33.0% | 31.6% | 32.2% |
| Global null | 21.9% | 47.7% | 43.8% | 47.7% |

AUROC는 posterior mean friction strength가 강한 표면에 더 높은 값을 주는지 측정한다. 0.5는 무작위 순위에 해당한다. Gate는 expected Brier gain을 기준으로 보정을 선택하므로, 같은 strength라도 방향 추정의 확신과 원래 decoder의 오차가 다르면 다른 결정을 내릴 수 있다. ¹보정이 실제로 유리했던 환경은 같은 held-out 환경에서 always apply와 matched physics의 Brier를 직접 비교한 기술값이다. 환경마다 물체 128개를 사용하므로 이 분류에는 측정 오차가 있다.

보정 적용률을 전체 local 평균에 맞춘 무작위 정책에 비해 실제 Gate4의 Δ는 -0.00151 [-0.00188, -0.00115]였다. 환경별 always-apply − never-apply 차이를 D, 적용 여부를 G라 하면 이 값은 E[GD]−E[G]E[D]다. 환경의 geometry와 sign에 따라 보정 여부를 잘 선택해도 좋아질 수 있는 지표이므로, 아래에는 같은 환경에서 local regime만 바꾼 대조도 제시한다.

| 표면 | Gate4의 평균 예측 gain | 모두 보정할 때 실제 gain | 선택된 환경의 실제 gain | 건너뛴 환경의 실제 gain | 분할 oracle Δ Brier |
|---|---|---|---|---|---|
| 세 local regime 평균 | -0.00230 | -0.00085 | +0.00346 | -0.00319 | -0.00261 |
| Local strong | -0.00184 | +0.00743 | +0.00893 | +0.00654 | -0.00672 |
| Local weak | -0.00240 | -0.00260 | +0.00217 | -0.00517 | -0.00128 |
| Local null | -0.00265 | -0.00740 | -0.00136 | -0.01037 | +0.00016 |
| Global null | -0.00036 | -0.00096 | +0.00035 | -0.00215 | -0.00041 |

이 표에서 gain은 matched physics의 loss에서 always-apply loss를 뺀 값이므로 양수가 보정에 유리하다. 선택·건너뜀 두 열은 Gate4가 고른 환경과 건너뛴 환경에서 같은 correction을 실제로 적용했을 때의 평균이다. 마지막 열은 환경 내 128개 deployment 물체를 64개씩 나눠 한쪽의 정답으로 보정 여부를 고른 뒤 다른 쪽에 적용하고 교대로 평가한 Δ다. 음수일수록 matched physics보다 좋다. 이 진단은 배포 전에 실제 정답을 추가로 요구한다.

## 네 번째 factual feature를 공통으로 제공한 대조

| 조건 | Gate4 − gate3+x4 Brier [paired 95%] | Gate3+x4 − gate3 Brier [paired 95%] | Gate3+x4 적용률 | Paid velocity를 보고 결정이 바뀐 비율 |
|---|---:|---:|---:|---:|
| local_equal | -0.001253 [-0.001699, -0.000790] | -0.000039 [-0.000082, -0.000001] | 7.0% | 33.9% |
| strong | -0.003292 [-0.003889, -0.002713] | -0.000075 [-0.000133, -0.000021] | 7.0% | 36.1% |
| weak | -0.000816 [-0.001304, -0.000316] | -0.000035 [-0.000078, +0.000003] | 7.0% | 33.8% |
| null | +0.000351 [-0.000136, +0.000873] | -0.000008 [-0.000048, +0.000035] | 7.0% | 31.8% |
| global_null | -0.000156 [-0.000594, +0.000328] | +0.000042 [-0.000012, +0.000105] | 23.8% | 42.6% |

같은 첫 세 접촉과 네 번째 factual feature에서 strong과 weak/null을 구별하는 posterior strength의 AUROC는 0.500, 네 번째 paid velocity까지 주면 0.524 [paired 95% 0.501, 0.546]였다. 세 local regime의 기존 관측들이 완전히 짝지어 있으므로 paid velocity 이전의 strong 및 weak/null score 분포는 같다. AUROC는 “보정 여부”가 아니라 source posterior strength의 순위를 평가한다. 강도 차이를 일부 읽어도 Brier 개선과 적용 여부의 순위는 달라질 수 있다.

세 local regime에서 같은 물체와 noise를 짝지은 상태로 gate decisions만 regime들 사이에서 무작위 permutation했을 때의 평균 loss와 실제 배정을 비교했다. 각 조건의 원래 deployment action, base probability와 실제 query label을 사용했다. 실제 − 무작위 배정 Brier는 -0.000179 [-0.000270, -0.000086]였다. 세 regime에서 gate decision이 모두 같은 경우는 85.7%였다. 이 대조는 환경별 평균 적용률과 네 번째 접촉에서 드러나는 공통 물체 성질을 고정한다. 여기에서 음수이면 강도 차이 때문에 바뀐 결정 자체가 유익했다는 근거가 된다.

각 CI는 조건별 95% paired cluster bootstrap이다. Object–surface와 첫 motion을 층화한 32개 block 안에서 orientation, regime, 환경과 32개 observation noise를 묶었다. Global-null은 4개 cell × 8개 독립 block에서 각각 실행했다. 보조 score는 동결한 source Gaussian을 marginalize해 계산했고 별도의 모델 학습이나 GPU timing은 수행하지 않았다. [수치](increment36.json) · [계산](increment36.py)

### 같은 gate에 conventional fourth와 query-matched fourth를 각각 제공한 결과

원래 v35의 conventional reader와 새 near gate의 차이에는 방향 추정 방식의 변화도 들어간다. 여기서는 두 gate가 같은 frozen source model, 처음 세 접촉의 signed proposal, 같은 β를 쓴다. 네 번째 conventional response는 실제 부호에 맞는 action likelihood로 갱신하고, near response는 query 부근 action의 likelihood로 갱신했다. 표에서 on 비율의 단위는 환경이다.

| 평가 범위 | Gate on, prefix3 | Gate on, conventional4 | Gate on, matched4 | 두 fourth의 결정이 달랐던 비율 | Far gate4 Brier | Near gate4 Brier |
|---|---:|---:|---:|---:|---:|---:|
| 세 local 균등 | 6.8% | 16.0% | 35.1% | 33.9% | 0.11289 | 0.11164 |
| 강한 방향성 | 6.8% | 16.0% | 37.3% | 35.0% | 0.12007 | 0.11767 |
| 약한 방향성 | 6.8% | 16.0% | 35.0% | 33.8% | 0.10957 | 0.10854 |
| query 부근 방향성 없음 | 6.8% | 16.0% | 33.0% | 33.0% | 0.10903 | 0.10872 |
| 전체 표면 방향성 없음 | 21.9% | 23.8% | 47.7% | 44.9% | 0.10640 | 0.10617 |
| 네 regime 균등 | 10.6% | 18.0% | 38.2% | 36.7% | 0.11127 | 0.11027 |

| 평가 범위 | Near4 − far4, 각자의 decoder | 네 번째 측정의 difference-in-differences | Near4 − far4, 같은 near decoder | Near4 − far4, 같은 far decoder |
|---|---:|---:|---:|---:|
| 세 local 균등 | -0.00125 [-0.00211, -0.00039] | -0.00109 [-0.00163, -0.00054] | -0.00112 [-0.00168, -0.00057] | -0.00109 [-0.00162, -0.00056] |
| 강한 방향성 | -0.00240 [-0.00354, -0.00127] | -0.00258 [-0.00328, -0.00187] | -0.00260 [-0.00331, -0.00189] | -0.00256 [-0.00320, -0.00191] |
| 약한 방향성 | -0.00103 [-0.00195, -0.00010] | -0.00075 [-0.00134, -0.00016] | -0.00079 [-0.00140, -0.00018] | -0.00074 [-0.00134, -0.00015] |
| query 부근 방향성 없음 | -0.00031 [-0.00113, +0.00050] | +0.00007 [-0.00060, +0.00073] | +0.00003 [-0.00065, +0.00071] | +0.00004 [-0.00063, +0.00070] |
| 전체 표면 방향성 없음 | -0.00023 [-0.00104, +0.00058] | -0.00012 [-0.00066, +0.00042] | -0.00013 [-0.00067, +0.00041] | -0.00010 [-0.00064, +0.00045] |
| 네 regime 균등 | -0.00099 [-0.00176, -0.00023] | -0.00085 [-0.00135, -0.00034] | -0.00087 [-0.00139, -0.00036] | -0.00084 [-0.00134, -0.00034] |

같은 near probabilities에 적용했을 때 세 local regime 평균은 -0.00112 [-0.00168, -0.00057]로 matched가 유리했다. Difference-in-differences는 -0.00109 [-0.00163, -0.00054]로 matched가 유리했다. 양쪽에서 네 번째 probe를 제공했을 때의 변화량을 각각 구했으므로, 이 지표는 처음 세 번만 읽은 정책에 대한 단순 우위를 넘어 어느 네 번째 동작이 더 유용했는지 평가한다.

세 local regime은 conventional 접촉 지점의 law를 공유한다. 실제로 같은 orientation·block에서 conventional fourth까지 사용한 gate decision이 세 regime 모두에서 같음을 모든 archive에서 확인했다. 그럼에도 conventional fourth가 첫 세 번의 방향 추정을 보완하거나 global-null을 구별하는 데 쓰일 수 있다. 각자의 decoder를 사용한 Brier에는 deployment action이 달라진 효과도 포함된다. 같은 decoder의 두 열에서는 같은 물체의 같은 sensor realization과 같은 proposal을 사용한다.

각 범위의 네 contrast에는 object × surface와 시작 부호를 층화한 paired bootstrap 12,000회의 simultaneous 95% interval을 사용했다. 새 conventional gate는 GPU가 저장한 관측과 prediction으로 CPU에서 평가했으며, 이 방법만의 GPU p90은 미측정이다. [보조 분석 계획](SUPPLEMENT_PLAN_KO.md) · [계산 코드](supplement36.py) · [수치](supplement36.json)

## 같은 예산에서 보정량만 줄이는 대조

| 표면 | 고정 32.5% 보정 | 세 측정 soft correction | 네 측정 soft correction | Gate4 − 고정 32.5% [95%] |
|---|---|---|---|---|
| 세 local regime 평균 | 0.11127 | 0.11123 | 0.11137 | +0.00037 [-0.00007, +0.00082] |
| Local strong | 0.11673 | 0.11616 | 0.11617 | +0.00094 [+0.00039, +0.00148] |
| Local weak | 0.10826 | 0.10834 | 0.10856 | +0.00028 [-0.00022, +0.00078] |
| Local null | 0.10881 | 0.10919 | 0.10937 | -0.00009 [-0.00069, +0.00050] |
| Global null | 0.10620 | 0.10622 | 0.10626 | -0.00003 [-0.00052, +0.00046] |

고정 32.5%는 네 source regime에서 가정한 near-field multiplier의 평균이다. Soft correction은 같은 βz에 각 환경의 posterior mean multiplier를 곱한다. 모두 동일한 matched physics probability와 deployment actions를 사용했다. 위 세 secondary 방법의 GPU p90은 따로 측정하지 않았다. [계산 코드](stats36.py)

Strong 표면의 실제 빈도가 달라지면 전체 효과도 달라진다. Weak와 local null의 비율을 서로 같게 두고 strong 비율만 바꿔 이미 측정된 loss를 재가중하면 Gate4−v35는 0%일 때 -0.00738, 25%일 때 -0.00462, 50%일 때 -0.00185, 75%일 때 +0.00091, 100%일 때 +0.00368였다. 이는 배포 집단 비율에 대한 sensitivity이며 각 비율에 맞춰 다시 fitting한 결과는 아니다.

## 한 번의 측정이 제공하는 정보량

| Probe 높이 | Tangential speed | Strong 대 local null의 정확도 상한 | 두 표면의 velocity mean이 같은 비율 | Weak 대 local null의 정확도 상한 |
|---|---|---|---|---|
| 0 mm | 1.1 | 50.0% | 100.0% | 50.0% |
| 23 mm | 0.425 | 63.8% | 46.7% | 54.7% |
| 23 mm | 0.5 | 68.7% | 33.6% | 56.4% |
| 23 mm | 0.75 | 76.9% | 16.4% | 59.7% |
| 23 mm | 1.1 | 83.5% | 3.3% | 62.5% |
| 25 mm | 0.425 | 64.0% | 45.7% | 54.8% |

각 물체의 latent geometry, orientation, 두 후보 표면의 정확한 law까지 아는 관찰자가 단 한 번의 noisy velocity로 어느 표면인지 추측하는 문제를 계산했다. 두 표면의 사전확률은 같다. 두 Gaussian likelihood의 평균 거리 D를 noise standard deviation으로 정규화하면 최적 정확도는 Φ(D/2)다. 같은 물체의 두 충돌이 모두 stick하면 tangential impulse가 friction coefficient에 의존하지 않아 D가 0이 될 수 있다. 이 열은 실제 gate의 달성 정확도와 구별되는 privileged upper bound다. 속도를 바꾼 행은 동일 물체에 대한 물리 계산이며 해당 동작을 쓰는 policy의 Brier나 latency 실측은 아직 없다. [계산식](diagnose36.py)

![v36 results](v36.png)

## 전체 GPU p90

| Gate4 p90 비교 대상 | 차이 (%) | 동시 95% interval | +2% 기준 |
|---|---|---|---|
| Conventional physics | +1.10% | [+0.83%, +1.37%] | 상한 이내 |
| v35 correction | +0.86% | [+0.70%, +1.02%] | 상한 이내 |
| Matched physics | +0.56% | [+0.21%, +0.91%] | 상한 이내 |
| Matched always apply | +0.22% | [-0.05%, +0.49%] | 상한 이내 |
| Matched gate · 세 측정 | +0.26% | [-0.11%, +0.64%] | 상한 이내 |

각 방법에 대해 8개 독립 timing block에서 세 번씩 총 24회 측정했다. 각 block에 모든 방법을 함께 넣어 randomized execution order로 실행했다. GPU p90은 한 대의 NVIDIA RTX PRO 6000 Blackwell Server Edition에서 환경 8개, 총 1,024개 deployment 물체를 처리한 wall time이다. CPU에서 실행된 gate, GPU synchronization과 결과의 host 반환을 포함한다. 실제 로봇 이동 시간은 별도로 검증해야 한다. 각 비교의 interval은 같은 timing block을 대응시켜 재표집하고 다섯 비교에 동시 보정했다.

## 네 번의 접촉을 어떻게 비교했나

예측할 사건은 **다음 물체가 높이 25 mm, incoming tangential speed +0.425인 충돌에서 stick하는가**이다. Physics decoder는 별도 deployment 물체마다 stick probability를 낸다. Brier loss는 이 확률과 0 또는 1인 실제 outcome 간의 squared error이다. Calibration에 쓴 물체의 다음 상태를 다시 맞히면 과제가 쉬워지므로, 모든 deployment 물체와 sensor noise를 calibration에서 독립적으로 생성했다.

- **Conventional fixed 4**는 네 물체에 높이 0, tangential speed ±1.10의 probe를 교대로 한다. 처음 방향을 균형화한다. v35 arm은 이 네 측정에서 기존 reader가 추정한 friction 방향을 β=0.8289760825845313의 log-odds correction에 넣는다.
- **Query-matched fixed 4**는 같은 네 물체와 같은 세 번의 접촉을 사용하되, 네 번째 물체에는 높이 23 mm, speed +0.50의 probe를 한다. 네 번째 속도 측정에 같은 Gaussian noise를 대응시킨다. Query와 비슷한 충돌을 하는 물체는 한 개다. 이때 측정하는 것은 outgoing velocity의 두 성분이며, simulator의 stick 판정을 gate에 주지 않는다.
- **Acquisition과 decoding**에는 해당 motion의 높이·속도에 맞는 likelihood를 쓴다. 각 환경에서 deployment 물체 128개 중 13개에 probe, 나머지에는 query view를 선택하고 공통 Coulomb decoder가 확률 p를 계산한다. 두 protocol 사이에는 관측 선택도 달라질 수 있다. Matched arm 안의 보정 정책들은 같은 선택, 같은 p와 같은 deployment outcome을 사용한다.
- **방향과 보정 적용은 별개의 추정이다.** Matched arm은 첫 세 번의 conventional 접촉으로 방향과 강도를 나타내는 공통 signed cue z를 제안한다. 항상 보정하는 arm은 p의 odds에 exp(βz)를 곱한다. Gate는 source에서 추정한 expected Brier gain이 양수인 환경에만 같은 보정을 적용한다. Gate3은 첫 세 측정으로 판단하고 Gate4는 같은 proposal을 유지하면서 네 번째 물체의 near-query velocity와 factual features를 함께 본다. Gate3도 실제 네 번째 관측을 acquisition에는 사용한다. 따라서 Gate4−Gate3가 네 번째 접촉을 최종 보정 판단에 제공한 효과를 측정한다. 두 protocol에서 reader가 사용할 수 있는 conventional 측정은 각각 네 번과 세 번이다.
- **평가 표면**은 두 물체 contour(sinusoid, quartic)와 두 friction response law(atan, algebraic)를 교차한다. 세 local regime은 높이 0에서 같은 directional friction을 보이고, 23 mm 이상에서는 원래 방향 효과의 100%, 30%, 0%를 보인다. 중간 높이에서는 smoothstep으로 연속적으로 바꾼다. 별도 global-null control은 표면 전체에서 방향성이 없다. 처음 세 응답과 conventional 네 번째 응답은 같은 orientation 안의 세 local regime에서 원소별로 같아야 한다.
- **표본과 비용**은 고정한 네 물체·표면 조합별로 독립 seed 8개, 총 32개 block이다. 한 block에 환경 8개, 각 환경에는 deployment 물체 128개가 있으므로 32,768개의 서로 다른 latent deployment 물체를 사용한다. 각 물체의 32개 sensor noise, 두 mirrored orientations, regime와 method는 block 안에서 대응시킨다. Interval은 이 32개 block을 재표집하며 반복 noise를 독립 표본으로 세지 않는다. GPU p90은 물체 1,024개를 한 번 처리한 end-to-end wall time이다. 공통 bank의 초기 구성, sensor를 생성하는 simulator, 실제 장비의 접촉 및 reset은 측정 범위 밖이다.

Held-out target은 v35에서 사용한 두 contour와 두 friction response 함수를 재사용한다. v36에서 새로 바꾼 것은 물체와 noise의 seed, 높이에 따라 방향 효과가 달라지는 law, 네 번째 접촉의 위치와 속도다. Source simulator의 contact law와 held-out target의 함수 형태는 다르다. 두 simulator 모두 query 근처에서 방향 효과가 약해질 수 있다는 구조를 모델링한다. 이 설계는 학습에 없는 geometry·law·world draws에서 주어진 네 후보 regime 사이의 구별을 평가한다. 실제 임의 재질의 regime 발견으로 일반화하려면 별도의 자료가 필요하다. 양 arm의 기본 physics decoder는 population proxy coupling을 공유한다.

## v36 보정의 적용 여부를 판단하는 방법

같은 환경에서 서로 다른 물체 세 개를 높이 0, tangential speed ±1.10으로 접촉시켜 해당 방향의 마찰과 반대 방향의 마찰 중 어느 쪽이 큰지 추정한다. Positive orientation과 negative orientation의 확률 차이를 z로 사용한다. 방향성이 없는 model은 z에 0을 기여한다. 보정을 선택하면 원래 물체별 stick probability p의 log odds에 v35의 고정 계수 0.8289760825845313과 z의 곱을 더한다. 네 번째 측정은 보정을 적용할지만 결정한다. z는 첫 세 측정으로 고정한다. v35의 conventional benchmark는 원래 네 번의 고속 접촉과 당시의 k=4 reader를 그대로 사용한다. Matched arm은 세 번만 고속 접촉하므로 이 arm의 z 추정기는 새로 구성했다.

**Source evidence.** Source는 v33의 cubic contour와 tanh directional friction을 사용한 네 가지 object–surface 조합이다. 각 조합에 두 orientation과 네 종류의 세기 조건을 같은 비중으로 둔다. Local strong/weak/null은 conventional 접촉에서 방향성이 같고 query 부근에서는 source law의 100%, 30%, 0%다. Global-null은 두 위치 모두에서 0%다. Source와 held-out target은 이 네 종류의 shift를 공유한다. Target의 두 contour와 두 friction response 함수, object와 noise seeds는 source와 다르다.

Far 방향 두 개와 near 조건 한 개에서 각각 contact likelihood를 만든다. Input은 noisy factual contour를 세 개의 orthonormal basis(평균, linear, quadratic)에 투영한 값, noisy factual velocity 두 성분, paid velocity 두 성분이다. Latent geometry·friction·restitution의 여섯 이진 variable로 64개 component를 나눈 뒤, component 안에서 연속 latent sample 384개가 낸 signal의 평균과 covariance를 계산한다. 알려진 sensor noise covariance를 더해 각 component를 Gaussian으로 근사한다. 이 근사는 contact equation을 직접 풀어 얻은 signal의 moment를 보존한다. Stick/slip 경계의 다봉성 및 환경 안에서 공유되는 proxy coupling α에는 오차가 남는다. Inference에서는 각 환경의 α를 쓰지 않고 source proxy flip을 독립적으로 적분한다. 모델당 64개 component의 prior는 균등하다.

세 번 또는 네 번의 관측 likelihood로 32개 model(4개 조합 × 2개 orientation × 4개 세기)의 posterior를 얻는다. 각 model에 대해 보정의 기대 Brier gain을 v34의 기존 fixed 4 deployment archive에서 추정했다. 8개 독립 block × 8개 signed family의 alternating archive 64개를 사용했다. 각 model의 source α·noise·deployment object, 선택된 action 및 decoder output을 유지한 채 query law로 stick label을 다시 계산했다. 원래 directional 및 nominal label 총 65,536개씩이 archive와 정확히 일치했다. z를 −1부터 +1까지 0.05 간격으로 바꾸어 얻은 41개 지점의 risk 차이를 선형 보간한다. 이 reference는 원래 conventional action과 source deployment decoder를 쓴다. Local weak/null은 far deployment sensor가 그대로여서 query label만 바꾸는 대응이 성립한다. Global-null의 gain table은 기존 directional 센서에서 나온 확률에 nominal label을 대입한 근사다. 실제 v36 global-null 평가는 paid 및 deployment sensor까지 처음부터 isotropic으로 생성한다. Source gain의 부호를 정답으로 target을 채점하지 않는다.

결정 시 source gain을 model posterior로 평균해 양수인 환경에서만 보정을 적용한다. 이 threshold는 0이며 target으로 고른 값이 없다. Near 강도에 대한 posterior mean은 각 model의 query multiplier를 posterior로 평균한 별도의 진단이다. 보정의 expected utility에는 부호가 틀릴 위험, 원래 확률을 움직이는 비용도 들어가므로 surface strength 분류와 apply 결정의 목적이 다르다. [실행용 gate](gate36.py) · [source fitting](train_gate36.py) · [계수 및 hash](gate36_freeze.json)

**Fourth factual measurement 대조.** Gate3는 첫 세 물체의 출력으로 posterior를 만들고 gate4는 네 번째 물체의 factual feature와 paid velocity를 함께 사용한다. 두 정책의 차이는 이 네 번째 record 전체의 효과다. 네 번째 물체에서 이미 얻은 factual feature만 제공하는 gate3+x4도 별도로 계산한다. 이때 각 source Gaussian에서 paid velocity 두 성분을 적분하고 factual feature 5개만 남긴 marginal likelihood를 사용한다. z는 동일하게 첫 세 개로 고정한다. Gate4와 이 대조의 risk 차이는 같은 네 번째 factual feature가 있을 때 paid velocity가 제공한 추가 decision utility를 나타낸다. 이 대조의 보정 방향은 네 번째 물체를 보기 전에 정한 값이다. [조건부 대조](increment36.py)

별도 source seed에서 512개 독립 draw에 orientation 및 세기 조건을 짝지어 4,096개 prefix를 평가했다. 네 번째 near-query 측정 뒤 local strong과 weak/null을 구별하는 posterior strength의 AUROC가 0.500에서 0.588로 바뀌었다. Gate4의 적용률은 strong 38.8%, weak 37.6%, local null 36.4%였다. Source simulator에서도 세기 순위가 조금 달라지는 것과 보정을 적용하는 빈도가 크게 갈리는 것은 구분해야 한다. Brier를 이용해 훈련한 gate는 강도 외에도 물체·표면별 원래 decoder의 bias와 방향 추정의 confidence를 고려한다. 이 independent-prefix 검산의 gain은 같은 source lookup에 기반한 surrogate이며 held-out deployment Brier는 v36 GPU 실험에서 따로 측정했다. [source 검산 원자료](gate36_source_check.json)

## 논문 기여 가능성

**예측 성능과 실험 설계의 증거는 크기가 다르다.** 세 local regime에서 첫 세 접촉과 네 번째 conventional 접촉까지 완전히 같은 응답을 갖는 paired worlds를 구현했다. Query 부근 측정을 넣으면 strong과 weak/null의 posterior 순위가 달라진다. 동일 matched decoder에서 네 번째 near 측정과 네 번째 conventional 측정으로 만든 gate의 차이는 -0.00112 [-0.00168, -0.00057]였다. 네 번째 factual feature는 양쪽에 주고 paid velocity만 추가한 차이는 -0.00125 [-0.00170, -0.00079]였다. 둘은 서로 다른 비교다. 전자는 동작을 바꾼 가치, 후자는 이미 얻은 geometry 위에서 velocity가 제공한 가치를 묻는다.

같은 environment에서 세 local regime의 gate 선택만 바꿔 배정한 대조에서 실제 선택의 Δ는 -0.00018 [-0.00027, -0.00009]였다. 85.7%의 짝에서는 세 표면 모두에 같은 결정을 내렸다. 이는 geometry 차이 또는 평균 보정률을 바꿔 얻은 이득과 분리되는 비교다.

Gate는 세 측정에서 네 측정으로 갈 때 전체 local 환경의 보정 적용률을 6.8%에서 35.1%로 높였다. Gate4는 strong 37.3%, weak 35.0%, local null 33.0%, global null 47.7%에서 보정을 켰다. 강한 환경을 더 자주 보정해 v35의 약·무방향 손해를 피하면서, strong에서 v35보다 덜 보정한 비용도 냈다. 이 비율을 보면 배포 환경에서 무방향 재질을 자동 검출하는 용도로는 아직 부족하다.

**단순한 대안과의 차이도 중요하다.** 같은 signed proposal에 source 평균 multiplier 0.325를 일률 적용한 방법의 Brier는 0.11127이고 binary gate와의 차이(Gate4−고정 shrinkage)는 +0.00037 [-0.00007, +0.00082]였다. 표면별로 기대되는 gain을 추정하는 모델을 기여로 내세우려면, 같은 budget의 단순 shrinkage보다 유리한 조건도 제시해야 한다. 약하거나 방향이 없는 표면에서 보정을 줄여 얻은 이득 전체를 새 probe의 효과로 산정하면 과장된다.

**현재 근거로 작성할 수 있는 결과**는 네 번이라는 작은 budget에서 서로 구분되지 않는 calibration histories를 구성하고, query 부근의 물리적 measurement로 얻는 표면 정보와 그 정보가 줄인 decision loss를 따로 정량화한 simulation 연구다. Contact geometry와 외형, 원래 decoder, source prior가 정해진 환경에서 얻은 결과다. GPU p90은 한 장치·한 session이고, 32개 독립 block은 고정한 4개 object–surface 조합 내부의 반복이다. 현재 단계에서 실제 새 재질, 다른 query 방향, physical probing을 포함한 로봇 cycle time으로 일반화할 근거는 부족하다. Source에서는 global-null gain을 directional deployment sensor로 근사했으며 correction 방향은 세 번의 far 접촉으로 고정했다.

**후속 검증은 물리 조건 하나를 먼저 바꾸는 편이 좋다.** 비용을 네 번으로 유지하고 같은 query 높이에서 (i) 지금 속도, (ii) 미끄러짐이 생기기 쉬운 속도, (iii) 기존 속도를 독립 배정한다. Noise-free identifiability 표의 후보 속도는 다음 실험 설계에만 쓴다. 다음 held-out panel에서 실제 GPU inference 및 Brier를 다시 측정하고, 강한 continuous directional physics posterior와 source-only shrinkage를 함께 비교한다. 실물에서는 동일한 표면 패치에서 probe와 별도 물체의 충돌을 짝지어 기록하고, 각 실험의 object ID와 surface session을 나누어야 한다. 높이·속도를 옮긴 비용, sensor saturation, 마모가 측정에 미친 영향도 같은 session에서 확인해야 한다.

### 어떤 접촉이 이후 충돌의 예측에 도움이 되는가

v36의 질문은 네 번째 접촉의 높이와 속도를 바꾸면, 이미 가진 방향 보정을 **어떤 환경에서 적용할지** 더 잘 결정할 수 있는가이다. 아래 표는 측정 동작, 추론 목표, 실제 task의 평가가 연결되는 여덟 연구를 목적 표집한 결과다. 각 논문의 원문에서 관측 대상과 평가 단위를 확인했다.

| 연구와 질문 | 방법과 표본 | 주요 결과와 저자의 결론 | v36에 남는 한계와 근거 |
|---|---|---|---|
| Fishel & Loeb (2012). 어떤 접촉 동작이 표면을 구분하는가? | BioTac으로 117개 texture를 탐색했다. Normal force와 sliding velocity 조합을 바꾸고 traction·roughness·fineness를 측정했다. Bayesian exploration이 다음 동작을 선택했다. | 이미 수집한 117개 후보의 texture identification은 중앙값 5회 탐색으로 95.4%에 도달했다. 비슷한 두 후보만 구별할 때는 99.6%였다. | 후보 집합과 평가 목표가 고정돼 있다. 처음 보는 contact law에서 다른 물체의 stick probability를 예측하려면 별도의 평가가 필요하다. [원문, Abstract·Results][F] |
| Tramsen et al. (2018). 비대칭 돌기에서 어느 방향에 friction이 더 큰가? | 인공 sawtooth와 substrate를 조합하여 양방향으로 당겼다. 상대 stiffness, aspect ratio, roughness를 바꾸고 pulling force를 기록했다. | 재료와 형상 조합에 따라 높은 friction을 보이는 방향이 반전됐다. 저자들은 접촉 면적과 mechanical interlocking의 상대적 역할로 설명했다. | 측정량은 pulling force다. 같은 재료의 impact에서 방향 효과의 크기, 또는 네 번의 접촉 뒤 Brier gain은 따로 검증해야 한다. [원문, Figs. 2–5][T] |
| Fazeli, Kolbert, Tedrake & Rodriguez (IJRR 2017). 운동과 외력을 어느 정도 알아야 contact dynamics를 식별하는가? | Planar rigid contact의 complementarity formulation에서 식별 가능한 parameter 조합을 유도했다. 실물 정사각형과 타원형 물체에 drop test 총 280회, 각 drop의 처음 세 bounce를 사용했다. | 알려진 외력이 없으면 inertial parameter와 contact force의 일부 비율이 식별된다. 타원의 mass/inertia 비율 추정치는 587 m⁻², 참값은 535 m⁻²였다. Noise가 커지면 추정 오차가 증가했다. | 실물 검증의 주대상은 inertia와 contact force다. Contact mode와 sensor noise가 식별 범위를 제한하는 선례로 읽을 수 있다. [원문, §§2·5.3–5.4·7][I] |
| Song & Boularias (RSS 2020). 몇 번의 push로 얻은 물성 추정이 다른 조작을 예측하는가? | Differentiable physics로 mass와 friction의 공간 분포를 맞췄다. 실물 5개, simulation 물체 8개를 사용했다. 실제 trajectory를 절반씩 나누는 25개 split으로 평가했다. | Hammer를 table edge로 보내 잡는 실험에서 friction 상한을 둔 방법은 16/16회, uniform mass는 12/16회, friction 상한을 없앤 방법은 13/16회 성공했다. 탁자 위 운동을 잘 맞춘 model이 상판 지지가 사라지면 실패할 수 있었다. | 실물 mass와 friction 참값을 알 수 없어 held-out pose와 task success로 평가했다. 한 hammer의 16회 시도는 16종의 독립 물체가 아니다. [원문, 실험·Final Remark][S] |
| Le et al. (2021 원고). 일부 표면의 접촉으로 다른 위치의 friction을 추정할 수 있는가? | Panda와 force–torque sensor, Kinect point cloud를 사용했다. 실물 15개 중 4개는 단일 재질이고 나머지는 2~4개 재질이었다. 국소 haptic 측정과 visual prior를 결합했다. | 여러 재질에서 위치별 friction 차이를 복원했다. 컵 두 종류를 사용한 grasp case study에서 추정된 friction이 높은 위치로 grasp candidates를 좁혔다. | 표면 내 visual similarity를 통해 측정값을 전달한다. 같은 재질에서 속도가 달라졌을 때의 friction law와 binary impact outcome은 별개의 변수다. [원문, §§III–V][L] |
| Attia, Alexanderian & Saibaba (2018). 예측할 quantity of interest가 달라지면 sensor 위치도 달라지는가? | Linear Gaussian inverse problem에서 목표 예측의 posterior covariance를 최적화했다. Advection–diffusion simulation에는 후보 위치 22개가 있다. Fig. 5의 thresholded design은 5개 위치를 사용한다. | 예측하려는 contaminant concentration의 범위와 차원이 달라지면 같은 후보 중에서도 유리한 sensor 배치가 달라졌다. | 선형 모델과 알려진 noise에서 계산한 design이다. 같은 표면에서 stick/slip transition이 만드는 비선형성과 model mismatch는 별도로 다뤄야 한다. [원문, §5][A] |
| Zhong et al. (SIAM/ASA JUQ, 2026). 비선형 predictive quantity에 적합한 실험을 어떻게 선택하는가? | Nested Monte Carlo, MCMC, kernel density estimation, Bayesian optimization을 결합했다. Convection–diffusion 예제는 source 위치 2,000개로 surrogate를 만들고 미래 농도를 예측할 네 지점을 바꿨다. | Parameter와 미래 농도에 대한 expected information gain을 각각 최적화하자 sensor 위치가 달라졌다. 일부 조건에서는 유체가 이동하는 방향을 고려해 목표 지점보다 상류에서 관측하는 가치가 나타났다. | 정확히 같은 위치나 조건이 항상 최적인 것은 실험으로 보장되지 않는다. 이 연구의 목적함수는 predictive information gain이고 v36의 목적함수는 downstream Brier loss다. [원문, §§4.3–4.4][Z] |
| Mozannar & Sontag (ICML 2020). 모델과 다른 예측자 중 누구에게 맡길지 어떤 loss로 학습하는가? | Cost-sensitive surrogate로 classifier와 rejector를 학습했다. CIFAR10H에서는 영상당 50개 annotator 응답에서 simulated expert를 만들고, 주석 있는 자료를 반으로 나눈 split을 10회 반복했다. | Table 2에서 imputation을 쓴 모델의 system accuracy는 96.29%, confidence baseline은 95.09%였다. Coverage는 각각 51.67%, 79.48%였다. | 두 방법이 다른 비율로 결정을 넘겼다. 이 논문의 classification accuracy와 v36의 연속 확률 Brier는 목적함수가 다르다. [원문, §6.3·Table 2][M] |

### 관측에서 결정까지의 연구 흐름과 증거

| 관계 | 근거 수준과 아직 식별되지 않은 부분 |
|---|---|
| 접촉 조건 → sensor response → 가능한 물성 | Force와 velocity를 바꿔 texture 판별력이 달라진 실물 측정, 재료 조합에 따른 friction 방향 반전, 여러 재질에서 위치별 friction 차이가 관측됐다. 재질·측정 방식·정답은 세 연구마다 다르다. 약한 impact anisotropy의 식별 한계에는 직접적인 수치가 없다. [Fishel & Loeb][F] · [Tramsen et al.][T] · [Le et al.][L] |
| 물성에 대한 적합도 → 새 intervention의 예측 | Fazeli et al.이 관측 가능한 contact dynamics의 조합을 수식과 실물 낙하로 검증했다. 이 연구를 인용하는 Song & Boularias의 hammer 실험에서 기존 동작과 잘 맞았던 물성이 변경된 지지 조건에 옮겨가지 못했다. Attia et al.과 Zhong et al.은 목표 예측을 직접 design에 넣으면 최적 측정 위치가 달라지는 수치 예를 보였다. 물리적 실패 사례와 design 원리는 서로를 보완한다. 네 번째 probe를 어느 높이에 둘지는 해당 contact model로 검증해야 한다. [Fazeli et al.][I] · [Song & Boularias][S] · [Attia et al.][A] · [Zhong et al.][Z] |
| Paid evidence → 보정 여부 → held-out loss | Mozannar & Sontag에서는 예측할 주체를 고르는 학습 목표가 system performance에 영향을 줬다. v36에 대응하는 질문은 관측된 접촉으로 두 decoder의 상대적인 downstream loss를 예측할 수 있는가이다. 논문에 제시된 수치만으로 새 표면의 correction gain을 산출할 수 없다. [Mozannar & Sontag][M] |

**서로 다르게 보이는 결과의 원인.** Fishel & Loeb의 높은 성공률은 미리 경험한 재질 중 정체를 맞히는 과제에서 나왔다. Song & Boularias의 일부 실패는 같은 물체에서 지지 조건까지 바꾼 뒤 나타났다. 표면을 구별하는 accuracy, 같은 조건의 pose error, 변경된 intervention의 task success는 서로 다른 측정량이다. Attia et al.과 Zhong et al.의 결과를 함께 보면 task와 관측 사이의 전달 관계를 알려주는 측정이 유리하다. 기하학적 거리가 가까운 동작에 대한 보편적인 우위까지 입증하는 자료는 없다. [Fishel & Loeb][F] · [Song & Boularias][S] · [Attia et al.][A] · [Zhong et al.][Z]

### 무엇을 측정해야 마지막 probe에 기여를 돌릴 수 있는가

아래 판정식은 v36의 paired 실험에 대한 분석 제안이다. 환경 e에서 같은 관측을 사용했을 때 보정한 Brier와 원래 Brier의 차이를 Δₑ, 해당 환경에서 보정을 켰으면 Gₑ=1이라고 두면 gate의 평균 변화는 E[GΔ]다. 같은 비율로 무작위로 켜는 policy의 기대 변화는 E[G]E[Δ]다. 두 값의 차이 Cov(G,Δ)가 음수이면 보정이 상대적으로 유리한 환경에 보정을 더 자주 적용했다는 뜻이다. 표면별 평균이 다른 경우 pooled covariance에는 표면군 구분의 효과가 포함된다. 각 군 안에서 계산하면 같은 surface regime 내의 구분 능력을 점검할 수 있다.

| 남은 공백 | 결과를 해석하거나 확장할 때 필요한 비교 |
|---|---|
| 보정을 덜 쓰는 것만으로 weak/isotropic 평균 loss가 내려갈 수 있다. Coverage와 accuracy가 함께 달라졌던 deferral 선례가 있다. [Mozannar & Sontag][M] | 항상 적용·항상 생략과 함께, 같은 적용률을 가진 randomized gate와 실제 gate의 차이를 보고한다. 모든 환경에서 correction이 꺼졌다면 약한 표면의 손실을 피한 효과는 확인할 수 있고 환경별 선택 효과는 0이다. |
| 네 번째 행동과 terminal decoder를 동시에 바꾸면 두 효과가 합쳐진다. 같은 물성 추정도 후속 과제에 따라 효용이 달랐다. [Song & Boularias][S] · [Zhong et al.][Z] | 같은 첫 세 접촉을 공유하고, 네 번째 접촉의 입력을 가린 gate 및 두 행동에서 같은 방식으로 학습한 gate를 함께 비교한다. 각 행동에서 강제 on·off를 같은 관측으로 채점한다. |
| Source에서 많이 보던 표면에 유리한 gate가 target mixture에서도 유리할 수 있다. 재료 조합에 따라 방향이 뒤집힐 수 있다. [Tramsen et al.][T] | Strong·weak·isotropic에서 보정 적용률과 paired Brier를 각각 계산한다. Source와 달라진 환경 비율에서의 pooled loss는 해당 비율을 명시한다. |
| Query와 가까운 동작이 friction에 민감한 동작인지는 contact state에 따라 달라진다. 선행연구에서는 force·속도 조합이나 upstream 위치가 측정의 가치를 바꿨다. [Fishel & Loeb][F] · [Zhong et al.][Z] | Friction cone 제약을 적용하면 sticking 상태에서 tangential response가 마찰 허용 범위 안에 있으면 여러 friction coefficient가 같은 측정을 낼 수 있다. Probe가 slip을 유발한 비율, plausible noise 변화, 별개의 deployment 물체에서의 gain을 함께 살핀다. [Fazeli et al., §2][I] |

### 후속 연구 질문 10개

| 질문 | 공백·중요성 | 조작할 변수와 검증 방법 |
|---|---|---|
| 1. 어느 높이와 속도에서 약한 방향성이 드러나는가? | 예측할 위치와 정보가 많은 측정 위치가 다를 수 있다. [Zhong et al.][Z] | 높이와 속도를 독립 교차하고 held-out query를 고정해 Brier와 강제 on의 상대 이득을 계산한다. |
| 2. 네 번째 접촉이 stick이면 무엇을 알 수 있는가? | 접촉 동작별로 잘 드러나는 물성이 다르고, 접촉 제약은 식별 가능한 물성 조합을 제한한다. [Fishel & Loeb][F] · [Fazeli et al.][I] | Normal impulse와 tangential speed를 독립 조작하고 probe stick/slip에 따른 측정 분포와 conditional Brier를 비교한다. |
| 3. Directional amplitude가 어느 범위에서 gate가 바뀌는가? | 재료 조합에 따라 방향과 크기가 변한다. [Tramsen et al.][T] | 연속 amplitude sweep에 방향이 없는 조건을 포함하고 적용률 및 oracle 대비 regret을 추정한다. |
| 4. Target에 isotropic 표면이 많아져도 gate가 probe의 정보를 활용하는가? | 예측자에게 맡기는 비율이 달라지면 평균 성능도 달라진다. [Mozannar & Sontag][M] | 같은 결과에 target prevalence를 다시 가중하고 같은 coverage의 무작위 대조와 비교한다. |
| 5. 같은 그림으로 보이는 두 접촉면에서 gate가 유지되는가? | Visuo-haptic friction 예측은 visual similarity와 물성의 관계에 의존한다. [Le et al.][L] | 표면 외관을 맞추고 force response만 바꾼 object–surface pair를 독립 holdout한다. |
| 6. Geometry와 friction이 서로 보상하면 방향 보정이 유리한가? | 서로 다른 물성이 같은 운동을 설명한 실물 사례가 있다. [Song & Boularias][S] | Curvature와 directional amplitude를 독립 조작하고 관측이 유사한 환경 쌍에서 다음 충돌의 확률을 평가한다. |
| 7. 단일 query에 맞춘 probe가 다른 query에도 유용한가? | Quantity of interest가 달라질 때 sensor design이 달라졌다. [Attia et al.][A] · [Zhong et al.][Z] | Calibration 후 query 높이·속도 범위를 넓힌 holdout을 사용해 query별 Brier와 worst-group loss를 계산한다. |
| 8. 약한 방향성을 binary gate와 연속 shrinkage 중 어느 쪽이 잘 다루는가? | Classification에서의 최적 routing 규칙은 연속 확률 예측의 손실에 그대로 대응하지 않는다. [Mozannar & Sontag][M] | Source에서 hard gate와 연속 interpolation weight를 따로 고정하고 동일 target에서 paired Brier를 비교한다. |
| 9. 같은 네 번으로 friction coefficient의 연속 posterior를 추정하는 방법과 차이가 있는가? | 실물에서 연속 물성 추정이 후속 manipulation에 쓰였다. [Le et al.][L] · [Song & Boularias][S] | Isotropic mass를 포함한 directional physics model과 frozen correction에 동일한 관측·계산 예산을 부여한다. |
| 10. 실제 robot에서 probe 교체에 드는 비용까지 포함하면 어느 방법이 유리한가? | 다른 motion의 force·속도와 과제 성공을 실물에서 측정한 선례가 있다. [Fishel & Loeb][F] · [Song & Boularias][S] | 물체와 surface session을 동시 holdout하고 reset·motion·sensor acquisition 포함 wall time, GPU p90, binary impact Brier를 기록한다. |

논문에서는 독립 deployment 물체의 event label, 네 번째 접촉을 가린 대조, 동일한 적용률의 대조가 각기 다른 주장을 뒷받침한다. 첫 번째는 최종 확률의 예측 성능, 두 번째는 추가 측정의 가치, 세 번째는 환경에 맞춰 보정을 선택한 가치를 검증한다. 새 동작이 잘 작동하는 물리적 이유를 주장하려면 별도 측정한 anisotropy나 contact state와 sensor response 사이의 대응도 필요하다. 실물 자료가 외부 타당성을 지지하는 범위는 각 문헌이 측정한 동작과 정답까지다. [Tramsen et al.][T] · [Le et al.][L] · [Song & Boularias][S]

[F]: https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00004/full
[T]: https://findresearcher.sdu.dk/ws/files/151776983/rsif.2017.0629.pdf
[S]: https://www.roboticsproceedings.org/rss16/p099.pdf
[L]: https://arxiv.org/pdf/2010.08277
[A]: https://arxiv.org/pdf/1802.06517
[Z]: https://arxiv.org/pdf/2403.18072
[M]: https://proceedings.mlr.press/v119/mozannar20b/mozannar20b.pdf
[I]: https://groups.csail.mit.edu/robotics-center/public_papers/Fazeli17a.pdf

## 재현 기록

GPU에서 32개 paired block을 완료했다. Raw archive 224개의 SHA-256을 대조하고 6개 방법의 예측에서 Brier를 다시 계산했다. θ로 재생성한 query label 및 near velocity도 일치했다. Local regimes에서 conventional 4개 응답·원래 예측·세 측정의 gate가 모두 원소별로 동일함을 확인했다. Calibration, deployment와 bank의 seed ID 576개는 서로 다른 block 간에 겹치지 않는다. [Source 고정값](gate36_freeze.json) · [GPU 코드](run36.py) · [실행 결과](results/completion.json) · [분석 수치](stats36.json) · [재현 방법](REPRODUCE_KO.md)

서버 보고서 경로는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/query_matched_probe_v36/REPORT_KO.md`다.
