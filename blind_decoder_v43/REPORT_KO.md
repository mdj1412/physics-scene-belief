# v43 · 이름을 가린 새 물체·표면·마찰 법칙에서 geometry와 correction을 분리한 결과

**사전 지정한 두 질문의 답은 다르다.** Source-trained robust decoder의 v38 optimizer는 temperature를 이기지 못했다. 접촉 위치가 어긋나면 projection이 손해가 되는 현상은 12개 catalog 모두에서 재현됐다. 다만 접촉점을 정확히 맞췄을 때 projection이 평균적으로 유리하다는 조건은 충족하지 못했다. V42에서 보인 완전한 손익 반전은 v43에서 형상에 따라 달랐다.

| 주 질문 | Blind OOD 결과 | 판정 |
|---|---|---|
| Robust decoder에서 optimizer를 써야 하는가? | Shifted Brier가 optimizer 0.143837, temperature 0.143771. Optimizer−temperature +0.000066, 동시 95% CI [−0.002586, +0.002719]. | 우위 미확인. 이 조건에서 optimizer가 Brier를 0.003 이상 줄인다는 가설은 구간 밖이다. |
| 변형 방향을 지우는 projection은 언제 손해인가? | 보정 없이 privileged−original이 nominal +0.000012 [−0.000680, +0.000703], shifted +0.004550 [+0.003872, +0.005227]. Interaction +0.004538 [+0.004206, +0.004870]. | 정확히 맞춘 때보다 어긋난 조건에서 projection의 상대 손해가 컸다. Pooled nominal 이득은 미확인이다. Fold에서만 nominal 이득과 shifted 손해가 함께 나타났다. |
| 새 decoder의 이득은 어디서 생겼는가? | 같은 구조·학습량에서 nominal-only ERM의 shifted Brier는 0.155147, offset 결과까지 학습한 robust는 0.143347. 후자가 12/12 panel에서 개선됐다. | Source에서 실제 접촉점이 바뀌었을 때의 outcome을 학습한 효과가 컸다. 접촉점이 정확할 때는 원래 decoder의 0%가 robust 0%보다 유리했다. |

동일 GPU에서 512개 물체의 warm p90은 **robust+temperature 474.75 ms**, **robust+optimizer 479.60 ms**였다. Optimizer가 4.85 ms 느렸다. 비율은 1.0102배 [paired bootstrap 95% CI 1.0077, 1.0124]로 사전 비용 한도 +2% 안에 들었지만, 실용적 Brier 개선 기준은 충족하지 못했다. 열두 조합을 같은 장치에서 비교한 표를 아래에 실었다.

![새 family에서의 보정·projection 손익·GPU 지연시간](blind_ood_v43.png)

그림 A는 여덟 비영 placement offset의 균등 평균이다. B의 음영은 아홉 offset에 대한 동시 95% 구간이며 양수는 projection의 손해다. C는 512개 물체의 paid evidence를 처리하는 warm GPU p90이다.

## 무엇을 숨겼고 무엇을 바꿨는가

예측 대상은 현재 센서로 보고 있는 외곽 자체가 아니라, 같은 물체가 이후 +0.425의 속도로 충돌했을 때의 stick 확률이다. 관측 중심은 19 mm 또는 31 mm이고, 실제 충돌점만 그 위아래로 이동한다. 세 decoder 모두 같은 관측을 받는다. Simulation evaluator는 외곽·표면·법칙·위치오차를 알지만 이 정보를 운영 예측기에 제공하지 않는다.

| 실험 단계 | 사용할 수 있는 정보와 비용 |
|---|---|
| Family 설계 | Source의 flare·roll 형상 및 saturating·crossover law에서 벗어나 fold·ripple object family, localized inlay·periodic woven surface field, rational·Gaussian-rate constitutive law를 새로 정의했다. 2×2×2 완전 교차. 세 수준의 contour variant에는 기존 이름 base·hollow·bulge를 재사용하되 새 basis로 변형했다. 같은 rigid point-impact solver와 latent coordinates를 사용한다. |
| Source 학습 | v42 source 12 catalog, 73,728개 row를 공통 neutral prior에서 다시 추론했다. Panel 0–7에서 2방향 shape holdout으로 hidden width를 고른 후 residual을 학습하고, 8–9에서 decoder별 T를 fit하고, 10–11에서 0%·fixed·T 중 단순 대안을 골랐다. T는 비영 offset의 NLL, robust residual은 nominal과 비영 offset의 Brier를 1:1로 가중한다. |
| Blind acquisition | 모든 환경에서 같은 neutral prior와 같은 source bank seed를 썼다. 각 환경에서 factual evidence로 128개 중 query view 115개, physical probe 13개를 구입했다. 채점 시 같은 설계에서 얻을 수 있었던 두 센서 중 미구입 값은 NaN으로 지운다. 환경별 실제 prior, family 이름, 실제 접촉점은 주지 않는다. |
| Original / robust | Original은 기존 16,384개 physics hypothesis와 센서 likelihood로 예측한다. Robust는 같은 base logit에 28개 paid feature에서 학습한 width-8 residual을 더한다. 같은 cutoff의 probability와 profile 기울기·곡률을 쓴다. 네 번째 calibration response는 residual 학습·예측에 들어가지 않는다. |
| Privileged | 같은 bank와 decoder에서 표준화한 query profile의 실제 변형 방향 하나만 제거한다. 그 축은 evaluator가 제공한다. 방향을 알 때의 진단 비교이며 실물에서 무료로 얻을 수 있는 정보로 보지 않는다. Probe branch의 likelihood는 같다. |
| 네 보정 | Zero는 같은 decoder의 p. Temperature는 source 하나의 T로 sigmoid(logit(p)/T)를 계산한다. Fixed와 optimizer는 같은 β와 첫 세 접촉의 signed cue z를 사용한다. 네 번째 response는 source gain table에서 optimizer의 보정 강도 w를 고르는 데 쓰인다. Fixed에서는 w=0.325다. |
| Target 채점 | 독립 12 catalog × 2 object × 2 surface × 2 law × 3 contour × 2 nominal height = 576개 archive. 매번 4개 환경 × 128개 물체, 센서 노이즈 4회. 물체·변형·센서를 대응시키고 9개 실제 높이를 채점한다. 같은 원형 물체를 서로 다른 조건에 적용하므로 반복을 독립 catalog로 세지 않았다. |
| GPU p90 | 동일 GPU에서 같은 512개 물체를 단 한 번의 sensor noise로 처리하는 12개 방법의 완전 crossover. Catalog별 1회 warm-up 후 3회 무작위 순회, 방법별 36개 기록. K256 acquisition, 선택한 sensor의 routing, posterior, 적용한 correction 및 CPU–GPU 전송·동기화를 포함한다. 물체를 움직이고 reset하는 시간과 offline source 준비 시간은 제외했다. |

V42와 달리 neutral prior가 acquisition과 decoding 양쪽에 적용됐다. 이전 GPU p90은 1,024개 물체 및 다른 acquisition 경로에서 측정했으므로 이번 512개 batch와 절대시간을 직접 비교하지 않는다. T는 original 3.296, robust 1.253, privileged 3.614였다. Source에서 선정한 최선의 단순 정책은 original 0%, robust 0%, privileged temperature였다. [학습 분할과 모델](MODEL_KO.md) · [물리 생성식](WORLD_KO.md) · [사전 판정 기준](PLAN_KO.md)

## 12개 조합의 held-out Brier와 GPU p90

512개 물체(4개 환경 × 128개)를 처리했다. 각 물체는 배정된 관측 하나를 사용했다. Brier는 낮을수록 좋다. Nominal은 실제 접촉점과 명목 높이가 일치한 경우, shifted는 여덟 비영 signed offset의 균등 평균이다. 각 조합은 동일한 물체·충돌 결과·관측을 공유한다.

| Decoder | Correction | Nominal Brier | Shifted Brier | Warm GPU p90, ms [95% CI] |
|---|---|---:|---:|---:|
| Original | 0% | 0.130837 | 0.147784 | 473.09 [472.59, 475.90] |
| Original | Temperature | 0.139136 | 0.150216 | 472.92 [472.40, 474.66] |
| Original | Fixed 32.5% | 0.130450 | 0.147661 | 477.42 [476.23, 478.70] |
| Original | v38 optimizer | 0.130715 | 0.149204 | 480.56 [478.89, 481.36] |
| Source-trained robust | 0% | 0.134838 | 0.143347 | 473.71 [473.14, 475.06] |
| Source-trained robust | Temperature | 0.135902 | 0.143771 | 474.75 [473.72, 476.65] |
| Source-trained robust | Fixed 32.5% | 0.134383 | 0.143218 | 478.51 [476.78, 480.31] |
| Source-trained robust | v38 optimizer | 0.133558 | 0.143837 | 479.60 [478.17, 480.45] |
| Privileged projection | 0% | 0.130849 | 0.152334 | 474.30 [472.19, 474.74] |
| Privileged projection | Temperature | 0.141538 | 0.154273 | 473.32 [472.10, 475.40] |
| Privileged projection | Fixed 32.5% | 0.130475 | 0.152210 | 478.65 [476.80, 479.78] |
| Privileged projection | v38 optimizer | 0.130367 | 0.153316 | 478.94 [478.41, 479.76] |

### 사전 지정한 네 대조

양수는 앞 방법의 Brier가 더 높다는 뜻이다. 12개 독립 catalog에 대해 paired t 구간을 구하고 아래 네 대조의 family-wise error rate를 5%로 보정했다. 반복 관측과 이동 높이는 같은 물체의 측정이다.

| 대조 | 평균 차이 [동시 95% CI] |
|---|---:|
| Robust에서 optimizer−temperature, shifted | +0.000066 [-0.002586, +0.002719] |
| Privileged−original, 보정 0%, nominal | +0.000012 [-0.000680, +0.000703] |
| Privileged−original, 보정 0%, shifted | +0.004550 [+0.003872, +0.005227] |
| 위 geometry 차이의 shifted−nominal interaction | +0.004538 [+0.004206, +0.004870] |

### 어떤 family와 어느 signed offset에서 재현됐는가

여덟 family 조합에서 같은 panel끼리 비교한 점추정이다. 동일한 두 수준의 family를 재조합했으므로 여덟 개의 독립적인 재현으로 세지 않는다.

| Object / surface / law | Robust optimizer−T | Geometry Δ nominal | Geometry Δ shifted | Interaction |
|---|---:|---:|---:|---:|
| fold/inlay/rational | +0.000697 | -0.001270 | +0.004396 | +0.005666 |
| fold/inlay/gaussian_rate | +0.001313 | -0.001443 | +0.004143 | +0.005585 |
| fold/woven/rational | -0.000794 | -0.000947 | +0.004503 | +0.005450 |
| fold/woven/gaussian_rate | +0.000018 | -0.001149 | +0.004269 | +0.005419 |
| ripple/inlay/rational | +0.000256 | +0.000292 | +0.004289 | +0.003997 |
| ripple/inlay/gaussian_rate | +0.000969 | -0.000374 | +0.003782 | +0.004156 |
| ripple/woven/rational | -0.001368 | +0.002752 | +0.005737 | +0.002986 |
| ripple/woven/gaussian_rate | -0.000560 | +0.002235 | +0.005280 | +0.003045 |

| 숨은 offset (mm) | Contour 변경에 따른 stick/slip 전환 | Geometry Δ, 0% [9점 동시 95% CI] |
|---:|---:|---:|
| -1 | 19.14% | +0.003227 [+0.002550, +0.003904] |
| -0.5 | 14.84% | +0.002754 [+0.002034, +0.003474] |
| -0.25 | 9.78% | +0.002517 [+0.001766, +0.003267] |
| -0.125 | 5.63% | +0.001830 [+0.001088, +0.002572] |
| +0 | 0.00% | +0.000012 [-0.000785, +0.000809] |
| +0.125 | 9.27% | +0.004170 [+0.003237, +0.005104] |
| +0.25 | 13.36% | +0.005602 [+0.004674, +0.006531] |
| +0.5 | 20.90% | +0.008600 [+0.007689, +0.009512] |
| +1 | 20.42% | +0.007699 [+0.006855, +0.008543] |

전환율은 같은 물체의 hollow·bulge와 base를 **같은 실제 높이**에서 비교했다. Geometry Δ는 base까지 포함한 전체 물체에서 privileged−original이다. [전체 수치와 panel별 결과](stats43.json)

## 무엇이 결과를 설명하는가

**Optimizer의 움직임에는 정보가 있었지만 평균 Brier의 이득으로 남지 않았다.** Robust decoder에서 temperature의 확률을 p, optimizer가 그 확률과 다른 양을 d라 하면, Brier 차이는 정확히 E[d²]−2E[d(Y−p)]다. 이번에는 각각 +0.003752와 −0.003686으로 거의 상쇄됐다. 첫 항은 예측을 움직인 크기의 비용, 둘째 항은 움직임이 실제 오차를 줄이는 방향과 얼마나 일치했는지를 나타낸다. 12개 catalog 중 6개에서만 optimizer가 우세했다. 사전 주 대조의 동시 구간 [−0.002586, +0.002719]은 실용적 개선 기준 −0.003을 배제한다. Source에서 고른 최선의 단순 보정인 0%와 비교해도 robust의 optimizer−0%는 +0.000491 [개별 95% CI −0.001275, +0.002256]이었다. 최저 shifted 점추정은 robust+fixed 0.143218이지만 target을 보고 고른 순위이므로 새 정책 우위의 근거로 쓰지 않는다. [보정별·panel별 통계](stats43.json)

**조합 순서와 clipping으로 대안 설명을 검사했다.** 같은 frozen optimizer를 robust temperature 뒤에 추가했을 때 Brier 차이는 +0.000598 [−0.001332, +0.002528]이었다. 환경별 cue를 세 가지 방식으로 다른 환경에 돌려 배정한 참조와 비교하면, 올바른 환경의 cue를 쓴 optimizer는 Brier가 0.001189 낮았지만 구간이 0을 포함했다. Robust의 확률 중 temperature 연산의 clipping 구간을 벗어난 비율은 0%였다. Original에서는 28.58%가 경계 밖이었지만 경계를 자르지 않고 같은 T를 적용했을 때의 Brier 차이는 −0.000074였다. 이 자료에서 큰 geometry 차이를 clipping 하나로 설명하기 어렵다. 이 항목은 주 판정에 포함하지 않은 보조 분석이다. [보조 분석 명세](DIAGNOSTIC_PLAN_KO.md) · [결과](residual43.json)

**왜 정확히 닿았을 때의 평균 이득이 사라졌나.** Source와 다른 두 형상에서 부호가 갈렸다. 보정 없이 fold의 projection−original은 nominal에서 −0.001202 [−0.001804, −0.000600], shifted에서 +0.004328 [+0.003772, +0.004884]였다. Ripple은 각각 +0.001226 [+0.000139, +0.002313], +0.004772 [+0.003781, +0.005763]였다. 이 네 구간은 같은 자료를 재분석한 형상별 동시 95% 구간이다. 평균 이득의 소실에는 fold와 ripple의 상쇄가 기여했다. 두 형상 모두에서 base/hollow/bulge는 명목 위치의 실제 slope와 label을 공유했다. **Contour만 바꾸어 label이 보존된다는 사실은 그 축을 제거했을 때 평균 예측이 개선된다는 충분조건이 아니다.** 같은 축에는 서로 다른 기본 물체의 국소 기울기를 읽는 신호도 있을 수 있다. 보정 없는 base만 보면 nominal 손익은 +0.000803 [−0.000071, +0.001678]이었다. [형상·변형별 paired 대조](split43.json)

**엇갈린 접촉에서는 물리 경로를 확인했다.** 명목 높이에서 contour만 바꾸면 실제 event 전환이 0%였지만 −0.125 mm에서 5.63%, +0.125 mm에서 9.27%였다. 변형된 fold/31 mm에서는 −0.5 mm와 +0.5 mm의 전환율이 각각 8.56%, 8.33%로 비슷했다. 같은 두 위치에서 projection은 Brier를 각각 0.010528 낮추고 0.005807 높였다. 단순한 전환 횟수보다 profile에서 보이는 기울기가 실제 접촉점의 기울기를 어느 방향으로 가리키는지가 중요했다. Simulator에서 실제 flip의 99.16%는 마찰 한계를 base에 고정해도 impulse demand 변화만으로 재현됐다. 7점 profile의 apparent slope와 실제 위치에서의 slope로 **target 채점 전에** 예측한 손익 부호는 변형 외곽의 32개 비영 조건 중 31개에서 맞았다. 모든 곳에서 손해라고 하면 28개가 맞는다. 이는 실제 위치와 형상을 아는 평가자의 진단이며, 숨긴 높이를 모르는 운영 정책의 정확도를 뜻하지 않는다. [물리 검산 및 부호 예측](PHYSICS_RESULTS_KO.md)

**가장 큰 차이는 decoder와 배포 위치 분포에 있었다.** 0%에서 robust−original은 nominal에서 +0.004002 [개별 95% CI +0.001922, +0.006081], shifted에서 −0.004438 [−0.006023, −0.002853]였다. 잘못 배치되는 비율을 q, 그중 여덟 signed offset이 균등하다고 가정하면 두 decoder의 평균 Brier가 같은 q는 **47.4%** [paired-panel bootstrap 27.4%, 64.6%]다. 기하 오차가 흔한 작업이라면 source에서 offset까지 학습하는 비용을 치를 근거가 있고, 접촉이 정확한 작업에서는 손해일 수 있다. 실제 장비의 q는 측정하지 않았다. 이 수치는 현재 설계의 배포 민감도다. [혼합 비율별 손익](split43.json)

**학습 자료의 효과를 같은 구조로 분리했다.** 같은 source 물체·관측, width 8, seed, 45 epoch를 사용해 nominal label만 학습한 ERM을 CPU에서 복원했다. 앞으로 올 물체의 결과는 훈련에 사용되지 않았다. 이 ERM의 Brier는 nominal 0.137090, shifted 0.155147로 두 위치 모두 original보다 높았다. Offset의 실제 결과까지 학습한 robust−nominal-only ERM은 nominal에서 −0.002252 [두 대조 동시 95% CI −0.003400, −0.001104], shifted에서 **−0.011800 [−0.012977, −0.010624]**였다. Shifted에서는 12개 panel 모두에서 개선됐다. 이 보조 ablation은 새 입력을 추가하지 않고 source에서 어떤 충돌의 label을 제공했는지만 바꿨다. 모델 용량만 늘렸다는 설명은 이 비교와 맞지 않는다. 단, 정확히 맞춘 접촉에서는 두 learned 모델 모두 original 0.130837을 넘어서지 못했다. 이 ERM의 GPU p90은 별도로 측정하지 않았다. [동일 구조·동일 학습량 검산](ERM_CONTROL_KO.md)

## 논문 기여와 다음 판정

1. **기여로 남길 결과.** 새 shape·surface·law를 동시에 숨긴 조건에서도, 접촉점이 조금 바뀌면 무해해 보였던 외곽 변형이 실제 결과를 바꾸고 그 정보를 지우는 비용이 증가했다. 같은 oracle 방향을 적용해도 fold와 ripple의 nominal 손익이 반대였다는 점이 중요하다. 기존 문헌의 일반적인 invariance 경고에 더할 수 있는 부분은, 같은 물체에서 local tangent·event 전환·예측 오차를 짝지어 방향과 크기를 설명한 결과다. [사전 예측표](geometry_forecast43.json) · [독립 물리 검산](physics_results43.json)
2. **축소할 주장.** 이 실험은 v38 optimizer의 blind OOD 우위나, nominal에서 projection이 항상 이롭다는 주장을 뒷받침하지 않는다. 직전 v42 temperature replay와 비교할 때 target family 외에도 bank prior와 acquisition을 neutral로 바꿨고 T를 별도 source에서 다시 학습했다. 두 버전의 차이를 특정한 하나의 변경에 귀속할 수 없다. 모든 방법이 같은 네 번을 지불했으므로 네 번째 probe의 구매 가치도 측정되지 않았다. [v42 temperature 재분석](../post_v42_review/temperature_replay.json)
3. **다음 확인.** 같은 물체·같은 local geometry에서 tip radius와 접촉 patch를 바꾸는 독립 solver를 쓰고, solver가 기록한 tangential velocity와 displacement로 stick/slip을 정의한다. Nominal과 signed offset에서 projection의 부호를 먼저 예고하고 두 original/robust decoder 모두에서 채점한다. 실물에서는 위치 센서로 q와 방향별 분포를 먼저 측정해야 실제로 어떤 decoder가 유리한지 계산할 수 있다. 추가 마찰 표본을 늘리는 연구는 그 뒤에 비용별 gain을 독립 평가할 때 설득력이 있다.

## 선행연구에 비추어 본 기여

아래 표는 v43의 두 가설을 반박할 수 있는 원저를 선정한 비교다. 각 논문에서 사용한 표본과 평가 단위가 달라 성능 수치를 합산하지 않았다. 이전 설계와의 직접 대조에는 [v42 연구 보고서](../height_offset_v42/REPORT_KO.md)를 사용했다.

| 질문을 던진 연구 | 방법·표본 | 저자들이 확인한 결과 | v43과의 차이 및 한계 |
|---|---|---|---|
| Guo et al. · 복잡한 모델의 confidence 오차를 단순하게 줄일 수 있는가? | ImageNet, CIFAR, Reuters 등의 분류 모델에서 validation NLL로 scalar temperature를 학습하고 교정법을 비교했다. | 여러 자료에서 temperature가 calibration을 개선했다. Reuters에는 같은 개선이 성립하지 않았다. | Source에서 학습하는 양의 스칼라 하나가 강한 대안이다. 접촉 시점이 바뀌는 binary collision에 그대로 전이하는지는 이 논문이 다루지 않았다. [원문 §4](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf) |
| Ovadia et al. · shift가 커져도 예측 불확실성을 신뢰할 수 있는가? | 이미지 corruption의 종류·강도, text, 광고 클릭 자료 등에서 여러 uncertainty 방법을 대조했다. | IID validation의 temperature 보정이 shift의 강도가 커질 때 calibration을 보장하지 않았다. | Guo et al.과의 차이는 validation과 test의 관계 및 shift의 강도다. 두 결과 모두 새 마찰 법칙에서의 승자나 Brier 효과 크기를 정해 주지 않는다. [원문](https://arxiv.org/abs/1906.02530) |
| Benton et al. · 어느 범위의 변환까지 invariance를 학습해야 하는가? | Augerino가 augmentation 분포와 예측기를 함께 학습했다. Mario–Iggy 회전 분류와 얼굴 각도 예측 등을 사용했다. | 목표 label에 따라 유용한 변환 범위가 달라졌다. | Invariance가 task에 의존한다는 원리는 선례가 있다. 같은 물체의 실제 접촉 위치와 운동 방정식으로 label 전환을 만드는 검증은 다르다. [원문](https://arxiv.org/abs/2010.11882) |
| Zhao et al. · invariant representation이 domain adaptation에 충분한가? | 이론적 반례와 bound, MNIST·USPS·SVHN의 transfer 실험을 사용했다. | Input distribution을 맞춰도 조건과 label 분포가 달라지면 target 성능이 악화될 수 있다. | 이론과 digit transfer는 point impact의 구체적인 sign reversal을 예측하지 않는다. [원문](https://proceedings.mlr.press/v97/zhao19a.html) |
| Meißenhelter et al. · 위치 오차가 접촉 이후 상태로 어떻게 전파되는가? | 서로 다른 3D mesh에서 위치와 contact normal의 불확실성을 근사하고 simulation sampling을 참조 분포로 비교했다. | 해당 예제에서 선형화보다 분포 근사가 나았고 대량 sampling보다 빠르게 계산했다. | 위치 불확실성의 물리 전파 자체는 알려져 있다. Decoder가 변형 축을 버릴 때 downstream Brier가 어떻게 변하는지는 그 논문의 지표와 다르다. [원문](https://cgvr.cs.uni-bremen.de/papers/icra40_24/ICRA40_2024_UncertainPhysics.pdf) |
| Fazeli et al. · impact solver가 실물 충돌을 어디까지 설명하는가? | 실험 장치의 2,000회 planar drop 중 품질 검사를 통과한 1,718회로 여섯 rigid-body contact model을 비교했다. | 일부 실측 impulse를 모형이 표현하기 어렵고, 최적으로 맞춘 parameter의 물리 해석에 한계가 있었다. | 한 solver에서 family를 바꿔도 이 외부 타당성 문제는 남는다. [원문](https://arxiv.org/pdf/1710.04979) |

**근거 지도.** Sensor와 contour가 만드는 확률 오차 → source에서 fit한 confidence 보정 → 미래 물체의 Brier라는 경로와, 접촉 높이 → local tangent → 마찰 한계 → 실제 stick event라는 경로가 만난다. 앞 경로에는 분류 calibration에 대한 실증 근거가 있고, 뒤 경로에는 접촉 위치 불확실성과 실물 solver 오차에 대한 독립 근거가 있다. 실물에서 두 경로의 상호작용이 재현되는지는 별도 가설이다. 같은 형상에서 label이 달라지는 simulator intervention은 모형 안의 원인을 검증한다. 서로 다른 독립 family에서의 평균 상관만으로 같은 인과 해석을 할 수 없다. [Guo et al.](https://proceedings.mlr.press/v70/guo17a.html) · [Meißenhelter et al.](https://cgvr.cs.uni-bremen.de/papers/icra40_24/ICRA40_2024_UncertainPhysics.pdf) · [Fazeli et al.](https://arxiv.org/pdf/1710.04979)

### 남은 질문과 검증 방법

| 연구 질문 | 공백·중요성 | 조작할 변수와 검증 |
|---|---|---|
| 1. Confidence를 조정한 다음에도 환경별 signed correction이 남는가? | 단일 T로 설명할 수 있는 개선과 마찰 cue의 독립 기여가 겹칠 수 있다. [Guo et al.](https://proceedings.mlr.press/v70/guo17a.html) | 별도 source에서 T를 먼저 fit하고 residual correction을 추가해 새 family Brier를 비교한다. |
| 2. 언제 source temperature 자체가 부적절한가? | Shift가 커질 때 calibration이 악화될 수 있다. [Ovadia et al.](https://arxiv.org/abs/1906.02530) | Label prevalence·위치오차 분포를 각각 변경하고 Brier와 reliability curve를 얻는다. |
| 3. 어떤 공간 해상도부터 기하 정보를 보존할 수 있는가? | 센서에서 보이지 않는 normal과 decoder가 놓친 normal은 대응이 다르다. [Meißenhelter et al.](https://cgvr.cs.uni-bremen.de/papers/icra40_24/ICRA40_2024_UncertainPhysics.pdf) | 7점·고밀도 profile × quadratic·local spline을 교차하고 slope error와 event 확률을 채점한다. |
| 4. 어떤 변형 축을 얼마나 약화해야 하는가? | Hard invariance가 task에 따라 불필요한 정보를 제거할 수 있다. [Benton et al.](https://arxiv.org/abs/2010.11882) | 같은 prior·decoder에서 hard, finite-variance, 원래 likelihood를 비교한다. |
| 5. 접촉 높이를 직접 재면 반전이 얼마나 줄어드는가? | 위치 uncertainty가 contact normal에 전파된다. [Meißenhelter et al.](https://cgvr.cs.uni-bremen.de/papers/icra40_24/ICRA40_2024_UncertainPhysics.pdf) | 절대오차를 아는 위치 측정과 geometry 해상도를 독립 조작한다. |
| 6. 새로운 법칙인가 새 표면인가, 어느 변화에서 정책이 깨지는가? | 두 요인을 동시에 홀드아웃하면 실제 배포 시험은 되지만 개별 요인의 기여는 분리되지 않는다. [Zhao et al.](https://proceedings.mlr.press/v97/zhao19a.html) | 동일 object family에서 single-axis 및 joint holdout을 비교한다. |
| 7. 다른 solver와 finite contact patch에서도 같은 부호인가? | 일부 실측 impulse를 point impact model이 설명하지 못했다. [Fazeli et al.](https://arxiv.org/pdf/1710.04979) | Tip radius·compliance·속도 threshold를 고정한 별도 solver에서 signed effect를 사전 예측한다. |
| 8. 몇 개 target label을 사면 고정보다 나은가? | V42에서 별도 물체의 target outcome을 준 경우와 네 번의 calibration만 준 경우 사이에 큰 차이가 있었다. [이전 검산](../height_offset_v42/SELECTOR_AUDIT_KO.md) | 환경별 0·4·16·64개 donor를 무작위 배정해 다른 물체의 비용 대비 Brier를 평가한다. |
| 9. 네 번째 물리 접촉을 구매하는 것이 유리한가? | 같은 네 번을 지불한 대조는 구매 여부의 효과를 식별하지 못한다. [기존 비용 분석](../adaptive_stopping_v29/REPORT_KO.md) | 같은 prefix에서 stop·buy를 무작위 배정해 reset·대기 시간을 포함한 break-even을 낸다. |
| 10. 실물에서 surface session이 달라져도 geometry 신호가 남는가? | 실측 planar pushing에는 위치·힘·형상 변이가 기록됐지만 v43의 단발 impact와 시간척도가 다르다. [Yu et al.](https://arxiv.org/pdf/1604.04038) | 공개 trajectory에서는 tangential velocity를 먼저 검증하고, 실제 impact에서는 다른 날짜·물체·표면 session별 paired probability를 평가한다. |

## 실행·검산·재현

- 서버 프로젝트는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/blind_decoder_v43`다. 2026년 11월 16일 같은 모델의 NVIDIA RTX PRO 6000 Blackwell Server Edition 두 장(node55·node56)이 유휴임을 확인했다. 할당 2744468·2744471에서 catalog를 나눠 실행하고, GPU p90은 node55의 2744468 한 장에서 재측정했다. [짝수 panel 기록](results/run_even.json) · [홀수 panel 기록](results/run_odd.json) · [latency 기록](results/run_timing.json)
- 독립 작업으로 world·물리 검산, source 학습·leakage 검산, GPU runner·timing을 나눴다. 사전 기하 예측은 18:11:43 KST, target 실행 코드와 robust model은 18:14:23 KST에 고정했다. 첫 target run 전에 이전 코드 의존성까지 checksum을 확인했다. 같은 원형 물체를 조건별로 재사용한 6,144번의 deployment draw에는 서로 다른 panel·population별 catalog prototype 5,951개가 있었다. Calibration prototype은 별도로 192개였다. 같은 물체의 반복 noise와 높이를 독립 표본으로 세지 않았다. [주 실험 동결 목록](freeze43.sha256) · [source 동결](model43.sha256) · [기하 예측 동결](geometry_forecast43.json)
- 576개 raw archive의 SHA-256을 검산하고 2,654,208개 binary event를 별도로 구현한 식과 대조했다. 12개 방법의 Brier를 직접 계산한 값과 맞췄다. Robust model의 모든 미구입 sensor 값을 큰 유한값으로 바꿔도 출력 변화는 0이었다. Registry의 family 이름·평가자 전용 latent는 operating function의 인자에 없다. 같은 bank를 모든 panel에 공유해 bank seed 자체의 불확실성은 이번 구간에 포함되지 않는다. [독립 확률 검산](audit43.json) · [물리 검산](physics_results43.json) · [사전 설계](PLAN_KO.md)
- 전체 결과는 `results/panel_NN/*.npz`, GPU 원측정은 `results/timing/panel_NN.npz`에 있다. 압축 파일에 paid sensor, 세 decoder의 12개 확률, 9개 높이의 label, correction cue가 있다. [재현 순서](REPRODUCE_KO.md) · [확률 분석](stats43.py) · [동일 GPU의 교차 측정 코드](runner43.py) · [독립 재채점](audit43.py)

14,155,776개 저장 예측의 단순 보정식을 독립 재계산한 최대 오차는 2.22×10⁻¹⁶, 동일 GPU timing path의 재계산 오차는 1.36×10⁻¹³이었다. 작업 종료 후 사용한 두 GPU 모두 0 MiB와 compute PID 0개를 확인했다. [GPU 검산](audit_gpu43.json) · [종료 상태](release43.log)
