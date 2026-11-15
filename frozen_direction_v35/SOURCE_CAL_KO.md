# v35에서 방향 정보 없이도 같은 개선을 얻을 수 있는가?

## Source에서 고정한 두 가지 대조군

V34의 fold 0에서는 estimated direction의 β를 **ridged_coarse**의 8개 block × 2개 orientation × 2개 schedule에서 공동으로 학습했다. 같은 32개 metrics archive에서 fixed 4의 같은 action을 선택했다. 각 stratum에서 8개 환경 × 128개 deployment worlds × 32개 sensor draws의 예측을 사용했다. 이 수는 1,048,576개 score이며, 같은 label과 prefix를 공유하는 반복을 포함한다. 두 source-only 보정은 target 자료 없이 CPU에서 학습·고정했다. [원래 fit](../bidirectional_probe_v34/anchored34.py) · [source-only fit 및 32개 입력 해시](source_cal35_frozen.json)

원 physics probability를 p, logit을 η라고 할 때 constant offset은 sigmoid(η + 0.01965140895), temperature scaling은 sigmoid(0.55849826698 η)다. T는 1.79051585140이다. 두 방법 모두 p를 [10⁻¹², 1−10⁻¹²]로 제한해 logit을 정의하고, binary log loss에 v34와 같은 0.005의 quadratic ridge를 더했다. Ridge의 기준은 identity에 해당하는 offset 0 및 inverse temperature 1이다. 원본 p가 경계일 때도 이 작은 clipping을 적용했다. Target family, schedule, sign에 관계없이 같은 계수를 사용했다. [학습 코드](fit_source_cal35.py) · [고정 계수와 입력 해시](source_cal35_frozen.json)

## 독립 target에서의 paired Brier

원래 fixed 4 action 및 관측은 다섯 방법에서 같다. T + signed는 먼저 temperature를 적용한 logit에 같은 v34 βz를 더해 sigmoid를 취한다. 음수는 각 contrast에서 왼쪽 방법의 Brier가 낮음을 뜻한다. [결과 확인 전 고정한 비교](SOURCE_CAL_PLAN_KO.md)

CI는 object × surface 내에서 block을 첫 probe 부호로 층화해 12,000회 재표집한 paired max-standardized interval이다. Mirrored orientation·두 schedule·동일 환경의 32개 sensor draws는 block 안에 함께 남겼다. Directional 표의 pooled 및 cell별 모든 30개 contrast에 대해 동시에 95%를 적용했다. Isotropic 표에서는 독립적으로 18개 contrast를 보정했다. 각 CI는 고정된 family의 block 변동에 관한 추정이다. [평가 코드](score_source_cal35.py) · [수치 원본](source_cal35.json)

| Object × surface | Schedule | Original | Offset | Temperature | Signed | T + signed | Signed − offset [95%] | Signed − temperature [95%] | (T + signed) − T [95%] |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pooled | same | 0.11885 | 0.11885 | 0.11808 | 0.11284 | 0.11393 | -0.00601 [-0.00711, -0.00491] | -0.00524 [-0.00631, -0.00416] | -0.00414 [-0.00547, -0.00281] |
| pooled | alternating | 0.11835 | 0.11835 | 0.11766 | 0.11270 | 0.11410 | -0.00565 [-0.00669, -0.00462] | -0.00497 [-0.00593, -0.00401] | -0.00357 [-0.00486, -0.00228] |
| sinusoid_atan | same | 0.12123 | 0.12103 | 0.12164 | 0.11764 | 0.12063 | -0.00339 [-0.00597, -0.00081] | -0.00400 [-0.00642, -0.00159] | -0.00102 [-0.00415, +0.00212] |
| sinusoid_atan | alternating | 0.12176 | 0.12156 | 0.12198 | 0.11757 | 0.12020 | -0.00398 [-0.00499, -0.00298] | -0.00441 [-0.00559, -0.00322] | -0.00178 [-0.00321, -0.00034] |
| sinusoid_algebraic | same | 0.12195 | 0.12194 | 0.12052 | 0.11458 | 0.11452 | -0.00735 [-0.00976, -0.00494] | -0.00594 [-0.00881, -0.00307] | -0.00600 [-0.00893, -0.00307] |
| sinusoid_algebraic | alternating | 0.12213 | 0.12211 | 0.12059 | 0.11551 | 0.11578 | -0.00661 [-0.00821, -0.00501] | -0.00508 [-0.00551, -0.00465] | -0.00481 [-0.00671, -0.00291] |
| quartic_atan | same | 0.11858 | 0.11864 | 0.11687 | 0.11052 | 0.11040 | -0.00813 [-0.01001, -0.00624] | -0.00636 [-0.00766, -0.00505] | -0.00647 [-0.00832, -0.00462] |
| quartic_atan | alternating | 0.11694 | 0.11702 | 0.11572 | 0.11041 | 0.11137 | -0.00661 [-0.00947, -0.00375] | -0.00531 [-0.00834, -0.00227] | -0.00435 [-0.00795, -0.00076] |
| quartic_algebraic | same | 0.11363 | 0.11378 | 0.11326 | 0.10861 | 0.11018 | -0.00517 [-0.00688, -0.00346] | -0.00465 [-0.00623, -0.00308] | -0.00308 [-0.00553, -0.00063] |
| quartic_algebraic | alternating | 0.11256 | 0.11270 | 0.11237 | 0.10729 | 0.10904 | -0.00541 [-0.00771, -0.00311] | -0.00507 [-0.00703, -0.00311] | -0.00333 [-0.00611, -0.00054] |

### Isotropic control

| Object × surface | Schedule | Original | Offset | Temperature | Signed | T + signed | Signed − offset [95%] | Signed − temperature [95%] | (T + signed) − T [95%] |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pooled | same | 0.10608 | 0.10613 | 0.10919 | 0.11217 | 0.11725 | +0.00604 [+0.00413, +0.00796] | +0.00298 [+0.00109, +0.00487] | +0.00806 [+0.00579, +0.01033] |
| pooled | alternating | 0.10568 | 0.10573 | 0.10892 | 0.11174 | 0.11695 | +0.00601 [+0.00287, +0.00914] | +0.00282 [-0.00022, +0.00585] | +0.00803 [+0.00387, +0.01220] |
| sinusoid_isotropic | same | 0.10662 | 0.10635 | 0.11010 | 0.11258 | 0.11829 | +0.00623 [+0.00402, +0.00844] | +0.00248 [+0.00015, +0.00481] | +0.00819 [+0.00548, +0.01091] |
| sinusoid_isotropic | alternating | 0.10633 | 0.10606 | 0.10992 | 0.11255 | 0.11839 | +0.00649 [+0.00153, +0.01145] | +0.00263 [-0.00219, +0.00746] | +0.00848 [+0.00159, +0.01536] |
| quartic_isotropic | same | 0.10554 | 0.10590 | 0.10827 | 0.11176 | 0.11621 | +0.00586 [+0.00271, +0.00900] | +0.00348 [+0.00048, +0.00649] | +0.00793 [+0.00428, +0.01159] |
| quartic_isotropic | alternating | 0.10504 | 0.10539 | 0.10792 | 0.11092 | 0.11551 | +0.00553 [+0.00161, +0.00944] | +0.00300 [-0.00075, +0.00675] | +0.00759 [+0.00278, +0.01240] |

## 해석

- same에서 signed correction과 constant offset의 pooled Brier 차이는 -0.00601이고 simultaneous 95% CI는 [-0.00711, -0.00491]다. signed correction의 이득을 지지한다.
- same에서 signed correction과 temperature scaling의 pooled Brier 차이는 -0.00524이고 simultaneous 95% CI는 [-0.00631, -0.00416]다. signed correction의 이득을 지지한다.
- alternating에서 signed correction과 constant offset의 pooled Brier 차이는 -0.00565이고 simultaneous 95% CI는 [-0.00669, -0.00462]다. signed correction의 이득을 지지한다.
- alternating에서 signed correction과 temperature scaling의 pooled Brier 차이는 -0.00497이고 simultaneous 95% CI는 [-0.00593, -0.00401]다. signed correction의 이득을 지지한다.
- Temperature를 먼저 적용한 뒤 같은 βz를 더하면 same에서 Brier 차이가 -0.00414 [-0.00547, -0.00281]다. 추가 이득을 지지한다.
- Temperature를 먼저 적용한 뒤 같은 βz를 더하면 alternating에서 Brier 차이가 -0.00357 [-0.00486, -0.00228]다. 추가 이득을 지지한다.
- 개별 directional family × schedule 8개 중 두 score-only 기준을 모두 앞선 것은 8개다. Temperature 후 βz의 추가 이득은 7개에서 simultaneous CI 상한이 0보다 작았다.
- Isotropic control에서 temperature 후 βz를 더한 pooled 차이는 same에서 +0.00806 [+0.00579, +0.01033]였다. 방향이 없는 target은 correction이 손해를 일으킬 수 있는 실험적 반례다.
- Isotropic control에서 temperature 후 βz를 더한 pooled 차이는 alternating에서 +0.00803 [+0.00387, +0.01220]였다. 방향이 없는 target은 correction이 손해를 일으킬 수 있는 실험적 반례다.

다섯 방법은 같은 sensor action을 쓰므로 이 결과가 비교하는 양은 관측 뒤 확률의 품질이다. Score-only baseline에 이겨도 추정한 sign이 단일 물리적 원인임을 식별하지는 못한다. Surface의 다른 속성과 reader 출력이 함께 달라질 수 있다. 원 score의 전반적인 보정과 환경 간 판별력이 각각 얼마나 변했는지는 독립 reliability·resolution 분석과 within-environment sign intervention으로 평가할 수 있다. 이 부록의 두 대조군에는 추가 GPU timing을 측정하지 않았다. [Brier decomposition](https://proceedings.neurips.cc/paper_files/paper/2019/file/8558cb408c1d76621371888657d2eb1d-Paper.pdf) · [physics residual의 선행 검증](https://journals.sagepub.com/doi/full/10.1177/0278364920954896)
