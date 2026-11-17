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
