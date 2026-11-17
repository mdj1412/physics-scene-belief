# v63 · 같은 수의 optical 판독을 남긴 정책의 downstream 결과

24개 catalog와 8개 board를 독립 복원추출했다. 수치의 단위는 명시한 곳을 제외하면 확률과 squared loss다. 비교의 기준은 confidence이며 음의 loss 차이일 때 confidence가 유리하다. 네 error cell은 같은 가중치다. [사전 분석 계획](SCORE_PLAN_KO.md) · [구현](score63.py)

| 정책 | Mean held-out Brier ↓ | Session loss p90 ↓ | P/C balanced accuracy | 저손실 정책 balanced accuracy | Fixed 선택률 |
|---|---:|---:|---:|---:|---:|
| Confidence | 0.155701576491 | 0.185992479 | 50.366% | 50.059% | 0.6592% |
| 같은 수 무작위 보류 | 0.155701575971 | 0.185992658 | 50.353% | 50.057% | 0.6517% |
| 항상 v54 gate | 0.155701480487 | 0.185992479 | — | 50.000% | 0.0000% |
| 기존 v61 reset | 0.155701645711 | 0.185992658 | 50.597% | 50.073% | 1.0605% |
| 광학 전부 보류 | 0.155701505764 | 0.185992479 | 50.000% | 50.037% | 0.2441% |
| 항상 fixed 3R+I | 0.155721643973 | 0.186032757 | — | 50.000% | 100.0000% |

| Confidence − 대조 | Mean Brier 차이 ×10⁻⁶ [동시 95% CI] | Session loss p90 차이 ×10⁻⁶ [동시 95% CI] |
|---|---:|---:|
| 같은 수 무작위 보류 | 0.0005 [-0.0098, 0.0108] | -0.1788 [0.0000, 0.2086] |
| 항상 v54 gate | 0.0960 [0.0339, 0.1771] | 0.0000 [-0.6407, 1.2815] |

Session loss p90은 discrete empirical quantile이다. Confidence − random의 관측 p90 차이는 음수지만 paired bootstrap 복원표본에서는 차이 0에 질량이 집중되어 percentile CI의 하한이 0이었다. 두 p90 비교 모두 0을 포함해 우위를 판정하기 어렵다.

같은 pilot 원본 판독이 n개이고 confidence가 k개를 남기는 경우, 가능한 n choose k개 subset을 모두 열거했다. 각 subset에서 보류된 물체의 두 전자센서와 결측 여부는 보존하고 광학 부호만 제거했다. 각 subset의 posterior를 계산해 행동을 고른 뒤 해당 행동의 Brier를 균등 평균했다. 무작위를 반영한 session p90은 두 행동의 binary replay loss 분포를 선택 확률로 혼합한 뒤 산출했다. 모든 pilot이 실제 사용할 예측을 출력한다.

| Transient 빈도 | Coherent 빈도 | Confidence 판독률¹ | 기존 판독률¹ | 선택 여지² | Confidence − random 평균 Brier ×10⁻⁶ [동시 95% CI] | Confidence − gate 평균 Brier ×10⁻⁶ [동시 95% CI] |
|---:|---:|---:|---:|---:|---:|---:|
| 0% | 0% | 5.263% | 9.258% | 4.079% | 0.0013 [-0.0094, 0.0109] | 0.0613 [0.0144, 0.1319] |
| 35% | 0% | 5.069% | 9.371% | 4.154% | 0.0024 [-0.0089, 0.0151] | 0.0596 [0.0120, 0.1297] |
| 0% | 35% | 5.136% | 9.091% | 4.049% | -0.0041 [-0.0376, 0.0262] | 0.1357 [0.0275, 0.2802] |
| 35% | 35% | 4.940% | 9.206% | 4.051% | 0.0025 [-0.0261, 0.0286] | 0.1275 [0.0261, 0.2654] |

¹ Paid reference–impact slot 중 실제 변형이 활성인 쌍을 분모로 삼고, 두 부호가 모두 읽힌 쌍을 분자로 삼았다. ² 무작위로 고를 subset이 여러 개인 0 < k < n인 pilot의 비율이다.

전체 paid pair 대비 confidence와 random은 각각 같은 2.566%를 사용했다. 원래 v61은 4.644%였다. 전체에서 confidence와 random이 다른 행동을 고를 확률은 0.08642%, P/C 예측이 달라지는 확률은 0.7767%였다. 같은 수에서 선택 여지가 있는 pilot은 4.0833 [1.0817, 7.8623]%였고, 그 안에서 행동을 바꿀 확률은 2.1163 [1.3166, 3.4531]%였다. 이 부분집합의 confidence − random 평균 Brier ×10⁻⁶은 0.0127 [-0.2744, 0.2533]였다. 이 단락의 CI는 pointwise 95%다.

## Solver와 target 배치에 대한 민감도

| 비교 | Confidence − random ×10⁻⁶ [동시 95% CI] | Confidence − gate ×10⁻⁶ [동시 95% CI] |
|---|---:|---:|
| ultra | 0.0003 [-0.0128, 0.0120] | 0.1114 [0.0412, 0.2064] |
| binary1 | 0.0014 [-0.0083, 0.0127] | 0.0973 [0.0207, 0.2091] |
| binary2 | -0.0014 [-0.0146, 0.0104] | 0.1085 [0.0362, 0.2039] |
| matched | -0.0045 [-0.0225, 0.0088] | 0.0106 [-0.0503, 0.0730] |
| reversed | 0.0056 [-0.0062, 0.0206] | 0.1814 [0.0839, 0.3006] |

Fine solver에서 A 환경의 48/48개 표면에서 fixed가 우세했고, B 환경의 48/48개 표면에서 gate가 우세했다. 표면을 같은 비율로 평균하면 gate − fixed의 차이는 A 41.07 ×10⁻⁶, B -81.39 ×10⁻⁶였다. 실제 더 낮은 surface별 Brier와 행동을 비교해 policy balanced accuracy를 얻었다. Board별로 confidence 평균 Brier가 random보다 낮은 경우는 3/8개, gate보다 낮은 경우는 0/8개였다.

| Count-matched random 대비 confidence 효과가 어떻게 변했는가? | Brier 차이의 차이 ×10⁻⁶ [pointwise 95% CI] |
|---|---:|
| Transient 주효과 | 0.0038 [-0.0031, 0.0130] |
| Coherent 주효과 | -0.0026 [-0.0236, 0.0155] |
| 두 빈도의 interaction | 0.0055 [-0.0080, 0.0206] |

행동 선택 효과는 optical accuracy와 다르다. 한 판독을 바꾸더라도 odds가 source cutoff를 넘지 않으면 두 정책의 deployment 예측은 같을 수 있다. 또한 selective가 정확한 판독을 남겨도 source에서 정해진 행동이 새 표면에서 손해일 수 있다. Bootstrap CI는 이 simulator, 새 catalog와 board profile의 표본 변동을 반영한다. 실제 장치의 misspecification, shared solver의 bias 또는 도메인 전체의 불확실성을 포함하지 않는다. [전체 수치](score63.json) · [판독 수 검산](score_validation63.log) · [입력·출력 checksum](score_results63.sha256)
