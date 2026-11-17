# v58 · 같은 8회 impact로 선택한 정책의 held-out 결과

## 무엇을 판별했는가

새 표면에서 8개 sacrificial object의 부호를 두 센서로 읽고, 이후 별도 session에서 사용할 v54 gate 또는 fixed 3R+I 중 하나를 고른다. 환경 A는 fixed, B는 gate가 유리하다는 v56 예측은 48개 새 synthetic surface 모두에서 재현됐다. Gate에서 fixed를 뺀 평균 Brier 차이는 A에서 +42.29×10⁻⁶, B에서 −84.80×10⁻⁶이다. 해당 표면의 환경, session persistence, sensor artifact 상태 및 실제 target label을 정책에 주지 않았다. 두 환경 후보, sensor mechanism과 그 parameter 후보는 알려져 있다. [사전 명세](SCORE_PLAN_KO.md) · [고정 정책](route58.py)

몸체별 참 mismatch는 두 센서가 공유하고 판독 가능한 물체의 net sign error는 sensor마다 독립적으로 3.92%다. Sensor-local latch는 해당 sensor가 해당 session 전체에서 고정된 mismatch bit를 출력한다. 4×2 dual MAP는 각 환경의 rho×h 조합에서 관측된 네 pair의 joint likelihood를 곱한 다음, 전체 pilot에 걸쳐 동일하게 유지되는 rho와 h를 marginalize한다. A posterior가 0.5를 초과하면 fixed, 동률이면 gate다. 8×1에서는 각 singleton의 관측분포가 A와 B에서 같다. Same-object sensor disagreement로 h를 추론할 수 있어도 A를 식별할 수는 없다. [390,625개 possible pilot의 정확 열거](exact_route58.json)

Primary는 24개 독립 catalog, 각각 두 surface와 두 law, 128개 held-out object에 대해 네 rho×h와 A/B를 균등 가중한다. 모든 arm이 같은 frozen continuous joint terminal을 쓴다. Mean held-out Brier는 알려진 binary event 확률에 대한 conditional expectation이다. Session loss p90은 128개 물체의 실제 binary draw 평균을 outcome으로 갖는 weighted empirical distribution에서 계산했다. 같은 surface·regime당 2,048개 pilot replicate는 선택 불확실성을 적분한다. Uncertainty interval은 24개 catalog를 paired bootstrap 5,000회로 재추출했다. Aggregate primary의 두 비교는 endpoint별 동시 95% interval, 나머지는 pointwise 95% interval이다. [서버 채점 코드](score58.py) · [전체 결과](score58.json)

| Pilot·정책 | A에서 fixed | B에서 fixed | Balanced accuracy | 평균 held-out Brier ↓ | Session loss p90 ↓ |
|---|---:|---:|---:|---:|---:|
| Dual 4×2 MAP | 37.995% | 34.678% | **51.659%** | 0.1605397313 | 0.189883530 |
| Dual 8×1 MAP · 항상 v54 gate | 0% | 0% | 50.000% | 0.1605330708 | 0.189880252 |
| Single 4×2 naive MAP | 41.654% | 38.687% | 51.483% | 0.1605406649 | 0.189884558 |
| Single 4×2 nuisance-aware MAP | 41.654% | 38.687% | 51.483% | 0.1605406649 | 0.189884558 |
| Dual 4×2 source-risk | 0.00992% | 0.00381% | 50.003% | 0.1605330702 | 0.189880252 |
| 항상 fixed 3R+I | 100% | 100% | 50.000% | 0.1605543232 | 0.189892814 |
| True regime privileged | 100% | 0% | 100.000% | 0.1605119246 | 0.189847335 |

Dual MAP의 fixed 선택은 36.34%, 그 최종 deployment impact 구매율은 58.67%다. 두 값은 서로 다른 결정이다. Single-channel naive는 v57의 단일 sensor LLR를 그대로 사용했다. 같은 sensor 판독만으로 rho×h를 적분해도 이 네 group에서 최종 MAP의 threshold를 넘는 관측 집합이 우연히 같았다. 390,625개 유한 관측을 전수 비교해 확인했다. 다른 group 크기·prior·sensor에서는 달라질 수 있다. [정책 구현](route58.py) · [열거 검산](verify_route58.py)

| 4×2 dual MAP − 대조 | Accuracy 차이 (percentage points) | Mean Brier 차이 (×10⁻⁶) | Session p90 차이 (×10⁻⁶) |
|---|---:|---:|---:|
| 8×1 dual MAP | **+1.659** [+1.539, +1.789] | **+6.661** [+4.201, +9.318] | +3.278 [−6.736, +22.100] |
| 4×2 single naive | **+0.176** [+0.124, +0.223] | **−0.934** [−1.244, −0.646] | −1.028 [−3.874, +0.522] |

8×1보다 환경 판별은 나아졌지만 MAP는 잘못 fixed로 바꿨을 때의 손실을 똑같은 횟수의 올바른 전환으로 회수할 수 없었다. 반면 같은 4×2에서 sensor를 하나 더 읽으면 single보다 평균 Brier가 낮아졌다. Source-risk 정책은 전환 odds를 v56 비용 비율 2.016으로 고정했다. 전체 786,432개 4×2 pilot의 약 0.00687%에서만 fixed를 선택했다. Gate 대비 mean 차이는 −0.000568×10⁻⁶ [−0.00222,+0.00108]이며, p90 차이는 0이었다. 두 센서를 추가한 adaptive policy가 두 기본 정책을 모두 능가한다는 근거는 없다. [Catalog별 결과](score58_cells.npz)

## 어떤 환경에서 grouping이 도움이 되는가

| 물리적 persistence | Sensor-local latch | Dual 4×2 정확도 | 이론상 dual 4×2 MAP 정확도 | Dual 4×2 − 8×1 mean (×10⁻⁶) | Dual 4×2 − single 4×2 mean (×10⁻⁶) |
|---|---|---:|---:|---:|---:|
| 없음 | 없음 | 50.020% | 50.000% | +6.789 | −0.538 |
| 없음 | 있음 | 49.980% | 50.000% | +7.678 | −1.234 |
| 있음 | 없음 | 53.589% | 53.685% | +5.555 | −0.567 |
| 있음 | 있음 | 53.047% | 53.111% | +6.620 | −1.395 |

Physical persistence가 없으면 두 regime의 관측분포가 같아서 A/B balanced accuracy의 기댓값은 모든 selector에서 50%다. 이때도 fixed 구매 빈도가 달라지면 평균 손실은 달라질 수 있다. Physical persistence가 있으면 같은 session 안의 두 관측에 A/B별 차이가 남는다. 네 조건을 합친 exact Bayes classification ceiling은 51.699%였고 실제 51.659%가 이에 가까웠다. 이 상한은 같은 binary 관측, 같은 candidate prior와 균등 0–1 판정 비용에서 얻었다. 여러 surface를 합쳐 메커니즘을 분석한 통계적 정밀도와 개별 surface에서 pilot 여덟 번만 보고 선택하는 정확도는 구분해야 한다. [정확 열거](exact_route58.json) · [Factorial 평가](score58.json)

두 independent deployment replay에서 dual 4×2 − 8×1의 실제 binary mean 차이는 +7.939, +5.228×10⁻⁶이었고, dual 4×2 − single 4×2는 −1.069, −0.771×10⁻⁶이었다. 같은 PyBullet solver의 fine에서 ultra로 해상도를 올렸을 때도 차이는 각각 +8.012, −1.083×10⁻⁶이었다. 실물에서는 pair별 sensor error의 조건부 독립, 같은 surface state의 유지, 두 번째 sensor의 추가 계측시간을 따로 확인해야 한다. [전체 수치](score58.json) · [사전 등록 checksum](score_freeze58.sha256)
