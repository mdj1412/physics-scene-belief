# v57 · 같은 marginal mismatch에서 grouping이 주는 정보와 손익

8×1의 완전한 pilot likelihood는 A와 B에서 같다. 4×2에서는 둘 다 판독 가능한 pair의 불일치 횟수 분포가 달라지고, 분류 성능이 올라간다. 이 차이가 실제 sensing policy의 선택에서 이득이 되는 정도는 **어느 환경에서 잘못 전환했는지**에 달렸다.

## 사전 지정과 정보 경계

A에서는 session별 e가 0 또는 0.5이며 비율은 42%, 58%다. B에서는 0.25 또는 0.5이며 비율은 84%, 16%다. 환경 prior는 반반이고, 실제 타입은 policy에서 숨긴다. 두 분포의 평균 e는 0.29이다. 환경 A이면 fixed 3R+I, B이면 v54 source-risk gate를 쓴다는 mapping은 v56의 결과로 고정했다. V57에서는 모든 48개 material에 대해 A에서 fixed I가, B에서 gate가 더 낮은 평균 Brier를 가졌다. [사전 평가 계획](SCORE_PLAN_KO.md) · [동결한 switch](policy57.py)

4×2에서 한 session의 유효한 부호 판독 수가 0 또는 1이면 A:B likelihood ratio는 1이다. 유효 판독이 2개일 때 불일치 횟수 0, 1, 2의 확률은 A에서 (0.565, 0.290, 0.145), B에서 (0.5125, 0.395, 0.0925)다. 네 session의 ratio를 곱한다. MAP는 odds가 1보다 클 때 fixed I를, 아니면 gate를 쓴다. Secondary loss-weighted rule은 같은 odds를 v56의 B에서 잘못 전환할 때의 손실 / A에서 올바르게 전환할 때의 이득 비율인 2.0161과 비교한다. 실제 구현은 보관된 full-precision 비율을 쓴다. 8×1에서는 어떤 판독이든 두 규칙 모두 gate를 고른다. 무작위 tie를 쓰더라도 balanced accuracy의 기댓값은 50%지만, 선택 비용이 달라질 수 있으므로 v56 mean loss가 더 낮은 gate를 tie action으로 고정했다. [선택 수식](policy57.py) · [v56 손익](../mismatch_switch_v56/identification56.json)

## 실제 분류와 held-out 결과

각 catalog 안의 두 material·두 rate law와 A/B를 맞춰 평가했다. 표면·환경별 4,096개 pilot replicate를 평균했으며 24개 fresh catalog를 독립 추론 단위로 삼았다. Mean held-out Brier는 독립 deployment 물체의 direction 전이 확률을 적분했다. Session p90은 같은 128개 binary deployment draw를 공유하는 두 행동 중 pilot이 산 행동의 session 평균 loss CDF에서 직접 구했다. Weighted sort로 p90 값을 독립 재계산해 일치하는지 확인했다. 동일한 pilot이 여러 배포 물체와 같은 material의 두 law에 공유되며, p90은 개별 deployment session 평균의 90번째 백분위수다. [서버 채점](score57.py) · [수치](score57.json)

| Pilot과 선택 규칙 | A에서 fixed 선택 | B에서 fixed 선택 | Balanced accuracy | 실제 deployment I 구매율 | 평균 Brier ↓ | Session p90 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| 8×1 MAP · 항상 v54 gate와 동일 | 0% | 0% | 50.000% | 35.113% | 0.158347117 | 0.190866411 |
| 4×2 MAP | 44.913% | 36.571% | **54.171%** | 61.549% | 0.158351504 | 0.190864995 |
| 4×2 loss-weighted | 0.651% | 0.240% | 50.205% | 35.402% | **0.158347068** | 0.190866411 |
| 항상 fixed 3R+I | 100% | 100% | 50.000% | 100% | 0.158364159 | 0.190873131 |
| True regime · privileged | 100% | 0% | 100% | 67.557% | 0.158325079 | 0.190833300 |

4×2 MAP의 A recall은 44.91%, B recall은 63.43%다. 8×1 대비 balanced accuracy가 **4.171 percentage points** 상승했다(95% CI 4.075–4.267). 같은 부호 쌍 중 판독 가능한 개수 자체가 A/B의 차이를 만들지는 않았다. 4×2에서 두 물체의 부호를 모두 읽은 session은 pilot당 평균 0.997개였고, 31.63%에서는 하나도 없었다. 두 설계가 식별 가능한 정보를 다르게 제공한다는 가설은 지지된다. [Catalog·pilot 단위](score57_cells.npz)

| 4×2 정책에서 대조를 뺀 차이 (Brier ×10⁻⁶) | 평균 [동시 95% CI] | Session p90 [동시 95% CI] |
|---|---:|---:|
| MAP − 8×1 · v54 gate | **+4.387** [+1.748, +6.984] | −1.416 [−19.215, +12.473] |
| MAP − fixed I | −12.655 [−16.710, −8.507] | −8.136 [−26.182, +18.605] |
| Loss-weighted − 8×1 · v54 gate | **−0.049** [−0.075, −0.024] | 0.000 [−1.207, 0.000] |
| Loss-weighted − fixed I | −17.091 [−23.760, −10.319] | −6.720 [−37.507, +36.375] |

각 rule의 두 mean 대비와 두 p90 대비를 각각 별도 family로 보정했다. Primary MAP의 평균은 24개 catalog 중 gate보다 6개에서만, fixed I보다 23개에서 우수했다. Secondary loss-weighted policy는 각각 19개, 21개에서 우수했다. 후자가 두 constant보다 mean이 낮다는 점은 predeclared secondary 검정 안에서 확인됐다. 더 강한 constant인 gate 대비 차이는 0.000000049로, 참 regime을 알 때 남아 있는 개선 가능량의 **0.223%**였다. 둘 다 p90에서 두 대조에 대한 우위를 확인하지 못했다. 0을 포함하거나 끝점이 0인 interval에서 동등성도 입증되지 않는다.

## 왜 더 잘 구별한 방법이 더 큰 손실을 냈는가

A에서 gate를 잘못 유지하면 fixed I 대비 mean 손실은 44.08×10⁻⁶이다. B에서 fixed I로 잘못 전환하면 gate 대비 손실은 78.16×10⁻⁶이다. 같은 prior 아래 MAP의 올바른 A 전환이 절약한 Brier는 전체 평균에 +9.897×10⁻⁶ 기여했지만, 잘못된 B 전환이 14.284×10⁻⁶을 소모했다. Loss-weighted rule의 해당 수치는 0.143, 0.094×10⁻⁶이었다. 이 규칙은 두 물체를 모두 읽은 session이 적어도 두 개였을 때만 fixed I로 전환했다. 같은 횟수의 measurements로 classification과 decision에서 다른 결과가 나오는 메커니즘을 직접 합산했다. [검산 결과](opportunity57.json) · [계산 코드](opportunity57.py)

Grouping에 해당하는 신호를 검사하기 위해, 같은 material·regime의 독립 pilot끼리 두 번째 물체의 네 판독을 순환 치환했다. 부호의 전체 주변 분포를 보존하면서 매칭을 끊자 MAP의 balanced accuracy가 50.005%로 감소했다. 원래 4×2 MAP의 평균은 이 깨진 pairing 대조보다 1.630×10⁻⁶ [1.333, 1.923] 낮았다. 별도 대조로 outer fixed-I 전환율이 정확히 같은 독립 random을 적분하자 원래 MAP가 2.556×10⁻⁶ [2.410, 2.703] 낮았다. 따라서 grouping의 incremental information은 유효하지만 항상 gate를 유지하는 행동까지 앞서지는 않는다. 두 대조의 CI는 탐색적 pointwise 구간이다.

## 독립 binary replay와 해상도

두 독립 deployment seed에서 MAP−gate의 mean 차이는 +5.231, +2.961×10⁻⁶으로 같은 방향이었다. Loss-weighted−gate는 −0.038, −0.062×10⁻⁶이었다. 첫 seed의 동시 95% CI [−0.082, +0.002]에는 0이 포함됐고 두 번째 [−0.107, −0.023]에는 포함되지 않았다. Ultra-resolution PyBullet에서 MAP−gate가 +5.228×10⁻⁶, loss-weighted−gate가 −0.041×10⁻⁶이었다. 같은 solver의 refinement와 replay를 확인한 결과이며, 실물 session의 손실 감소나 reset 시간 절약을 보장하지 않는다. [전체 결과](score57.json) · [Source freeze](score_plan57.sha256)
