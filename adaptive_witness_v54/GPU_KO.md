# v54 · 네 번째 probe 선택의 시간과 물리 비용

## 어떤 시간과 관측을 비교했는가

세 정책은 교체 전 4쌍과 교체 후 첫 3개 weak/reference 쌍을 공유한다. Fixed 4R은 네 번째 weak/reference, fixed 3R+I는 네 번째 weak/impact를 구입한다. Adaptive는 교체 후 첫 3쌍의 여섯 displacement만으로 네 번째 probe를 선택한다. 네 번째 물체의 weak measurement도 이 시점에는 주어지지 않는다. Terminal continuous joint는 선택이 끝난 뒤 교체 전 네 쌍, 교체 후 첫 세 쌍, 네 번째 물체의 weak와 실제 구입한 한 관측을 받는다. 한 batch에서 reference와 impact를 선택한 session을 나눠 각 posterior를 구한 후, 네 session의 512개 배포 물체에 같은 GPU physics decoder를 적용했다.

NVIDIA GPU가 하나만 보이는 Slurm allocation에서 12개 catalog 각각 네 session을 구성했다. 각 session에는 128개 물체가 있고 한 정책당 catalog별 warmup 1회 후 6회, 총 72개 timed batch를 썼다. 세 정책의 6개 순열을 모두 사용해 순서와 직전에 실행한 정책을 균형화했다. 모든 정책에서 GPU가 준비될 때까지 동기화하고 시작하며, 선택·terminal estimator·기존 factual acquisition·decoder·host 출력까지 실행한 다음 다시 동기화했다. 재현 코드가 보관된 정책별 512개 예측과 일치하는지 확인했다. 고른 관측의 반대편 reference를 NaN, impact 방향을 범위 밖 값으로 대체하는 검증을 실제 route와 의도적으로 혼합한 route 모두에 실시했다.

GPU p90은 이런 synchronized wall time의 90번째 백분위수다. 실제 camera exposure, 충돌, 표면 교체, 로봇 motion, 원시 image 건너오기는 시간 범위 밖이다. 48개 session timing fixture는 latency 측정 단위다. 배포 수익이나 probe 사용률은 모든 held-out session에서 따로 계산한다. GPU p90 차이는 12개 catalog를 쌍을 유지해 16,000회 재표집했다. Adaptive와 두 fixed 정책을 비교하는 interval은 두 비교를 합친 95% Bonferroni interval이다.

## 같은 sensing·reset 예산의 의미

| 교체 후 정책 | Weak / Reference / Impact contacts | 전체 contacts / resets | Session당 wall time 가정 |
|---|---|---|---|
| Fixed 4R | 4 / 4 / 0 | 8 / 8 | 5.92 s |
| Fixed 3R+I | 4 / 3 / 1 | 8 / 8 | 4.88 s + T |
| Adaptive, impact 선택률 p | 4 / (4−p) / p, session 평균 | 8 / 8, 모든 session | 5.92 s + p(T−1.04 s), session 평균 |

T는 impact 시행, 방향 판독, reset 및 안정화까지의 wall time이다. Weak와 reference에 각각 0.44초, 1.04초를 둔 v52·v53 비용 모델을 사용했다. T가 1.04초이면 세 정책 모두 교체 후 5.92초, 128개 배포 물체당 46.25 ms를 사용한다. T가 짧으면 동일 slot을 예약해 남은 시간을 기다릴 수 있고, T가 길면 세 방법의 contact·reset 수만 일치한다. T가 2.08초인 예에서 adaptive의 추가 probe 시간은 fixed 4R보다 8.125p ms/object 더 길며, fixed 3R+I보다 8.125(1−p) ms/object 짧다. 교체 전의 공통 8 contacts·8 resets는 이미 지급했고, 배포 물체마다 별도로 weak/contact 한 번과 reset 한 번이 필요하다. 이 예산 모델은 impact에 따른 오염·마모나 희생 물체의 재조달 비용을 수치로 추정하지 않았다.

Cross-fitted six-bin rule의 held-out impact 구입률은 62.37%다. T=2.08초이면 교체 후 session 평균은 6.569초다. 한 물체당 fixed 4R보다 5.068 ms 더 길고, fixed 3R+I보다는 3.057 ms 짧다. 같은 조건의 fixed 4R 대비 평균 Brier gain 0.0000242을 쓰면, 기대 비용함수가 Brier + λ × seconds/object인 경우 break-even λ는 0.00478 Brier-equivalent/s다. 추가 시간을 이보다 높게 평가하면 해당 gate의 평균 이득이 상쇄된다. 실제 장치 비용과 조직의 시간 가치를 모르는 상태에서의 시나리오다. [교차 학습 결과](sign_bins54.json)

## 빈 GPU에서 측정한 주 비교

Slurm 2744471(node56)의 NVIDIA RTX PRO 6000 Blackwell Server Edition이 사용률 0%, 0 MiB, 실행 중인 compute process 0개일 때 작업을 시작했다. 점검 당시 36–40 GB 사용 중이던 별도 할당 두 개는 사용하지 않았다. 주측정에서 최대 절대 prediction replay error는 4.45×10⁻¹⁶ 미만이었고 측정 중 다른 GPU process는 관측되지 않았다.

| 정책 | GPU p90 (ms / 512 objects) | 네 session의 host 선택+terminal 계산 p90 (ms) | Adaptive에서 fixed를 뺀 GPU p90 차이, simultaneous 95% CI (ms) |
|---|---:|---:|---|
| Nominal EVSI | 466.815 | 0.243 | 기준 |
| Fixed 4R | 466.831 | 0.160 | −0.016 [−0.463, +0.492] |
| Fixed 3R+I | 466.776 | 0.160 | +0.039 [−1.207, +0.932] |

Timing fixture에서 nominal gate가 impact를 고른 session은 48개 중 0개였고, 이 gate 계산만 따로 재면 네 session에서 p90 0.0882 ms였다. Terminal·decode까지 포함한 주 비교에서는 두 GPU latency 차이 모두 0과 양립했다. [GPU 실측과 bootstrap](timing54.json) · [실행 코드](bench54.py)

주측정 뒤 추가 sensitivity를 위해 policy54.py에 robust mode를 붙였다. 주측정 때 적재된 [원본 정책](timing_nominal_policy54.py)의 SHA-256은 timing_freeze54.sha256의 policy54.py와 일치한다. 주측정을 그대로 재실행하려면 별도 checkout에서 이를 policy54.py로 복원한 후 run_bench54.sh를 실행한다. 새 정책에서는 기본 nominal routing 수식과 입력이 동일하다.

## Cross-fitted sign-bin gate의 별도 crossover

6개 sign pattern의 규칙표를 해당 catalog를 제외한 fold에서 학습하고, 그 규칙표만 배포 시점에 미리 적재했다. 실제 timed call에서는 매번 첫 세 쌍에서 readable 개수와 sign 일치 개수를 계산하고 해당 session의 네 번째 sensor를 고른다. 별도 동일 GPU에서 각 catalog마다 네 정책의 4×4 Williams crossover를 수행했다. 48개 batch/정책에서 sign-bin, nominal EVSI, fixed 4R, fixed 3R+I의 GPU p90은 각각 **464.69, 465.15, 465.04, 465.02 ms/512 objects**였다. Sign-bin에서 EVSI, fixed 4R, fixed 3R+I를 각각 뺀 차이는 −0.46 [−2.07, +0.64], −0.35 [−3.64, +0.20], −0.32 [−1.98, +0.19] ms였다. 세 동시 95% interval이 모두 0을 포함한다. 네 session의 gate-only host p90은 sign-bin 0.0261 ms, nominal EVSI 0.0869 ms였다. 같은 48개 timing session에서 실제 impact 선택률은 sign-bin 62.5%, EVSI 0%였다. 이 표의 GPU p90은 앞 절의 다른 시각에 측정한 절댓값과 직접 빼지 않는다. [별도 timing](bins_timing54.json) · [원시 시간과 구입 분기](bins_timing54.npz)

Robust gate는 held-out session의 35.29%에서 impact를 구입했다. T=2.08초인 시나리오에서 fixed 4R 대비 추가 비용은 2.868 ms/object, 평균 Brier gain은 0.0000251이다. 이 경우 break-even λ는 0.00876 Brier-equivalent/s다. [전체 deployment 집계](score54.json)

## Robust gate의 별도 crossover

Source에서 acquisition을 평가할 때의 reference error prior를 넓게 두고, 실제 예측에는 v53의 nominal continuous joint를 그대로 쓴 robust gate를 측정했다. 같은 GPU·fixture에서 4-way crossover, 정책당 48개 batch의 GPU p90은 robust **466.20**, sign-bin **465.83**, fixed 4R **465.68**, fixed 3R+I **465.91 ms/512 objects**였다. Robust에서 각 대조를 뺀 p90 차이는 sign-bin +0.37 [−0.66, +2.35], fixed 4R +0.52 [−0.33, +2.03], fixed 3R+I +0.29 [−0.87, +2.00] ms였다. 3개 simultaneous 95% interval 모두 0을 포함했다. 네 session의 robust gate-only host p90은 0.1194 ms였다. Timing용 48개 session의 impact 구입률은 27.08%였고, 모든 held-out 조건을 합친 구입률은 35.29%였다. 세 timing run에서 guard가 기록한 다른 GPU process는 없었고 512-object prediction의 최대 재생 오차는 모두 4.45×10⁻¹⁶ 미만이었다. [Robust paired timing](robust_timing54.json) · [원시 session 및 GPU 기록](robust_timing54.npz)
