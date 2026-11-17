# Source에서 선택한 soft projection이 held-out contour로 전이되는가?

**부분적으로 성분을 남기면 hard projection의 손해는 줄었지만, 처음 보는 두 family에서는 전체 profile을 유지하는 편이 더 정확했다.** Source의 두 family에서 다섯 후보 중 최저 Brier를 준 강도는 α=0.75였다. 이 값을 고정해 terrace와 dimple에 적용하자 항상 전체 profile을 유지했을 때보다 mean Brier가 **0.006473** 증가했다. [Source 선택 기록](soft_choice.json) · [Target 결과](soft_transfer.json)

## 비교를 공정하게 한 방법

예를 들어 α=0.75는 estimated deformation axis 방향 residual의 25%를 남긴다. Gaussian log likelihood에서 그 방향의 squared residual에 주는 가중치는 0.0625다. α=0에서는 원래 full likelihood를, α=1에서는 기존 hard projection을 복원한다. 이 비교에서는 measurement, 구매한 관측, axes, hypothesis bank, camera로 얻은 위치와 그 위치의 routing 결정이 모두 같다. 기존 위치 규칙이 projection을 선택한 경우에만 강도를 적용한다. Paid probe branch에는 profile 항이 없어 모든 강도의 forecast가 동일하다. α<1일 때 residual 변환은 가역적이므로 이 대조는 고정된 likelihood의 가중치 변화가 미치는 영향을 측정한다. [분석 사전 명세](soft_plan.md) · [구현](soft_likelihood.py)

V45의 fold/ripple 12개 source catalog에서 α를 고르고 결과·코드 hash를 기록한 뒤 v46의 terrace/dimple 12개 catalog에서 점수를 구했다. 각 split은 4개 환경 × 128개 draw × 12 catalog를 기반으로 하고 모든 구성, 네 번의 paid measurement noise, 세 contact offset, 두 finite pressure area를 동일 가중했다. 반복·조합을 독립 catalog로 세지 않았다. V46은 과거에 hard와 full 결과를 이미 분석한 archive이므로 새 family를 처음 수집한 확인 실험으로 해석할 수 있는 범위는 없다. [Source 기록](soft_choice.json) · [실행자 파일](soft_likelihood.py)

| α | Source fold/ripple Brier | Target terrace/dimple Brier | Terrace Brier | Dimple Brier |
|---:|---:|---:|---:|---:|
| 0 | 0.150023 | **0.155170** | **0.155592** | **0.154749** |
| 0.25 | 0.149771 | 0.156743 | 0.158270 | 0.155217 |
| 0.50 | 0.149606 | 0.159107 | 0.162136 | 0.156079 |
| **0.75** | **0.149598** | 0.161644 | 0.166205 | 0.157082 |
| 1.00 | 0.149655 | 0.162776 | 0.168009 | 0.157544 |

12개 target catalog를 단위로 Bonferroni 보정한 두 paired 95% interval을 계산했다. α=0.75 minus full은 **+0.00647324 [0.00598495, 0.00696153]**, α=0.75 minus hard는 **−0.00113262 [−0.00121647, −0.00104876]**다. Source에서 α=0.5와 0.75 차이는 0.0000074에 불과해 최적 강도를 정밀하게 찾았다고 할 수 없다. 두 target family 각각에서 기술적 평균은 강도에 따라 순서대로 증가했다. 이 다섯 점으로 그 사이의 모든 강도나 다른 미관측 family에 대한 단조성을 보장하지 않는다. [Catalog별 결과](soft_transfer.json)

## 검산과 논문에서 쓸 수 있는 주장

Allocated GPU가 비었는지 process와 memory를 확인한 뒤 Slurm 2744468의 RTX PRO 6000 Blackwell에서 source와 target을 포함해 50.2초 동안 계산했다. 이 시간은 실험 전체 wall time이다. 두 endpoint의 예측은 v45·v46 archive와 최대 절대오차 1.47×10⁻¹³ 이내로 일치했다. 기존의 별개 scoring 코드로 계산한 각 catalog의 Brier와 새 집계의 차이도 1.12×10⁻¹⁶ 미만이었다. [독립 검사기](soft_audit.py) · [검산 기록](soft_audit.json)

논문에는 **동일한 고정 위치 정책과 physics likelihood 아래에서 source가 선호한 partial projection도 두 held-out shape family에 전이되지 않았고, hard projection보다 손해가 작았다**고 쓸 수 있다. 고정된 likelihood의 오차에는 표현의 predictive information과 downstream model의 적합성 양쪽이 관여한다. 최적의 추론기가 잃는 Brier 정보량을 정량화하는 데는 별도의 근거가 필요하다. 이 수치는 neural augmentation이나 target calibration을 수행한 결과로도 해석할 수 없다.
