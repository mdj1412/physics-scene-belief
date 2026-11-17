# v65 · 실물 pilot에서 구매 전환을 관찰하는 데 필요한 표면과 session

**장비 검증은 12개 표면 × 4개 session으로 시작할 수 있다. 이 크기에서 전환이 0건일 확률은 기준 시나리오에서 92.0%다.** 한 건을 볼 가능성 자체를 95%까지 높이려면 표면당 8개 유효 session, surface 내 ICC 0.05 가정에서 236개 독립 표면이 필요하다. [사전 계획](POWER_PLAN_KO.md) · [서버의 전체 수치](power65.json)

## 하나의 관측과 p의 의미

같은 표면의 새 **8-impact calibration pilot 1개와 독립적으로 사전 지정한 deployment query 1개**를 한 paired session으로 센다. 두 조명에서 같은 여덟 접촉을 각각 판독하고 두 frozen policy를 같은 deployment prefix에 적용한다. 두 policy가 최종적으로 서로 다른 네 번째 관측(reference 또는 impact)을 선택하면 1건이다. 한 pilot 안에 8번의 impact와 8번의 reference, 4개의 내부 접촉 묶음이 있어도 구매 전환을 셀 기회는 한 번이다. 같은 calibration bank를 여러 query에 재사용하면 추가적인 bank-level cluster가 생기므로 여기 표의 session 수로 단순 환산할 수 없다.

v64 raw optical code에서 각 lamp의 상위 gate 선택을 다시 계산했다. 두 lamp의 상위 선택이 달라지고 같은 deployment query에서 v54가 reference를 살 때에만 실제 구매가 달라진다. 이 확률을 24개 catalog, 2개 surface instance, 8개 board, 두 angular width, 잠재 환경·query 배정에 대해 평균하면 0.0018629868825가 재현된다. 이 값은 98,304 impacts에서 실제 사건 수를 세어 얻은 발생률이 아니다. 아래 세 planning scenario는 simulation 평균과 v64의 catalog×board bootstrap 95% CI 양끝이다. 실물의 p가 이 범위에 들어간다는 근거는 없으며 p=0이면 아무리 많이 측정해도 한 건의 진짜 전환을 관찰할 수 없다. 48개 모사 surface의 conditional 평균도 0–0.633%였고 15개는 0이었다. 유한한 simulation 지원 범위에서 얻은 값으로 field ICC를 추정하지 않았다. [원래 v64 집계](../reflective_field_v64/sim_score64.py) · [v65 원자료 검산](power65.py)

## 세 가지 표본 수 목표

서로 다른 surface는 독립인 물리 표본이다. 같은 물리 표면을 다른 날에 다시 사용한 것은 같은 surface의 반복으로 묶는다. 공정, 원재료 batch, board, operator, day가 표면 사이에 공통 효과를 주면 별도 blocking 또는 variance 추정이 필요하다. 같은 surface 안의 paired session이 exchangeable이고 pₛ를 조건으로 독립이라 두었다. pₛ ~ Beta(pκ,(1−p)κ), κ=1/ICC−1이면 각 surface의 전환 수 Xₛ는 beta-binomial이다. ICC=0은 독립 binomial의 극한이다. S개의 surface, surface당 m회일 때 전체 전환 수는 Xₛ의 합이다. 한 번이라도 전환이 있는 서로 다른 surface 수는 정확히 Binomial(S,1−Pr[Xₛ=0])이다. 희귀사건의 분포 꼬리 확률은 ICC 하나로 정해지지 않으므로 수치에는 beta 혼합분포 가정도 필요하다. [계산·경계 검산](power65.py) · [Cluster 설계의 배경](https://academic.oup.com/aje/article-abstract/114/6/906/124295)

| 목표 | 0.0491% 가정 | 0.1863% 가정 | 0.3525% 가정 | 기준 시나리오의 calibration impact |
|---|---:|---:|---:|---:|
| 95% 확률로 전환 ≥1건 | 895 × 8 | **236 × 8 = 1,888** | 125 × 8 | **15,104** |
| 90% 확률로 전환 ≥10건 | 3,839 × 8 | **1,011 × 8 = 8,088** | 535 × 8 | **64,704** |
| 90% 확률로 전환 표면 ≥10개 | 4,245 × 8 | **1,121 × 8 = 8,968** | 594 × 8 | **71,744** |

각 cell은 독립 surface 수 × 유효 paired session 수/표면이다. 두 번째 목표는 한 표면의 전환을 여러 번 셀 수 있다. 세 번째 목표는 현상이 여러 표면에서 재현되는 사례를 얻으려는 목적이다. 두 목표는 사용자에게 비교 근거를 제공하려고 고른 설계 예시다. 표의 비용에 더해 calibration reference가 impact와 같은 수만큼 필요하고, deployment에서 세 reference와 선택한 관측·후속 label을 모으는 비용은 별도다. 물리적 조명 두 방향이 같은 impact를 공유하면 impact 수를 두 배로 세지 않는다.

| 기준 p에서의 설계 | ≥1건, 95%에 필요한 표면 | ≥10건, 90% | ≥10개 전환 표면, 90% |
|---|---:|---:|---:|
| m=1, ICC 무관 | 1,607 | 7,623 | 7,623 |
| m=8, ICC=0 | 201 | 953 | 957 |
| m=8, ICC=0.01 | 208 | 965 | 990 |
| m=8, ICC=0.05 | 236 | 1,011 | 1,121 |
| m=8, ICC=0.10 | 270 | 1,065 | 1,282 |
| m=4, ICC=0.05 | 433 | 1,958 | 2,054 |
| m=16, ICC=0.05 | 136 | 536 | 649 |
| m=32, ICC=0.05 | 85 | 295 | 404 |

같은 384개 paired session을 쓴다면 기준 시나리오에서 한 건 이상 볼 확률은 24표면×16회 41.1%, 48×8회 45.7%, 96×4회 48.6%다. 희귀한 고위험 표면 한 곳에 반복이 몰려 10건을 채울 수도 있어 전환 수와 전환 표면 수를 함께 보고한다. [모든 설계](power65.json)

## 정밀도와 0건의 해석

**한 건을 봤다는 것과 발생률을 정밀하게 추정했다는 것은 다르다.** 기준 p에서 독립 paired session을 가정해 95% Clopper–Pearson interval의 *총 폭의 절반*이 p의 절반 이하일 확률을 90% 이상으로 요구하면, 100회 단위 grid에서 처음 충족한 수는 11,800회다. 기대 전환 수 22.0건, 실제 계산한 assurance는 91.37%다. 폭의 절반을 p의 1/4로 줄이면 40,200회, 기대 전환 수 74.9건, assurance는 90.80%다. 이는 양쪽 끝이 점추정치와 정확히 대칭이라는 뜻이 아니다. Cluster가 있는 분석에서 같은 iid CI를 적용하면 적용 범위 밖이다. [원전](https://doi.org/10.1093/biomet/26.4.404) · [확률 계산](power65.py)

Cluster 설계에서 평균 발생률의 상대 표준오차는 RSE²=(1−p)[1+(m−1)ICC]/(Smp)다. 기준 p, m=8, ICC=.05일 때 RSE≤25%는 **1,447표면×8=11,576회**, RSE≤15%는 **4,019표면×8=32,152회**가 필요하다. RSE는 CI 폭이 아니라 sampling standard deviation과 p의 비율이다. 표면간 이질성이 크거나 day 단위 공통 오류가 있으면 표를 재계산해야 한다. [Cluster variance](https://www.bmj.com/content/316/7142/1455.1.full)

장비 검증용 **12×4=48회**의 기대 전환 수는 0.089건이고 0건일 확률은 92.02%다. 0건 후 one-sided 95% upper bound는 지정한 beta-binomial/ICC=.05 모형에서 **6.51%**다. 독립 이항의 upper bound나 이 모형에서 계산한 upper bound를 실물의 모형 불문 보장으로 취급하지 않는다. 작은 pilot에서는 sync 성공률·독립 marker identity 판독률·paired data 완결률이 주요 판단 기준이다. 0건을 조명 효과가 없다는 결론이나 장비 검증 실패로 쓰지 않는다.

**누락 비용.** 표면당 8회를 예약하되 각 paired session이 독립적으로 90% 확률로만 완결된다고 가정했다. 실제 전환 사건이 발생해도 결측이면 관찰되지 않은 것으로 두어 surface count distribution을 정확히 thinning했다. 기준 p·ICC=.05에서 세 목표를 맞출 모집 수는 각각 **258표면×8회, 1,117×8회, 1,227×8회**다. 예컨대 첫 목표는 2,064회 예약, 16,512회 calibration impact다. 추적이 틀릴수록 독립 strain도 읽기 어려워지면 이 보정은 충분하지 않다. 판독 보류는 정상적인 정책 출력으로 유지하고 실제 탈락(공통 timebase·행동 입력 또는 기준 검증 실패)과 구분한다. 구매 전환이 검증 신호 없이도 재생 가능하면 별도 분모로 보존해야 한다. [Thinning 계산](power65.py)

## 검산과 의사결정

Datacenter3의 Slurm compute node에서 CUDA를 숨기고 실행했다. 3가지 p × 5가지 m × 4가지 ICC의 60개 조건을 계산했다. 한 건은 zero-count probability, ≥10건은 surface count 생성함수의 0–9차 항을 정확하게 합성해 계산했다. 두 확률 목표 각각 최소 정수 S와 바로 전 S−1의 통과/미통과를 확인했다. ICC=0에서 iid binomial 결과를 재현했고, 5가지 크기·ICC에서 각각 100,000개 panel을 multinomial로 모사해 사건·전환 표면 확률을 검산했다. 전 schedule에서 기대 전환 수는 Smp다. [실행 기록](power65.log) · [분포와 seed](power65.json)

**논문의 Brier 개선 주장을 검정하기 위한 표본 수는 별도다.** 구매 전환 빈도만으로는 전환된 사례의 loss 차이의 부호·크기·변동을 정할 수 없다. 같은 lamp 간 구매 전환과 같은 lamp 안에서의 confidence–random 또는 confidence–gate disagreement도 서로 다른 사건이다. Field pilot에서 각 주요 비교의 paired loss, 정책이 갈린 빈도와 surface 간 분산을 먼저 기록한 뒤, 실용적 최소 Brier 차이를 사전 지정하고 confirmatory surface를 별도로 확보해야 한다.
