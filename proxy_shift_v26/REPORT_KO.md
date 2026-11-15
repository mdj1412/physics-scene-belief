# v26 · 두 factual proxy의 약화와 반전에서 동결된 v25 gate의 실패 경계

**V25의 factual gate와 고정 K256이 모두 기존 prior를 사용하면, 어느 조건에서도 gate의 선행 실패를 검출하지 못했다.** 36개 조건 중 relative failure는 0개였다. 대신 두 방법의 held-out Brier loss가 함께 상승했다. Profile 관계만 바꾼 실험에서는 αc=0.75(중심값을 12.5% 확률로 뒤집음)부터, velocity 관계만 바꾼 실험에서는 αf=0.5(25%)부터 양쪽 모두 원본 대비 +0.003을 넘는 악화가 검출됐다. 두 관계를 동시에 바꾸면 각각 12.5%인 첫 단계부터 양쪽이 악화됐다.

**K256·fallback·공통 decoder에 변화된 모집단 prior를 알려 주자, frozen gate의 상대 실패가 드러났다.** 같은 256개 worlds에서 두 관계를 동시에 75% 확률로 뒤집은 조건이 첫 검출점이었다. Gate 0.14473, K256 0.10441로 차이는 +0.04032였다. Profile만 바꾸면 100% 반전에서 처음 검출됐고 차이는 +0.07064였다. Velocity만 100% 뒤집었을 때도 gate가 평균 +0.02748 나빴지만 simultaneous lower bound가 +0.00174여서 사전 margin +0.003에 못 미쳤다. 두 실험 모두 ‘K256 자체 악화가 +0.003 이내’까지 요구하는 더 엄격한 판정은 36개 조건 모두에서 성립하지 않았다.

**배포 관점에서 gate의 confidence는 shift를 경고하지 못했다.** 두 관계를 완전히 뒤집자 gate가 fallback 없이 직접 선택한 비율은 80.6%에서 83.6%로 늘었고, 이 branch에서 realized winner와 다른 선택은 13.2%에서 49.5%로 늘었다. Source prior에서 이 조건의 GPU p90은 gate 3.954 ms, K256 3.834 ms였다. 같은 decoder를 쓴 always-query의 Brier는 0.17529로 두 adaptive 방법(0.27549, 0.27200)보다 낮았다.


## 1. 무엇을 고정하고 무엇을 바꿨나

V25 factual gate의 96개 RBF centers, normalization, regression 계수, confidence cutoff(0.0478034), decision rule을 원본 hash와 대조해 그대로 불러왔다. 추가 학습이나 threshold tuning은 없었다. Gate는 raw factual evidence로 Brier(probe)−Brier(query view)를 예측한다. 절댓값이 cutoff 이상이면 action을 직접 선택하고, 작으면 K1024/M2048 posterior acquisition을 사용한다. 고정 K256은 같은 likelihood와 M2048, label-stratified support를 사용하며 각 world에서 Q/P를 다시 비교한다. 어떤 방법이든 선택이 같으면 같은 16,384-hypothesis inner decoder로 최종 stick probability를 계산하므로 Brier도 같다.

각 world의 curvature spread와 friction spread는 독립적인 두 binary factors다. V25에서 넓은 curvature의 물체는 평균 boundary offset이 +0.00115, 좁은 물체는 −0.00115였다. 넓은 friction의 물체는 restitution이 평균 0.795, 좁은 물체는 0.605였다. Factual 충돌이 stick하므로 restitution이 큰 쪽의 outgoing horizontal velocity는 더 음수다. Gate에 보이는 profile mean과 velocity는 각각의 spread에 대한 불완전한 단서이며, cell이나 latent parameter를 입력으로 쓰지 않는다.

두 관계의 세기를 αc, αf로 표시했다. 각 관계에서 world별로 확률 (1−α)/2만큼 **반대 폭 집단의 조건부 중심값**을 사용하고 원래의 개별 jitter는 유지한다. α가 1이면 v25 관계를 유지한다. 0.75는 불일치 확률 12.5%, 0.5는 25%, 0은 50%, −0.5는 75%, −1은 100%다. α=0에서 조건부 중심과 해당 폭은 모집단에서 독립이다. 관측 feature에는 geometry·noise도 들어가므로 측정한 Pearson correlation이 α나 정확한 0일 필요는 없다. 두 flip에는 서로 독립인 난수를 쓰고 severity 간에는 동일 난수를 재사용했다. 이 방식은 모집단의 두 중심 혼합과 그 jitter 분포를 유지하고 coupling만 바꾼다.

Offset 변화는 profile 전체를 평행 이동하므로 slope와 참 stick label을 보존한다. Restitution 변화는 factual velocity, probe response 및 일부 query label을 물리적으로 바꾼다. 두 경우 모두 새 물리를 반영해 실제 sensor observations와 label을 다시 계산했다. 실제 장비를 구동하지 않은 synthetic intervention이다.

Query 위치·속도, Coulomb law, 후보 actions Q/P, factual/추가 sensor calibration 모두 v25와 같다. 1,024개 새 worlds(각 spread cell 256개), 새 독립 bank pairs 5쌍, world·action당 32개 noise repeats를 사용했다. 변화량을 격리하기 위해 36개 조건에서 world, latent jitter와 표준화된 noise를 공유했다. 한 world에서 실제 가능한 변경은 네 가지(원본/velocity 관계만 뒤집힌 물리/offset 관계만 뒤집힌 물리/둘 다)여서 source-prior policy의 같은 입력은 정확히 한 번 계산해 재사용했다. GPU latency는 각 조건에서 별도로 실행한 실측값이다.

## 2. ‘먼저 실패’의 사전 정의

Primary에서는 gate와 K256, 그 공통 decoder 모두 v25의 원래 물리 prior를 유지했다. Δ(s)=L_gate(s)−L_K256(s)라 하고, intact 조건 (1,1)을 0으로 쓴다. 실용적으로 의미 있는 **상대 실패**는 Δ(s)의 simultaneous 95% lower bound가 +0.003을 넘고, Δ(s)−Δ(0)의 lower bound도 0을 넘는 최초 tested condition이다. +0.003은 v24부터 사용한 연구용 absolute Brier margin이다. 실제 robot 사용자의 허용 비용으로 측정한 값은 없다.

**Gate 단독 선행 악화**는 위 두 조건에 더해 L_gate(s)−L_gate(0)의 lower bound가 +0.003을 넘고, L_K256(s)−L_K256(0)의 upper bound가 +0.003 미만인 경우다. 이것은 두 방법이 동시에 크게 나빠지지만 한 방법이 조금 더 나쁘다는 결과와 구분된다. 물리 시뮬레이터의 시간이 흐르면서 고장이 발생하는 지점이 아니라, 사전 선택한 severity grid에서 통계적 조건을 처음 만족하는 지점이다.

4개 physical cells 안에서 world를, 별도로 bank를 resample한 10,000회 crossed bootstrap을 썼다. 모든 방법·조건에 같은 draws를 적용했다. 36개 조건 × 4개 Brier contrasts에 대해 bootstrap 표준편차로 표준화한 최대 절댓값의 95 percentile을 사용했다. 아래 Brier bands는 이 144개 값에 대한 simultaneous bands다. 동일 관측의 noise repeats나 bank를 별도 독립 world로 세지 않았다. Timing은 12 worlds/cell × 3 repeats × 5 banks로 각 조건·방법당 720개다. 순서를 무작위화하고 batch-1 host→GPU factual 전송부터 선택한 observation의 posterior update까지 CUDA synchronization을 포함했다. Sensor를 이동시키거나 probe를 물체에 가하는 시간과 static bank setup은 제외했다. P90 intervals는 world·bank·repeat를 짝지어 재추출한 3,000회 pointwise bootstrap이다. Timing에는 Brier의 +0.003 기준을 적용하지 않았다.

## 3. 실패가 처음 관측된 조건

| 원본에서 severity를 키운 경로 | Δ 하한 >0 및 상대 악화 | 상대 실패 (+0.003) | 그 뒤 모두 상대 실패 | Gate 단독 선행 악화 | Gate 자체 악화 하한 >+.003 | K256 자체 악화 하한 >+.003 |
|---|---|---|---|---|---|---|
| Profile 관계만 | 검출 없음 | 검출 없음 | 검출 없음 | 검출 없음 | (0.75, 1) | (0.75, 1) |
| Velocity 관계만 | 검출 없음 | 검출 없음 | 검출 없음 | 검출 없음 | (1, 0.5) | (1, 0.5) |
| 두 관계 동시 | 검출 없음 | 검출 없음 | 검출 없음 | 검출 없음 | (0.75, 0.75) | (0.75, 0.75) |

| 상대 실패 민감도 | Margin 0 | Margin +0.001 | Margin +0.003 | Margin +0.005 |
|---|---|---|---|---|
| Profile 관계만 | 검출 없음 | 검출 없음 | 검출 없음 | 검출 없음 |
| Velocity 관계만 | 검출 없음 | 검출 없음 | 검출 없음 | 검출 없음 |
| 두 관계 동시 | 검출 없음 | 검출 없음 | 검출 없음 | 검출 없음 |

36개 조합에서 각 좌표를 따로 바꾸므로 모든 조건을 단일 순서로 정렬할 수는 없다. 위 표는 두 단독 경로 및 같은 세기로 함께 바꾼 대각선 경로를 따른다. ‘검출 없음’은 이 표본과 기준으로 입증하지 못했다는 뜻이다.

| (αc, αf) | Gate Brier | K256 Brier | Δ [simultaneous 95%] | 원본 대비 Δ 변화 [simultaneous 95%] | 직접 선택률 | Gate / K256 p90 ms | p90 ratio [pointwise 95%] |
|---|---|---|---|---|---|---|---|
| (1, 1) | 0.07322 | 0.07476 | -0.00154 [-0.00635, +0.00328] | +0.00000 [+0.00000, +0.00000] | 80.6% | 3.753 / 3.789 | 0.990 [0.269, 1.118] |
| (0, 1) | 0.15162 | 0.15215 | -0.00053 [-0.00596, +0.00490] | +0.00101 [-0.00526, +0.00727] | 78.9% | 4.005 / 3.800 | 1.054 [0.285, 1.134] |
| (-1, 1) | 0.22169 | 0.21848 | +0.00322 [-0.00378, +0.01021] | +0.00475 [-0.00418, +0.01369] | 80.7% | 4.098 / 3.824 | 1.072 [1.038, 1.142] |
| (1, 0) | 0.12424 | 0.12450 | -0.00026 [-0.00668, +0.00616] | +0.00128 [-0.00537, +0.00793] | 81.0% | 3.918 / 3.785 | 1.035 [0.271, 1.122] |
| (1, -1) | 0.17551 | 0.17483 | +0.00068 [-0.00580, +0.00717] | +0.00222 [-0.00519, +0.00963] | 80.3% | 4.041 / 3.821 | 1.057 [1.018, 1.140] |
| (0, 0) | 0.18589 | 0.18401 | +0.00189 [-0.00459, +0.00837] | +0.00343 [-0.00458, +0.01143] | 79.9% | 3.894 / 3.796 | 1.026 [0.271, 1.117] |
| (-1, -1) | 0.27549 | 0.27200 | +0.00349 [-0.00431, +0.01129] | +0.00502 [-0.00444, +0.01449] | 83.6% | 3.954 / 3.834 | 1.031 [0.272, 1.132] |

## 4. Proxy가 실제로 바뀌었는지와 선택 오류의 경로

| (αc, αf) | r(profile mean, curvature spread) | r(vx, friction spread) | 원본 대비 label 변경 | 예측 margin과 실현 margin의 r | Gate / K256 probe 비율 | Gate와 K256이 달랐던 고유 worlds | 갈린 선택에서 평균 Δ |
|---|---|---|---|---|---|---|---|
| (1, 1) | 0.718 | -0.657 | 0.0% | 0.483 | 26.8% / 26.1% | 119 | -0.03265 |
| (0, 1) | 0.010 | -0.657 | 0.0% | -0.051 | 26.7% / 26.2% | 121 | -0.01182 |
| (-1, 1) | -0.709 | -0.657 | 0.0% | -0.408 | 26.1% / 25.3% | 102 | +0.08406 |
| (1, 0) | 0.718 | -0.019 | 5.7% | 0.258 | 27.6% / 26.3% | 111 | -0.00580 |
| (1, -1) | 0.718 | 0.657 | 11.0% | 0.056 | 26.9% / 25.7% | 96 | +0.01766 |
| (0, 0) | 0.010 | -0.019 | 5.7% | -0.137 | 27.5% / 26.4% | 117 | +0.04415 |
| (-1, -1) | -0.709 | 0.657 | 11.0% | -0.454 | 26.8% / 26.1% | 83 | +0.11749 |

실현 margin은 같은 실제 world에서 probe의 Brier에서 query view의 Brier를 뺀 값이다. 이 값은 추가 관측 이전에는 알 수 없다. 두 policy의 Δ는 선택이 갈린 비율과 그 경우의 평균 Brier 차이의 곱이다. 따라서 confidence가 유지된다는 사실만으로 gate의 판단이 유효하다고 평가할 수 없다.

| (αc, αf) | 직접 branch의 잘못된 부호¹ | Fallback branch의 gate−K256 | Pure cue replay Brier | Replay의 원본 gate 대비 변화 [pointwise 95%] | Replay에서 바뀐 action |
|---|---|---|---|---|---|
| (1, 1) | 13.2% | -0.00551 | 0.07322 | +0.00000 [+0.00000, +0.00000] | 0.0% |
| (0, 1) | 29.9% | -0.00429 | 0.10610 | +0.03288 [+0.02671, +0.03932] | 18.2% |
| (-1, 1) | 44.6% | -0.00098 | 0.13943 | +0.06621 [+0.05790, +0.07444] | 35.4% |
| (1, 0) | 20.7% | +0.00053 | 0.08169 | +0.00847 [+0.00472, +0.01240] | 7.2% |
| (1, -1) | 27.3% | +0.00359 | 0.08846 | +0.01524 [+0.01008, +0.02041] | 12.2% |
| (0, 0) | 34.7% | +0.00027 | 0.11303 | +0.03981 [+0.03345, +0.04645] | 22.3% |
| (-1, -1) | 49.5% | -0.00393 | 0.15251 | +0.07928 [+0.07161, +0.08734] | 38.4% |

¹ Bank·world별 반복 noise 평균으로 정의한 realized winner와 direct branch의 예측을 대조했다. **Pure cue replay**는 gate의 score에만 shifted input을 주고, fallback·decoder·실제 observation과 query label은 모두 원본 world의 것을 사용한다. 다른 성능 저하 경로를 차단한 인공 진단이므로, coherent physical shift에서 이 수치만큼 gate가 더 나빠졌다고 해석할 수 없다.

### 같은 물리 결과에서 action만 바꾸면

원본과 shifted world 각각에서 원래 action과 변경된 action 모두의 Brier를 계산했다. 두 적용 순서의 평균인 Shapley decomposition으로, observation 선택이 바뀐 몫(action)과 같은 action에서도 factual reweighting·sensor response·decoder prediction·실제 label이 바뀐 몫(environment)을 분리했다. 두 성분의 합은 각 방법의 원본 대비 loss 변화다. 이 표의 5,000회 paired bootstrap intervals는 탐색적 pointwise 95%다.

| 물리적 변화 (αc, αf) | Gate action 성분 | Gate environment 성분 | K256 action 성분 | K256 environment 성분 | 두 방법의 action 성분 차이 [95%] |
|---|---|---|---|---|---|
| (1, 1) | +0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 [+0.00000, +0.00000] |
| (0.75, 1) | +0.01134 | +0.00882 | +0.01148 | +0.00870 | -0.00015 [-0.00222, +0.00141] |
| (1, 0.75) | +0.00237 | +0.00835 | +0.00229 | +0.00836 | +0.00008 [-0.00213, +0.00233] |
| (-1, 1) | +0.08892 | +0.05955 | +0.08604 | +0.05767 | +0.00288 [-0.00133, +0.00757] |
| (1, -1) | +0.03280 | +0.06949 | +0.03151 | +0.06856 | +0.00129 [-0.00236, +0.00497] |
| (-1, -1) | +0.09762 | +0.10465 | +0.09332 | +0.10392 | +0.00430 [-0.00018, +0.00901] |

원본 복제에서 v25와 v26의 latent parameters, cell, label이 일치했다. CPU에서 v25와 v26의 원본 inner·outer banks가 동일했다. 균형 추출한 8 worlds × 2 actions × 32 repeats의 512개 확률에서 CPU 재현값과 저장된 GPU 결과의 최대 차이는 2.23×10⁻¹⁵ 이하였다. 완전 velocity 반전에서는 좁은 curvature·좁은 friction cell의 113/256개에서 실제 stick label이 바뀌었다. 다른 세 cell 및 profile-only 반전에서는 변경이 0건이었다. [원본 재현 및 성분별 수치](results/counterfactual.json)

### 두 proxy의 차이와 joint interaction

두 단독 경로의 Δ 차이는 Δ(α,1)−Δ(1,α)다. Interaction은 Δ(α,α)−Δ(α,1)−Δ(1,α)+Δ(1,1)이다. 두 표준 오차 계열의 동시 범위는 각각 6개 축 비교, 36개 interaction에 대해 별도로 구했다.

| 남은 관계 α | Profile 경로 Δ − velocity 경로 Δ [simultaneous 95%] | Interaction [simultaneous 95%] |
|---|---|---|
| 1 | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] |
| 0.75 | -0.00009 [-0.00424, +0.00406] | +0.00043 [-0.00160, +0.00246] |
| 0.5 | -0.00072 [-0.00512, +0.00368] | +0.00146 [-0.00211, +0.00502] |
| 0 | -0.00027 [-0.00611, +0.00557] | +0.00114 [-0.00514, +0.00742] |
| -0.5 | +0.00147 [-0.00484, +0.00777] | +0.00133 [-0.00797, +0.01064] |
| -1 | +0.00254 [-0.00433, +0.00940] | -0.00195 [-0.01392, +0.01002] |

두 관계를 완전히 뒤집었을 때도 두 단독 경로의 상대 gap 차이와 interaction의 구간은 0을 포함한다. Profile 쪽의 더 큰 pure cue sensitivity를 coherent world에서 K256 대비 더 큰 불이익이라고 단정할 수 없다.

### 어느 worlds에서 손실이 컸나

| (αc, αf) | 좁은 curvature·좁은 friction Δ | 좁은 curvature·넓은 friction Δ | 넓은 curvature·좁은 friction Δ | 넓은 curvature·넓은 friction Δ | Always-query Brier |
|---|---|---|---|---|---|
| (1, 1) | -0.00050 | -0.00415 | -0.00096 | -0.00053 | 0.10036 |
| (-1, 1) | +0.00003 | +0.00030 | +0.00892 | +0.00363 | 0.12696 |
| (1, -1) | +0.00445 | -0.00438 | +0.00044 | +0.00222 | 0.14341 |
| (-1, -1) | +0.00021 | +0.00005 | +0.00225 | +0.01144 | 0.17529 |

두 축 완전 반전에서 gate와 K256의 차이가 가장 컸던 집단은 넓은 curvature·넓은 friction이었다. 그 집단에서는 실제 label이 보존됐다. 원래 분포에서 always-query는 두 selector보다 손실이 컸지만 두 축 반전에서는 더 낮은 Brier를 냈다. 이는 공통 decoder를 쓰는 두 adaptive selector의 상대 비교만으로 관측을 선택할 가치를 판단하기 어렵다는 독립적인 대조다. Cell별 차이는 descriptive mean이며 별도 subgroup 우월성 검정을 하지 않았다.


## 5. 같은 shifted prior를 두 방법에 제공하면

두 번째 실험에서는 frozen gate의 계수·cutoff를 유지하면서, gate fallback·K256·공통 decoder의 numerical banks가 해당 (αc, αf)를 안다고 가정했다. 개별 world의 실제 flip은 제공하지 않았다. 조건별 prior를 만드는 16,384개의 후보와 독립 flip draws로 likelihood·stick labels를 다시 계산했다. 첫 세 bank, 각 cell에서 사전 정한 64개씩 총 256개 worlds를 사용했다. 비교하는 source-prior 값도 정확히 같은 부분표본이다. 더 작은 secondary 표본의 경계를 full-cohort primary와 혼합하지 않았다. Matched condition별 모든 선택을 고정한 뒤 그 실험의 evaluator가 private outcomes를 열었다.

| 경로 | Matched 상대 실패 첫 조건 | Matched gate 단독 선행 악화 | 동일 부분표본 source-prior 상대 실패 | 부분표본 source-prior 단독 선행 악화 |
|---|---|---|---|---|
| Profile 관계만 | (-1, 1) | 검출 없음 | 검출 없음 | 검출 없음 |
| Velocity 관계만 | 검출 없음 | 검출 없음 | 검출 없음 | 검출 없음 |
| 두 관계 동시 | (-0.5, -0.5) | 검출 없음 | 검출 없음 | 검출 없음 |

| (αc, αf) | Matched gate / K256 Brier | Source gate / K256 Brier (같은 worlds) | Matched Δ [simultaneous 95%] | Matched Δ−source Δ [simultaneous 95%] | Matched gate / K256 p90 ms |
|---|---|---|---|---|---|
| (1, 1) | 0.08367 / 0.08256 | 0.08367 / 0.08256 | +0.00110 [-0.00821, +0.01041] | -0.00000 [-0.00000, -0.00000] | 3.892 / 3.484 |
| (0, 1) | 0.11891 / 0.10077 | 0.14567 / 0.14241 | +0.01814 [-0.00685, +0.04314] | +0.01488 [-0.01140, +0.04116] | 미측정 |
| (-1, 1) | 0.13314 / 0.06250 | 0.22818 / 0.21667 | +0.07064 [+0.03856, +0.10273] | +0.05913 [+0.02327, +0.09499] | 4.083 / 3.516 |
| (1, 0) | 0.10312 / 0.09506 | 0.12055 / 0.12073 | +0.00806 [-0.01014, +0.02627] | +0.00825 [-0.00915, +0.02564] | 미측정 |
| (1, -1) | 0.10272 / 0.07524 | 0.17612 / 0.17368 | +0.02748 [+0.00174, +0.05322] | +0.02504 [+0.00098, +0.04909] | 4.001 / 3.504 |
| (0, 0) | 0.13508 / 0.11554 | 0.17358 / 0.17018 | +0.01954 [-0.00835, +0.04743] | +0.01614 [-0.01054, +0.04282] | 3.945 / 3.535 |
| (-1, -1) | 0.14629 / 0.06287 | 0.27624 / 0.27303 | +0.08342 [+0.05199, +0.11485] | +0.08021 [+0.04836, +0.11207] | 3.952 / 3.525 |

사전 지정한 timing 다섯 조건은 (1,1), (−1,1), (1,−1), (−1,−1), (0,0)이다. 각 방법에 대해 조건별 3 banks × 16 worlds × 2 repeats로 측정했다. Online 시간에 matched prior의 offline 구성 비용은 포함하지 않았다.

대각선에서 처음 relative failure가 검출된 (−0.5, −0.5)는 사전 timing 대상에 없었다. 이후 같은 GPU에서 고정한 정책과 선택을 그대로 사용하고 48 worlds × 3 repeats × 3 banks를 추가 측정했다. Gate / K256 p90은 4.039 / 3.503 ms였다. Ratio 1.153의 pointwise 95% 범위는 [0.277, 1.161]였다. 이 보완 측정은 Brier 판정을 바꾸지 않는다. [추가 timing](results/matched/onset_latency.json)

36개 조건 전체에서 matched relative failure가 검출된 곳은 10개였다. 다른 검출점보다 두 축 모두에서 더 심하게 바꿀 필요가 없었던 경계 조합은 (0.5, −1), (−0.5, 0), (−1, 1)이었다. 일부 인접 조건에서는 구간이 다시 넓어져 판정이 달라지므로 연속적인 안전 임계값은 아직 추정할 수 없다.

| αc ↓ / αf → | 1 | 0.75 | 0.5 | 0 | -0.5 | -1 |
|---|---|---|---|---|---|---|
| 1 | +0.00110 | +0.00235 | +0.00265 | +0.00806 | +0.01328 | +0.02748 |
| 0.75 | +0.00320 | +0.00316 | +0.00041 | -0.00135 | +0.00661 | +0.02197 |
| 0.5 | +0.00662 | +0.00332 | +0.00195 | +0.00101 | +0.00858 | +0.02571 **∗** |
| 0 | +0.01814 | +0.01562 | +0.01662 | +0.01954 | +0.01935 | +0.02828 |
| -0.5 | +0.03687 | +0.03694 | +0.03356 | +0.04060 **∗** | +0.04032 **∗** | +0.04563 **∗** |
| -1 | +0.07064 **∗** | +0.07841 **∗** | +0.07934 **∗** | +0.08397 **∗** | +0.08710 **∗** | +0.08342 **∗** |

각 값은 matched-prior gate−K256 Brier다. ∗는 +0.003 relative failure를 표시한다. 직접 비교에서는 동시 반전 75%에서 matched Δ가 source Δ보다 커진 양의 simultaneous interval이 0을 포함했다. Profile 완전 반전과 두 축 완전 반전에서는 그 차이의 lower bound가 각각 +0.02327과 +0.04836이었다.

## 6. 전체 36개 조건

| αc | αf | Gate Brier | K256 Brier | Gate 자체 악화 | K256 자체 악화 | Δ [simultaneous 95%] | Gate / K256 p90 ms | +.003 상대 실패 | Gate 단독 선행 악화 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.07322 | 0.07476 | +0.00000 | +0.00000 | -0.00154 [-0.00635, +0.00328] | 3.753 / 3.789 | — | — |
| 1 | 0.75 | 0.08394 | 0.08541 | +0.01072 | +0.01065 | -0.00147 [-0.00750, +0.00456] | 3.925 / 3.804 | — | — |
| 1 | 0.5 | 0.09890 | 0.09979 | +0.02567 | +0.02503 | -0.00090 [-0.00687, +0.00508] | 3.952 / 3.798 | — | — |
| 1 | 0 | 0.12424 | 0.12450 | +0.05102 | +0.04974 | -0.00026 [-0.00668, +0.00616] | 3.918 / 3.785 | — | — |
| 1 | -0.5 | 0.15131 | 0.15062 | +0.07808 | +0.07586 | +0.00069 [-0.00546, +0.00684] | 3.950 / 3.783 | — | — |
| 1 | -1 | 0.17551 | 0.17483 | +0.10229 | +0.10007 | +0.00068 [-0.00580, +0.00717] | 4.041 / 3.821 | — | — |
| 0.75 | 1 | 0.09338 | 0.09494 | +0.02016 | +0.02018 | -0.00156 [-0.00593, +0.00281] | 3.929 / 3.795 | — | — |
| 0.75 | 0.75 | 0.10151 | 0.10257 | +0.02828 | +0.02781 | -0.00106 [-0.00655, +0.00442] | 3.963 / 3.790 | — | — |
| 0.75 | 0.5 | 0.11277 | 0.11297 | +0.03954 | +0.03821 | -0.00020 [-0.00603, +0.00563] | 3.977 / 3.777 | — | — |
| 0.75 | 0 | 0.13370 | 0.13333 | +0.06047 | +0.05856 | +0.00037 [-0.00561, +0.00635] | 3.988 / 3.759 | — | — |
| 0.75 | -0.5 | 0.15857 | 0.15718 | +0.08535 | +0.08242 | +0.00140 [-0.00458, +0.00738] | 3.991 / 3.764 | — | — |
| 0.75 | -1 | 0.17966 | 0.17880 | +0.10643 | +0.10404 | +0.00086 [-0.00525, +0.00696] | 4.026 / 3.791 | — | — |
| 0.5 | 1 | 0.10993 | 0.11155 | +0.03671 | +0.03679 | -0.00162 [-0.00602, +0.00279] | 3.925 / 3.807 | — | — |
| 0.5 | 0.75 | 0.12032 | 0.12064 | +0.04710 | +0.04588 | -0.00032 [-0.00535, +0.00470] | 3.959 / 3.799 | — | — |
| 0.5 | 0.5 | 0.13125 | 0.13077 | +0.05802 | +0.05601 | +0.00048 [-0.00505, +0.00601] | 3.979 / 3.784 | — | — |
| 0.5 | 0 | 0.15143 | 0.15034 | +0.07821 | +0.07558 | +0.00109 [-0.00450, +0.00669] | 3.984 / 3.794 | — | — |
| 0.5 | -0.5 | 0.17509 | 0.17301 | +0.10187 | +0.09825 | +0.00208 [-0.00341, +0.00758] | 4.008 / 3.861 | — | — |
| 0.5 | -1 | 0.19727 | 0.19550 | +0.12404 | +0.12074 | +0.00177 [-0.00362, +0.00716] | 4.019 / 3.838 | — | — |
| 0 | 1 | 0.15162 | 0.15215 | +0.07839 | +0.07739 | -0.00053 [-0.00596, +0.00490] | 4.005 / 3.800 | — | — |
| 0 | 0.75 | 0.15992 | 0.16012 | +0.08669 | +0.08536 | -0.00021 [-0.00632, +0.00591] | 4.005 / 3.791 | — | — |
| 0 | 0.5 | 0.16825 | 0.16854 | +0.09502 | +0.09378 | -0.00029 [-0.00660, +0.00601] | 4.015 / 3.793 | — | — |
| 0 | 0 | 0.18589 | 0.18401 | +0.11267 | +0.10924 | +0.00189 [-0.00459, +0.00837] | 3.894 / 3.796 | — | — |
| 0 | -0.5 | 0.20498 | 0.20222 | +0.13175 | +0.12746 | +0.00276 [-0.00338, +0.00890] | 3.985 / 3.820 | — | — |
| 0 | -1 | 0.22649 | 0.22408 | +0.15327 | +0.14932 | +0.00241 [-0.00396, +0.00879] | 3.938 / 3.789 | — | — |
| -0.5 | 1 | 0.19025 | 0.18809 | +0.11702 | +0.11333 | +0.00216 [-0.00422, +0.00853] | 4.043 / 3.761 | — | — |
| -0.5 | 0.75 | 0.19573 | 0.19355 | +0.12251 | +0.11879 | +0.00218 [-0.00460, +0.00897] | 4.021 / 3.789 | — | — |
| -0.5 | 0.5 | 0.20278 | 0.20108 | +0.12956 | +0.12632 | +0.00170 [-0.00523, +0.00863] | 3.994 / 3.794 | — | — |
| -0.5 | 0 | 0.21799 | 0.21357 | +0.14477 | +0.13881 | +0.00442 [-0.00295, +0.01179] | 3.950 / 3.878 | — | — |
| -0.5 | -0.5 | 0.22938 | 0.22366 | +0.15615 | +0.14890 | +0.00571 [-0.00158, +0.01301] | 3.973 / 3.728 | — | — |
| -0.5 | -1 | 0.24862 | 0.24441 | +0.17539 | +0.16964 | +0.00421 [-0.00283, +0.01126] | 3.902 / 3.803 | — | — |
| -1 | 1 | 0.22169 | 0.21848 | +0.14847 | +0.14371 | +0.00322 [-0.00378, +0.01021] | 4.098 / 3.824 | — | — |
| -1 | 0.75 | 0.22717 | 0.22367 | +0.15395 | +0.14891 | +0.00350 [-0.00398, +0.01098] | 4.112 / 3.837 | — | — |
| -1 | 0.5 | 0.23183 | 0.22880 | +0.15860 | +0.15404 | +0.00302 [-0.00471, +0.01076] | 4.057 / 3.816 | — | — |
| -1 | 0 | 0.24823 | 0.24354 | +0.17500 | +0.16878 | +0.00468 [-0.00355, +0.01291] | 4.070 / 3.787 | — | — |
| -1 | -0.5 | 0.25809 | 0.25233 | +0.18487 | +0.17757 | +0.00576 [-0.00215, +0.01366] | 4.024 / 3.797 | — | — |
| -1 | -1 | 0.27549 | 0.27200 | +0.20227 | +0.19724 | +0.00349 [-0.00431, +0.01129] | 3.954 / 3.834 | — | — |

![Gate robustness under two proxy shifts](results/v26.png)

두 heatmap은 0을 중심으로 같은 대칭 색 범위를 사용한다. 점은 matched relative failure 검출을 표시한다. 오른쪽 timing 그래프의 값은 point estimate이며 uncertainty는 앞 표에 제시했다.

## 7. 논문 기여 가능성과 적용 범위

**논문에서 제시할 수 있는 결과는 frozen observation policy의 실패를 찾아내는 비교 실험이다.** 같은 query와 sensor calibration에서도, 동일한 오래된 prior를 공유하는 두 방법만 비교하면 두 방법의 절대 성능이 함께 나빠지는 구간을 놓치기 쉽다. 동일 worlds와 bank seeds에서 양쪽에 올바른 population prior를 주면 K256은 action을 크게 바꾸지만, gate는 high-confidence branch의 결정을 그대로 유지한다. 두 축 완전 반전에서 prior를 바꾸자 K256은 bank·world 쌍의 54.0%, gate는 4.9%에서 action을 바꿨다. 이 조건의 matched Δ에서 동일 부분표본 source Δ를 뺀 값의 simultaneous 95% 범위는 [+0.04836, +0.11207]이다. 이 대조에서는 두 방법의 공통 decoder도 함께 바뀌므로 K256의 개선 전체를 acquisition 개선에만 귀속할 수 없다.

**현재 결과의 강도는 synthetic diagnostic 기여에 집중된다.** Profile-only 반전은 참 label을 1개도 바꾸지 않고 원래 gate의 action 선택을 교란한다. Velocity-only 반전은 일부 world의 정답까지 바꾼다. 두 실험을 분리하고, 동일 world의 두 potential observations와 고정된 action을 교차 평가했으므로 이 차이를 추적할 수 있다. 반면 frozen gate는 joint reversal에서 Brier나 p90 우위를 보이지 않았고, 이 실험은 shift를 자동 추정하거나 교정하는 새 방법의 성능을 입증하지 않는다. Known-prior 보조 실험의 256개 worlds는 primary의 부분집합이고 bank도 세 쌍뿐이다. 같은 물체의 다른 실제 재질, 다른 contact law, 다른 sensor, 다른 GPU로의 전이는 측정하지 않았다. P90의 추정 범위는 fallback이 몰리는 worlds의 구성에 민감하다. 원본에서도 gate/K256의 p90 ratio 범위는 [0.269, 1.118]이었다.

**다음 논문 실험의 통과 조건.** Deployment batch의 factual evidence만 사용해 두 coupling의 방향이나 불확실성을 추정하고, boundary profile과 velocity가 해당 world에서 믿을 만한지에 따라 query/probe를 선택하는 policy를 개발할 수 있다. 학습에 쓰지 않은 환경 단위로 별도 test set을 만들고, source K256·estimated-prior K256·known-prior K256·always-query를 동일 decoder에서 비교한다. 두 priors가 factual marginal상 구분되지 않는 환경 쌍도 설계해, 추가 calibration probe나 소수의 paired outcomes가 있어야 회복되는 조건을 함께 측정해야 한다. Sensing 시간까지 포함해 사전 지정한 Brier margin과 latency budget을 동시에 만족하고, 실제 물체 또는 독립 simulator에서 방향이 재현될 때 방법론적 기여 주장을 강화할 수 있다. 현재는 이 통과 조건이 남아 있다.


## 문헌 검토와 논문 기여의 범위

### 1. 논문별 비교

아래 일곱 편의 원문에서 실제로 조작한 변수, 연구 단위, 측정값을 추출했다. 마지막 열은 해당 결과를 v26의 frozen observation policy에 적용할 때 추가로 확인할 점이다.

| 연구 | Research question · methodology | Sample · data source | 주요 결과와 저자의 결론 | v26에 적용할 때의 한계 |
|---|---|---|---|---|
| Arjovsky et al., *Invariant Risk Minimization* | 서로 다른 training environments에서 같은 predictor가 최적이 되도록 학습하면 label과 색의 관계가 뒤집혀도 일반화하는가? ERM과 IRM을 비교했다. | Colored MNIST의 binary labels. 색과 label의 일치 확률이 training 두 환경에서는 80%, 90%, test에서는 10%다. Table 1은 10 runs를 평균했다. | Reversed test accuracy는 ERM 17.1%, IRM 66.9%, grayscale oracle 73.0%였다. 이 조건에서 training environments 사이의 변동을 활용할 수 있었다. | 색을 바꿔도 digit label은 유지된다. Restitution까지 바꾸는 물리 실험에는 추가적인 경로가 있다. [원문 Table 1](https://arxiv.org/pdf/1907.02893#page=17) |
| Kamath et al., *Does Invariant Risk Minimization Capture Invariance?* | Practical IRM relaxation이 찾는 predictor와 full IRM의 해는 언제 달라지는가? Two-bit 환경에서 population risk를 계산하고 반례를 구성했다. | Binary label과 두 noisy bits의 명시적 분포. 원문의 한 반례는 두 training environments와 한 reversed test environment를 쓴다. Population 계산에는 유한 피험자 수가 없다. | 반례의 test squared loss는 full IRM 0.18, ERM 0.28, 논문의 scalar relaxation 0.38이었다. 같은 binary class를 골라도 확신의 크기가 달라 risk가 갈렸다. | 원문의 loss scale은 v26의 stick Brier와 다르다. 특정 relaxation에 대한 반례로 모든 invariance 방법의 순위를 정할 수 없다. [원문 §3, pp. 4–5](https://proceedings.mlr.press/v130/kamath21a/kamath21a.pdf#page=4) |
| Sagawa et al., *An Investigation of Why Overparameterization Exacerbates Spurious Correlations* | Minority group을 reweight해도 모델 크기에 따라 worst-group error가 증가하는 이유는 무엇인가? Image experiments와 synthetic linear model을 함께 분석했다. | CelebA와 Waterbirds 두 image datasets. 각 task에서 binary label과 binary attribute를 교차해 네 그룹을 평가했다. | 연구한 설정에서는 reweighting과 model size의 조합이 worst-group error에 영향을 줬다. 다수 그룹을 subsample하는 방법과 reweighting도 다른 결과를 냈다. | Training 및 group 구성 변경을 다룬다. 이미 동결한 gate의 원인이나 실패점을 이 결과에서 계산할 수 없다. [원문 §§2–3, 6](https://proceedings.mlr.press/v119/sagawa20a/sagawa20a.pdf#page=2) |
| Hoftijzer, Burghouts & Spreeuwers, *Language-Based Augmentation to Address Shortcut Learning in Object-Goal Navigation* | Room type과 wall color를 연결해 훈련한 ObjectNav policy가 색을 교체하면 어떻게 행동하는가? Deceptive와 nondeceptive 변형 및 language-based augmentation을 비교했다. | Simulated ObjectNav. Test set당 1,080 episodes를 9 target objects, 5 layouts와 가능한 color permutations에 배분했다. | Fig. 6에서 target room 한 곳의 색을 바꾸면 EmbCLIP success가 45%→14%, augmented model은 39%→30%였다. 상충하는 단서는 agent를 잘못된 방으로 유도할 수 있었다. | Episode success와 경로 길이는 one-step Brier나 GPU p90과 다르다. Augmentation 모델은 재학습했다. [원문 Fig. 6, Appendix B](https://ris.utwente.nl/ws/portalfiles/portal/418166870/Language-Based_Augmentation_to_Address_Shortcut_Learning_in_Object-Goal_Navigation.pdf#page=6) |
| Sundaralingam & Hermans, *In-Hand Object-Dynamics Inference using Tactile Fingertips* | 미끄러지기 직전의 fingertip force와 surface normal에서 실제 friction coefficient를 추정할 수 있는가? Tilting platform으로 얻은 reference와 비교했다. | YCB 실물 8개, 물체당 10 trials. 두 용기에서는 plastic과 label 표면을 따로 측정했다. | 물체 전체에서 friction coefficient 추정의 median error는 0.0056이었다. 힘을 받을 때 휘어지는 표면에서 오차가 컸다. | 접촉면, tactile force, excitation이 명시된 실험이다. 충돌 후 수평 velocity만으로 friction spread를 식별한다는 증거로 일반화하기 어렵다. [원문 §§IV–V, Fig. 6](https://arxiv.org/pdf/2003.13165#page=6) |
| Go & Isaac, *Robust Expected Information Gain for Optimal Bayesian Experimental Design Using Ambiguity Sets* | Prior가 달라질 때 experiment ranking이 얼마나 민감한가? KL ambiguity set에서 robust EIG를 구성했다. | §3의 분석 예시는 두 가지 binary tests와 유병 확률이 변하는 한 모집단이다. | 예시에서 prior probability 0.5를 경계로 두 tests의 EIG 순위가 뒤집힌다. 모호한 prior 아래에서 design을 비교하는 근거를 제공한다. | EIG는 관측 후 stick probability의 실현 Brier와 다른 objective다. [원문 §§3–5](https://proceedings.mlr.press/v180/go22a/go22a.pdf) |
| Huang et al., *Amortized Bayesian Experimental Design for Decision-Making* | 다음 design과 최종 decision을 함께 amortize하면 downstream utility가 개선되는가? Transformer Neural Decision Process를 학습했다. | 여러 sequential design tasks와 retrosynthesis planning 사례를 평가했다. 연구 단위는 task별 simulated trajectories 및 planning problems다. | 저자들은 여러 task에서 decision-aware designs가 유용했다고 보고했다. | 다음 observation과 decision을 공동 학습한다는 아이디어에는 선례가 있다. v26 비교는 두 frozen selectors가 서로 다른 두 물리적 proxy shift에서 어떤 action을 고르는지 추적한다. [원문 §§3–5](https://proceedings.neurips.cc/paper_files/paper/2024/file/c59f05d7ab3638b138cc61f32e1a7cd1-Paper-Conference.pdf) |

### 2–3. Literature review와 knowledge map

**Training에서 정보가 되던 단서가 test에서 오도하는 경우.** Colored MNIST에서 색과 label의 관계를 반전했고, ObjectNav에서는 목표로 가는 방향과 연결됐던 색을 다른 방에 배치했다. 전자는 class prediction, 후자는 잘못된 탐색 행동을 관찰한다. V26에서는 *profile mean*과 *horizontal velocity*가 직접 예측하는 대상이 최종 stick label인지, 다음 observation의 상대적 가치인지 분리해야 한다. V25 gate의 학습 대상은 후자였다. [Arjovsky et al.](https://arxiv.org/pdf/1907.02893#page=17) · [Hoftijzer et al.](https://arxiv.org/pdf/2402.05090#page=6) · [V25 gate 구현](../within_calibration_v25/gate25.py)

**이론적 보장과 실측 단위.** Two-bit 반례에서는 population level에서도 predictor의 확신과 loss가 달라졌다. Group 연구에서는 평균과 worst-group error의 평가 대상이 다르다. 실제 마찰 실험에서는 sensor signal이 어떤 조건에서 friction을 드러내는지 관측했다. 따라서 v26의 knowledge map은 **물성 분포의 변화 → factual cue와 spread의 관계 → gate score 및 fallback → query/probe 선택 → 해당 observation으로 갱신한 stick probability → held-out Brier**로 쓸 수 있다. 계산 비용에는 별도의 **fallback 빈도 → full acquisition 실행 → end-to-end GPU p90** 경로가 있다. 물성의 직접적인 효과도 존재한다. Restitution은 contact law와 label·probe response에 관여하므로, 두 경로를 구분해야 한다. [Kamath et al.](https://proceedings.mlr.press/v130/kamath21a/kamath21a.pdf#page=5) · [Sagawa et al.](https://proceedings.mlr.press/v119/sagawa20a/sagawa20a.pdf#page=2) · [Sundaralingam & Hermans](https://arxiv.org/pdf/2003.13165#page=6) · [V25 물리 모델](../within_calibration_v25/physics25.py)

### 4–7. Research gaps, 상충하는 연구, methodology comparison, evidence synthesis

| 비교할 evidence | 차이를 설명하는 조건과 남은 gap | V26에서의 검증 또는 해석 |
|---|---|---|
| IRM은 한 Colored MNIST 반전에서 ERM보다 높은 accuracy를 냈다. Two-bit 반례에서는 scalar relaxation이 ERM보다 높은 squared loss를 냈다. [Arjovsky et al.](https://arxiv.org/pdf/1907.02893#page=17) · [Kamath et al.](https://proceedings.mlr.press/v130/kamath21a/kamath21a.pdf#page=5) | Dataset, 입력 정보, 목적 함수, invariant solution과 relaxation의 관계가 다르다. 같은 환경에 같은 metric을 적용한 replication으로 해석할 수 없다. | V26에서는 IRM을 훈련하지 않았다. Gate가 취약해지더라도 IRM이 이를 복구한다고 주장하려면 별도 학습과 독립 비교가 필요하다. |
| Sagawa 등은 label×attribute groups를 따로 평가했다. Hoftijzer 등은 cue를 제거하거나 오도하도록 바꿨을 때 episode 행동을 비교했다. [Sagawa et al.](https://proceedings.mlr.press/v119/sagawa20a/sagawa20a.pdf#page=2) · [Hoftijzer et al.](https://arxiv.org/pdf/2402.05090#page=6) | Group별 error는 누가 손해를 보는지 보여준다. Cue를 설계해 바꾸는 실험은 어느 경로가 취약한지 확인하는 데 유리하다. 두 방법 모두 다른 실제 물체로의 일반화를 별도로 요구한다. | 두 축 각각의 weakening·reversal과 결합 변형에서 gate–K256 차이를 측정하고, pooled risk에 가려진 physical cell의 손실을 제시한다. |
| ObjectNav에서는 wall color를 교체해도 목표 object의 정의는 유지된다. 실제 접촉에서는 표면 부위에 따라 friction response와 reference가 달라진다. [Hoftijzer et al.](https://arxiv.org/pdf/2402.05090#page=6) · [Sundaralingam & Hermans](https://arxiv.org/pdf/2003.13165#page=7) | 물리량에 대한 intervention은 proxy의 예측력 외에 실제 outcome distribution도 바꿀 수 있다. 실물 검증에는 object·surface·reset 단위를 구별해야 한다. | 같은 shifted world에서 두 selector를 비교한 차이와, nominal world 대비 각 policy 자체의 loss 변화를 구분한다. Gate가 보는 입력만 바꾼 대조는 물리적 world shift와 별도 estimand다. |

**근거의 강도.** 반전된 단서가 성능을 해칠 수 있다는 데에는 분류와 embodied simulation의 직접 근거가 있다. 특정 IRM relaxation의 한계에는 구성적 반례가 있다. 물체·접촉면에 따라 정보를 얻는 조건이 달라진다는 근거는 실제 robot에서 측정됐지만 물체 수와 sensor가 제한적이다. 이 원문 중 v25 gate와 K256의 우열, 두 proxy 중 더 취약한 축, friction shift 이후 GPU p90을 직접 측정한 연구는 없다. 각 논문의 accuracy, squared loss, navigation success, friction error를 합산할 근거도 없다. [Arjovsky et al.](https://arxiv.org/pdf/1907.02893#page=17) · [Kamath et al.](https://proceedings.mlr.press/v130/kamath21a/kamath21a.pdf#page=5) · [Hoftijzer et al.](https://arxiv.org/pdf/2402.05090#page=6) · [Sundaralingam & Hermans](https://arxiv.org/pdf/2003.13165#page=7)

### 8. 근거에서 도출한 연구 질문 10개

| Research question과 gap | 중요성 · 변수 · 가능한 검증 | 근거 |
|---|---|---|
| 1. 기존 단서의 무정보화와 완전 반전에서 행동이 얼마나 다른가? | 단서가 없을 때와 확신을 가지고 틀릴 때의 비용을 비교한다. 같은 calibration의 proxy strength별 action, fallback, Brier를 측정한다. | [Hoftijzer et al.](https://arxiv.org/pdf/2402.05090#page=6) |
| 2. 두 proxy 중 어느 축에서 gate−K256 차이가 먼저 양수가 되는가? | 축별 취약성을 확인하려면 한쪽만 바꾼 동일-seed contrast와 여러 strength에 대한 uncertainty가 필요하다. | [Arjovsky et al.](https://arxiv.org/pdf/1907.02893#page=17) |
| 3. 두 관계를 함께 뒤집었을 때 피해가 단일 반전 합보다 큰가? | 한 단서가 다른 단서를 보상하는지 밝힌다. Two-factor intervention의 interaction contrast를 paired worlds에서 계산한다. | [Kamath et al.](https://proceedings.mlr.press/v130/kamath21a/kamath21a.pdf#page=4) |
| 4. 평균보다 먼저 실패하는 physical cell은 무엇인가? | 좁은 curvature와 넓은 friction처럼 probe의 정보가 큰 cell에서 손해가 집중될 수 있다. 균등가중·cell별·worst-cell Brier를 함께 평가한다. | [Sagawa et al.](https://proceedings.mlr.press/v119/sagawa20a/sagawa20a.pdf#page=2) |
| 5. Shift는 advantage의 부호와 confidence 중 무엇을 흔드는가? | 부호는 direct action, 절댓값은 fallback을 바꾼다. Frozen scores의 부호/크기, direct coverage, conditional loss를 연결한다. | [Kamath et al.](https://proceedings.mlr.press/v130/kamath21a/kamath21a.pdf#page=5) |
| 6. Gate가 올바른 prior를 받는 K256보다 나빠지는 경계와 stale K256보다 나빠지는 경계는 같은가? | Baseline도 과거 cue 관계에 의존할 수 있다. 같은 shifted data에서 bank prior만 바꾸고 decoder 조건을 명시한다. | [Arjovsky et al.](https://arxiv.org/pdf/1907.02893#page=17) · [Kamath et al.](https://proceedings.mlr.press/v130/kamath21a/kamath21a.pdf#page=5) |
| 7. Factual cue의 변화와 실제 restitution 변화가 각각 얼마의 Brier 차이를 만드는가? | 수평 velocity는 물리적 결과다. Gate-input-only negative control과 실제 물성 intervention을 분리하고 query label·potential observations의 변경률을 기록한다. | [Sundaralingam & Hermans](https://arxiv.org/pdf/2003.13165#page=6) |
| 8. 낯선 물체 표면에서도 수평 velocity가 friction spread를 예측하는가? | 실제 proxy 타당성을 확인한다. 물질·표면 부위·충돌 위치를 교차하고 robot과 독립 friction measurement로 평가한다. | [Sundaralingam & Hermans](https://arxiv.org/pdf/2003.13165#page=7) |
| 9. 실패한 조건을 보고 재학습한 방법이 다음 독립 반전에서 복구되는가? | 평가 grid에 맞춘 사후 선택을 막는다. 별도 환경에서 inverse association의 강도와 joint shift를 새로 추출해 frozen challenger를 검증한다. | [Arjovsky et al.](https://arxiv.org/pdf/1907.02893#page=17) · [Hoftijzer et al.](https://arxiv.org/pdf/2402.05090#page=6) |
| 10. Shift로 fallback이나 probe가 늘어날 때 실제 관측 비용까지 감수할 가치가 있는가? | GPU p90, probe 수행·reset 시간, realized Brier를 분리 측정하고 사전 지정한 utility로 비교한다. | [Hoftijzer et al.](https://arxiv.org/pdf/2402.05090#page=6) · [Sundaralingam & Hermans](https://arxiv.org/pdf/2003.13165#page=6) |

### 9. Research assistant review

논문에서 가장 방어하기 쉬운 기여 후보는 **하나의 physical query에 대해, frozen observation policy가 어떤 proxy 관계의 변화에서 posterior-based K256보다 나빠지는지 paired decision loss로 위치를 좁히는 것**이다. Sign reversal 자체, synthetic shortcut의 존재, 불변 표현의 가치에는 이미 선례가 있다. 분야 전체의 신규성은 이 일곱 편만으로 확정할 수 없다. 기존 source 환경에서 두 정책의 평균이 비슷했다는 결과 역시 두 정책이 같은 information을 썼다는 증거가 되지 않는다. [Arjovsky et al.](https://arxiv.org/pdf/1907.02893#page=17) · [Hoftijzer et al.](https://arxiv.org/pdf/2402.05090#page=6)

결과에서 ‘최초 실패’는 실제 비교한 grid와 사전 판정 기준에 종속된다. 차이가 검출되지 않은 인접 셀은 무해성이 입증된 셀로 분류할 수 없다. Gate와 K256의 action이 갈린 비율, 그때의 paired loss, 실제 fallback 비율과 GPU p90을 함께 제시하면 proxy shift가 어디까지 전달됐는지 보인다. 두 정책에 같은 decoder와 같은 observation을 주고도 예측이 달라진다면 구현 차이를 먼저 점검해야 한다. 실제 다른 재료·마감·접촉 위치에서 두 proxy 관계의 분포를 측정하고, simulator에서 얻은 실패 경계를 원래 단위로 대조하는 것이 외적 타당성을 높이는 직접 검증이다. [Sagawa et al.](https://proceedings.mlr.press/v119/sagawa20a/sagawa20a.pdf#page=2) · [Sundaralingam & Hermans](https://arxiv.org/pdf/2003.13165#page=7)


## 재현 자료

서버 폴더는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/proxy_shift_v26`이다. [사전 protocol](PROTOCOL.md), [물리적 shift 구현](physics26.py), [GPU 실행](run26.py), [primary 통계](stats26.py), [matched-prior 실행](matched26.py), [matched-prior 통계](matched_stats26.py), [원본 및 선택 경로 검산](counterfactual26.py), [실행 장치](results/device.json), [GPU 종료 기록](results/worker_exit.json), [추가 p90 측정 코드](onset_timing26.py), [36개 조건 원수치](results/summary.json), [matched 결과](results/matched/summary.json), [동결된 v25 gate](../within_calibration_v25/results/selected.json)를 보존했다. GPU worker는 한 장의 RTX PRO 6000 Blackwell Server Edition만 순차적으로 사용했다. 이전 v18–v25 모듈과 상대 경로를 유지한 새 서버 output directory에서 빈 Slurm GPU를 확인하고 v18 guard로 run26.py를 실행한다. 별도 worker log directory에서 matched26.py를 실행한 후 compute-node CPU로 stats26.py, counterfactual26.py, matched_stats26.py를 실행한다. 원본 선택·parameter hashes와 실제 timing, per-world losses는 results에 보존했다.
