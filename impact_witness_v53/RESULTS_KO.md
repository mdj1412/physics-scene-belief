# v53 · 동일 observation에서의 예측과 네 번째 probe 교체의 가치

## 사전 지정 비교

세 정책 모두 교체 전 네 쌍과 교체 후 세 weak/reference 쌍, 네 번째 sacrificial specimen의 weak/impact 쌍을 받았다. 후자를 3R+I라고 적는다. 관측은 simulation에서 얻은 변형 방향이다. 최종 평가에 쓸 물체의 stick/slip label은 정책이 볼 수 없다. 각 환경의 128개 deployment 물체는 calibration specimen과 다른 support에서 뽑았다. 알려진 교체 시점에 같은 contact law의 다른 표면으로 옮긴다.

12개 paired catalog를 독립 추론 단위로 두고 각 catalog에서 4개 환경을 평가했다. κ 다섯 값, object-level reference-to-impact error e 세 값, session-level complement 확률 g 네 값을 균등 혼합했다. 각 조합에서 calibration prefix를 256회 샘플링했다. 평균 Brier는 frozen forecast와 각 prefix를 사용해 held-out 물체·관측·impact 조건부 분포를 적분했다. Session loss p90은 128개 binary outcome의 squared error 평균을 먼저 구한 뒤 session 모집단에서 취한 90번째 백분위수다. 조건별 p90을 평균하지 않았다. p90 차이는 두 marginal quantile의 차이다. 두 지표 모두 낮을수록 좋다.

| Calibration | Prediction policy | 평균 held-out Brier | Session loss p90 |
|---|---|---:|---:|
| 3R+I | Continuous joint mixture | 0.158519 | 0.182310 |
| 3R+I | 유지·재추정·unsigned 중 선택하는 v52 selector | **0.158458** | 0.182443 |
| 3R+I | 같은 세 행동의 sign vote | 0.158489 | 0.182498 |
| 3R+I | 교체 후 네 쌍만 쓰는 simple sign vote | 0.158606 | 0.182587 |
| 3R+I | 교체 전후의 sign만 쓰는 continuous joint | 0.158543 | 0.182343 |
| 3R+I | 네 번째 impact 관측을 가린 continuous joint | 0.158681 | 0.182451 |
| 4R | 기존 v52 continuous joint | 0.158526 | 0.182367 |
| 4R | 기존 v52 selector | 0.158520 | 0.182592 |
| 4R | 기존 v52 같은 세 행동의 sign vote | 0.158552 | 0.182656 |
| 4R | 기존 v52 post-only vote | 0.158656 | 0.182722 |
| 공통 deployment touch | Always unsigned | 0.159297 | 0.182935 |

| 3R+I continuous joint에서 차감한 same-information 대조 | 평균 Brier 차이 [동시 95% CI] | Session p90 차이 [동시 95% CI] |
|---|---:|---:|
| 3R+I v52 selector | +0.00006078 [+0.00004585, +0.00007572] | −0.00013265 [−0.00023317, −0.00006199] |
| 3R+I 같은 세 행동의 sign vote | +0.00002965 [+0.00001382, +0.00004548] | −0.00018716 [−0.00027127, −0.00012903] |

평균과 p90이 동시에 낮아지는 가설은 primary solver에서 성립하지 않았다. 연속 혼합은 평균에서 두 대조보다 높았으며, p90에서 두 대조보다 낮았다. 네 contrast에 대한 paired-catalog t interval과 5,000회 paired-catalog bootstrap percentile interval에 Bonferroni를 적용했다. 이전 방식을 현재 rollout에서 다시 계산해 v52의 평균과 p90을 복원했다. 한층 단순한 post-only vote에 대해서는 joint의 평균이 0.00008758, p90이 0.00027661 낮았다. 원래의 4R v52 selector와 비교한 joint의 평균 차이는 −0.00000060 [−0.00001756, +0.00001636], p90 차이는 −0.00028114 [−0.00034507, −0.00016720]이었다. 이 대조는 observation과 policy가 함께 바뀐다.

## Impact를 본 가치와 2.0 N reference를 포기한 비용

아래는 secondary contrast다. Interval은 각각 nominal 95%이며 서로 겹치는 대조를 여러 번 수행했다.

| 비교 | 평균 Brier 차이 [95% CI] | Session p90 차이 [95% CI] |
|---|---:|---:|
| Joint, 3R+I − 3R+masked I | −0.00016197 [−0.00016942, −0.00015452] | −0.00014098 [−0.00017965, −0.00009198] |
| Joint, 3R+I − 4R | −0.00000740 [−0.00001251, −0.00000230] | −0.00005676 [−0.00008506, −0.00000277] |
| 세 행동 selector, 3R+I − 4R | −0.00006138 [−0.00007020, −0.00005257] | −0.00014849 [−0.00016708, −0.00006378] |
| 세 행동 sign vote, 3R+I − 4R | −0.00006285 [−0.00007096, −0.00005474] | −0.00015871 [−0.00018323, −0.00008214] |

네 번째 impact sign을 읽으면 가렸을 때보다 예측이 좋아진다. 그러나 같은 접촉·reset 예산에서 네 번째 reference도 유용하다. Signed impact를 가렸을 때를 기준으로 한 평균 이득 0.000162는 실제 probe 교체 이득 0.0000074의 약 22배다. 세 행동 selector는 같은 교체로 평균 Brier를 0.0000614 낮췄다. Paired difference-in-differences는 +0.00005398 [0.00004910, 0.00005886]이었다. 이 design과 고정된 calibration 방식에서는 selector가 교체 효과를 더 크게 회수했다.

같은 evidence와 prior에서 sign-only continuous 대신 magnitude까지 쓰는 joint는 평균 −0.00002457 [−0.00002753, −0.00002160], p90 −0.00003220 [−0.00006758, −0.00001501]이었다. 반대로 sign만 유지하고 세 행동에서 연속 혼합으로 옮기면 평균 +0.00005422, p90 −0.00015496였다. 측정 크기의 효과와 연속 혼합의 tail 이득을 분리할 수 있다. [Sign-only continuous 검산](vote_score53.json)

## 어떤 조건에서 결론이 바뀌는가

- e=0에서는 네 g 모두에서 3R+I continuous의 평균이 4R continuous보다 0.000010–0.000086 높았다. e=0.25에서는 0.000087–0.000209 높았다. e=0.5에서만 0.000169–0.000234 낮았다. 그러나 이 조건에서도 absolute mean은 항상 unsigned인 대조보다 0.001133–0.001574 높았다. 한 번의 직접 관측으로 표면에 대한 신뢰도를 정확히 복원하기는 어렵다.
- e의 prior 평균을 0.25로 바꿔 평가 mixture의 평균과 맞춘 경우, 4R과 3R+I에 동일 prior를 사용했을 때 교체의 평균 차이는 오히려 +0.00002098 [+0.00001776, +0.00002419]였다. p90 차이는 +0.00000793 [−0.00001404, +0.00004146]였다. 이 uniform prior의 평균만 discrete evaluation과 같으며 분포 전체를 아는 oracle은 아니다.
- e=.10으로 고정한 모델과 비교하면 e의 불확실성까지 적분한 mixed joint는 평균이 0.00000661 낮았다. p90 차이는 −0.00000176 [−0.00001374, +0.00001040]이었다. Impact sign에 10% sign reversal을 넣고 sensor model이 이 확률을 알고 있으면 평균 0.158579, p90 0.182346이었다. 이때 clean 4R continuous 대비 평균은 0.000053 높아졌다.
- 48개 환경으로 구성한 fleet를 256회 다시 실현했을 때 mixed joint가 mixed selector보다 실제 fleet 평균에서 진 비율은 66.8%, fleet p90에서 진 비율은 43.4%였다. 같은 세 행동 vote에 대해서는 각각 55.9%, 43.0%였다. 모두 같은 catalog를 쓰는 반복이다. 독립 배포 noise seed에서는 mixed joint의 p90이 두 대조보다 각각 0.0001367, 0.0001881 낮았다.

## Solver 해상도가 평균 순위를 바꿨다

PyBullet time step을 0.0125→0.00625 ms, mesh spacing을 0.025→0.0125 mm로 바꾸고 solver iteration을 80→160으로 늘렸다. 동일한 물체·관측·정책 예측에서 deployment target label만 바뀌었다. Primary에서 joint가 mixed selector와 same-menu vote를 평균에서 이긴 catalog는 각각 0/12, 1/12였다. Ultra에서는 각각 8/12, 12/12였다. Joint의 평균 차이는 ultra에서 **−0.00000878 [−0.00002568, +0.00000811]** 및 **−0.00004208 [−0.00006055, −0.00002361]**였다. 두 비교 모두 12개 catalog 전체에서 contrast가 joint 쪽으로 이동했다. Ultra가 실물 ground truth라는 근거는 없다.

Label table의 884,736개 entry 중 29,069개(3.286%)가 바뀌었다. 28,736개는 stick→slip, 333개는 slip→stick이었다. 평평한 state를 여러 family에서 공유한 table이다. Tip radius 0.25 mm에서 0.282%, 2.0 mm에서 6.290%가 바뀌었다. 이 비대칭 때문에 작은 평균 Brier 격차에 대해 solver 해상도 하나로 정책 순위를 일반화하기 어렵다. [Resolution과 matched prior 검산](resolution53.json)

공유된 flat-state trajectory를 한 번만 세면 663,552개 중 21,101개(3.180%)가 바뀌었다. 같은 deployment random draws로 ultra binary outcome을 다시 채점한 joint·mixed selector·same-menu vote의 session p90은 0.188393, 0.188599, 0.188650이었다. 두 차이의 동시 interval은 −0.00020604 [−0.00029485, −0.00015219]와 −0.00025678 [−0.00035445, −0.00019577]였다. 따라서 **mean 순위는 resolution에 민감하고, joint의 tail 개선은 두 solver resolution에서 유지됐다.** [Ultra session 재실행](ultra_tail53.json)

## 재현 경로

[SCORE_PLAN_KO.md](SCORE_PLAN_KO.md)와 [score53.py](score53.py)는 v53 결과 채점 전에 server에 SHA-256으로 고정했다. 정책 스크립트가 12개 catalog의 blind prefix별 선택을 저장한 후 Slurm compute node에서 score53.py를 실행한다. 스크립트는 기존 4R continuous의 weight와 v52 loss를 수치 허용오차 내에서 복원하는지 검사한다. [전체 수치](score53.json) · [조건부 평균](catalog53.npz) · [두 deployment seed의 session loss](session53.npz) · [별도 sign-only 대조](vote_score53.py) · [Resolution audit](resolution53.py) · [Ultra tail 재현](ultra_tail53.py)
