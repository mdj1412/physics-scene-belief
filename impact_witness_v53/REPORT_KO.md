# v53 · 마지막 reference를 impact 방향 측정으로 바꿀 가치가 있는가?

**네 번째 calibration 물체에서 impact 방향을 읽으면 같은 관측을 받은 post-only simple sign vote보다 continuous joint mixture의 평균 held-out Brier와 session loss p90이 모두 낮았다.** 그러나 교체 전 기록까지 활용하는 v52의 세 행동을 같은 새 관측에 적용하면 결론이 갈렸다. 기본 solver에서 세 행동의 likelihood selector가 **평균 Brier 0.158458**로 continuous joint의 **0.158519**보다 좋았다. 반면 continuous joint의 **session p90은 0.182310**으로 같은 selector의 **0.182443**보다 낮았다. 문자 그대로의 v52 원본(네 쌍 모두 reference)과 비교해도 평균 차이는 −0.0000006으로 불확실했다. 따라서 연구 질문의 두 지표가 강한 대조군 모두에서 동시에 개선됐다는 가설은 지지되지 않았다.

| 교체 후 관측 | 정책 | 평균 held-out Brier ↓ | Session loss p90 ↓ | GPU p90, ms/512개 ↓ |
|---|---|---:|---:|---:|
| 3R+I | Continuous joint | 0.158519 | 0.182310 | 461.54 |
| 3R+I | V52 세 행동 selector | 0.158458 | 0.182443 | 461.68 |
| 3R+I | 같은 세 행동의 sign vote | 0.158489 | 0.182498 | 461.87 |
| 3R+I | Post-only simple sign vote | 0.158606 | 0.182587 | 461.45 |
| 4R | Continuous joint | 0.158526 | 0.182367 | 461.34 |
| 4R | 기존 v52 selector | 0.158520 | 0.182592 | 461.26 |
| 4R | 기존 v52 세 행동의 sign vote | 0.158552 | 0.182656 | 461.41 |

Session loss p90은 물체 128개의 평균 Brier가 큰 쪽 10% 경계다. GPU p90은 같은 장치에서 512개 물체를 처리하는 데 걸린 시간의 90번째 백분위수다. 두 p90은 서로 다른 물리량이다.

## 1. 무엇을 바꾸고 무엇을 고정했는가

Calibration 물체 한 개에서 0.4 N으로 측정한 **weak displacement의 방향이 후속 impact에서 유지되는가**가 관심 대상이다. 그 방향을 알면 다른 배포 물체에서 받은 weak displacement를 어느 쪽으로 보정할지 결정할 수 있다. 물체마다 영향이 달라서 방향을 알고도 특정 배포 물체의 stick/slip 정답이 주어지지는 않는다.

| 단계 | 네 reference 쌍 · 4R | 세 reference 쌍과 한 impact 쌍 · 3R+I |
|---|---|---|
| 교체 전 | 네 weak/reference 쌍 | 같은 측정 |
| 교체 직후 물체 1–3 | 0.4 N weak + 2.0 N reference | 같은 물체와 같은 측정 |
| 교체 직후 물체 4 | 0.4 N weak + 2.0 N reference | 같은 0.4 N weak + impact 순간 변형 방향 |
| 배포 | 별도 128개 물체에서 각각 query-local weak preload 한 번 후 stick probability 예측 | 같은 물체, 관측, decoder와 outcome |
| 교체 후 예산 | 8 contacts, 8 resets | 8 contacts, 8 resets |

첫 번째 독립 변수는 **어떤 측정을 구매했는가**다. 두 번째는 **구매한 측정을 예측에 어떻게 반영하는가**다. Continuous joint는 보정 방향을 유지한 예측과 뒤집은 예측을 posterior mean으로 혼합한다. 같은 정보에 맞춘 v52 selector는 교체 전 보정 유지, 교체 후 보정 재추정, unsigned 중에서 source의 Brier risk로 선택한다. 같은 세 행동의 vote는 displacement 크기를 쓰지 않고 부호 일치만 센다. Post-only simple sign vote는 교체 후 부호만 세어 방향 혼합 비율을 정한다. 원래의 v52 정책도 4R 그대로 재생했다.

혼합 조건의 첫 번째 실험에서 impact 방향은 **simulator가 지정한 impact 상태에서 두 지점의 변형 차이를 읽은 이상적 센서값**이다. 네 번째 reference는 정책에서 가렸다. 독립적인 calibration 물체를 사용했으며 그 표면의 이름, 실제 reversal rate, deployment stick/slip은 입력에서 제외했다. 정책별 산출을 검산했고 사용하지 말아야 할 네 번째 reference 값을 교란해도 mixed policy 출력이 같았다. 표면·물체 support는 v50–52와 공유하므로 이 연구에서 held-out은 기존 모델의 training/deployment 분리를 뜻한다. [실험 계획](PLAN_KO.md) · [통계 계획](SCORE_PLAN_KO.md) · [정책 검산](POLICY_KO.md)

## 2. 같은 관측의 정책 비교와 같은 정책의 관측 비교

아래 그림의 점은 3R+I continuous joint의 loss에서 각 대조의 loss를 뺀 값이다. 왼쪽이 continuous joint에 유리하다. 위 두 행은 두 endpoint와 두 comparator를 함께 보정한 95% interval, 나머지는 탐색적인 개별 95% interval이다. 같은 정보에서 selector를 이기는지, 같은 decoder에서 네 번째 물리 관측을 바꿀 가치가 있는지는 다른 비교다.

![정책 변화와 네 번째 관측 교환에 따른 Brier 차이](comparisons53.png)

**Impact 방향의 추가 가치와 교환 가치가 달랐다.** 같은 세 reference에 네 번째 impact의 부호를 가린 조건보다 평균 Brier가 **0.0001620** 낮아졌다. 그러나 네 번째 reference를 포기하는 비용까지 포함하면 4R continuous 대비 개선은 **0.0000074**였다(개별 95% CI −0.0000125 ~ −0.0000023). 같은 비교의 session p90 차이는 **−0.0000568**(−0.0000851 ~ −0.0000028)이었다. 숨겨 둔 impact sign을 보여 주는 비교만으로는 센서 교체의 이점을 크게 추정하게 된다.

**어느 환경에서 이득이 났는가.** 아래는 drift rate와 초기 retention을 평균한 기본 solver의 held-out Brier다. Strong vote와 selector보다 좋아지는지에 앞서, 방향 정보를 쓰는 것 자체가 이로운 환경인지가 중요하다.

| Reference와 impact의 active sign 불일치율 e | 3R+I continuous | 4R continuous | 3R+I selector | Unsigned |
|---|---:|---:|---:|---:|
| 0 | 0.156213 | 0.156169 | 0.155954 | 0.159297 |
| 0.25 | 0.158667 | 0.158526 | 0.158654 | 0.159297 |
| 0.50 | 0.160678 | 0.160883 | 0.160766 | **0.159297** |

마지막 행에서는 impact 관측으로 바꾸자 잘못된 방향 보정의 손실을 일부 줄였지만 unsigned가 여전히 더 좋았다. 또 e의 prior 평균을 v52에서 가져온 0.10에서 평가 혼합의 평균인 0.25로 바꾸면, 3R+I가 4R보다 **평균 Brier 0.0000210 높아졌다**. Prior의 평균만 일치시킨 민감도 검사이며 true e를 제공한 대조가 아니다. [채점 결과](score53.json)

## 3. 한 번 본 impact로 무엇을 식별했는가

Weak→reference 방향 유지율을 k, reference→impact 반전 확률을 e라 하면 weak→impact 유지율은 q = e + (1−2e)k다. Fourth witness 하나에서 weak와 impact만 읽으면 q에 대해 배울 수 있지만 k와 e를 서로 분리하기 어렵다. 네 번째 물체의 **49.23%**에서 변형 방향이 0이 아니었다. 나머지에서는 방향에 관한 증거를 얻지 못했다. 실제 e를 0, 0.25, 0.50으로 바꿔도 posterior mean e의 평균은 **0.0994, 0.1002, 0.1010**이었다. 반면 q의 posterior mean은 impact sign을 가렸을 때보다 절대값으로 평균 0.0300, 0.0327, 0.0353만큼 움직였다. 한 번의 측정으로 예측을 조정하는 것과 표면의 error rate를 추정하는 일 사이의 격차다. [Calibration 결과](calibration53.json) · [식별성 검산](identifiability53.json)

### 이상적인 부호를 영상으로 읽으면

네 번째 calibration 물체의 family·높이·충돌 상태에 맞춰 **충돌 전/충돌 순간 × 두 marker × 49 grayscale pixels**를 별도로 렌더링했다. 두 marker는 접촉 중심으로부터 각각 1.5 mm 떨어져 있다. 각 영상의 intensity centroid로 두 marker의 움직임 차이를 읽고 ±0.15 mm 안은 무응답으로 처리했다. 광학 배율은 0.1 mm/pixel, 두 시점에 공통인 translation과 노이즈 모델은 모든 정책 비교에서 공유했다. 아래 셋은 동일한 continuous posterior를 사용한다.

| Impact readout | 평균 held-out Brier | Session loss p90 |
|---|---:|---:|
| 이상적 profile sign | 0.158519 | 0.182310 |
| Marker jitter 0.20 pixel | 0.158519 | 0.182310 |
| Marker jitter 0.85 pixel | 0.158558 | 0.182360 |

0.85-pixel 조건에서는 실제 inactive인 관측 중 약 **37.4%**에 방향이 잘못 검출됐다. 그 결과 평균 손실이 원래의 4R continuous(0.158526)보다 높아졌다. 이 sensor는 지정된 충돌 순간과 보이는 marker를 가정한다. 실제 영상에서의 onset 검출, 가림, 재료 내부의 변형까지 해결한 실험은 아니다. 표의 confusion count는 같은 물체·노이즈를 여러 환경 조건에 재사용한 simulation count다. 독립 물체 수와 다르다. 네 session의 synthetic line-scan 판독은 별도 CPU timing에서 p90 **0.0355 ms**였다. [픽셀 판독 코드](optical53.py) · [측정 오차와 손실](optical53.json) · [별도 readout timing](scan_latency53.json)



## 통계 결과와 해상도 검산

### 사전 지정 비교

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

### Impact를 본 가치와 2.0 N reference를 포기한 비용

아래는 secondary contrast다. Interval은 각각 nominal 95%이며 서로 겹치는 대조를 여러 번 수행했다.

| 비교 | 평균 Brier 차이 [95% CI] | Session p90 차이 [95% CI] |
|---|---:|---:|
| Joint, 3R+I − 3R+masked I | −0.00016197 [−0.00016942, −0.00015452] | −0.00014098 [−0.00017965, −0.00009198] |
| Joint, 3R+I − 4R | −0.00000740 [−0.00001251, −0.00000230] | −0.00005676 [−0.00008506, −0.00000277] |
| 세 행동 selector, 3R+I − 4R | −0.00006138 [−0.00007020, −0.00005257] | −0.00014849 [−0.00016708, −0.00006378] |
| 세 행동 sign vote, 3R+I − 4R | −0.00006285 [−0.00007096, −0.00005474] | −0.00015871 [−0.00018323, −0.00008214] |

네 번째 impact sign을 읽으면 가렸을 때보다 예측이 좋아진다. 그러나 같은 접촉·reset 예산에서 네 번째 reference도 유용하다. Signed impact를 가렸을 때를 기준으로 한 평균 이득 0.000162는 실제 probe 교체 이득 0.0000074의 약 22배다. 세 행동 selector는 같은 교체로 평균 Brier를 0.0000614 낮췄다. Paired difference-in-differences는 +0.00005398 [0.00004910, 0.00005886]이었다. 이 design과 고정된 calibration 방식에서는 selector가 교체 효과를 더 크게 회수했다.

같은 evidence와 prior에서 sign-only continuous 대신 magnitude까지 쓰는 joint는 평균 −0.00002457 [−0.00002753, −0.00002160], p90 −0.00003220 [−0.00006758, −0.00001501]이었다. 반대로 sign만 유지하고 세 행동에서 연속 혼합으로 옮기면 평균 +0.00005422, p90 −0.00015496였다. 측정 크기의 효과와 연속 혼합의 tail 이득을 분리할 수 있다. [Sign-only continuous 검산](vote_score53.json)

### 어떤 조건에서 결론이 바뀌는가

- e=0에서는 네 g 모두에서 3R+I continuous의 평균이 4R continuous보다 0.000010–0.000086 높았다. e=0.25에서는 0.000087–0.000209 높았다. e=0.5에서만 0.000169–0.000234 낮았다. 그러나 이 조건에서도 absolute mean은 항상 unsigned인 대조보다 0.001133–0.001574 높았다. 한 번의 직접 관측으로 표면에 대한 신뢰도를 정확히 복원하기는 어렵다.
- e의 prior 평균을 0.25로 바꿔 평가 mixture의 평균과 맞춘 경우, 4R과 3R+I에 동일 prior를 사용했을 때 교체의 평균 차이는 오히려 +0.00002098 [+0.00001776, +0.00002419]였다. p90 차이는 +0.00000793 [−0.00001404, +0.00004146]였다. 이 uniform prior의 평균만 discrete evaluation과 같으며 분포 전체를 아는 oracle은 아니다.
- e=.10으로 고정한 모델과 비교하면 e의 불확실성까지 적분한 mixed joint는 평균이 0.00000661 낮았다. p90 차이는 −0.00000176 [−0.00001374, +0.00001040]이었다. Impact sign에 10% sign reversal을 넣고 sensor model이 이 확률을 알고 있으면 평균 0.158579, p90 0.182346이었다. 이때 clean 4R continuous 대비 평균은 0.000053 높아졌다.
- 48개 환경으로 구성한 fleet를 256회 다시 실현했을 때 mixed joint가 mixed selector보다 실제 fleet 평균에서 진 비율은 66.8%, fleet p90에서 진 비율은 43.4%였다. 같은 세 행동 vote에 대해서는 각각 55.9%, 43.0%였다. 모두 같은 catalog를 쓰는 반복이다. 독립 배포 noise seed에서는 mixed joint의 p90이 두 대조보다 각각 0.0001367, 0.0001881 낮았다.

### Solver 해상도가 평균 순위를 바꿨다

PyBullet time step을 0.0125→0.00625 ms, mesh spacing을 0.025→0.0125 mm로 바꾸고 solver iteration을 80→160으로 늘렸다. 동일한 물체·관측·정책 예측에서 deployment target label만 바뀌었다. Primary에서 joint가 mixed selector와 same-menu vote를 평균에서 이긴 catalog는 각각 0/12, 1/12였다. Ultra에서는 각각 8/12, 12/12였다. Joint의 평균 차이는 ultra에서 **−0.00000878 [−0.00002568, +0.00000811]** 및 **−0.00004208 [−0.00006055, −0.00002361]**였다. 두 비교 모두 12개 catalog 전체에서 contrast가 joint 쪽으로 이동했다. Ultra가 실물 ground truth라는 근거는 없다.

Label table의 884,736개 entry 중 29,069개(3.286%)가 바뀌었다. 28,736개는 stick→slip, 333개는 slip→stick이었다. 평평한 state를 여러 family에서 공유한 table이다. Tip radius 0.25 mm에서 0.282%, 2.0 mm에서 6.290%가 바뀌었다. 이 비대칭 때문에 작은 평균 Brier 격차에 대해 solver 해상도 하나로 정책 순위를 일반화하기 어렵다. [Resolution과 matched prior 검산](resolution53.json)

공유된 flat-state trajectory를 한 번만 세면 663,552개 중 21,101개(3.180%)가 바뀌었다. 같은 deployment random draws로 ultra binary outcome을 다시 채점한 joint·mixed selector·same-menu vote의 session p90은 0.188393, 0.188599, 0.188650이었다. 두 차이의 동시 interval은 −0.00020604 [−0.00029485, −0.00015219]와 −0.00025678 [−0.00035445, −0.00019577]였다. 따라서 **mean 순위는 resolution에 민감하고, joint의 tail 개선은 두 solver resolution에서 유지됐다.** [Ultra session 재실행](ultra_tail53.json)

### 재현 경로

[SCORE_PLAN_KO.md](SCORE_PLAN_KO.md)와 [score53.py](score53.py)는 v53 결과 채점 전에 server에 SHA-256으로 고정했다. 정책 스크립트가 12개 catalog의 blind prefix별 선택을 저장한 후 Slurm compute node에서 score53.py를 실행한다. 스크립트는 기존 4R continuous의 weight와 v52 loss를 수치 허용오차 내에서 복원하는지 검사한다. [전체 수치](score53.json) · [조건부 평균](catalog53.npz) · [두 deployment seed의 session loss](session53.npz) · [별도 sign-only 대조](vote_score53.py) · [Resolution audit](resolution53.py) · [Ultra tail 재현](ultra_tail53.py)


## GPU 측정과 sensing·reset 비용

### 같은 예산에서 어떤 측정이 달라지는가

교체 전 공통 예산은 weak 0.4 N + reference 2.0 N 네 쌍이다. 교체 후 mixed protocol은 처음 세 쌍을 그대로 유지한다. 네 번째 물체에서는 weak 접촉 후 reset하고, 같은 위치에서 impact를 일으켜 변형 방향을 한 번 읽은 뒤 reset한다. 기존 네 번째 reference를 생략한다. 배포 시에는 calibration 물체와 별개인 128개 물체에 query-local weak 접촉을 각각 한 번 쓴다. 교체 후 표의 수량은 한 session 기준이다.

| Post-swap protocol | Weak contacts | Reference contacts | Impact contacts | Contacts / resets | 전체 calibration time |
|---|---:|---:|---:|---:|---:|
| Reference four | 4 | 4 | 0 | 8 / 8 | 5.92 s |
| Three reference + one impact | 4 | 3 | 1 | 8 / 8 | 4.88 s + T |

T는 impact 시행, 방향 판독과 reset까지 마치는 시간이다. 같은 mixed protocol을 받는 continuous joint, 세 행동 selector, sign vote 사이에는 장치 속도와 무관하게 sensing과 reset 차이가 없다. 두 protocol을 비교할 때 **횟수와 소요 시간은 별개의 예산**이다. v52 동작 시간 가정에서 weak 접촉 한 번은 0.44 s, reference는 1.04 s다. 같은 길이 1.04 s의 네 번째 slot을 예약하면 T가 이 안에 들어오는 장치에서만 같은 시간 예산이 성립한다. T가 더 작으면 남는 시간을 대기 시간으로 처리할 수 있다. 이 시뮬레이션은 고속 센서로 부호를 얻는 실제 시간, 에너지, 마모, 표면 교체, specimen handling을 측정하지 않았다. T에 판독 지연과 안정화를 모두 포함해야 한다.

| T 가정 | Mixed session | Reference four와 차이 | Mixed 추가 시간 / 배포 물체 |
|---:|---:|---:|---:|
| 0.52 s | 5.40 s | −0.52 s | 42.19 ms |
| 1.04 s | 5.92 s | 0.00 s | 46.25 ms |
| 2.08 s | 6.96 s | +1.04 s | 54.38 ms |

등시간 가정에서 두 방식은 교체 전후 합계 16 contacts/16 resets, 11.84 s다. 새 calibration의 session 비용과 물체당 0.44 s deployment touch를 합하면 486.25 ms/object다. 교체 전 네 쌍도 이 session에 전부 청구하면 532.50 ms/object다. Impact 방향 판독은 물리 slot에 포함한다. GPU 시간은 물리 slot 종료 후 지불하는 별도 비용이다.

### GPU 측정 경로

같은 서버 GPU에서 12 catalog를 사용한다. 각 timed batch는 calibration 기록이 서로 다른 네 session, 각 session의 배포 물체 128개로 이루어진다. 계수와 GPU decoder는 미리 올리지만, 매 batch마다 old/new calibration의 raw readings부터 posterior, 행동 또는 혼합 계수를 다시 계산한다. GPU에서 factual acquisition과 original physics decoder를 실행하고, 같은 약한 접촉을 해당 정책의 signed mixture 또는 unsigned forecast에 반영한다. 처음과 끝의 CUDA synchronization을 포함한 wall time을 GPU p90으로 보고한다. Host에서 측정하는 posterior 계산 시간도 별도로 기록한다. 물리 slot과 impact camera 또는 센서의 raw image 처리 시간은 GPU timing 밖에 있다. GPU가 받는 것은 이미 판독된 한 개의 ternary gauge sign이다.

각 catalog에서 각 정책의 warmup 한 번 뒤 8-arm Williams crossover를 한 번 완주해 96개 512-object batch를 정책별로 얻는다. 각 위치에 각 정책이 한 번씩, 각 선행–후행 조합도 한 번씩 나타나도록 순서를 균형화한다. Catalog를 단위로 16,000회 paired bootstrap하여 marginal p90 차이의 interval을 계산한다. Mixed continuous joint에서 mixed selector, 같은 세 행동의 vote, post-only vote를 각각 뺀 세 비교는 Bonferroni 동시 95% interval을 쓴다.

정확한 측정 경로를 확인하기 위해, 매 batch의 실제 GPU 출력을 보관된 independent-object forecast와 단일 session씩 재계산한 policy weights로 조합한 예측에 대조한다. Mixed 정책을 호출할 때 숨겨진 네 번째 reference 값을 +999 mm 교란한 후 action과 mixture weight가 그대로인지 검사한다.

별도의 optical diagnostic에서는 실제로 렌더링한 네 session의 uint8 line scan(각각 2시점 × 2마커 × 49픽셀)을 입력했다. Server CPU의 centroid와 differential-displacement 방향 판독은 8,640개 four-session batch에서 median **0.02751 ms**, p90 **0.03549 ms**였다. Trigger, exposure, sensor readout과 hardware reset은 포함하지 않았다. 이 microbenchmark와 GPU 측정의 p90을 더한 값은 end-to-end p90을 보장하지 않는다. [Line-scan 측정 코드](scan_latency53.py) · [수치](scan_latency53.json)

예를 들어 T=2.08 s이고 128개 물체에 비용을 분담하면 네 번째 reference 대신 impact를 쓰는 추가 비용은 0.008125 s/object다. 평균 Brier를 목적함수로 쓸 때, continuous joint의 sensor 교체 이득 0.00000740은 대기 시간 1 s를 Brier 0.00091 이하로 평가할 때에만 이 시간 비용을 상쇄한다. 세 행동 selector의 sensor 교체 이득 0.00006138을 같은 방식으로 나누면 0.00755 Brier/s다. 이는 지정한 T와 본 catalog 평균의 조건부 환산이며, 실제 생산 비용은 별도 측정해야 한다.

### GPU p90 결과

| 관측·정책 | GPU p90 ms/512 [pointwise 95% CI] | Cold posterior host p90 ms/4 sessions |
|---|---:|---:|
| 3R+I · continuous likelihood | 461.54 [460.96, 464.24] | 0.14959 |
| 3R+I · v52-style selector | 461.68 [461.16, 464.75] | 0.16734 |
| 3R+I · same-action sign vote | 461.87 [461.07, 464.74] | 0.12289 |
| 3R+I · post-only simple vote | 461.45 [460.41, 464.73] | 0.02119 |
| 3R+I · impact sign masked | 461.38 [460.50, 463.55] | 0.13690 |
| 4R · continuous joint | 461.34 [460.72, 463.57] | 0.13652 |
| 4R · frozen v52 selector | 461.26 [460.91, 464.13] | 0.10829 |
| 4R · frozen vote selector | 461.41 [461.00, 464.25] | 0.06766 |

같은 3R+I continuous joint에서 selector·same-action vote·post-only vote를 뺀 p90 차이는 차례로 **-0.14 ms** [-1.07, +1.70], **-0.32 ms** [-1.55, +0.42], **+0.09 ms** [-1.95, +0.94]였다(동시 95% CI). 세 차이 모두 zero를 포함한다. 이 batch와 장치에서 빠르다고 단정할 수 없다.

Node56의 Slurm allocation 2744471, NVIDIA RTX PRO 6000 Blackwell Server Edition, Torch 2.7.1+cu128, CUDA 12.8에서 실행했다. 첫 검사에서 메모리 0 MiB, 이용률 0%, 다른 compute process가 없었고 guard는 foreign process 없이 정상 종료했다. 모든 방법의 maximum replay error는 4.44×10⁻¹⁶이었다. [측정 결과](timing53.json) · [각 batch의 raw timing](timing53) · [GPU guard 기록](timing_guard53/results/worker_exit.json)

### 보조 대조 · continuous blending을 유지한 sign count

같은 GPU에서 12 catalog마다 두 정책을 역순으로 교차해 각각 48회 측정했다. 3R+I continuous likelihood 465.70 ms와 continuous sign-count mixture 465.40 ms를 비교한 차이는 +0.30 ms [95% paired catalog bootstrap -1.37, +1.37]였다. Raw witness부터 같은 인코딩·decoder·최종 혼합까지 실행했고 sign-count weights는 평가 캐시와 일치했다. 이 두 수치는 같은 보조 run 안에서 비교하며, 위 8-arm run의 시계값과 직접 차감하지 않는다. [보조 GPU 수치](vote_timing53.json) · [보조 raw timing](vote_timing53.npz)


## 논문에 남길 주장

네 쌍의 예산에서는 **비용을 들여 얻은 impact 방향의 가치가 어느 관측을 포기했는지에 따라 달라지고, 혼합 정책의 순위는 평균과 session tail에서 갈린다**는 것이 가장 직접적인 기여다. 두 단계의 분리가 중요하다. 같은 posterior에서 source Brier를 최소화하는 연속 가중치와 unsigned fallback을 허용하니, 사후 진단에서 평균은 0.158470으로 내려갔지만 session p90은 0.182391로 올라갔다. 관측이 얼마나 유익한지와 그 정보를 마지막 예측에 어떻게 쓰는지는 서로 다른 원인이다. [사후 decision rule 검산](action_risk53.json)

이 증거만으로 네 번째 probe를 일괄 교체할 이유는 약하다. 다음 확인 실험의 주요 비교는 세 reference가 주어진 시점에서 네 번째 물체의 impact를 볼지 네 번째 reference를 볼지를 사전에 정한 정책과 fixed 4R이다. Independent surface/day holdout에서 실제 acquisition·reset 시간을 지불하고, 영상의 무응답과 false activation을 기록하며, 첫 세 probe만 보는 정책까지 포함해야 한다. 결과를 채점할 때는 평균과 session p90 중 무엇을 개선해야 하는지와 실용적 최소 효과를 사전에 정한다.



## 문헌과 논문 기여의 범위

이번 검토의 질문은 “같은 네 쌍을 쓸 때, 마지막 reference를 impact 관측으로 교환할 이유가 있는가?”다. 아래 네 원문을 선택했다. 첫 연구는 정보의 가치를 정의하고, 다음 세 연구는 실제 충돌과 접촉을 어떻게 측정하고 검증했는지 보여준다. 이 범위의 검토로 분야 전체에서 최초라는 주장을 뒷받침할 수는 없다.

### 1. 원 논문 비교

| 원 논문 | 질문·방법 | 표본과 평가 단위 | 주요 결과·결론 | v53에 적용할 때의 한계 |
|---|---|---|---|---|
| Goel & DeGroot, *Comparison of Experiments and Information Measures* | 고정한 prior에서 서로 다른 관측 실험의 정보량을 어떻게 비교하는가? Expected Bayes risk로 실험 순서를 정의하고 분포족에 대해 조건을 유도했다. | 수리적 분석. 실물 표본과 센서 시간은 없다. | 어느 실험이 모든 decision problem에서 더 유익하다는 주장은 특정 loss에서의 승리보다 강하다. | v53은 하나의 예측 문제, 특정 prior, 두 summary에서의 교환 가치만 측정한다. [원문](https://doi.org/10.1214/aos/1176344790) |
| Pfrommer et al., *ContactNets* | 관측된 자세에서 discontinuous contact dynamics를 학습할 수 있는가? Signed distance와 contact-frame Jacobian을 학습해 rollout했다. | 10 cm acrylic cube, 목재 표면, 148 Hz 카메라 세 대. 750회 중 고품질 toss 570회. | Contact geometry를 구조화한 모델은 적은 training toss에서도 end-to-end 모델과 비교해 특히 회전 예측이 좋았다. | Camera가 추적한 rigid pose와 v53이 요구하는 국소 탄성 변형 부호는 관측량이 다르다. [원문](https://proceedings.mlr.press/v155/pfrommer21a/pfrommer21a.pdf) |
| Acosta et al., *Validating Robotics Simulators on Real-World Impacts* | 실물 trajectory를 기준으로 solver와 contact parameter의 적합도를 비교했다. | Cube toss 550개, Cassie jump 22개. Cube는 세 solver, Cassie는 두 solver를 비교했다. | 비탄성 충돌과 큰 움직임은 재현해도 restitution과 일부 edge/corner impact에서 오차가 남았다. Cassie trajectory는 마찰 계수를 식별할 slip을 충분히 만들지 않았다. | Solver를 정교하게 하거나 시행 수를 늘려도 필요한 contact regime이 빠질 수 있다. 앞선 cube 계열 자료와 수집 계보가 겹쳐 독립적인 재질 반복으로 세기 어렵다. [원문](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) |
| Dong et al., *Maintaining Grasps within Slipping Bound by Monitoring Incipient Slip* | GelSlim의 displacement field와 rigid motion의 차이로 incipient slip을 검출했다. | 일상 물체 10개 각 24회, 총 240회. 수동 push/rotation, 사후 사람 판독. | 평균 검출 정확도 86.25%. 접촉 신호가 약한 두 물체에 false negative가 집중됐다. | Grasp slip 검출은 짧은 impact에서의 변형 부호 판독과 다르다. 이 정확도를 v53 sensor의 noise rate로 옮길 수 없다. [원문](https://arxiv.org/pdf/1810.13381) |

### 2–3. 증거의 흐름과 지식 지도

관측 예산 → 가능한 두 실험 → 각 실험이 만드는 posterior → 같은 downstream decision의 손실, 이 연결은 Goel & DeGroot의 risk 관점에서 나온다. V53에 적용하면 세 reference에 impact를 **더하는** 효과와 네 번째 reference를 **교체하는** 효과에 각각 다른 comparator가 필요하다. [Goel & DeGroot](https://doi.org/10.1214/aos/1176344790)

측정 가능한 위치·자세 → contact model → 미래 trajectory는 ContactNets가 실물에서 다룬 경로다. 이에 비해 v53의 경로는 weak displacement → impact 순간의 국소 변형 방향 → 다른 물체들의 stick probability다. 두 중간 변수가 실제로 관측되는지는 별도의 문제다. 힘을 가한 상태에서 surface motion을 직접 측정한 Dong 등의 결과와 재료·충돌 상태에 따른 solver 오차를 측정한 Acosta 등의 결과가 이 간극을 드러낸다. [ContactNets](https://proceedings.mlr.press/v155/pfrommer21a/pfrommer21a.pdf) · [Dong et al.](https://arxiv.org/pdf/1810.13381) · [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf)

### 4–7. 연구 공백, 상충해 보이는 결과와 증거 강도

- **강한 근거**. 구조화된 contact model로 같은 cube의 일부 trajectory를 잘 예측할 수 있고, 일부 탄성·모서리 충돌은 계속 어렵다. 두 결과는 평가 지표, 충돌 조건, 비교 모델이 다르다. 평균 회전 오차가 줄었다는 결과는 모든 contact regime의 국소 변형을 맞힌다는 뜻으로 확대하기 어렵다. [ContactNets](https://proceedings.mlr.press/v155/pfrommer21a/pfrommer21a.pdf) · [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf)
- **제한적인 근거**. 접촉 신호의 세기에 따라 방향 관련 측정의 활용도가 달라질 수 있다. GelSlim 실험은 물체·하중에 따른 검출 차이를 보였지만 저하 원인을 실험 모든 물체에서 하나로 식별하지는 않았다. V53의 weak-to-impact 조건에 대한 물리적 인과 효과는 독립 측정이 필요하다. [Dong et al.](https://arxiv.org/pdf/1810.13381)
- **근거 부족**. 네 논문은 같은 교체 후 session에서 weak/reference와 weak/impact를 맞교환하고 다른 128개 물체의 held-out Brier 분포를 비교하지 않는다. 특히 calibration 대상을 희생하면서 생긴 surface wear와 impact 방향 readout 오류의 공동분포는 남은 공백이다. [Dong et al.](https://arxiv.org/pdf/1810.13381) · [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf)

### 8. 판별 가능한 후속 연구 질문 열 가지

| 질문 | 공백과 중요한 이유 | 독립 변수와 검증 방법 | 근거 |
|---|---|---|---|
| 1. 서로 다른 센서가 같은 impact 방향을 읽는가? | 단일 센서 오류가 정답에 섞일 수 있다. | 고속 측면 영상과 tactile field의 blind annotation을 active 물체에서 비교한다. | [Dong](https://arxiv.org/pdf/1810.13381) |
| 2. 방향 판독이 유효한 최소 exposure는 얼마인가? | 빠른 충돌은 pose track 사이에서 끝날 수 있다. | 같은 고속 영상을 시간적으로 downsample하고 부호와 무응답률을 계산한다. | [ContactNets](https://proceedings.mlr.press/v155/pfrommer21a/pfrommer21a.pdf) |
| 3. Inactive 물체를 피하도록 네 번째 대상을 고를 수 있는가? | 변형이 없으면 직접 관측도 방향을 주지 않는다. | 기존 세 쌍의 정보와 구입 전 영상만 이용해 사전 선택하고 무작위 선택과 비교한다. | [Dong](https://arxiv.org/pdf/1810.13381) |
| 4. Reference 오류가 session 안에서 군집하는가? | 공동 오차는 128개 예측의 tail을 바꾼다. | 날짜·표면별 반복을 모아 independent와 hierarchical noise를 비교한다. | [Acosta](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) |
| 5. Impact를 가한 이력이 뒤의 마찰을 바꾸는가? | 측정 자체가 이후의 contact law를 바꿀 수 있다. | 새 표면을 짝짓고 impact와 reference 순서를 무작위화한 뒤 wear·slip을 기록한다. | [Acosta](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) |
| 6. Query에서 떨어진 위치에서도 방향을 옮길 수 있는가? | 하나의 위치가 전체 물체를 대표하지 않을 수 있다. | Probe offset과 query height를 교차하고 새 표면에서 Brier를 비교한다. | [ContactNets](https://proceedings.mlr.press/v155/pfrommer21a/pfrommer21a.pdf) |
| 7. 약한 negative evidence를 무응답으로 처리하면 tail이 줄어드는가? | 낮은 contrast에서 false sign의 비용이 크다. | Source에서 abstention을 고정하고 held-out object에서 coverage와 p90을 함께 보고한다. | [Dong](https://arxiv.org/pdf/1810.13381) |
| 8. Reference–impact prior의 폭을 실물에서도 고정할 수 있는가? | 고정 prior에서의 실험 순위가 다른 배포 혼합에서 달라질 수 있다. | Source surface로 prior를 정하고 untouched surface/date holdout에서 두 probe를 다시 짝짓는다. | [Goel & DeGroot](https://doi.org/10.1214/aos/1176344790) |
| 9. 실제 reset 시간으로도 교체가 이득인가? | 같은 횟수의 contact가 같은 시간이나 손상을 보장하지 않는다. | Reset 실패·교체 포함 wall time으로 예산을 맞추고 전체 cycle의 gain을 측정한다. | [Goel & DeGroot](https://doi.org/10.1214/aos/1176344790) |
| 10. 더 강한 contact decoder에도 효과가 남는가? | 단순 decoder의 representation error가 보정 이득을 만들 수 있다. | 구조화한 learned decoder와 원 decoder에 같은 calibration·split을 교차 적용한다. | [ContactNets](https://proceedings.mlr.press/v155/pfrommer21a/pfrommer21a.pdf) |

### 9. 심사 관점

Bayesian averaging 자체를 새로운 기법으로 제시하기는 어렵다. 이 실험의 기여 후보는 *동일한 두 번의 접촉으로 reference 또는 impact 중 하나만 관측할 때, 그 선택이 평균 및 나쁜 session의 손실을 어떻게 다르게 바꾸는가*라는 검증 가능한 질문이다. 외부 타당성을 판단하려면 서로 다른 날과 미사용 surface specimen, 독립적인 고속 impact 측정, 실제 reset 시간이 필요하다. 특히 cube pose 데이터만으로는 local deformation sign이나 동일 specimen의 weak/reference 반응을 복원할 수 없다. [Goel & DeGroot](https://doi.org/10.1214/aos/1176344790) · [ContactNets](https://proceedings.mlr.press/v155/pfrommer21a/pfrommer21a.pdf) · [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf)
