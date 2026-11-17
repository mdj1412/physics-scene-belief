# v51 · 표면을 교체한 뒤에도 네 쌍의 calibration을 믿을 수 있는가?

**두 오류가 각각 10%일 때 네 쌍을 사용한 query-local soft의 held-out Brier loss는 0.157858로, simple sign vote(0.157898)와 unsigned policy(0.159297)를 모두 앞섰다. 두 오류가 각각 25%가 되면 soft는 0.159994로 vote(0.160017)보다 조금 좋았지만 unsigned에 졌다.** 여기서 첫 오류는 2.0 N reference와 impact 사이에서 각 물체의 변형 부호가 바뀌는 사건이고, 둘째는 표면 교체 후 한 session 전체에서 하중 간 방향 관계가 반대로 바뀌는 사건이다.

**두 사건은 평균 손실에는 같은 효과를 낼 수 있지만 배포 위험은 다르다.** 물체별 오류만 25%일 때와 session drift만 25%일 때의 기대 Brier는 같았다. 반면 48개 session을 한 번씩 운영하는 모의 배포에서 soft가 unsigned에 진 비율은 각각 **12%와 28%**였다. 논문으로 발전시킬 부분은 작은 vote 대비 개선의 크기보다, 어떤 물리적 실패가 여러 물체에 동시에 전파되는지 분리한 결과다.

## 1. 두 오류를 어떻게 나눴나

| 구분 | 조작과 관측 |
|---|---|
| Calibration | 교체 전 surface에서 서로 다른 물체 4개를 사용한다. 각 물체의 같은 위치에서 0.4 N weak와 2.0 N reference의 signed differential displacement를 각각 읽고 매번 reset한다. Impact label을 calibration에 제공하지 않는다. |
| 실제 surface pairing | Mosaic에서 calibration하면 nap에 배포하고 nap에서 calibration하면 mosaic에 배포한다. 같은 catalog와 rate law를 유지하고 surface별 friction field와 물체는 교체한다. 모든 primary cell에 이 교체가 들어간다. g=0에서도 surface 자체는 바뀐다. |
| 교체 전 방향 유지율 κ | 교체 전 active 물체에서 weak와 2.0 N reference의 부호가 같을 확률이다. 0, 0.25, 0.5, 0.75, 1의 다섯 조건을 균등 가중한다. Inert 상태는 유지한다. |
| Session drift 확률 g | 표면 교체 후 확률 g로 해당 session의 유지율이 κ에서 1−κ로 바뀌고, 나머지는 κ를 유지한다. 한 session의 128개 물체가 같은 drift event를 공유한다. |
| Reference–impact 불일치율 e | 새 표면에서 각 active 물체의 2.0 N reference 부호가 impact에서 독립적으로 뒤집힐 확률이다. Impact의 signed contour에 해당하는 PyBullet 충돌 결과로 stick/slip을 채점한다. |
| 실험 단위 | e와 g를 각각 0, 0.10, 0.25, 0.50, 0.75, 1로 바꿔 36개 조건을 만들었다. 12개 catalog, 48개 환경, 6,144개 held-out deployment 물체를 사용했다. Calibration에는 별도 support의 물체만 썼다. |

**Impact는 독립 contact solver에서의 충돌이다.** v50의 held-out 물체·두 surface·두 해상도의 conditional PyBullet trajectory를 재사용해 solver와 물체가 달라지면서 생기는 차이를 통제했다. 각 조건에서 최종 변형 방향을 정한 뒤 그 물체·형상·접촉 조건에서 계산해 둔 impact outcome을 선택한다. 같은 episode의 reference 부호와 impact 부호를 구분했으며, e를 바꾼다고 stick/slip label 자체를 무작위로 뒤집지는 않았다. 예를 들어 e=25%, g=0에서 active 물체의 실현 불일치율은 24.91%였고 전체 binary impact label이 바뀐 비율은 **4.56%**였다. e와 g는 재료에서 측정한 값이 아니라 이 시험에서 지정한 값이다. [설계와 검산](WORLD_KO.md) · [Intervention 구현](world51.py) · [사전 계획](PLAN_KO.md)

## 2. 같은 budget에서 무엇을 비교했나

- **Likelihood soft.** 네 weak/reference 쌍의 측정 크기와 부호, 동결된 sensor likelihood를 이용해 κ의 posterior mean α를 구한다. 배포 물체에서는 query 위치에 weak preload를 한 번 가해 d를 얻고, frozen direct의 두 출력 p(d)와 p(−d)를 α·p(d)+(1−α)·p(−d)로 섞는다.
- **Simple sign vote soft.** 같은 네 쌍과 같은 dead zone을 사용한다. 읽을 수 있는 쌍이 n개이고 그중 부호가 같은 쌍이 s개일 때 α=(1+s)/(2+n)로 **동일한 두 확률**을 섞는다. 동률일 때 원래 부호로 강제하는 hard majority보다 강한 대조다. Hard vote도 보조 비교에 포함했다.
- **Unsigned fit.** Query 위치에서 동일한 d를 읽고 source에서 학습한 p와 |d|에 대한 보정식을 적용한다. Target label을 쓰지 않는다. 허용된 네 calibration 쌍은 생략해도 예측이 동일하다. 실제 배포에서 아낄 수 있는 비용은 §6에서 계산했다.

측정 budget과 비교 조건을 동일하게 하기 위해 각 정책은 같은 noisy prefix와 같은 deployment readout을 받는다. 세 정책은 e·g·실제 κ·발생한 drift event를 볼 수 없다. 추가로 오류율을 알려준 진단과 사건 자체를 알려준 진단을 별도로 채점했다. [Frozen policy와 GPU 실행](bench51.py) · [채점 코드](score51.py)

## 3. Held-out Brier loss가 낮아지는 조건

| e · reference 불일치 | g · session drift | Likelihood soft | Vote soft | Unsigned fit | 두 대조에 동시 우위 |
|---:|---:|---:|---:|---:|---|
| 0% | 0% | 0.155886 | 0.155942 | 0.159297 | 확인 |
| 10% | 10% | 0.157858 | 0.157898 | 0.159297 | 확인 |
| 25% | 0% | 0.158625 | 0.158658 | 0.159297 | 확인 |
| 0% | 25% | 0.158625 | 0.158658 | 0.159297 | 확인 |
| 25% | 10% | 0.159172 | 0.159202 | 0.159297 | unsigned 대비 불확실 |
| 25% | 25% | 0.159994 | 0.160017 | 0.159297 | unsigned 우위 |
| 50% | 0% | 0.161363 | 0.161375 | 0.159297 | unsigned 우위 |
| 100% | 0% | 0.166840 | 0.166808 | 0.159297 | vote와 unsigned 우위 |
| 100% | 100% | 0.155886 | 0.155942 | 0.159297 | 확인 · 두 반전이 상쇄 |

각 catalog 안에서 5개 κ·4개 환경·calibration prefix 반복·geometry와 noise를 먼저 평균했다. **36개 cell × 2개 primary contrast**에 paired catalog Student t와 Bonferroni를 적용해 두 simultaneous 95% upper bound가 모두 0 미만일 때만 우위로 판정했다. 같은 12개 catalog와 관측을 여러 cell에 재사용하므로 36번의 독립 재현으로 해석할 수 없다.

- **동시에 이긴 9개 cell.** (e,g)=(0,0), (0,.10), (.10,0), (.10,.10), (0,.25), (.25,0), (.75,1), (1,.75), (1,1)이다. 낮은 오류 영역에서 6개, 두 반전이 상쇄되는 영역에서 3개였다. e=g=0에서 soft−vote는 **−0.0000560 [−0.0000667, −0.0000454]**, soft−unsigned는 **−0.003411 [−0.003614, −0.003208]**이었다. 측정 크기를 반영한 estimator의 추가 효과는 같은 측정의 부호를 전부 버렸을 때와의 차이보다 작다.
- **우위가 사라지는 경계.** e=.25, g=.10의 soft−unsigned는 −0.000125 [−0.000254, +0.000005]였다. e=g=.25에서는 +0.000697 [+0.000565, +0.000829]로 손실이 확실히 늘었다. 이때도 soft−vote는 −0.000023 [−0.000027, −0.000018]이었다. **Vote를 이기는 것만으로 유료 calibration이 정당화되지는 않는다.**
- **교체 전 κ도 중요하다.** 두 오류가 모두 0이어도 κ=.5인 환경에서는 soft 0.160743, vote 0.160759, unsigned 0.159297이었다. 애초에 weak 부호와 reference 부호의 관계가 무작위라면, 네 쌍에서 보이는 우연한 방향 차이를 미래 물체에 옮기는 것이 손해다. 이 표의 경계는 다섯 κ를 균등 가중한 결과다. [36개 cell·κ별 원자료 및 CI](score51.json) · [Catalog별 risk](catalog51.npz)

![두 대조 정책과의 Brier 차이 및 평균이 같은 두 조건의 session별 손실 분포](reference_drift_v51.png)

그림 A·B의 흰 원은 두 대조에 동시에 이긴 조건이다. 파란색일수록 soft의 손실이 작다. Vote와의 차이가 더 작아서 A와 B의 척도가 다르다. 그림 C의 가로축은 soft−unsigned이며, 오른쪽 꼬리는 한 session에서 soft가 크게 손해를 본 경우를 나타낸다.

## 4. 평균 Brier로 두 원인을 구별할 수 없는 이유

교체 전 active 방향 유지율이 κ이면, 배포 때 weak 부호가 impact까지 유지될 주변확률은 **0.5+(κ−0.5)(1−2e)(1−2g)**다. 두 단계 중 정확히 하나만 반전될 확률을 **h=e+g−2eg**로 놓으면, 현재 frozen 정책의 기대 Brier는 h에 선형이다. 이는 κ가 1일 때를 제외하면 h가 weak–impact 불일치율 자체라는 뜻은 아니다. e와 g의 자리를 바꿔도 평균 순위는 같지만, 한 표면에서 연달아 예측할 때 두 물체의 오류가 함께 발생할 가능성은 달라진다.

Primary 평균 Brier의 교차점은 unsigned 대비 **h=0.311**(12개 catalog bootstrap의 exploratory 95% 구간 0.307–0.316), vote 대비 **h=0.631**(0.593–0.680)이었다. 따라서 더 엄격한 점추정 경계는 unsigned와의 0.311이다. 이는 관측 뒤 도출한 연속 교차점이며 사전 지정된 36개 cell의 동시 유의 경계와 구별된다. 예를 들어 e와 g가 모두 25%이면 h=0.375이고, 어느 한쪽만 25%이면 h=0.25다. [교차점과 수치 검산](score51.json)

**평균이 같은 경우에도 손해가 어느 session에 몰리는지 확인했다.** Calibration prefix 하나를 뽑고 128개 물체마다 실제 binary simulator outcome 하나를 쓰는 48-session 배포를 250회 반복했다. 아래 p90은 *session의 Brier 차이*에 대한 값이다. GPU p90과 단위가 다르다.

| Soft−unsigned의 배포 지표 | 개별 물체 오류만 e=.25 | 교체 후 session drift만 g=.25 |
|---|---:|---:|
| Exact expected Brier 차이 | −0.000672 | −0.000672 |
| Rollout 평균 차이 | −0.000689 | −0.000744 |
| 같은 catalog·환경·κ에서 반복한 session 차이의 표준편차¹ | 0.003397 | 0.007512 |
| 전체 session 차이의 p90 | +0.003739 | +0.009363 |
| 48개 session 평균에서 soft가 unsigned에 진 배포 비율 | 12% | 28% |

¹ 각 catalog·환경·κ 조합 안에서 반복한 손실 차이의 표준편차를 계산해 평균했다. 표준편차 증가분은 +0.004115 [paired catalog bootstrap 95% CI +0.003891, +0.004316]이었다. Active 물체 둘의 weak–impact 일치 지표의 공분산은 이 모형에서 g(1−g)(1−2e)²(2κ−1)²다. κ=1에서 두 열의 이론값은 0과 0.1875다. 데이터가 같은 평균만 보여도 개별 물체에서 나는 오류와 표면 전체에 공통인 오류를 구별하려면 여러 독립 session의 반복이 필요하다. [Rollout 검산](rollout51.json) · [공분산 진단](covariance51.json)

오류가 전혀 없는 조건에서도 250개 배포의 **39.6%에서 단순 vote의 평균 Brier가 soft보다 낮았다.** 첫 48-session 배포의 soft−vote는 +0.000030 [−0.000045, +0.000106]이었다. Catalog와 prefix를 평균한 기대 이득을 한 번의 배포 성능으로 약속하기 어려운 크기다. [Realized rollout 상세](WORLD_KO.md)

## 5. 대안 설명과 민감도

- **표면 교체와 drift를 구별했다.** 실제로 다른 surface에서 측정한 calibration을 전달했으며, e=g=0의 반대 표면 실험은 0.155886이었다. 원래 표면을 그대로 쓴 v50 null control도 원래 점수까지 재현했다. 이 합성 자료에서는 surface 이름이 달라도 부호 관계가 같으면 transfer가 잘된다. 두 surface마다 미지의 물리 drift를 데이터에서 추정했다는 뜻은 아니다. [Same-surface replay](replay51.json)
- **반전 대신 무관한 표면을 받는 경우.** Drift event에서 유지율을 1−κ로 정하는 대신, 교체 전 κ와 독립적으로 다섯 값 중 하나를 균등 추출했다. g=e=1일 때 원래 실험의 soft 0.155886은 이 control에서 **0.161363**이 되어 unsigned 0.159297에 졌다. 손익 지도 오른쪽 위의 회복은 두 반전이 상쇄된 결과이고 임의 표면 교체에 일반화되지 않는다. [독립 재추출 control](controls51.json)
- **Calibration 표본과 solver 해상도.** 같은 catalog의 8개 후보 중 처음 네 물체 대신 겹치지 않는 뒤 네 물체만 쓰면 두 대조 동시 우위는 9개에서 11개 cell로 바뀌었다. 처음 네 쌍을 유지하고 time step과 mesh spacing을 절반으로 줄인 ultra solver에서도 11개였다. Ultra에서 unsigned 교차점은 h=0.330 [0.325, 0.334]였다. 두 오류를 0.25와 0.10으로 둔 경계 두 cell의 판정이 해상도에 민감했다. CPU로 독립 집계한 primary와 GPU 집계 사이 최대 Brier 차이는 1.67×10⁻¹⁶이었다. [새 네 쌍과 CPU 검산](controls51.json) · [Ultra solver](score51_ultra.json)
- **오류율을 알고 있다면.** 알려준 e·g로 α를 h+(1−2h)α로 변환한 privileged diagnostic은 e=1,g=0에서 0.155884까지 회복했다. 같은 조건의 frozen soft는 0.166840이었다. 반대로 e=.5이거나 g=.5이면 이 진단은 부호를 대칭 평균해 0.159637이며, 별도로 학습한 unsigned fit은 0.159297이었다. e·g를 알려주는 진단에는 배포 정책에 없는 정보를 썼다. [진단별 loss](score51.json)

## 6. GPU p90과 sensing·reset 비용

서버의 네 할당 중 다른 process가 약 40 GB씩 쓰는 두 GPU는 그대로 두었다. 빈 node55 GPU에서 두 해상도의 held-out Brier를 집계하고, 빈 node56의 RTX PRO 6000 Blackwell에서 4개 정책을 같은 batch로 실행했다. 각 batch는 네 환경의 물체 512개이고, calibration estimator가 필요한 정책은 환경별 네 쌍의 실제 저장된 noisy 측정값으로 그때마다 다시 계산했다. 실행 순서와 직전 정책을 균형화한 Williams crossover에서 정책당 96개 timed batch를 얻었다. Offline physics·로봇 움직임을 제외한 factual acquisition·구매한 sensor 데이터 처리·original decoder·최종 보정·synchronization이 측정 범위다.

| 정책 | GPU p90, ms/512 objects [pointwise 95% CI] |
|---|---:|
| Likelihood soft | 465.59 [464.08, 467.03] |
| Simple sign vote soft | 466.34 [464.31, 466.88] |
| Unsigned fit | 466.47 [464.66, 467.54] |
| Hard vote | 465.50 [464.14, 466.98] |

Soft−vote는 **−0.75 ms [−1.69, +0.70]**, soft−unsigned는 **−0.88 ms [−1.61, +0.20]**였다(두 primary timing 비교에 대한 simultaneous 95% CI). GPU 속도 우위는 확인되지 않았다. 네 환경의 estimator만 따로 잰 host p90은 likelihood 0.04548 ms, vote 0.00833 ms였다. e와 g는 예측 때 숨겨지므로 각 cell의 오류 발생으로 GPU 계산을 추가 실행하지 않는다. [Timing 숫자](timing51.json) · [실행·검산 상세](GPU_KO.md)

**모든 정책의 배포 preload는 1 contact·1 reset이다.** Likelihood와 vote는 환경당 네 calibration 물체에 0.4 N weak와 2.0 N reference를 각각 한 번씩 가하므로 **추가 8 contacts·8 resets**가 든다. Unsigned는 같은 budget을 제공해 비교했지만 실배포에서는 여덟 번을 모두 생략할 수 있다. 128개 물체가 이 calibration을 공유하면 물체당 추가 비용은 각각 0.0625 contact·reset이다.

이전 실험과 동일하게 접근·측정·후퇴·안정화 100+50+100+40 ms, loading 4 N/s, unloading 8 N/s로 가정했다. Weak 440 ms, reference 1,040 ms이므로 네 쌍은 **5.92초/session**, 128개에 나누면 **46.25 ms/object**다. 공통 deployment touch까지 합치면 soft와 vote는 486.25 ms/object, calibration을 생략한 unsigned는 440 ms/object다. 물체·표면 교체, 실제 reset 실패와 잔류 변형은 미계측이다. 교체 시간은 세 정책에 공통이며 포함하지 않았다. e=g=.10에서 soft가 unsigned보다 얻는 0.001439 Brier를 위 시간으로 나누면 1초당 약 **0.0311 Brier-equivalent**가 손익분기점이다. e=g=.25에서는 물리 시간에 가격을 매기기 전부터 unsigned의 loss가 낮다.

## 7. 논문 기여 가능성과 다음 검증

**현재 지지되는 기여**는 reference의 신뢰도 저하와 session drift를 별도 수준에서 조작하고, 같은 기대 Brier가 서로 다른 session 손실 분포를 가릴 수 있음을 관측 비용과 함께 보였다는 것이다. Soft vote까지 동일 조건에서 비교했기 때문에 likelihood에서 측정 크기를 쓰는 추가 이득과, calibration 자체에서 얻는 방향 정보의 가치를 구분할 수 있다. 이 likelihood 계산 방식이나 31%라는 임계값 자체의 보편성은 근거가 약하다.

**가장 필요한 다음 검증은 두 오류를 실물에서 따로 재는 것이다.** 같은 물체·위치에서 2.0 N reference의 marker 이동과 별도 high-speed impact 영상의 변형 방향을 독립적으로 판독해 e를 추정한다. 교체 전·후 여러 물체에서 weak/reference 쌍을 얻고 이 교체를 다른 날과 surface specimen으로 반복해 session별 κ 분포와 잔류 변형을 측정한다. 기기 좌표 반전, near-zero 판독, 마모, reset 순서는 따로 기록한다. 사전 고정할 대조는 같은 budget의 simple vote와 calibration을 생략하는 unsigned다. 평가에는 평균 Brier와 session loss p90을 함께 둔다. 실물에서 관측한 drift가 전면 반전보다 독립 재추출에 가깝다면 §5의 대안 모형이 비교 기준이 된다.

## 선행연구와 독립 검토

검토할 질문은 *교체 전에 같은 물체에서 약한 하중과 2.0 N 하중을 짝지어 관찰하면, 교체 뒤 다른 물체가 충돌할 때의 stick 확률을 얼마나 잘 예측하는가*다. 두 하중에서 기록한 변형 부호, 충돌 중 변형 부호, 최종 stick/slip label은 서로 다른 측정 대상이다. 아래 여섯 편의 원문에서 관련된 부분을 비교했다.

| 원논문 | 질문·방법 | 표본·데이터 | 결과·결론 | v51로 일반화할 때의 한계 |
|---|---|---|---|---|
| Tuononen, *Scientific Reports*, 2016 | 같은 rubber–glass interface가 어떻게 미끄러지기 시작하는가? High-speed imaging, digital image correlation, force 기록과 finite-element analysis를 병행했다. | Dry/wet glass의 여러 normal load. 600 N 영상을 제시했고 400 N에서 dwell time 2–600 s를 바꿨다. | Dry에서는 접촉 가장자리에서 순차적으로 분리됐고 wet에서는 훨씬 급격하게 미끄러졌다. Macroscopic force만으로 onset 시점을 고르기 어려웠다. | 실물에서 환경과 관측 시점이 중요하다는 근거다. 2.0 N과 충돌 사이의 polarity mismatch 비율이나 v51 센서 분해능을 알려주지는 않는다. [원문](https://pmc.ncbi.nlm.nih.gov/articles/PMC4904280/) |
| Ma & Rodriguez, *RA-L*, 2018 | 반복 planar pushing의 여러 결과 군집은 왜 생기는가? 기존 데이터의 순서·초기 자세를 분석하고 anisotropic friction simulation과 비교했다. | 이전에 수집된 반복 push 2,000회의 sequence 등을 재분석 | 표면의 마찰 방향성과 자동 수집 과정이 다음 초기 자세의 분포를 바꿔 관측된 군집을 설명할 수 있었다. | 반복 횟수가 많아도 독립적인 표면 교체 횟수는 작을 수 있다. 이 자료에서 v51의 부호 반전을 직접 관측한 것은 아니다. [원문](https://arxiv.org/pdf/1802.10089) |
| Dong et al., *ICRA*, 2019 | 진행 중인 grasp에서 incipient slip을 검출할 수 있는가? GelSlim의 dense motion field와 rigid-motion fit의 차이를 측정했다. | 실물 10개, 240회 검출 평가와 bottle-cap control 실증 | 24 Hz에서 86.25% detection accuracy를 보고했다. | 주로 rigid object를 가정하고 이미 접촉 중인 물체를 관찰한다. Surface 교체 후 독립 물체의 미래 충돌에 대한 probabilistic forecast와 시간 순서 및 endpoint가 다르다. [원문](https://arxiv.org/pdf/1810.13381) |
| VanderWeele, *Biometrics*, 2013 | Surrogate에서 관찰한 방향을 최종 outcome으로 옮길 충분조건은 무엇인가? Causal models와 이론적 반례를 분석했다. | 수리적 모형·반례. 물리 실험 표본 없음 | Surrogate와 outcome의 양의 상관만으로 intervention effect의 방향 보존을 보장할 수 없음을 보였다. | 논문은 treatment effect에 관해 증명한다. V51은 미래 outcome 예측 과제로, 이론이 접촉 부호의 반전을 증명하지 않는다. 두 관계를 따로 검증해야 한다는 방법론적 선례다. [원문](https://projects.iq.harvard.edu/files/applied_stats/files/surrogates_biom.pdf) |
| Gneiting & Raftery, *JASA*, 2007 | 어떤 scoring rule이 정직한 probability forecast를 유도하는가? Strict propriety와 score divergence를 분석했다. | 확률분포에 대한 정리와 forecast 사례. 로봇 접촉 표본 없음 | True conditional probability를 보고하면 expected quadratic loss를 최소화한다. | 어떤 estimator가 이동한 환경의 conditional probability를 더 잘 근사하는지는 데이터에 달려 있다. [원문](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf) |
| Cameron, Gelbach & Miller, *Review of Economics and Statistics*, 2008 | Cluster 수가 적을 때 추론의 크기 왜곡을 줄일 수 있는가? Cluster bootstrap-t 접근을 Monte Carlo와 응용에서 비교했다. | 생성한 회귀 데이터와 기존 경제 데이터 기반 실험 | 검토한 조건 일부에서 cluster 구조를 보존한 추론이 과도한 귀무가설 기각을 줄였다. | 접촉 Brier 차이에 대한 특정 CI의 정확도를 보장하지 않는다. Calibration prefix와 session 변화가 공유되는 실험에서 독립 단위를 명시해야 하는 이유를 제공한다. [저자 원고](https://www.nber.org/system/files/working_papers/t0344/t0344.pdf) |

### 문헌에서 남는 간격과 평가 기준

**근거 지도.** Surface condition과 접촉 이력은 local detachment와 초기 자세를 바꾸고, dense tactile observation은 진행 중인 contact의 slip을 검출할 수 있다. 이 두 연결에는 실물 근거가 있다. 그다음 연결인 **교체 전 reference → 교체 후 impact deformation → 미래 stick 확률**에 대해, 위 실물 연구들은 하나의 자료에서 세 단계를 함께 측정하지 않았다. [Tuononen](https://pmc.ncbi.nlm.nih.gov/articles/PMC4904280/) · [Ma & Rodriguez](https://arxiv.org/pdf/1802.10089) · [Dong et al.](https://arxiv.org/pdf/1810.13381)

**엇갈려 보이는 관찰.** Dong et al.이 실물 slip을 검출한 결과와 Tuononen이 macro force에서 단일 onset을 정하기 어렵다고 한 결과는 함께 성립한다. 전자는 연성 sensor의 국소 영상에서 일부 영역이 미끄러지는지를 판단했다. 후자는 rubber–glass의 분리 전파를 보며 국소와 전체 sliding의 시점이 다를 수 있음을 관찰했다. 표본·측정 위치·시간 해상도·판정 기준을 맞춘 head-to-head 비교가 없어 방법 간 일반적 우위를 정할 근거는 부족하다. [Dong et al.](https://arxiv.org/pdf/1810.13381) · [Tuononen](https://pmc.ncbi.nlm.nih.gov/articles/PMC4904280/)

**왜 단순 sign vote가 강한가.** V51에서 양쪽 신호가 threshold를 넘은 쌍의 부호가 모두 믿을 만하고 같은 환경을 공유한다고 가정하면, 부호 일치·불일치는 retention을 알려주는 Bernoulli trial이다. Beta(1,1) prior에서 같은 부호가 a개이고 유효한 쌍이 n개이면 posterior mean은 (1+a)/(2+n)이다. Soft vote도 이 값으로 두 endpoint를 혼합하므로 likelihood만의 추가 정보는 주로 측정 크기별 신뢰도 차이다. Binary Brier에서 두 정책의 conditional expected loss 차이는 참 stick 확률과 각 예측의 제곱거리 차이이므로, κ 추정 RMSE나 sign accuracy만으로 winner를 정할 수 없다. 이는 이 설계에 대한 계산이며, probability forecast를 점수화하는 이론적 근거는 다음과 같다. [Gneiting & Raftery](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)

**무엇을 분리해 주장할 수 있는가.** V51의 symmetric flip 모형에서는 calibration 환경의 centered retention (2κ−1)에 reference–impact 부호 오류가 (1−2e), session이 반전될 확률이 (1−2g)를 곱한다. 조건이 독립이면 같은 곱을 가진 (e,g)는 같은 물체의 평균적인 방향 전달률을 낼 수 있다. 둘이 모두 반전하면 부호가 복구될 수도 있다. 그러나 e는 물체별이고 g는 환경별이므로 **한 session에 여러 물체를 배포했을 때의 공동 실패 양상**은 다르다. 이는 실험 모형의 성질이다. 평가자가 e와 g를 독립 조작했다는 사실은 배포 정책이 두 원인을 관측만으로 분리할 수 있다는 증거가 아니다. 실제 표면의 intervention·측정 절차를 분리해야 하는 근거가 있는 연구는 다음과 같다. [VanderWeele](https://projects.iq.harvard.edu/files/applied_stats/files/surrogates_biom.pdf) · [Ma & Rodriguez](https://arxiv.org/pdf/1802.10089)

**논문 주장 점검.** 다수 물체와 noise draw는 한 session에서 공유한 네 쌍의 calibration 정보량을 늘리지 않는다. Catalog별 paired 차이, 독립 replacement session의 분포, 작은 이득의 불확실성을 함께 보고해야 한다. 지정된 e·g에서의 실험은 입력 조건별 정책 선택의 근거이고, 실제 표면에서 그 e·g가 얼마나 흔한지는 별도의 실험 대상이다. 단순 vote 대비 이득과 unsigned 대비 이득은 분리해서 읽어야 한다. 특히 장비에서 unsigned가 calibration을 생략할 수 있을 때는 소요 시간과 물체 교체·마모를 포함해야 한다. [Cameron et al.](https://www.nber.org/system/files/working_papers/t0344/t0344.pdf) · [Tuononen](https://pmc.ncbi.nlm.nih.gov/articles/PMC4904280/)

### 후속 연구 질문 10개

| 질문 | 근거와 필요성 | 검증 변수와 방법 |
|---|---|---|
| 1. Reference sign과 impact sign의 불일치는 어느 접촉 시각에 집중되는가? | Detachment가 시공간에 걸쳐 전파된다. [Tuononen](https://pmc.ncbi.nlm.nih.gov/articles/PMC4904280/) | Impact 전후 high-speed optical marker 궤적, impulse 시각, 부호 역전을 함께 기록한다. |
| 2. Surface만 교체했을 때의 drift와 reset 자세 변화의 효과는 각각 얼마인가? | 초기 자세·수집 순서가 반복 push의 분포를 바꿨다. [Ma & Rodriguez](https://arxiv.org/pdf/1802.10089) | 동일 재질의 sham replacement와 다른 재질로의 교체 순서를 무작위화하고 자세를 별도 측정한다. |
| 3. 표면 교체 뒤 한 쌍만 재측정해도 session-wide 반전을 검출할 수 있는가? | 교체 뒤 초기 상태가 달라질 수 있다. [Ma & Rodriguez](https://arxiv.org/pdf/1802.10089) | 총 네 쌍을 교체 전·후 4+0, 3+1, 2+2로 나눠 환경별 실패와 손실을 비교한다. |
| 4. 약한 신호의 부호 판독 오차와 진짜 deformation reversal을 분해할 수 있는가? | 국소 영상과 전체 force가 서로 다른 접촉 현상을 관측한다. [Tuononen](https://pmc.ncbi.nlm.nih.gov/articles/PMC4904280/) | 독립 camera 두 시점과 force channel을 동기화하고 near-zero 범위를 미리 정한다. |
| 5. 단일 scalar에 크기에 따른 신뢰도 정보가 실제로 남는가? | Dense motion field에는 단일 부호에 포함되지 않은 공간 정보가 있다. [Dong et al.](https://arxiv.org/pdf/1810.13381) | 같은 paid pairs에서 sign vote, amplitude likelihood, 고정 길이 영상 요약을 비교하고 resolution을 변화시킨다. |
| 6. 동일 평균 Brier라도 일부 session에서 여러 물체를 함께 실패하는가? | 공유된 오차는 독립 관측처럼 취급할 수 없다. [Cameron et al.](https://www.nber.org/system/files/working_papers/t0344/t0344.pdf) | 물체별·session별 failure correlation과 session loss의 상위 분위수를 별도 추정한다. |
| 7. 물체별 flip이 특정 형상·속도에 집중되는가? | 서로 다른 지점·조건에서 slip 전파가 달랐다. [Tuononen](https://pmc.ncbi.nlm.nih.gov/articles/PMC4904280/) | Contour, tip radius, impact velocity에 따라 e를 직접 계측하고 blind group에서 재검증한다. |
| 8. 어떤 경우에는 교체 사실만 알아도 unsigned로 돌아가는 편이 유리한가? | Surrogate의 방향 전달에는 조건이 필요하다. [VanderWeele](https://projects.iq.harvard.edu/files/applied_stats/files/surrogates_biom.pdf) | Replacement indicator만 공개한 policy를 고정하고, blind·recalibration·oracle 사이의 loss를 비교한다. |
| 9. 실제 sensor 비용을 넣어도 네 쌍의 calibration 가치가 남는가? | 장시간의 dwell과 초기 상태가 friction에 영향을 줬다. [Tuononen](https://pmc.ncbi.nlm.nih.gov/articles/PMC4904280/) · [Ma & Rodriguez](https://arxiv.org/pdf/1802.10089) | 접근·reference contact·reset·재시도·배포 개수별 비용을 기록하고 session당 순이득을 계산한다. |
| 10. 참 κ 추정이 조금 개선되어도 Brier가 바뀌지 않는 경우는 무엇인가? | Proper score는 최종 event probability를 평가한다. [Gneiting & Raftery](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf) | 사전 고정한 두 endpoint 확률의 절대 차이 구간별로 paired Brier 차이와 estimator 오차를 함께 본다. |

## 재현 자료

서버 경로는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/reference_drift_v51/`다. 사전 계획은 한국 시각 2026-11-17 00:56:13에 기록했다. 멀티 에이전트가 물리 intervention·독립 rollout, GPU inference와 latency, 문헌 검토를 분담했고 primary risk는 별도 GPU 코드와 CPU 구현으로 대조했다. v50 원본은 읽기 전용 의존성으로 사용했다. Slurm GPU node에서 같은 프로젝트의 Python 환경으로 `score51.py`, CPU node에서 `controls51.py`와 `rollout51.py`를 실행하면 primary 결과와 보조 검증을 재현할 수 있다. [동결 비교 계획](plan51.sha256) · [GPU scorer](score51.py) · [CPU control](controls51.py) · [독립 rollout](rollout51.py) · [문헌별 증거와 후속 질문](LITERATURE_KO.md)
