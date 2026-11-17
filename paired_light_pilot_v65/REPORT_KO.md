# v65 · 같은 충격을 두 조명에서 읽으면 다음 관측 선택이 달라지는가?

**첫 실물 실험으로 권하는 규모는 독립 표면 시편 12개 × 4개 session이다.** 같은 8회 impact를 두 조명에서 읽고, 각 조명의 기록만으로 같은 후속 물체의 네 번째 관측을 고른다. 이 48개 비교에서 한 번도 선택이 달라지지 않을 확률은 v64 기준 **92.0%**다. 따라서 첫 실험에서 먼저 확인할 것은 동일 사건의 촬영 성공률, 독립적으로 식별할 수 있는 물질점의 비율, 같은 위치에서 측정한 strain의 신뢰도다. 구매 전환 한 건 이상을 95% 확률로 보려면 기준 가정에서 **236개 표면 × 8개 session**이 필요하다. [서버 표본 수 계산](POWER_KO.md) · [현장 촬영 순서표](planned_run_sheet65.csv)

## 1. 한 session은 어떤 질문에 답하는가?

예를 들어 한 표면에서 서로 다른 물체 여덟 개에 같은 절차를 적용한다. 각 물체의 2.0 N reference 한 번과 impact 한 번을 조명 A와 B에서 모두 촬영한다. 네 쌍의 둘째 물체 직전에 전자센서 하나를 무작위 reset한다. 그 뒤 같은 표면에 별도로 예약한 **아홉 번째 물체**의 첫 세 weak/reference 쌍을 읽는다. 동일한 이 prefix를 양쪽에 주고, 각각 네 번째 관측으로 reference(R)와 impact(I) 중 하나를 고르게 한다. 이때 한쪽이 R, 다른 쪽이 I면 구매 전환 1건이다. 두 추천을 먼저 봉인한 뒤 무작위로 지정한 한쪽의 행동만 집행한다. 사전 지정한 독립 운영자가 1:1 배정을 보관하고 봉인 후 집행자에게 공개하며, 실제 receipt와 실패 여부를 기록한다. [현장 기록 규약](METROLOGY_RECORDS_KO.md) · [v64 재생 결과](DRY_REPLAY_KO.md)

| 분석 단위 | 측정값 | 이 값에 필요한 독립 기록 |
|---|---|---|
| Calibration impact × lamp | 같은 잘못된 물질점 ID를 3개 연속 nominal frame 이상 추적한 비율 | NIR의 coded material marker와 blind 판독. 반사상만 있고 대응하는 ID가 없으면 unassigned로 분류 |
| Calibration event | 초기/후기 시간창 사이 signed local strain과 uncertainty, optical displacement와의 부호 일치 | 같은 면의 정해진 위치에서 측정한 gauge εxx, 독립 NIR extensometer, common clock |
| 8-impact bank | 두 조명에서 상위 router가 fixed 3R+I와 v54 gate 중 서로 다르게 고른 비율 | 양쪽 calibration raw, router 입력·출력 |
| **8-impact bank + 공통 후속 물체 하나** | **Primary endpoint**인 두 최종 R/I 추천의 불일치. A→I/B→R과 그 역방향도 별도 집계 | 동일 prefix hash, 두 봉인 추천. 실제 실행 여부와 실행한 행동은 별도 열 |
| 후속 물체 | 가능하면 실행한 branch의 예측확률과 실제 stick/slip에서 Brier loss를 기록 | 무작위 배정확률, 구매한 관측, frozen decoder, 독립 outcome. 비실행 branch의 loss는 미관측 |

하나의 잘못된 ID가 두 lamp에서 똑같이 선택될 수도 있다. 반대로 ID가 맞아도 timing이나 strain 판독의 차이가 행동을 바꿀 수 있다. 그래서 **모든 event에 두 종류의 reference를 연결한 뒤** 두 조명 사이에서 바뀐 지점이 marker, sign/abstention, 상위 router, 최종 R/I 중 어디인지 추적한다. 같은 물체의 frame, 같은 bank에 재사용한 query, 두 lamp의 사진은 독립 session 수에 더하지 않는다. [정책 경로 검산](dry_replay65.json)

## 2. 실물 pilot 설계와 예산

- **시편과 순서.** Polished, brushed, satin finish에서 별개로 제작·조달한 coupon을 4개씩 사용한다. Lot을 교차 배정하고 lot·마모·재장착을 기록한다. 각 coupon을 이틀에 걸쳐 board X와 Y에서 한 번씩 측정한다. 첫날 X→Y, 둘째 날 Y→X이며 각 block 안에서는 시편 순서를 무작위화한다. 같은 coupon을 닦거나 다시 설치해도 같은 surface ID다. Object reservation은 계획상 자리이고, 실험 전 물리 serial과 대응시킨다. 두 board와 이틀만으로 다른 board·날짜의 일반화 성능을 주장하기 어렵다. [예약 순서표](planned_run_sheet65.csv) · [생성 코드](schedule65.py)
- **두 조명의 같은 impact.** 같은 spectrum의 LED를 좌우 ±30°에 두고 공통 광축의 global shutter head 또는 multi-tap sensor로 발광을 짧게 교대한다. 설계 예시는 8 μs pulse, 12 μs exposure, 중심 간격 20 μs다. 두 시점의 시간차를 측정하며, 8개 물체에서 선행 발광 순서를 4:4로 균형화한다. 각 lamp의 photodiode, 실제 shutter 응답, contact switch, 별도 timing LED를 같은 clock에 기록한다. 장착·광량·포화·공간 등록·시간차·잔광을 development coupon에서 검사한 뒤 실험 도중 reader와 threshold를 고정한다. [광학 설계·공차](RIG_KO.md) · [Yoda et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5876938/)
- **Marker identity.** 움직이는 물질점, 인접 distractor, 고정 anchor에 visible spot과 공위치인 고유 NIR code를 부여한다. 별도 NIR camera에서 두 판독자가 visible 정책과 outcome을 가린 채 ID를 읽는다. 거울상·가림·후보 중복은 두 번째 보정 시점(view)이나 제3 판독자로 확인하고 여전히 불분명하면 unresolved로 보존한다. 각 사건에서 confirmed swap, unassigned reflection, transient jump, correct를 구분한다. Ink·접착 전후 blank 시편 대조도 포함한다. [계측 설계](RIG_KO.md) · [Li et al.](https://arxiv.org/abs/1603.08120)
- **Local strain.** 같은 면의 물질 좌표에서 접선 x 방향으로 부착한 short-grid foil strain gauge의 축방향 평균 εxx를 읽고, gauge footprint 양 끝 NIR fiducial 간 거리 변화로 검산한다. Gauge 위치와 contact mark 간 거리를 기록한다. 순수 병진에서는 marker가 움직여도 strain은 0일 수 있다. 무부하·양/음 mechanical control·rigid translation에서 gain, polarity, deadband, 시간 지연을 고정한다. 두 기준이 불일치하거나 gauge가 포화되면 그대로 기록한다. Gauge 신호와 NIR 영상은 평가용으로만 연다. [계측 설계](RIG_KO.md) · [관련 impact 원자료](https://doi.org/10.25532/OPARA-655)

| 촬영 묶음 | Session 수 | Calibration impacts | 2.0 N references | Mechanical resets |
|---|---:|---:|---:|---:|
| 서로 다른 조명 A/B | 48 | 384 | 384 | 768 |
| 같은 조명을 두 슬롯에 둔 별도 sham(A/A 6, B/B 6) | 12 | 96 | 96 | 192 |
| **총 예약** | **60** | **480** | **480** | **960** |

각 bank에 전자센서 reset 4회, 총 240회가 추가된다. 두 조명이 같은 접촉을 공유하므로 이중 촬영에 필요한 추가 impact는 0회다. 표에는 commissioning과 새 후속 물체의 비용이 빠져 있다. 후속 물체 60개에는 총 180개의 weak/reference prefix 쌍과 60개의 실제 네 번째 관측, Brier를 평가한다면 별도 outcome 측정이 필요하다. 순서표에는 calibration slot 480개가 있고 이 값은 수집 예정량이다. [촬영 순서표](planned_run_sheet65.csv)

## 3. 서버에서 계산한 surface·session 수

V64 archive에서 원래 12,288개 simulated bank를 재생했다. 상위 선택의 차이는 0.287882%였다. 상위 선택이 갈린 조건에서 같은 후속 물체의 v54 gate가 R을 선택할 확률은 64.713%였다. 나머지는 양쪽 모두 I를 고른다. 이 두 확률을 곱한 **0.186299%**를 표본 수 계산의 기준으로 사용했다. 모든 가능한 조합을 열거한 468,836/251,658,240은 같은 bank·prefix를 교차 사용한 수치다. 이 분모를 독립 현장 측정 수로 해석하면 불확실성을 크게 과소평가한다. [서버 replay](dry_replay65.json) · [별도 계산](operations65.json)

서로 다른 coupon은 독립이고, 같은 coupon 안에서는 구매 전환 경향을 공유한다고 가정했다. 표면별 전환 확률을 Beta distribution에서 뽑고 그 표면의 session을 조건부 독립 시행으로 두었다. 아래 기준값은 표면당 유효 session 8개, intraclass correlation(ICC) 0.05다. v64 simulation interval 양끝의 평균 발생률과 ICC 0·0.01·0.05·0.10도 계산했다. 세 기준 모두 최소 정수 surface 수와 바로 한 개 적을 때의 통과 여부를 검사했고, 5개 시나리오는 각각 100,000개 panel로 Monte Carlo 검산했다. [가정과 전체 sensitivity](POWER_KO.md) · [기계 판독 결과](power65.json)

| 확보하고 싶은 결과 | 확률 | 표면 수 × session/표면 | 유효 session | Calibration impacts |
|---|---:|---:|---:|---:|
| 전환 ≥1건 | 95% | **236 × 8** | 1,888 | 15,104 |
| 전환 ≥10건 | 90% | **1,011 × 8** | 8,088 | 64,704 |
| 전환이 있는 서로 다른 표면 ≥10개 | 90% | **1,121 × 8** | 8,968 | 71,744 |

- 실물 전환율이 0.0491% 또는 0.3525%라면 첫 목표는 각각 **895개 또는 125개 표면**이다. 이 범위는 simulation에서 선택한 sensitivity 값이다. 실물 전환율이 0일 가능성도 열려 있다. 기준 발생률에서 ICC만 0에서 0.10으로 바꾸면 첫 목표는 201개에서 270개 표면으로 변한다. 각 session이 사건 여부와 독립적으로 10% 누락된다고 가정하면 기준 목표 3개의 예약 표면 수는 258, 1,117, 1,227개다. [전체 표](POWER_KO.md)
- 첫 pilot **12×4**의 예상 전환 수는 0.089건이다. 0/48일 때 정해 둔 beta-binomial 모형의 단측 95% 상한은 **6.51%**다. 한 건을 찾는 비용과 발생률을 정밀 추정하는 비용도 다르다. 독립 session을 가정해 95% interval 반폭을 기준 발생률의 절반 이내로 만들 확률을 90% 요구하면, 100회 단위 탐색에서 처음 충족한 크기는 **11,800회**다. [정밀도 계산](POWER_KO.md)
- 현장 결과에서는 surface별 비율과 전체 분모, 두 방향 전환 수, 전환이 생긴 고유 표면 수를 함께 낸다. 사전 등록 N개 중 f개가 전환이고 m개는 어느 한쪽 영상을 잃었다면 가능한 발생률 범위는 [f/N, (f+m)/N]이다. 정상 영상에서 발생한 frozen abstention은 유효한 의사결정이다. Identity·strain의 판정 불능은 그 평가값의 분모에 표시한다. 같은 lamp의 sham에서도 같은 기준을 적용한다. 소규모 pilot에서 ICC를 정밀 추정할 수 없으므로 민감도와 raw counts를 유지한다. [평가 규약](METROLOGY_RECORDS_KO.md)

## 4. 장비 없이 실제로 검증한 부분

| 검증 | 자료와 절차 | 결과와 해석 |
|---|---|---|
| v64 재현과 정보 누출 검사 | 실제 보존된 simulation raw fixture 48개를 다시 decode하고, 전체 archive에서 122,880개 prefix instance의 첫 세 쌍만으로 v54를 재계산 | 3,072개 encoded 값과 768개 상위 추천이 원본과 일치. 최종 전환 0.186299% 재현. [재생 보고서](DRY_REPLAY_KO.md) |
| **실측 영상의 round-trip 검사** | DiLiGenT 물체 10개, 기존 337개 patch에서 조명 순서 정·역방향을 각각 평가. 기준 patch에서 세 다른 lamp로 찾은 뒤 매번 원래 영상으로 되돌렸다. 양방향 NCC ≥0.8, 세 왕복 모두 1 pixel 이내를 요구 | 정방향에서 기존 62개 수용/9개 일관 오등록이 **17개 수용/2개 오등록**으로 바뀌었다. 두 사례는 cat과 cow다. 역순에서는 64/14가 **10/0**으로 바뀌었다. 정지 표면에서 세 lamp 모두 같은 방향으로 ≥2 pixel 벗어난 좌표 오차다. 두 방향은 같은 물체의 재분석이고, 실제 충격의 marker swap 빈도는 아직 미측정이다. [좌표별 결과](static_closure65.csv) · [사전 기준](STATIC_PLAN_KO.md) · [원자료](https://sites.google.com/site/photometricstereodata/single) |
| 같은 lamp의 timing stress test | 같은 simulation fixture 영상에서 한쪽 photodiode에만 16 kHz 기준 offset을 주입 | +4 sample(0.25 ms)에서 최종 구매 차이 0.250%, +8 sample(0.50 ms)에서 1.000%, 0·−4·−8 sample에서 0%. 특정 bank에서 timing만으로 기준 0.186%와 비슷하거나 더 큰 값이 나왔다. 실물 지연 빈도는 측정해야 한다. [주입 계획](OPERATIONS_PLAN_KO.md) · [서버 결과](operations65.json) |
| 두 조명의 시간 분리 | 20 μs 간격, LED 잔광과 노출 중심 오차를 바꿔 적분 | 잔광 time constant가 2→5 μs로 바뀌면 최대 광량 혼입은 0.457%→11.47%. 4 px/ms이면 위치차 상한이 0.088 px, 10 px/ms이면 0.22 px다. 실제 속도·광량을 함께 측정해야 쓸 수 있는 설계다. [서버 결과](rig_timing65.json) |
| 현장 데이터 QA | 두 raw, 공통 prefix, 독립 reference, 봉인 시각과 하나의 actuator receipt를 연결하고 14개 오류·누락 시나리오를 실행 | 중복 event, prefix 바꿔치기, hash·receipt 불일치 등을 검출했다. 한쪽 영상 유실은 미상으로 유지했고, 별도 strain·receipt가 없어도 유효하게 봉인한 두 추천은 분석에 남겼다. [검사 코드](field_qc65.py) · [실행 결과](qc65.json) |

**어떤 비용을 줄일 수 있는가.** 상위 선택이 동일하고 frozen code와 공통 prefix가 같으면 최종 선택도 반드시 같다. 대규모 *전환율 전용* 조사에서 상위 불일치 전수와 상위 일치의 무작위 5%에만 후속 prefix를 수집하는 설계를 별도로 계산했다. v64 분포에서 후속 prefix가 필요한 bank는 **5.2735%**이고 모든 bank는 분모에 남는다. 현장 가능성을 보는 첫 48회는 전수 수집한다. 8회 calibration, 독립 identity·strain 확인 또는 Brier에 이 절감률을 적용할 근거는 없다. 같은 조명 sham 100개에서 0건이어도 단측 95% 상한은 2.95%다. 독립 sham에서 0건을 전제로 그 상한을 반올림 전 기준 발생률의 절반 아래로 내리는 데만 3,215개가 필요하다. [검산](operations65.json)

## 5. 문헌 비교와 연구 판단

| 연구 | 질문·자료·방법 | 발견과 적용 범위 |
|---|---|---|
| Shi et al., DiLiGenT | 여러 조명에서 일반 재질 surface normal을 얼마나 복원하는가? 실물 10개 × 96개 lamp 방향과 독립 normal reference로 photometric stereo를 비교 | 알려진 고정 형상에서 조명 의존 오류를 시험할 수 있다. 움직이는 material ID, 접촉 strain과 행동은 제공하지 않는다. [논문·자료](https://sites.google.com/site/photometricstereodata/single) |
| Li et al. | 비강체 optical flow를 실제 변형에서 어떻게 평가하는가? 동시 RGB/NIR 영상의 비가시 marker로 8개 RGB 방법을 비교 | 조명·가림이 있는 영상에서 다른 spectrum으로 위치 기준을 만들 수 있다. 반사 coupon의 ID 오인식률과 국소 strain은 새로 검증해야 한다. [원문](https://arxiv.org/abs/1603.08120) |
| Yoda et al. | 움직이는 물체에서 서로 다른 조명의 정보를 시간상 가깝게 분리할 수 있는가? Multi-tap CMOS prototype으로 움직이는 대상의 photometric stereo를 시험 | 시간 분할 조명의 구현 근거다. 이 연구의 prototype이 v65 impact 속도와 afterglow 허용범위를 충족하는지는 계산·계측이 별도로 필요하다. [원문](https://pmc.ncbi.nlm.nih.gov/articles/PMC5876938/) |
| Duran Vergara et al. | 충격파와 변형을 영상과 기계 측정으로 복원할 수 있는가? Concrete cylinder 4개에서 초고속 영상과 bar gauge를 비교 | 동적 영상을 독립 전기 신호와 연결했다. Bar 신호로 추정한 시편 global strain은 v65의 같은 면 local εxx와 다른 공간 척도다. [원문](https://doi.org/10.1088/1361-6501/adb16f) · [공개 자료](https://doi.org/10.25532/OPARA-655) |
| Eldridge, Ashby & Kerry | 반복 수가 다른 cluster에서 필요한 표본 수가 어떻게 달라지는가? Cluster 크기와 분석법의 영향을 수식과 설계 사례로 평가 | 시편 수·반복 수·ICC를 분리할 근거다. 본 물리 과제의 ICC 수치를 제공하지 않는다. [원문](https://doi.org/10.1093/ije/dyl129) |

**근거의 연결.** Shi와 Yoda의 방법은 조명 입력을, Li의 방법은 물질점 대응을, Duran Vergara 등의 방법은 기계적 변형을 검증한다. V65에서는 이 셋을 동일 event에 연결하고, 해당 event가 속한 bank의 최종 R/I까지 따라간다. 서로 다른 방식이 같은 값을 측정하는 구간은 먼저 development coupon에서 확인해야 한다. 한 예로 v64에서 OPARA 4건의 peak global compression 부호는 일치했지만, 본 정지 표면 실험에서는 왕복 확인 후에도 오등록이 남았다. 서로 다른 공간 척도, 다른 endpoint와 표본으로 얻은 두 결과는 양립한다. [OPARA 재분석](../reflective_field_v64/MECHANICS_KO.md) · [실측 round-trip 결과](static_closure65.json)

**근거의 강도와 논문 기여.** 대칭적인 조명 input, 공통 prefix, frozen policy에서 상위 불일치가 최종 구매까지 전달되는 수학적 조건은 직접 검산됐다. 조명이 정지 표면의 일부 등록을 흐트러뜨린다는 근거는 실측이지만 표면·tracker에 의존한다. 실물의 persistent swap이 구매를 바꾸는 빈도, 그 구매가 실제 Brier를 개선하는지는 미확인이다. 독립 reference를 갖춘 동일 충격으로 *어떤 계측 오류가 어디서 소거되고 어떤 오류가 행동까지 전파되는지* 보여주면 논문의 실험적 기여가 된다. Multi-light 촬영 자체의 독창성이나 조명을 바꾸면 성능이 좋아진다는 주장은 현재 근거로 지지하기 어렵다. [광학 선행연구](https://pmc.ncbi.nlm.nih.gov/articles/PMC5876938/) · [오류 전파 검산](DRY_REPLAY_KO.md)

## 6. 후속 연구 질문 10개

| 연구 질문 | 남은 문제·의미 | 측정과 접근 |
|---|---|---|
| 1. 어느 finish에서 실제 ID swap이 반복되는가? | 정지 patch 결과를 접촉에 적용할 근거가 부족하다. | NIR confirmed swap을 finish·coupon별로 추정하고 가림 분모를 함께 보고한다. [DiLiGenT](https://sites.google.com/site/photometricstereodata/single) |
| 2. 거울상도 동일 NIR code로 읽히는가? | 검증 채널에도 shared optical error가 생길 수 있다. | 각도·반사 정도와 한 시점/두 시점의 ID ambiguity를 blind 비교한다. [Li et al.](https://arxiv.org/abs/1603.08120) |
| 3. Displacement sign은 언제 local strain sign과 갈라지는가? | 병진과 국소 변형은 다른 물리량이다. | Translation·bending·contact에서 encoder, NIR 길이와 같은 면 εxx를 대조한다. [기계 계측 근거](https://doi.org/10.1088/1361-6501/adb16f) |
| 4. 시간차만으로 R/I가 갈리는가? | 합성 신호에서는 지연 방향에 따라 결과가 달랐다. | A/A·B/B에서 실제 midpoint 차이와 switch를 짝짓고 exposure order를 교환한다. [주입 실험](operations65.json) |
| 5. 어떤 속도까지 두 광원을 분리할 수 있는가? | 잔광 감소를 위한 긴 간격은 motion error를 키울 수 있다. | 속도·pulse 간격·잔광을 교차하고 광량 혼입과 위치 오차를 함께 잰다. [Timing 결과](rig_timing65.json) |
| 6. 마모나 재부착이 오류를 같은 표면에 집중시키는가? | 사건 수와 사건 표면 수에 필요한 비용이 다르다. | coupon lot·wear·day·board를 교차해 표면별 분포와 ICC sensitivity를 갱신한다. [표본 수](POWER_KO.md) |
| 7. ID 변경이 실제 어느 단계까지 전파되는가? | 틀린 track이 생겨도 상위 router나 구매는 동일할 수 있다. | Event 단위 ID·sign·abstention에서 bank의 router와 공통 query R/I까지 transition count를 보존한다. [정책 재생](dry_replay65.json) |
| 8. 확인 불능이 반사 표면에 집중되는가? | 정답이 있는 사례만 고르면 오류 빈도가 작아 보일 수 있다. | 전체/판정 가능 분모와 극단 경계, 독립 adjudication 수율을 finish별로 비교한다. [현장 규약](METROLOGY_RECORDS_KO.md) |
| 9. 두 단계 표집으로 field 전환율을 보존하는가? | 후속 prefix 비용이 대규모 모집의 병목일 수 있다. | 전수 pilot과 같은 결과가 나오는지 검산 후 discordant 전수·concordant 무작위 audit를 적용한다. [비용 계산](operations65.json) |
| 10. 관측이 달라진 사례에서 query prediction도 개선되는가? | 구매 전환만으로 gain의 크기와 부호를 모른다. | 새 surface에 실제 행동을 무작위 배정하고 propensity·독립 outcome으로 Brier 차이와 cluster variance를 추정한다. [v64 분석 계획](../reflective_field_v64/SIMULATION_PLAN_KO.md) |

## 7. 다음 단계로 넘어가는 기준

첫 48개 A/B session에서 제안하는 운영 기준은 **44개 이상에서 두 추천과 공통 prefix가 복원되고, 각 finish의 paid impacts 중 80% 이상에서 독립 ID와 gauge 판독을 모두 확보**하는 것이다. 이는 설계자가 정한 feasibility 기준이며 실제 장비 가동 전에 확정한다. 같은 lamp의 switch는 원시 clock·광량·정렬로 조사하고 모든 실패를 남긴다. Sham과 A/B의 차이를 판별하기 어려우면 최초 논문 주장을 계측과 오류 전파 사례까지로 제한한다. 미리 정한 threshold에 실패하면 원인을 고쳐 새 surface에서 검증하며, 이미 확인한 coupon은 development에 둔다.

대규모 모집을 결정하는 시점에는 필요한 것이 최초 전환 한 건, 여러 표면의 재현, 발생률의 정밀 추정, 혹은 Brier 개선 중 무엇인지 먼저 고정한다. **현재 근거로는 12개 표면에서 계측부터 검증하는 편이 타당하다.** 그 결과가 장비 문제를 해결하고 여러 표면에서 audit가 가능함을 보이면, 별도 holdout과 최소 실용 Brier 차이로 confirmatory experiment를 설계할 수 있다. v65에서 새로 수행한 계산은 datacenter3 Slurm allocation 2744469의 CPU에서 실행했다. 실물 impact, 실물 구매 효과, 실물 Brier와 장비 latency는 아직 측정되지 않았다. [실물 프로토콜](RIG_KO.md) · [재현 manifest](report65.sha256)
