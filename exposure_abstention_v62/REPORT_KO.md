# v62 · LED pulse와 shutter exposure가 달라질 때, 어느 판독을 보류할 가치가 있는가?

**Selective reset은 기존 v61 reset보다 평균 held-out Brier loss가 0.0164 × 10⁻⁶ 낮았다. 기존 sham과의 차이는 불확실했고, 항상 v54 gate를 쓰는 정책보다는 0.0336 × 10⁻⁶ 높았다.** 기존 v61 reader가 판독한 79,701개 reference–impact 쌍에서 관측된 mismatch 오판은 0건이었다. 보류한 42,253쌍은 전부 맞는 판독이었고, 같은 수의 쌍을 무작위로 남겨도 비슷한 손익이 나왔다. 이 실험에서 확인한 작은 개선은 관측을 덜 쓰면서 새 표면에서 평균 손실이 더 컸던 fixed 3R+I를 덜 선택한 결과다. [정책 검증](SCORE_KO.md) · [동일 판독 수 대조](thinning62.json)

## 1. 무엇을 고정하고 무엇을 바꿨나

새 합성 sensor board 8개 각각을 새 표면 48개 모두에서 평가했다. 표면당 4개 session을 만들고 session마다 무작위 순서로 서로 다른 물체 2개를 검사했다. 한 물체에는 2.0 N reference와 impact를 한 번씩 가한다. 모든 정책은 표면당 **8 impacts, 8 references, 16 mechanical resets**를 지불한다. Reset arm만 각 session의 둘째 물체 직전에 전자센서 한 대를 무작위로 골라 reset한다. Sham에는 같은 센서 ID를 배정하고 reset 동작만 생략한다. 보류해도 추가 impact나 재촬영 기회가 생기지 않는다. [계측 계획](VISION_PLAN_KO.md) · [비용 장부](cost62.json)

- **광학 요인.** 1,000 fps에서 LED pulse 폭 0.20·0.80 ms와 global shutter exposure 0.25·0.80 ms를 독립적으로 교차했다. LED 밝기는 발광 구간과 노출 구간이 겹친 시간에 따라 달라진다. 물체 marker의 영상도 exposure 안의 네 시점에서 적분해 motion blur를 반영했다. LED 발광 지연과 camera trigger 지연은 각각 0·2 ms로 설정했다. 이 네 지연 조합을 모든 8-impact pilot에 두 번씩 무작위 배치하고 네 광학 조건에서 같은 물체와 noise를 재사용했다. 16 kHz ADC의 contact switch와 photodiode를 통해 지연을 추정했다. [광학 구현](vision62.py) · [v61의 시간 보정](../switch_diode_v61/REPORT_KO.md)
- **보류하는 정책.** 기존 v61 reader가 부호를 읽은 뒤, 관측한 LED 최고 contrast가 48 이상이고 두 번째가 28 이하이며, marker 상대변위가 0.75 pixel 이상이고, 겹치지 않는 두 frame 쌍의 변위가 같은 방향으로 각각 0.20 pixel 이상일 때만 그 부호를 유지했다. 어느 하나라도 실패하면 해당 clip을 결측 처리한다. Reference와 impact 중 하나만 결측이어도 그 물체의 optical mismatch는 쓰지 않는다. 두 전자센서의 기록은 계속 쓰고, 모든 후속 물체에 stick 확률을 출력한다. Cutoff는 target 결과를 보기 전에 고정했다. [Reader 사전 명세](VISION_PLAN_KO.md) · [동결 시각](vision_plan62.utc)
- **원인과 행동.** P에서는 같은 session의 실제 물리적 mismatch가 상관되고, C에서는 실제 mismatch가 독립이어도 전자센서에 common-mode fault가 생긴다. 원인 판별은 P/C를 맞히는 문제다. 행동은 이후 deployment에서 네 번째 관측을 항상 impact로 사는 **fixed 3R+I** 또는 세 reference를 본 뒤 선택하는 **v54 gate** 중 하나다. 실제로 손실이 낮은 행동을 맞혔는지가 정책 판별률이다. 주 분석에서는 C의 장치 fault 빈도를 이 행동의 유불리와 독립시켰다. Frozen source likelihood와 odds cutoff 2.016을 모든 reader에 동일하게 적용했다. [채점 사전 명세](SCORE_PLAN_KO.md) · [Frozen likelihood](../sensor_reset_v59/route59.py)

24개 새 catalog에서 기존 두 rate law로 12,288개 held-out 물체를 생성했다. Calibration까지 합친 36,864개 parameter vector는 과거 archive 313,744개와 겹치지 않았다. 변형이 활성인 6,144개 물체 중 2,236개에서 변형 방향을 바꾸면 PyBullet stick label이 달라졌다. Fine/ultra solver 간에는 24,576개 endpoint 중 820개에서 차이가 났다. 평균 held-out Brier는 frozen decoder의 stick 확률에 대해 두 변형 방향의 binary outcome을 생성 확률로 적분한 squared loss다. Session loss p90은 별도 binary replay에서 128개 물체별 session 평균을 구해 그 분포의 90번째 백분위수를 취했다. [World audit](world62.json) · [중복 검사](support62.json)

## 2. 같은 8회 impact의 정책 결과

| 정책 | Active 쌍 판독률 | 원인 판별률 | 저손실 정책 판별률 | 평균 held-out Brier ↓ | Session loss p90 ↓ | GPU p90, ms / 512 objects ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Selective reset + camera | 2.372% | 50.283% | 50.029% | 0.157078333 | 0.188423350 | 480.932 |
| V61 reset + camera | 5.047% | 50.582% | 50.028% | 0.157078349 | 0.188423350 | 480.983 |
| V61 sham + camera | 5.047% | 50.490% | 50.026% | 0.157078345 | 0.188423350 | 480.828 |
| 항상 v54 gate | — | — | 50.000% | 0.157078299 | 0.188423350 | 480.108 |

판독률의 분모는 물리적으로 변형이 활성인 reference–impact 쌍이다. 쌍을 읽으려면 두 접촉 모두 부호가 있어야 한다. 전체 유료 쌍을 분모로 삼으면 selective는 **1.190%**, 기존 v61은 **2.534%**를 읽었다. 상시 gate에는 같은 pilot 물리 비용을 부과하지만 판독이나 P/C 진단을 요청하지 않는다. 원인과 정책 판별률은 balanced accuracy다. [판독률 분모](VISION_KO.md) · [모든 정책 수치](score62.json)

| 대조 정책 (차이 = selective reset − 대조) | 평균 Brier 차이, ×10⁻⁶ [동시 95% CI] | Session loss p90 차이, ×10⁻⁶ [동시 95% CI] |
|---|---:|---:|
| V61 reset + camera | **−0.0164** [−0.0430, −0.0023] | 0.000 [−0.805, 0.000] |
| V61 sham + camera | −0.0124 [−0.0445, +0.0088] | 0.000 [−0.492, +0.447] |
| 항상 v54 gate | **+0.0336** [+0.0125, +0.0596] | 0.000 [−0.104, +1.118] |

24개 catalog와 8개 board를 각각 재추출한 6,000회 paired crossed bootstrap으로 각 endpoint의 세 비교를 동시 보정했다. 8개 board 각각에서 기존 reset보다 selective가 유리하고, selective보다 v54 gate의 평균 loss가 낮았다. 기존 sham과의 순위는 board에 따라 갈렸다. Ultra solver와 두 binary replay에서도 selective − 기존 reset의 평균 차이는 음수였다. Gate 대비 손해는 세 검산에서 모두 같은 방향이었으나 한 binary replay는 CI가 0을 포함했다. 네 주 정책의 session loss p90은 두 번째 replay에서도 서로 같았다. 원인 판별률은 기존 reset보다 **0.299 pp 감소** [−0.514, −0.110]했다. [Board별 결과와 민감도](SCORE_KO.md)

![네 광학 조건의 active 판독률과 세 대조의 Brier 차이](exposure_abstention62.png)

왼쪽은 같은 물체에서 읽을 수 있었던 비율이다. 오른쪽에서 0 왼쪽은 selective reset이 더 낮은 평균 Brier를 냈다는 뜻이다. 가로 구간은 세 대조에 대한 동시 95% CI다. [그림 코드](figure62.py)

## 3. Pulse 폭과 shutter exposure의 독립 효과

| Pulse / exposure (ms) | V61 판독률 → selective 판독률 | 보류 후 optical 쌍 0개인 pilot | Selective − v61 reset 평균 Brier, ×10⁻⁶ | Selective − gate 평균 Brier, ×10⁻⁶ |
|---|---:|---:|---:|---:|
| 0.20 / 0.25 | 4.614% → 2.532% | 91.06% | −0.0129 | +0.0321 |
| 0.20 / 0.80 | 0.423% → 0.001% | 99.996% | −0.0015 | +0.0208 |
| 0.80 / 0.25 | 7.518% → 4.268% | 85.92% | −0.0196 | +0.0475 |
| 0.80 / 0.80 | 7.635% → 2.685% | 90.92% | −0.0317 | +0.0339 |

**0.20 ms pulse에 0.80 ms exposure를 쓰면** 발광 구간이 shutter 안에 전부 들어와도 빛이 켜진 시간은 exposure의 25%뿐이다. 이 조건에서 selective가 읽은 4/394,760쌍은 같은 물체 slot을 네 P/C–A/B counterfactual로 평가한 결과다. 반대로 0.80 ms pulse에서 exposure를 0.25 ms에서 0.80 ms로 늘리면 기존 v61 판독률은 비슷했지만 selective는 4.268%에서 2.685%로 떨어졌다. 광학 설정마다 달라지는 부호·contrast의 confidence를 한 threshold로 판정한 결과다. 네 cell의 전체 평균이 가장 나쁜 조건을 가릴 수 있다. [실제 건수·상호작용](vision62.json)

## 4. 어떤 판독을 보류한 것이 이득이었는가?

**오판 제거로 설명할 수 있는 이득은 0이었다.** 기존 v61의 79,701개 유효 판독 중 selective가 남긴 37,448쌍도, 제거한 42,253쌍도 모두 정확했다. Camera 부호를 결측 처리해도 전자센서는 그대로였다. 실제 P/C와 stick label을 policy에 넣지 않고, 별도 privileged 진단에서만 원래 유효했던 광학 부호를 정답으로 교체했다. 교정할 부호가 없어 네 cell 모두의 loss가 그대로였다. 올바른 판독만 제거하는 경로와 틀린 판독만 제거하는 경로를 독립 구현으로 다시 계산해도 평균 변화 전부가 올바른 판독을 제거하는 경로에서 나왔다. [부호 수리](sign_repair62.json) · [제거 효과 분해](attribution62.json)

**같은 수를 무작위로 남겨도 개선이 유지됐다.** 각 pilot에서 v61의 유효 쌍이 n개이고 selective가 k개를 남겼다면, 그 n개에서 정확히 k개를 남기는 모든 조합의 posterior와 행동을 구하고 균등 평균했다. 따라서 같은 board·표면·물체·판독 수에서 *어느 쌍을 남겼는지*의 효과만 비교한다. 이 대조는 k를 selective에서 가져오므로 k 자체를 정하는 가치까지 제거하지 않는다. [사전 계획](ABLATION_PLAN_KO.md) · [Exact subset enumeration](thinning62.py)

| 같은 reset에서 사용한 광학 정보 | 평균 유효 쌍 / 8회 | 평균 held-out Brier | Fixed 3R+I 선택률 |
|---|---:|---:|---:|
| 기존 v61이 읽은 전부 | 0.20269 | 0.157078349199 | 0.3497% |
| Confidence로 선택 | 0.09524 | 0.157078332766 | 0.2604% |
| 같은 수를 무작위로 유지 | 0.09524 | 0.157078332403 | 0.2626% |
| 광학을 모두 보류하고 전자센서만 사용 | 0 | 0.157078320013 | 0.1933% |

Selective − 동일 수 무작위의 평균 Brier 차이는 **+0.00036 × 10⁻⁶** [pointwise 95% CI −0.00091, +0.00185]였다. 선택을 달리할 여지가 있는 **0 < k < n인 pilot은 1.969%**뿐이었다. 그 pilot만 보아도 차이는 +0.0184 × 10⁻⁶ [−0.0371, +0.1355]로 불확실했다. 동등성 검정을 한 것은 아니다. 광학을 아예 사용하지 않은 reset은 selective보다 평균이 **0.01275 × 10⁻⁶ 낮았고** [selective − electronics-only 95% CI +0.00447, +0.02405], 상시 v54 gate는 그보다도 낮았다. [무작위 보류의 전체 수치](thinning62.json) · [선택 여지가 있던 부분집합](conditional62.json)

**실제로 바뀐 것은 gate로 되돌아간 몇 번의 행동이었다.** 기존 reset과 selective 사이에 행동이 달라진 pilot은 0.1035%였다. 전체의 0.0964%가 fixed에서 gate로, 0.0071%는 gate에서 fixed로 전환했다. A에서는 fixed가 gate보다 평균 42.50 × 10⁻⁶ 유리했고 B에서는 잘못된 fixed 선택의 손해가 평균 80.51 × 10⁻⁶였다. 주 실험에서는 장치의 fault 빈도와 미래 행동의 이익을 독립시켰으므로, source 단서를 덜 쓰는 것이 이 비대칭 손실을 조금 줄였다. Fault–행동의 source 대응을 복원하면 selective − v61 reset은 +0.0181 × 10⁻⁶ [−0.0015, +0.0386], 그 대응을 뒤집으면 −0.0509 × 10⁻⁶ [−0.1014, −0.0148]였다. [환경별 손익](SCORE_KO.md)

## 5. 기존 판독에도 실제 오류를 넣으면 어떻게 되는가?

주 실험에서 기존 v61 판독이 모두 맞았기 때문에, confidence가 **오류를 감지하는 능력**을 주 실험만으로 평가하기는 어렵다. 사후 양성 대조로 저장된 같은 raw fixture에서 impact의 위쪽 marker patch만 바꾸었다. Reference 영상, LED, switch·photodiode 파형, 실제 물리 outcome과 threshold는 그대로 유지했다. 한 조건은 임의의 frame 하나에 3 pixel의 수평 추적 오차를 넣었다. 다른 조건은 fixture 대비 marker의 상대변위를 시간 전반에 걸쳐 반대 방향으로 따르도록 만들어 일관되게 잘못된 대상을 추적하는 경우를 모사했다. 시뮬레이터의 실제 부호는 교란을 만드는 데 쓰지 않았다. [Raw-frame 대조 구현](track_stress62.py)

| 저장된 fixture의 교란 | V61 오판 / 판독 | Selective 오판 / 판독 |
|---|---:|---:|
| 원래 영상 | 0 / 140 | 0 / 80 |
| 임의 frame 하나의 3-pixel 이동 | 24 / 163 | 0 / 72 |
| 시간상 일관된 상대 운동 반전 | 140 / 140 | 72 / 72 |

한 frame의 오류가 새 변형처럼 보여 기존 reader의 판독 건수가 오히려 늘어날 수 있었다. 두 변위 추정의 일치를 요구하는 규칙은 그 조건의 오판을 제거했다. 전체 궤적이 일관되게 틀릴 때는 selective가 통과시킨 판독도 모두 틀렸다. 이는 24개 catalog의 제한된 fixture를 네 광학 조건과 paired 가설에서 재생한 메커니즘 검사다. 실제 표면의 반사 발생률 또는 이 교란을 받은 deployment의 Brier를 측정한 결과로 확대할 수 없다. [양성 대조 원자료](track_stress62.json)

## 6. GPU 및 sensing 비용

Node55의 빈 NVIDIA RTX PRO 6000 Blackwell Server Edition 한 장에서 raw clip과 ADC waveform 판독, confidence 판정, 관측 선택, 공통 decoder, 최종 GPU forecast까지 측정했다. 호출당 2개 표면, 총 512개 deployment object다. Warmup 뒤 24개 catalog × 4개 광학 조건에서 6개 정책을 Williams crossover로 실행했다. 정책당 96회이며, 전후 CUDA synchronization을 포함한다. 장비의 촬영·접촉·reset 대기와 파일 읽기·모델 초기화는 제외했다. [실행 증거와 benchmark](GPU_WORLD_KO.md)

Selective reset에서 기존 reset·기존 sham·gate의 GPU p90을 뺀 차이는 각각 **−0.050 ms** [동시 95% CI −13.667, +0.977], **+0.105 ms** [−1.148, +1.037], **+0.824 ms** [−0.172, +1.639]였다. Board를 먼저 재추출한 민감도에서도 세 구간 모두 0을 포함했다. 별도 CPU 측정으로 32개 clip 판독과 routing만 비교했을 때도 selective − 기존 reset은 +0.025 ms [−0.141, +0.228]였다. 96개 block에서 네 주 정책의 후속 관측 선택은 모두 같았다. 따라서 이 p90은 실제로 각 reader를 실행한 대표 표본의 비용이고, 드물게 행동이 바뀌는 경우의 시간은 별도로 평가했다. [GPU 원자료](timing62.json) · [경로 확인](route_timing62.json) · [CPU 원자료](reader_cost62.json)

실제로 후속 관측이 달라지는 경우를 따로 확인했다. Target label을 쓰지 않고 서로 다른 16개 catalog에서 한 pilot씩 골라 원래 영상·파형을 재생했다. Selective가 기존 reset의 fixed 선택을 gate로 바꾼 경우는 13개, 반대는 3개였다. 같은 512개 물체 단위로 정책당 32회 측정한 GPU p90은 selective 555.024 ms, 기존 reset 553.494 ms, 상시 gate 555.187 ms였다. 이 표본은 전환 사례만 모았고 board 7개를 포함하므로 전체 deployment의 p90은 위의 96개 block으로 판단한다. [전환 사례의 GPU 계측](RARE_TIMING_KO.md) · [원자료](rare_timing62.json) · [희귀 전환 checksum](rare_gpu62.sha256)

한 표면에서 두 camera arm의 16개 clip은 128개 frame, 51,840 bytes다. Dual reader의 두 ADC 채널은 16개 contact에 대해 4,096 bytes를 기록한다. Selective와 기존 reset은 각각 전자 reset 4회, sham은 0회다. Peak LED power가 일정할 때 16회 발광의 총 on-time은 pulse 폭에 따라 3.2 또는 12.8 ms다. 128개 frame의 총 노출 시간도 shutter에 따라 32 또는 102.4 ms로 바뀌지만 같은 frame 주기 안의 값이다. 두 값을 장비 wall time에 더하지 않았다. 같은 수의 impact가 같은 광학 에너지라는 가정도 두지 않았다. 장비 setup과 물리 대기 시간은 미측정이다. [비용 장부](cost62.json) · [계측 조건](GPU_WORLD_KO.md)

## 7. 논문 기여와 다음 검증

이 결과는 **보류 정책 자체의 일반적 우월성**을 뒷받침하지 않는다. 논문에서 근거가 있는 주장은 세 가지다. 첫째, 유료 영상 대부분에서 두 부호를 모두 얻지 못하면 계측기의 판독률과 실제 행동의 변화율은 크게 다를 수 있다. 둘째, 정확한 판독을 버려도 source의 행동 선택 규칙이 새 환경에 맞지 않으면 강건해 보이는 평균 개선이 생길 수 있다. 셋째, 같은 수의 무작위 보류와 비교해야 어떤 판독을 골랐는지의 기여를 따로 평가할 수 있다. 이번 주 실험에서는 그 추가 기여가 확인되지 않았다. [판독률](vision62.json) · [같은 수 대조](thinning62.json) · [v61 선행 실험](../switch_diode_v61/REPORT_KO.md)

다음으로 우선할 검증은 threshold를 새로 고르는 것보다 **기존 reader가 그럴듯하게 틀린 사례를 독립 기준으로 확보하는 것**이다. 실제 board·반사 표면·날짜를 분리하고 force 또는 독립 영상으로 reference–impact 부호를 기록해 결측 사례도 평가하는 방식이 적합하다. 단발성 추적 오류와 일관된 marker 오인식을 구분하고, 같은 수의 무작위 보류 및 상시 gate와 비교해야 한다. 이번 primary에는 exposure로 정규화한 밝기와 고정 noise, global shutter를 사용했다. Exposure에 따른 shot noise, 실측 sensor response, 실제 반사율·마모를 반영한 실험은 남아 있다. [실물 detector의 timing response](https://www.volkerschatz.de/science/doi-10.1088_1361-6501_aa5d8a/camtimpap.pdf) · [실물 slip 검증](https://www.mdpi.com/1424-8220/18/2/333)

## 문헌 검토와 연구 심사

### 1·6. 연구 질문과 방법을 원문별로 비교

| 논문과 질문 | 방법·표본 | 결과와 결론 | v62 적용 범위 |
|---|---|---|---|
| Schatz (2017). Trigger와 exposure의 실제 관계를 알 수 있는가? | LED 점등 시각과 camera trigger의 상대 지연을 바꾸어 산업용 camera 3종의 밝기 응답을 측정했다. | 발광과 노출의 겹침으로 trigger delay 및 exposure duration을 추정했다. 한 CMOS 장치에는 nominal exposure가 끝난 뒤에도 잔여 감도가 있었다. | 직사각형 pulse–exposure overlap 모형이 출발점이 될 근거이며, 실제 detector의 시간 응답을 모두 설명하지 않는다. [원문 §§2,7](https://www.volkerschatz.de/science/doi-10.1088_1361-6501_aa5d8a/camtimpap.pdf) |
| Meyer et al. (2026), *RocSync*. 광학 clock으로 여러 camera의 exposure를 정렬할 수 있는가? | 두 camera를 사용한 독립 14분 recording 5회. 별도로 multi-view pose와 3D reconstruction을 평가했다. | Hardware 기준에 대한 평균 RMSE는 1.34 ms였다. 빠른 움직임에서 timing과 interpolation을 함께 개선하자 재구성이 나아졌다. | Camera 간 정렬이 핵심인 과제다. 한 sensor의 판독을 보류한 뒤 다른 sensor로 선택을 계속해야 하는 v62의 gain을 직접 예측하지는 않는다. [원문 §5](https://www.mdpi.com/1424-8220/26/3/1036) |
| Geifman & El-Yaniv (2017). 일부 예측을 보류해 목표 risk를 만족시킬 수 있는가? | CIFAR와 ImageNet에서 학습된 classifier의 confidence를 사용해 risk–coverage를 평가하고 calibration 표본에서 threshold를 정했다. | 논문의 ImageNet 예에서는 약 60% test coverage에서 top-5 error 2%의 risk bound를 보고했다. | Accepted classification의 error에 관한 결과다. 이 논문의 보장이나 threshold를 v62의 heuristic reader나 전체 deployment Brier로 옮길 수 없다. [원문 §§2–4](https://papers.neurips.cc/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf) |
| Jones et al. (2021). 전체 selective accuracy가 좋아지면 각 집단도 좋아지는가? | Vision·NLP 5개 dataset에서 group별 risk와 coverage 및 margin distribution을 비교했다. | 전체 accuracy가 올라가도 일부 worst group은 악화됐다. 예를 들어 CelebA에서는 낮은 coverage에서 worst group accuracy가 내려갔다. | 데이터에서 관측된 그룹 차이로 sensor board 간 실제 인과관계를 정할 수 없다. 그룹별 판독 기회와 오판을 따로 기록할 방법론적 이유가 된다. [원문 Fig. 2–3](https://openreview.net/pdf?id=N0M_4BkQ05i) |
| Rigi et al. (2018). 실제 물체의 incipient slip을 영상으로 판독할 수 있는가? | 5개 물체에 대한 37회 실험에서 DAVIS의 contact area 변화와 독립 1,000 fps 영상을 대조했다. | 평균 detection latency 44.1 ms였고 성능은 물체별로 달랐다. | Slip onset latency는 shutter exposure, GPU latency, terminal Brier와 각각 다른 측정치다. [원문 §4](https://www.mdpi.com/1424-8220/18/2/333) |

### 2·3. 주제별 근거와 knowledge map

**Pulse와 shutter의 겹침 → optical contrast 및 motion blur → 부호와 판독 보류 → 남은 전자·광학 관측 → P/C posterior → 관측 정책 → held-out stick 확률**이 이번 질문의 연결 구조다. 첫 연결에는 실물 optical bench 근거가 있다. 중간의 confidence–coverage 교환에는 supervised classification 근거가 있다. 실물 slip을 영상으로 볼 수 있다는 근거도 있지만, 마지막 정책 선택의 손익까지 동일 표면·동일 예산·새 board에서 검증한 결과는 이 문헌 집합에서 확인되지 않았다. [Schatz](https://www.volkerschatz.de/science/doi-10.1088_1361-6501_aa5d8a/camtimpap.pdf) · [Geifman & El-Yaniv](https://papers.neurips.cc/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf) · [Rigi et al.](https://www.mdpi.com/1424-8220/18/2/333)

### 4·5·7. 공백, 상반돼 보이는 결과와 근거 강도

- **근거가 강한 부분.** Exposure와 발광 구간을 맞추는 것이 광학 readout을 바꾼다는 근거는 직접 측정에서 나온다. 실제 camera의 sensitivity kernel, rolling shutter, 반사·가림 등은 setup에 따라 달라져 기기 간 정확도를 하나의 숫자로 환산할 수 없다. [Schatz](https://www.volkerschatz.de/science/doi-10.1088_1361-6501_aa5d8a/camtimpap.pdf) · [Meyer et al.](https://www.mdpi.com/1424-8220/26/3/1036)
- **조건부로 지지되는 부분.** Confidence로 보류하면 평균 판독 정밀도가 좋아질 수 있다. 전체 평균을 다룬 selective risk와 group 내부의 selective risk는 분모가 다르다. 두 문헌의 결과는 양립하며, 특정 장치에서 거부한 모든 사례가 어렵거나 오판이었다는 근거는 되지 않는다. [Geifman & El-Yaniv](https://papers.neurips.cc/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf) · [Jones et al.](https://openreview.net/pdf?id=N0M_4BkQ05i)
- **실물 근거가 부족한 부분.** Reject한 뒤 다른 센서로 계속하는 정책의 손익은 유효 판독의 accuracy만으로 계산할 수 없다. 실물 contact dataset에서는 표면·물체에 따라 감지 성능이 달랐고, 같은 판독 수의 무작위 보류와 비교한 미래 stick 예측은 보고되지 않았다. v62는 이 부분을 합성 데이터에서 검증한다. [Rigi et al.](https://www.mdpi.com/1424-8220/18/2/333) · [Jones et al.](https://openreview.net/pdf?id=N0M_4BkQ05i)

### 8. 문헌과 미해결 결과에서 나온 연구 질문 열 가지

| 질문 | 왜 필요한가 | 변수와 접근 |
|---|---|---|
| 1. 실제 camera의 temporal response가 길면 최적 exposure가 바뀌는가? | Nominal window 밖의 감도가 측정됐다. [Schatz](https://www.volkerschatz.de/science/doi-10.1088_1361-6501_aa5d8a/camtimpap.pdf) | Photodiode와 camera별 측정 kernel로 동일 fixture를 재생한다. |
| 2. LED 폭을 늘려도 광량 총합을 고정하면 coverage 변화가 남는가? | 시간 겹침과 photon 수가 함께 바뀔 수 있다. [Schatz](https://www.volkerschatz.de/science/doi-10.1088_1361-6501_aa5d8a/camtimpap.pdf) | Peak irradiance 고정 및 total photon budget 고정 arm을 교차한다. |
| 3. Rolling shutter에서 LED와 물체 표식의 행 차이가 보류를 유발하는가? | 영상 위치에 따른 timing bias가 가능하다. [Meyer et al.](https://www.mdpi.com/1424-8220/26/3/1036) | Row offset과 exposure를 독립 변경해 sign risk와 coverage를 측정한다. |
| 4. Sub-frame interpolation이 판독을 더 구해도 Brier가 낮아지는가? | 영상 geometry에서 효과가 보고됐다. [Meyer et al.](https://www.mdpi.com/1424-8220/26/3/1036) | 동일 물체에서 frame interpolation 유무만 바꾸고 downstream 행동 전환을 센다. |
| 5. 유광·연성 표면에서 보류된 물체가 더 어려운가? | 실제 물체별로 slip detection의 정확도와 latency가 달랐다. [Rigi et al.](https://www.mdpi.com/1424-8220/18/2/333) | 별도 force reference를 두고 미판독 물체까지 층화한다. |
| 6. Board가 바뀌어도 calibrated risk–coverage를 유지하는가? | 기존 selective risk 보장은 해당 calibration 분포에 기대고 있다. [Geifman & El-Yaniv](https://papers.neurips.cc/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf) | Board family를 통째로 hold out해 coverage와 calibration error를 측정한다. |
| 7. 평균 coverage가 같아도 최악 표면의 판독 기회가 줄어드는가? | Group disparity가 평균에 가려질 수 있다. [Jones et al.](https://openreview.net/pdf?id=N0M_4BkQ05i) | Board·surface별 coverage, risk와 worst-group interval을 비교한다. |
| 8. 부호를 버리는 대신 불확실한 likelihood로 입력하면 어떤가? | Threshold 기반 방법은 선택된 사례의 위험을 다루며 후속 행동 비용을 학습하지 않는다. [Geifman & El-Yaniv](https://papers.neurips.cc/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf) | Source에만 맞춘 sign-error likelihood와 hard abstention을 같은 물체에서 비교한다. |
| 9. 결측 사유가 실제 접촉과 장치 문제를 구분하는가? | 물체별 검출 실패와 group별 coverage가 동시에 바뀔 수 있다. [Rigi et al.](https://www.mdpi.com/1424-8220/18/2/333) · [Jones et al.](https://openreview.net/pdf?id=N0M_4BkQ05i) | LED 미검출·변위 부족을 별도 code로 기록하고 보지 못한 board에서 검증한다. |
| 10. 판독을 포기한 후 실제 장비의 다음 행동이 비용을 회수하는가? | Reconstruction과 slip onset 지표는 policy risk와 단위가 다르다. [Meyer et al.](https://www.mdpi.com/1424-8220/26/3/1036) · [Rigi et al.](https://www.mdpi.com/1424-8220/18/2/333) | 같은 8회 impact에서 관측별 대기, reset 시간과 실제 binary outcome을 측정한다. |

### 9. 심사 관점

Confidence threshold와 exposure control 각각에는 선행연구가 있다. v62의 논문 가능성은 판독 수를 맞춘 대조와 한 번도 읽지 못한 물체까지 포함한 Brier가 가장 단순한 gate와 어떤 차이를 보이는지에 달려 있다. 실물 광학·접촉 계측과 target 장치에서 유효한 risk calibration이 없으므로 새 하드웨어에서의 우월성 또는 risk 보장으로 확장할 근거는 부족하다. 이번 문헌 집합은 질문에 직접 관련된 다섯 원문을 읽은 비교이며, 최초성에 대한 체계적 전수 조사는 별도 과제다. [Schatz](https://www.volkerschatz.de/science/doi-10.1088_1361-6501_aa5d8a/camtimpap.pdf) · [Geifman & El-Yaniv](https://papers.neurips.cc/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf) · [Jones et al.](https://openreview.net/pdf?id=N0M_4BkQ05i)


## 서버 및 재현 자료

서버 디렉터리는 `datacenter3:/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/exposure_abstention_v62`다. GPU forecast와 latency는 실행 전 memory·process를 재확인하고 guard를 켠 채 수행했다. Optical simulation, physics solver, statistical inference는 Slurm compute node에서 CUDA를 숨기고 실행했다. 계측, 정책 채점, GPU 실행은 분담했으며 exact random thinning과 privileged sign repair는 원래 채점에서 독립된 코드 경로로 검산했다. Primary에는 사전에 고정한 cutoff만 사용했다. Raw tracking 교란 및 희귀 전환 timing은 사후 진단으로 구분했다. [Physics·forecast checksum](deployment62.sha256) · [광학 checksum](vision_bundle62.sha256) · [정책·Brier checksum](score_results62.sha256) · [GPU checksum](gpu_results62.sha256)
