# v61 · LED가 늦었는지, camera가 늦었는지 알면 다음 물체의 stick/slip을 더 잘 예측하는가?

**새 합성 board 8개와 표면 48개에서 contact switch와 독립 photodiode는 LED 발광 지연과 camera 촬영 지연을 분리했다.** 두 지연을 함께 추정할 수 있었던 impact의 비율은 91.06%였고, 각 지연의 MAE는 0.068 ms와 0.211 ms였다. 같은 8회 impact에서 이 보정은 LED-only reset의 평균 Brier loss를 0.765 × 10⁻⁶ 줄였다. 그러나 **두 계측기를 모두 쓰는 조건에서 reset을 추가한 이득은 sham과 구별되지 않았고, 항상 v54 gate를 쓴 정책보다 평균 loss가 0.190 × 10⁻⁶ 높았다.** 이 주 평가는 장치의 common-mode fault 빈도와 배포 시 유리한 행동의 관계를 독립시킨 경우다. [계측 결과](vision61.json) · [정책 검증](SCORE_KO.md)

## 1. 두 지연을 어떻게 분리했나

물체에 **처음 접촉한 순간**, LED가 **실제로 빛난 순간**, camera가 **그 빛을 담은 frame**을 구분했다. Contact switch는 접촉에 반응하고 독립 photodiode는 실제 발광에 반응한다. 두 전압을 같은 16 kHz ADC의 서로 다른 채널에 기록한다. Camera는 자기 buffer 안의 LED 밝기를 1,000 fps로 기록한다. Switch와 photodiode에는 서로 다른 response bias·gain·noise·dropout이 있고, switch에는 bounce도 생긴다. ADC의 공통 시간 원점은 contact마다 달라진다. 물리와 광학을 서로 다른 센서로 보되 두 전압의 차이를 잴 수 있는 공통 timebase를 가정했다. [계측 사전 명세](VISION_PLAN_KO.md) · [구현](vision61.py)

Switch와 photodiode가 임계값을 넘는 시각을 각각 *s*, *p*, 영상의 LED frame index를 *j*(0부터 시작)라고 하자. 정시에 접촉하면 buffer index 2에 해당하도록 촬영을 설정했다. **LED lag 추정값은 p − s**, **camera lag 추정값은 (2 − j) × 1 ms + (p − s)**다. 이 실험에서 camera lag는 접촉에 대한 *실효 촬영 구간의 offset*이다. 입력으로 camera trigger TTL, 물리 contact 정답, board별 지연표를 주지 않았다. 실제 장치에서 전기 trigger 전달 지연과 shutter 내부 지연까지 나누려면 exposure strobe를 따로 측정해야 한다. [Reader 코드](vision61.py) · [Exposure와 photodiode를 직접 비교한 실측](https://research.monash.edu/files/334027026/279962385_oa.pdf)

Impact에서 LED와 camera에 각각 0 또는 +2 ms를 주입한 네 경우를 완전히 교차했다. 0은 추가 지연 설정이 0이라는 뜻이며, board 고유 지연과 clip jitter는 남는다. 세 reader는 같은 contact와 raw 데이터를 받는다. **Nominal**은 고정 frame, **LED-only**는 영상의 발광 frame, **dual**은 발광 frame과 두 ADC 파형을 사용한다. Dual은 edge나 표식이 없거나 필요한 판독 구간이 buffer에 없으면 해당 부호를 결측 처리한다. [설계와 광학 결과](VISION_KO.md)

## 2. 새 board와 표면에서 무엇을 예측했나

- **원인 판별.** 같은 표면의 두 물체에서 reference와 impact의 변형 부호가 함께 바뀌는 현상을 만들었다. P에서는 실제 mismatch가 session 안에서 상관되고, C에서는 실제 물체별 mismatch가 독립이지만 두 전자센서가 공통의 잘못된 upstream 신호를 받을 수 있다. 한 센서만 reset하면 그 센서의 local latch가 해제된다. 공통 upstream fault와 물리 표면은 그대로다. P/C의 정답은 평가에서만 사용했다. [P/C likelihood](../sensor_reset_v59/route59.py) · [새 센서 모형](vision61.py)
- **같은 예산의 정책 판별.** 4개 session마다 다른 물체 2개를 무작위 순서로 검사한다. 각 물체에 2.0 N reference와 impact를 1회씩 가한다. 각 session의 둘째 물체 직전에 무작위로 선택한 전자센서 한 대를 reset하거나, 동일한 ID로 sham을 시행한다. 이 pilot을 마치면 후속 deployment에서 네 번째 관측으로 항상 impact를 사는 **fixed 3R+I** 또는 앞선 세 reference에 따라 네 번째 관측을 고르는 **v54 gate**를 선택한다. 세 번째 대조군은 항상 v54 gate를 쓴다. [정책과 원인 판정 계획](SCORE_PLAN_KO.md)
- **어려운 주 평가.** 유리한 정책이 다른 A/B 환경의 marginal mismatch는 둘 다 29%다. C에서 source의 높은 fault 빈도(29.58%)와 낮은 빈도(4.08%)는 주 평가의 각 A/B에 반반 배정한다. P/C나 장치 빈도를 알아낸 것만으로 정답 행동이 결정되지 않게 했다. Source에서 정한 likelihood와 odds cutoff 2.016, downstream decoder는 동결했다. P/C와 A/B는 각각 균등 가중하고 네 timing cell도 균등 가중했다. Source의 fault–A/B 관계를 유지하거나 뒤집은 평가는 별도로 보고한다. [채점 코드](score61.py)
- **일반화 단위.** 이전에 사용하지 않은 합성 sensor board 8개 각각을 새 표면 48개 모두에 배치했다. 한 pilot의 16개 contact는 같은 board를 쓴다. 한 표면·timing cell·가설별 512개 pilot에는 board당 64개 반복이 들어 있다. 독립 deployment는 24개 catalog, 2개 기존 rate law, held-out 물체 12,288개다. Board와 catalog를 각각 재추출하는 paired crossed bootstrap 6,000회로 정책 차이의 불확실성을 구했다. 8개 board는 하나의 회로 모형에서 gain, noise, timing, latch 등을 새로 추출한 profile이다. [World 설계](WORLD_PLAN_KO.md) · [채점 사전 명세](SCORE_PLAN_KO.md)

## 3. 시간 정렬의 정확도와 실패 조건

각 cell의 고유 impact clip 196,608개 중 **179,038개(91.06%)**에서 두 지연을 함께 추정했다. 유효 clip의 **LED lag MAE는 0.068 ms**(bias −0.056 ms), **camera lag MAE는 0.211 ms**(bias +0.074 ms)였다. Board별 MAE는 각각 0.034~0.103 ms와 0.141~0.357 ms였다. LED에만 +2 ms를 주면 추정된 LED lag가 2.000 ms 늘어났고 추정 camera lag의 변화는 0.00001 ms 미만이었다. Camera에만 +2 ms를 주면 추정된 camera lag가 2.000 ms 늘고 LED lag의 변화는 0이었다. 이는 같은 noise를 공유한 2 ms intervention에 대한 차이다. 개별 clip의 절대 시간이 이 정도 정밀도로 측정됐다는 뜻으로 읽을 수 없다. [Board·cell별 timing 원자료](vision61.json)

| 추가 지연 (LED, camera), ms | LED-only 변형 쌍 판독률 | Dual 변형 쌍 판독률 | 같은 쌍에서 LED-only 오판 → dual 오판 |
|---|---:|---:|---:|
| (0, 0) | 50.38% | 45.49% | 0 → 0 / 178,581쌍 |
| (0, 2) | 53.71% | 48.44% | 0 → 0 / 190,181쌍 |
| (2, 0) | 0% | 45.49% | 공통 판독 0쌍. Dual 단독 판독 178,653쌍, 오류 0건 |
| (2, 2) | 33.87% | 48.44% | 118,639 → **0** / 118,671쌍 |

변형 쌍 판독률의 분모는 reference와 impact의 진짜 변형이 활성인 392,688개 paired counterfactual이다. 두 시각을 추정한 91.06%와 달리 실제로 부호를 읽으려면 **두 접촉 모두** 충분한 변형·가시성·시간 구간을 가져야 한다. LED가 정시에 켜질 때는 두 ADC edge를 추가로 요구하면서 판독을 포기하는 일이 늘었다. 둘 다 늦을 때 LED-only는 복원 운동을 첫 변형으로 오해했다. [공통 판독 집합과 분모](vision61.json)

**같은 frame에 보이는 두 사건.** 별도의 raw fixture에서 같은 384개 impact를 (0,0)과 (2,2)로 짝지었다. 발광 frame의 histogram은 두 조건에서 완전히 같았다. 각각 331건이 index 2, 34건이 index 3, 19건이 결측이었다. 이 한 값으로 두 조건을 구별하는 최선의 empirical balanced accuracy는 **50%**였다. 두 파형을 추가한 사전 지정 판별은 조건당 **342/384건을 분류했고 모두 맞혔다**. 나머지 42건을 무작위로 추측하게 한 전체 기대 정확도는 **94.53%**였다. 영상 전체를 학습한 분류기의 한계에 관한 주장은 아니다. [사전 명세](CLOCK_PLAN_KO.md) · [대응 대조](clock61.json)

**두 기준선이 함께 움직이는 경우와 한쪽만 움직이는 경우.** 같은 raw ADC 파형 두 개를 함께 1 ms 늦추면, 각 cell에서 유효 부호 수·오류 수·선택 행동이 모두 같았다. 반대로 switch만 1 ms 늦추면 이 fixture에서 완전히 읽은 reference–impact 쌍이 0개가 됐다. Diode만 늦추면 (2,2)의 판독은 376쌍에서 24쌍으로 줄었다. 한 채널 전체를 가리거나 영상 속 LED를 지우면 dual reader는 결측을 반환했다. 이 대조가 검증한 것은 공통 시간 원점의 소거와 differential bias에 대한 취약성이다. Fixture의 pair를 독립적인 새 장치로 세지 않았다. [Clock intervention 결과](clock61.json) · [채널 차폐 대조](vision61.json)

**유한한 pulse와 shutter를 고려한 추가 검사.** 주 renderer는 보이는 LED를 가장 가까운 한 frame에 기록한다. 사후 대조에서는 같은 384개 impact fixture의 원래 ADC 파형과 광학 밝기를 유지하고, LED pulse와 exposure window가 겹친 길이에 비례해 밝기를 다시 계산했다. (2,2)에서 두 지연을 읽은 비율은 원래 342/384건(89.1%)에서 0.20 ms pulse·0.25 ms exposure일 때 **67/384건(17.4%)**으로 떨어졌다. 0.80 ms pulse·0.25 ms exposure에서는 263/384건을 읽었으나 유효 표본의 camera lag MAE가 0.191 ms에서 **0.560 ms**로 늘었다. 광학 channel만 바꾼 fixture 검사이므로 아래 정책 Brier와 GPU 표는 주 renderer 조건에 해당한다. 이 결과 때문에 91%의 판독률을 실물 장비의 예상 성능으로 사용할 수 없다. [Pulse–exposure 구현](exposure61.py) · [측정값](exposure61.json)

## 4. Held-out 정책 비교

주 endpoint는 frozen decoder의 stick 확률과 binary outcome 사이의 squared error를 물리 생성 확률에 대해 적분한 평균 held-out Brier다. Session loss p90은 별도 binary replay에서 session별 128개 물체의 평균 loss를 구한 뒤 pooled 분포를 반전해 계산했다. 정책 판별률은 각 표면에서 실제 loss가 낮았던 행동에 대한 balanced accuracy다. 상시 gate는 P/C 진단을 출력하지 않는다. [평가 설계](SCORE_PLAN_KO.md)

| 정책 | 원인 판별률 | 실제 저손실 정책 판별률 | 평균 held-out Brier ↓ | Session loss p90 ↓ | GPU p90, ms/512 objects ↓ |
|---|---:|---:|---:|---:|---:|
| Dual reset + camera | 54.722% | 50.016% | 0.158664300 | 0.189778805 | 489.066 |
| Dual sham + camera | 54.097% | 50.024% | 0.158664302 | 0.189778909 | 488.846 |
| 항상 v54 gate | — | 50.000% | 0.158664110 | 0.189778909 | 480.114 |
| LED-only reset + camera | 52.184% | 50.194% | 0.158665066 | 0.189778909 | 480.668 |
| Nominal reset + camera | 52.298% | 50.401% | 0.158666485 | 0.189779237 | 480.844 |

**Reset이 sham을 이겼는가?** 원인 판별률은 **+0.625 pp** [95% CI +0.412, +0.839] 높아졌다. 반면 reset − sham의 평균 Brier는 **−0.002 × 10⁻⁶** [동시 95% CI −0.066, +0.057]였다. Reset − gate는 **+0.190 × 10⁻⁶** [+0.105, +0.287]로 손해였다. Session loss p90의 차이는 두 대조 모두 −0.104 × 10⁻⁶였고 구간은 각각 [−1.073, +0.298], [−1.267, +1.281]이었다. Board 8개 각각에서 gate의 평균 Brier가 낮았으며, 어떤 board 하나를 제외해도 이 결론이 유지됐다. Ultra solver와 독립 binary replay 두 번에서도 reset과 sham의 평균 차이는 불확실했고 gate가 reset보다 유리했다. [대응 CI와 검산](SCORE_KO.md)

**두 지연을 구별한 이득은 언제 컸는가?** (2,2)에서 dual reset은 LED-only reset의 잘못된 부호를 복구하면서 평균 Brier를 **3.198 × 10⁻⁶ 낮췄다**. 반면 (2,0)에서 LED-only는 영상 판독을 모두 포기해 원인 판별률이 50%였다. Dual은 원인을 54.60%까지 맞혔으나 평균 Brier는 LED-only보다 **0.176 × 10⁻⁶ 높았다**. 결측일 때 gate에 가까운 행동을 유지한 편이, 원인 단서를 가지고 source 기준으로 행동을 바꾼 것보다 이 조건에서 나았다. 사전 지정한 네 조건을 균등 평균하면 dual reset − LED-only reset의 Brier는 −0.765 × 10⁻⁶ [−1.197, −0.400]였다. [Timing별 대응 비교](score61.json) · [Factorial interaction](factorial61.json)

![같은 frame에 LED가 보이는 두 경우의 ADC 파형과 지연별 정책 손익](switch_diode61.png)

왼쪽은 같은 물체에서 저장한 raw contact switch와 photodiode trace다. 두 사건에서 LED가 같은 camera frame에 있지만 switch에서 발광까지의 간격이 달라진다. 오른쪽은 label을 보지 않는 두 reader가 선택한 정책에서 항상 v54 gate의 평균 Brier를 뺀 값이다. [그림 코드](figure61.py) · [정책별 원자료](score61.json)

## 5. 계측이 정확해진 뒤에도 손실이 남는 이유

A에서는 fixed 3R+I가 v54 gate보다 평균 **46.626 × 10⁻⁶** 이득이고 B에서는 **77.858 × 10⁻⁶** 손해였다. 이 순위가 A/B 모두 48개 표면 각각에서 확인됐다. 그럼에도 dual reset이 fixed를 고른 비율은 A **1.293%**, B **1.261%**로 유사했다. 원인 정보와 유리한 행동을 연결하는 source의 관계를 유지하자 reset은 두 대조를 이겼다. 같은 관측의 대응만 바꾸니 손익의 부호가 바뀌었다. [행동 손익·표면별 수치](score61.json)

| C의 fault 빈도와 A/B 관계 | Dual reset 원인 판별률 | Dual reset 정책 판별률 | Reset − sham 평균 Brier, ×10⁻⁶ | Reset − gate 평균 Brier, ×10⁻⁶ |
|---|---:|---:|---:|---:|
| Source와 일치 | 54.722% | 50.859% | −0.147 [−0.220, −0.084] | −0.329 [−0.433, −0.221] |
| 서로 독립 | 54.722% | 50.016% | −0.002 [−0.066, +0.057] | +0.190 [+0.105, +0.287] |
| Source와 반대 | 54.722% | 49.173% | +0.144 [+0.063, +0.223] | +0.710 [+0.534, +0.883] |

**원인을 완벽히 알면 달라지는가?** 별도 사전 지정 분석에서 C의 장치 fault 빈도를 A/B와 독립인 nuisance로 적분했다. True cause를 주지 않고도 원래 frozen rule보다 평균 Brier가 0.189 × 10⁻⁶ 낮아졌지만, gate 대비 차이는 **+0.0006 × 10⁻⁶** [pointwise 95% CI −0.0064, +0.0113]였다. 이 방법이 fixed를 선택한 비율은 0.0107%에 불과했다. 다음으로 **true P/C와 유료 관측을 함께** 주고 P에서만 조건부 odds를 사용한 privileged 진단에서도 gate 대비 차이는 **−0.0194 × 10⁻⁶** [−0.0401, +0.0032]였다. P/C만 단독으로 알려주는 경우에는 두 환경의 prior가 그대로여서 gate를 고른다. 반면 어떤 행동이 더 낮은 loss를 내는지 표면별로 미리 알면 이론상 회수할 수 있는 차이는 평균 **23.313 × 10⁻⁶**였다. 원인을 충분히 잘 맞히지 못한 탓만으로 현재 정책의 한계를 설명하기는 어렵다. 이 보조 계산은 8,000회 crossed bootstrap을 사용했고, true cause는 실제 세 정책에 주지 않았다. [사전 명세](VALUE_PLAN_KO.md) · [구현](value61.py) · [수치](value61.json)

## 6. GPU와 sensing 비용

빈 GPU로 확인한 node55의 NVIDIA RTX PRO 6000 Blackwell Server Edition 한 장에서 forecast와 latency를 측정했다. 한 호출은 2개 표면의 raw clip 32개와 전자센서 및 ADC 파형에서 시작해 512개 deployment object의 stick 확률을 반환한다. Preprocessing·정책 선택·frozen GPU decoder를 포함하고 앞뒤에서 CUDA를 동기화했다. 촬영·물리 reset·디스크 읽기·모델 초기화 시간은 제외했다. 24개 catalog마다 네 timing cell에서 같은 데이터로 6개 arm의 실행 순서를 균형화해 정책당 96개 measured block을 얻었다. [Latency 사전 명세](TIMING_PLAN_KO.md) · [실측값](timing61.json)

GPU p90의 reset − sham은 **+0.220 ms** [동시 95% CI -11.807, +13.277], reset − gate는 **+8.952 ms** [-0.999, +16.128]였다. CI는 16,000회 paired catalog bootstrap을 사용했다. Timing은 board당 3개 catalog를 사용했으므로 8개 board를 먼저 뽑는 nested bootstrap도 계산했다. 같은 두 차이의 구간은 각각 [-13.767, +13.776], [-3.424, +17.250] ms다. 별도 CPU microbenchmark에서 32개 clip의 판독과 routing p90은 dual reset 0.5642 ms, LED-only reset 0.4066 ms였다. [GPU 항목별 검산](GPU_WORLD_KO.md) · [Board cluster 민감도](timing_board61.json) · [CPU 비용](reader_cost61.json)

**물리 예산.** 모든 정책이 표면당 8 impacts + 8 references + 16 mechanical resets를 지불했다. Reset arm에만 4 electronic resets가 추가된다. 두 camera arm은 각 16개 clip(51,840 bytes)을 사용하고, dual reader는 여기에 2개 ADC 채널의 16개 waveform씩(4,096 bytes)을 사용한다. 배선·장비 setup·전자 reset 대기 시간은 미측정이다. 항상 v54 gate는 같은 예산이 배정돼도 쓰지 않는 pilot 영상을 읽을 필요가 없다. 후속 네 번째 deployment observation으로 impact를 구매한 비율은 reset 35.763%, sham 35.738%, gate 34.931%였다. [예산과 구매율](cost61.json)

## 7. 논문 기여와 다음 검증

v60에서 남았던 모호성은 LED와 camera가 같은 방향으로 늦으면 발광 표식이 제대로 맞은 것처럼 보인다는 점이었다. v61은 같은 impact에서 그 모호성을 직접 구성하고, 보정 덕분에 **잘못된 부호를 구한 경우**와 단지 **결측을 유효 판독으로 바꾼 경우**의 downstream 손익이 반대로 갈 수 있음을 보였다. 논문에서 가장 방어하기 좋은 기여는 시간 정렬 알고리즘의 신규성보다 **측정 정확도·원인 식별·정책 가치가 갈라지는 조건을 paired intervention으로 입증한 결과**다. Source와 일치할 때의 작은 평균 이득, 새 표면에서 강한 gate에 진 점, 손실 꼬리에 대한 불확실성을 함께 제시하는 것이 타당하다. [v60 결과](../trigger_fault_v60/REPORT_KO.md) · [v61 조건별 수치](score61.json)

첫 실물 검증은 실제 접촉을 독립 force transducer로 표시하면서 switch, photodiode, exposure strobe를 함께 기록하는 것이다. 같은 물체·표면에서 board와 날짜를 교차하고, 판독 가능 비율을 포함한 두 지연의 오차를 먼저 확인해야 한다. 이후 deployment에서 동일 8회 pilot과 상시 gate를 무작위 배정해 실제 stick outcome과 전자 reset 시간을 측정하면 계측 개선과 정책 개선을 독립적으로 판단할 수 있다. 지금의 8개 board와 두 law는 합성 모형의 반복이고, 유효한 frame에서의 정확도는 광학 pulse와 shutter의 겹침에 민감했다. [실물 timing 측정의 선례](https://research.monash.edu/files/334027026/279962385_oa.pdf) · [실물 slip 검증의 선례](https://www.mdpi.com/1424-8220/18/2/333)

## 문헌 검토와 연구 심사

### 1·6. 논문별 질문, 방법, 표본, 결론과 적용 한계

| 연구와 질문 | 방법 및 평가 단위 | 관측·결론 | v61에 적용할 때의 한계 |
|---|---|---|---|
| Schatz (2017). Trigger와 실제 exposure 사이 지연을 측정할 수 있는가? | 공통 timer로 LED와 camera를 구동하고 상대 지연을 훑어 detector response를 복원했다. 산업용 camera 3종을 측정했다. | Trigger 지연과 exposure duration을 sub-microsecond 수준에서 측정했다. 한 CMOS camera는 nominal exposure 뒤에도 잔여 감도가 있었다. | 공통 timer로 가한 입력은 변형이 시작된 물리 시점을 직접 측정하지 않는다. [원문 §§2,7](https://www.volkerschatz.de/science/doi-10.1088_1361-6501_aa5d8a/camtimpap.pdf) |
| Duke et al. (2019). 저가 high-speed camera의 shutter 신호가 실제 exposure와 얼마나 맞는가? | Chronos를 두 scientific camera와 비교하고, pulsed LED·fast photodiode·exposure TTL·oscilloscope로 timing을 검증했다. | 시험한 Chronos의 exposure start jitter는 21.7 ± 3.4 ns였다. Command 대비 고정 지연은 frame interval에 따라 달랐다. | 장치별 bench 결과다. 실물 contact switch가 언제 닫히는지, 재질에 따라 달라지는지는 측정하지 않았다. [원문 §3.1](https://research.monash.edu/files/334027026/279962385_oa.pdf) |
| Klumpp et al. (2025), *Syntalos*. 여러 장치의 timestamp를 맞추면 실제 사건도 맞는가? | 24 h 이상 clock alignment를 추적하고 6개 장치의 offset을 1 Hz LED로 약 15분 측정했다. | 내부 clock을 맞춰도 장치 간 약 40 ms까지 일정한 offset이 남아 독립 측정이 필요했다. | 가장 빠른 장치를 기준으로 삼은 상대 시각이다. 특정 camera의 지연값을 v61에 대입할 수 없다. [원문 Fig. 3–5](https://www.nature.com/articles/s41467-025-56081-9) |
| Meyer et al. (2026), *RocSync*. 영상에 보이는 clock으로 여러 camera를 맞추면 downstream geometry도 개선되는가? | 두 camera의 14분 녹화를 5회 반복했다. 별도로 움직이는 장면에서 pose와 3D reconstruction을 비교했다. | 평균 synchronization RMSE 1.34 ms였다. Sub-frame interpolation을 함께 썼을 때 빠르게 움직이는 손과 도구의 재구성이 개선됐다. | 두 camera가 같은 순간을 보는 문제와 그 순간이 실제 contact인지 아는 문제는 다르다. 빠르게 움직이는 두 시점의 3D 예시는 정책 Brier의 근거가 아니다. [원문 §§5.1,5.4–5.5](https://www.mdpi.com/1424-8220/26/3/1036) |
| Rigi et al. (2018). 실물 접촉에서 incipient slip을 독립적으로 확인할 수 있는가? | DAVIS의 contact area 변화와 별도 1,000 fps 영상을 대조했다. 5개 물체, 37회 실험이다. | 평균 slip detection latency는 44.1 ms였고 물체별 차이와 noise가 컸다. | Slip detector의 반응 시간과 광학 경로의 time offset은 정의가 다르다. 동일 물체에서 sensing 정책을 바꾼 Brier도 보고하지 않았다. [원문 §4, Table 2](https://www.mdpi.com/1424-8220/18/2/333) |

### 2–5·7. 종합, knowledge map, 공백과 결과 차이

- **Timing의 계측.** Trigger command → 실제 exposure, LED drive → 실제 발광은 각기 다른 경로다. 공통 timer·photodiode·optical clock의 실측은 장치의 경로별 지연을 별도로 검증할 수 있다는 강한 근거를 준다. 반복 측정해 jitter가 작아도 고정 offset이 남을 수 있다. [Schatz](https://www.volkerschatz.de/science/doi-10.1088_1361-6501_aa5d8a/camtimpap.pdf) · [Duke et al.](https://research.monash.edu/files/334027026/279962385_oa.pdf) · [Klumpp et al.](https://www.nature.com/articles/s41467-025-56081-9)
- **Contact와의 연결.** 물체의 실제 변형 → switch가 contact를 검출 → LED 발광을 photodiode가 검출 → camera의 frame index에서 두 시간축을 연결 → reference/impact 변형 부호 → 원인 posterior → 정책 → 미래 stick/slip Brier가 v61의 knowledge map이다. 기존 실물 연구는 slip이나 exposure를 각각 계측했다. 위 전 경로를 새 board와 표면에서 같은 예산으로 비교하는 증거는 검토한 논문에 없다. [Duke et al.](https://research.monash.edu/files/334027026/279962385_oa.pdf) · [Rigi et al.](https://www.mdpi.com/1424-8220/18/2/333)
- **결과가 다르게 보이는 이유.** Controlled bench의 ns 단위 jitter와 multi-device 실험의 ms 단위 offset은 시간의 변동과 평균 차이라는 서로 다른 양이다. RocSync의 reconstruction은 동시성이 직접 필요한 geometry task다. v61에서 timing이 downstream 성능을 바꾸려면 추가로 행동이 바뀌고 그 행동의 loss가 낮아야 한다. 전자에는 실험적 근거가 있고, 후자의 실물 입증은 남아 있다. [Duke et al.](https://research.monash.edu/files/334027026/279962385_oa.pdf) · [Klumpp et al.](https://www.nature.com/articles/s41467-025-56081-9) · [Meyer et al.](https://www.mdpi.com/1424-8220/26/3/1036)

### 8. 후속 연구 질문 열 가지

| 질문 | 확인할 공백과 중요성 | 변수·방법 |
|---|---|---|
| 1. Switch가 실제 접촉보다 얼마나 늦는가? | 시간 기준 자체의 bias가 두 지연 추정을 모두 움직일 수 있다. [Rigi et al.](https://www.mdpi.com/1424-8220/18/2/333) | Stiffness·impact speed를 교차해 switch onset을 별도 force transducer와 비교한다. |
| 2. LED driver와 photodiode amplifier 중 어디서 lag가 생기는가? | 광학·전자 지연은 별도 경로다. [Duke et al.](https://research.monash.edu/files/334027026/279962385_oa.pdf) | LED current, 발광, diode TTL을 같은 oscilloscope에서 동시에 기록한다. |
| 3. Clock drift만으로 접촉 정렬이 언제 깨지는가? | 장기 synchronization 결과를 짧은 high-speed event에 옮길 근거가 부족하다. [Klumpp et al.](https://www.nature.com/articles/s41467-025-56081-9) | Session 길이와 온도를 바꾸고 중간 calibration 유무를 교차한다. |
| 4. Rolling shutter에서 LED와 물체가 다른 행에 있으면 어떠한가? | Frame index가 같은 두 부분의 exposure가 다를 수 있다. [Meyer et al.](https://www.mdpi.com/1424-8220/26/3/1036) | Row 간격·readout direction을 무작위화해 sign error를 측정한다. |
| 5. Detector의 잔여 감도가 반동을 첫 변형으로 보이게 하는가? | 실제 temporal response는 단순 window보다 길 수 있다. [Schatz](https://www.volkerschatz.de/science/doi-10.1088_1361-6501_aa5d8a/camtimpap.pdf) | Exposure duration과 rebound 속도를 바꿔 detector별 timing kernel을 추정한다. |
| 6. Analog threshold를 바꾸면 어떤 재질에서 contact가 먼저 누락되는가? | 물체별 slip 판독 성능 차이가 있다. [Rigi et al.](https://www.mdpi.com/1424-8220/18/2/333) | 표면의 광택·탄성으로 층화하고 독립 force label로 누락까지 센다. |
| 7. 독립 exposure strobe를 추가하면 single LED의 모호성이 풀리는가? | Frame의 노출 시점에 다른 기준을 추가할 수 있다. [Duke et al.](https://research.monash.edu/files/334027026/279962385_oa.pdf) | 같은 영상에 exposure TTL만 추가해 두 지연의 MAE를 비교한다. |
| 8. Calibration이 다른 기종의 board로 전이되는가? | 장치별 고정 offset의 크기가 다르다. [Klumpp et al.](https://www.nature.com/articles/s41467-025-56081-9) | Manufacturer·회로·날짜를 통째로 hold out하고 동일 표면을 모든 board에서 측정한다. |
| 9. Sub-frame interpolation으로 같은 8회에서 얻는 행동 가치가 늘어나는가? | 알려진 이득은 주로 multi-view reconstruction이다. [Meyer et al.](https://www.mdpi.com/1424-8220/26/3/1036) | Decoder를 고정하고 fractional timing, 실제 선택 전환률, paired Brier를 기록한다. |
| 10. 실제 장비에서 reset과 두 기준선 구입이 그 비용을 회수하는가? | Hardware timing 자체가 downstream policy 효용을 정하지 않는다. [Rigi et al.](https://www.mdpi.com/1424-8220/18/2/333) · [Duke et al.](https://research.monash.edu/files/334027026/279962385_oa.pdf) | 동일 impact 예산과 별도 wall-clock 예산에서 session을 무작위 배정하고 loss와 reset 시간을 함께 측정한다. |

### 9. 심사 관점

Photodiode와 trigger의 독립 계측에는 명확한 선행연구가 있다. 새 기여를 평가하려면 동일한 영상으로 어느 지연이 식별되는지, 새 표면에서 그 정보가 실제로 행동을 얼마나 바꾸는지, 손익이 남는지를 순서대로 보여줘야 한다. 실물 contact onset의 독립 기준과 실제 board 간 재현이 없으면 측정한 시간이나 효과 크기를 장비의 사양으로 제시할 수 없다. 이 다섯 원문은 질문에 맞춰 고른 문헌이며 exhaustive novelty search는 별도로 필요하다. [Schatz](https://www.volkerschatz.de/science/doi-10.1088_1361-6501_aa5d8a/camtimpap.pdf) · [Duke et al.](https://research.monash.edu/files/334027026/279962385_oa.pdf) · [Rigi et al.](https://www.mdpi.com/1424-8220/18/2/333)

## 재현 및 데이터 검사

서버의 `datacenter3:/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/switch_diode_v61`에 code, frozen 계획, simulation archive, GPU timing과 본 보고서를 저장했다. 첫 catalog 후보 5100–5123에서 calibration vector 5,375개가 v60의 old support와 겹친 것을 채점 전에 확인했다. 후보 전체를 제외하고 **6000–6023에서 label과 GPU forecast를 전부 다시 생성**했다. 확정된 support와 deployment를 합친 36,864개 vector는 기존 233,472개 및 제외한 후보와 겹치지 않는다. PyBullet fine/ultra 간 label 차이는 879/24,576 endpoint였다. 통계와 timing에는 최종 holdout만 사용했다. [중복 검산](support61.json) · [실행·복원 순서](GPU_WORLD_KO.md) · [Deployment checksum](deployment61.sha256)

계측·정책·서버 실행을 분담해 수행하고, 별도 구현으로 재계산한 frozen reset의 평균 Brier가 주 분석과 절대 오차 10⁻¹⁴ 이내에서 일치하는지 확인했다. 384개 raw impact에 대한 clock injection 및 유한 exposure 검사는 전체 정책 채점에서 분리했다. [계측 archive checksum](vision_bundle61.sha256) · [채점 archive checksum](score_results61.sha256) · [GPU archive checksum](gpu_results61.sha256) · [원인 정보 재계산](value61.py)
