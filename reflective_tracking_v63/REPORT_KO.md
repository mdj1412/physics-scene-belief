# v63 · 반사 표면에서 어느 tracking error를 보류할 가치가 있는가?

**단발성 tracking error만 있을 때 confidence 보류는 같은 수를 무작위로 남기는 것보다 변형 부호의 오류를 줄였다. 이 개선은 held-out Brier loss로 이어지지 않았다.** 일관된 marker 오인식을 넣자 두 방법 모두 유효 판독의 약 3분의 1에서 틀렸다. 네 오류 조건을 같은 비중으로 합치면 confidence와 무작위 보류의 Brier 차이는 0에 가까웠고, 상시 v54 gate의 평균 loss가 두 방법보다 낮았다. 새 board·반사 표면과 독립 strain sensor는 모두 simulation으로 구현했다. [정책 결과](SCORE_KO.md) · [계측 결과](OPTICS_KO.md)

## 1. 실험에서 분리한 세 가지 문제

1. **영상을 읽을 수 있는가?** 원래 reader가 reference·impact의 변형 방향을 모두 읽은 경우에만 두 방향이 다른지 결정한다. Confidence는 그 판독 중 일부를 보류한다. 저신뢰 쌍을 버려도 같은 물체의 전자센서 신호는 남는다.
2. **남긴 판독이 정확한가?** 영상과 다른 noise·dropout을 쓰는 differential strain pickup으로 두 접촉의 부호를 독립적으로 읽었다. Pickup도 틀리거나 읽지 못할 수 있어 simulator truth와의 차이, camera와 동시에 읽힌 비율을 따로 기록했다.
3. **정확해진 판독이 행동을 개선하는가?** 각 표면의 8회 calibration impact를 보고, 이후 물체에서 네 번째 관측을 늘 impact로 사는 fixed 3R+I와, 세 reference를 보고 결정하는 v54 gate 중 하나를 고른다. 새 deployment 물체의 stick 확률을 실제 solver outcome에 대해 Brier loss로 평가했다. 원인 P는 같은 session에서 물리적 변형 방향의 mismatch가 지속되는 상황이고, C는 전자센서의 common-mode fault가 그처럼 보이는 상황이다. 원인 판별과 저손실 정책 판별에는 balanced accuracy를 썼다. [판독 명세](VISION_PLAN_KO.md) · [채점 명세](SCORE_PLAN_KO.md)

## 2. 같은 물체·같은 판독 수·같은 impact 예산

24개 새 catalog의 반사 표면 48개 각각에 새 board 8개를 모두 교차했다. 표면당 256개 pilot, board 하나당 32개다. Pilot은 4개 session × 서로 다른 물체 2개로 구성한다. 물체마다 2.0 N reference 1회와 impact 1회, pilot당 **8 references·8 impacts·16 mechanical resets**다. 두 보류 정책은 둘째 물체 직전에 session당 1회씩 전자센서 하나를 reset한다. 상시 v54 gate에도 같은 calibration impact를 배정하되 그 판독을 사용하지 않는다. 독립 검증은 같은 접촉에서 병렬로 얻으며 재촬영은 없다. 98,304개 물체–impact slot을 네 fault 조건과 P/C·환경 가설에서 재사용했다. 가설이나 replay 수를 독립 표본 수로 세지 않았다. [전체 설계](OPTICS_KO.md)

1,000 fps에서 LED pulse 0.80 ms, shutter exposure 0.25 ms를 고정했다. 표면마다 반사의 밝기·위치·ghost 비중을 새로 뽑았다. **Transient fault**는 임의의 한 impact frame에서 위쪽 marker patch를 좌우 3 pixel 옮긴다. **Coherent fault**는 실제 frame에서 측정한 fixture 상대 운동을 뒤집은 ghost를 상부 patch에 86–99% 비중으로 합성한다. 두 사건의 배정률을 각각 0·35%로 교차하고 독립 난수로 추출했다. 둘 다 허용한 조건의 실제 건수는 무교란 41,430, transient만 22,524, coherent만 22,500, 둘 다 11,850으로 두 배정 간 상관은 −0.0072였다. 한 개의 밝은 반사가 지속적으로 tracking을 지배하는 경우를 겨냥한 stress test다. 기준 접촉에는 교란을 가하지 않았다. [Raw renderer](optics63.py) · [Fault 집계](optical_results63.json)

- **Confidence**는 v62에서 고정한 peak LED contrast, 상대변위 크기, 겹치지 않는 두 frame 쌍의 방향 일치 조건을 모두 요구한다. Target label과 strain 판독, fault 종류나 surface 이름은 보지 않는다. [Frozen reader](../exposure_abstention_v62/VISION_PLAN_KO.md)
- **Same-count random**은 같은 pilot에서 기존 v61 reader가 읽은 n쌍 가운데 confidence가 남긴 수 k를 구하고 정확히 k쌍을 균등하게 선택한다. 평균 Brier는 가능한 모든 subset에서 실제로 행동을 고른 후 그 행동의 loss를 평균했다. Session p90은 행동별 loss 분포를 그 선택 확률로 혼합한 뒤 계산했다. GPU timing에서는 한 subset을 실제로 추출했다. 이 대조는 *어떤 쌍을 남기는가*를 분리하며, k 자체를 추정하는 능력은 두 방법이 공유한다. [선택·채점 구현](score63.py) · [GPU 구현](bench63.py)
- **상시 v54 gate**는 어느 표면에서도 동일한 source-trained acquisition rule을 쓴다. 이후 물체에서는 처음 세 reference에 따라 네 번째 구매가 달라질 수 있다. 세 정책의 terminal decoder는 공통이다. Primary에서는 전자 common-mode fault의 빈도와 실제로 유리한 deployment 행동을 독립시켰다. [비교 정책과 배포 조건](SCORE_PLAN_KO.md)

새 deployment 물체는 12,288개다. Calibration support를 포함한 새 parameter vector 36,864개는 앞선 실험의 고유 vector 350,608개와 겹치지 않았다. 기존 두 contact-law family와 PyBullet을 재사용했다. 변형이 활성인 6,144개 물체 중 2,199개는 변형 방향을 바꾸면 stick/slip label이 바뀌었다. Fine·ultra solver에서 24,576개 binary endpoint 중 796개가 달랐다. 광학 pilot은 부호와 변형 크기에서 영상·strain을 생성하며, 반사를 넣는 과정이 물리 마찰까지 바꾸도록 설계하지 않았다. [World audit](world63.json) · [Archive 중복 검사](support63.json)

## 3. 독립 센서가 확인한 부호 오류

Differential strain pickup의 두 채널을 별도 gain, surface coupling, drift, noise, dropout, ADC quantization으로 생성했다. 접촉 전후 평균의 차이가 5 ADC count 이상일 때만 부호를 읽었다. 두 접촉을 모두 읽은 활성 쌍은 **146,429 / 197,792 (74.03%)**, 그중 reference–impact mismatch 판정 오류는 **1,134건 (0.774%)**이었다. 비활성 쌍에서도 양쪽 센서 부호가 출력된 경우가 5,064/195,424건 있었다. 독립성은 참 변형을 공유하면서 측정 noise와 반사 교란을 분리했다는 뜻이다. 부호 검증은 offline에만 썼다. [Strain waveform·truth 대조](OPTICS_KO.md)

| Transient / coherent 배정률 | 활성 쌍 판독률¹ | Confidence의 오류율² | Random의 오류율² | 독립 strain과 불일치³, confidence / random |
|---|---:|---:|---:|---:|
| 0% / 0% | 5.263% | 0.000% | 0.000% | 0.501% / 0.501% |
| 35% / 0% | 5.069% | **0.269%** | **1.404%** | **0.749% / 1.603%** |
| 0% / 35% | 5.136% | 33.931% | 33.541% | 34.252% / 33.916% |
| 35% / 35% | 4.940% | 33.640% | 33.690% | 33.896% / 33.896% |

¹ 두 보류 정책은 pilot마다 같은 수를 읽는다. ² Simulator truth 기준으로 남긴 reference–impact 쌍 중 mismatch를 오판한 비율이다. ³ 해당 정책과 strain 양쪽이 읽은 쌍에서 비교했다. 이 분모는 정책에 따라 달라지며, strain도 오판할 수 있다. [건수·분모·interval](optical_results63.json)

Transient만 있을 때 confidence − random의 오류율 차이는 **−1.135 pp** [pointwise 95% CI −1.657, −0.569]였다. Coherent만 있을 때는 **+0.390 pp** [−0.456, +1.261], 두 fault가 모두 가능한 조건에서는 **−0.050 pp** [−0.871, +0.760]였다. 두 fault가 한 impact에 *실제로 동시에 발생하고 판독까지 된* 경우만 보면 confidence는 99.09%, random은 94.48%가 틀렸다. 같은 방향으로 잘못 움직이는 ghost는 두 구간의 방향 일치 검사를 통과할 수 있다. [사건별 진단](OPTICS_KO.md)

## 4. Held-out Brier와 실제 후속 선택

| 정책 | 평균 held-out Brier ↓ | Session loss p90 ↓ | 원인 판별률 | 저손실 정책 판별률 | GPU p90, ms / 512 objects ↓ |
|---|---:|---:|---:|---:|---:|
| Confidence 보류 | 0.155701576491 | 0.185992479 | 50.366% | 50.059% | 483.573 |
| 같은 수 무작위 보류 | 0.155701575971 | 0.185992658 | 50.353% | 50.057% | 496.661 |
| 상시 v54 gate | 0.155701480487 | 0.185992479 | — | 50.000% | 496.339 |

같은 수 무작위 보류 대비 confidence의 평균 Brier 차이는 **+0.00052 × 10⁻⁶** [동시 95% CI −0.00985, +0.01084], 상시 v54 gate 대비 **+0.09600 × 10⁻⁶** [+0.03390, +0.17708]였다. 24 catalog와 8 board를 각각 복원추출한 6,000회 paired bootstrap으로 primary 두 비교를 보정했다. 네 cell별 비교 8개를 함께 보정해도 각 cell에서 무작위와의 CI는 0을 포함했고, 상시 gate 대비 CI는 양수였다. 단발성 오류를 줄인 cell만 보더라도 confidence − random은 +0.00238 × 10⁻⁶ [−0.00892, +0.01511]였다. 사전에 정한 두 대조 모두에서의 우위는 성립하지 않았다. 동등성 검정은 수행하지 않았다. [비교별 구간](SCORE_KO.md)

Session loss p90은 별도의 realized binary replay에서 구했다. 두 번째 replay에서는 세 정책 모두 0.186042935였다. 첫 replay의 bootstrap 차이 역시 0을 포함했으며 discrete quantile 때문에 관측 차이가 percentile interval 바깥에 있기도 했다. Fine 결과를 ultra 또는 두 binary replay로 바꿔도 무작위 대비 결론은 불확실했고, gate 대비 평균 손해는 유지됐다. 장치 fault와 행동의 관계를 source처럼 맞춰 주는 보조 조건에서는 gate 대비 차이의 CI도 0을 포함했다. [민감도·board별 성능](score63.json)

![오류 유형별로 같은 판독 수에서 confidence와 random의 오류율 및 Brier 차이를 비교한 그림](tracking_abstention63.png)

왼쪽은 수용한 부호 판독의 오류율 차이와 pointwise 95% CI다. 오른쪽은 모든 deployment 물체의 Brier 차이와 cell별 8개 비교에 대한 동시 95% CI다. Transient만 있는 조건에서는 왼쪽의 interval 전체가 0 아래에 있지만 오른쪽은 0을 포함한다. [그림 코드](figure63.py)

## 5. 같은 수로 어느 판독을 남길 때 실제로 행동이 달라지는가?

- **판독을 바꿔 고를 여지.** 전체 pilot 중 **4.083%**만 0 < k < n이었다. 대부분은 아무 쌍을 읽지 못했거나, 기존 판독을 모두 남겼거나, 모두 보류해서 비교할 subset이 하나뿐이었다. 전체 paid 쌍 기준 두 방법의 판독률은 2.566%, 적어도 한 쌍을 남긴 pilot은 17.212%였다. [채점 원자료](score63.json)
- **행동을 바꿀 여지.** 같은 k의 모든 subset을 별도 구현으로 열거했다. 어떤 쌍을 남기느냐에 따라 fixed 3R+I와 gate 양쪽을 실제로 선택할 수 있었던 pilot은 **0.1836%** [pointwise 95% CI 0.0407, 0.3612]였다. Confidence와 uniform random이 실제로 다른 행동을 고를 확률은 전체의 **0.0864%**였다. 이 두 수치는 가능한 전환과 실제 전환을 각각 센 것이다. [사전 명세](HEADROOM_PLAN_KO.md) · [독립 열거 결과](headroom63.json)
- **바꿀 수 있을 때의 가치.** 각 catalog·surface·환경의 두 행동 중 어느 쪽의 평가 손실이 낮은지 미리 안다고 가정하고, 가능한 subset에서 항상 유리한 행동을 고르는 privileged 진단을 했다. 이때조차 평균 Brier는 **0.155701521961**, 상시 gate는 **0.155701480487**이었다. Confidence가 이 진단까지 줄일 수 있는 손실은 0.05453 × 10⁻⁶ [pointwise 95% CI 0.01171, 0.10884]였다. 이는 현재 k와 frozen router, 이 평가 panel에 한정된 낙관적 한계다. Privileged − gate 차이는 +0.04147 × 10⁻⁶이고 pointwise 95% CI [−0.00098, +0.08640]은 0을 포함한다. 배포 가능한 정책의 성능이거나 다른 k·decoder에도 적용되는 한계로 해석할 수 없다. [Subset별 risk](headroom63.py) · [Gate 대비 paired 진단](gate_bound63.json)

실제로 fixed를 쓰면 A의 평균 Brier는 gate보다 41.07 × 10⁻⁶ 낮고 B에서는 81.39 × 10⁻⁶ 높았다. Confidence의 저손실 행동 판별률은 상시 gate보다 0.059 pp 높았지만 잘못된 fixed 선택의 손해가 더 컸다. 두 보류 정책이 상시 gate를 거의 그대로 쓰다가 드문 사례에서 fixed로 전환했기 때문에, **accuracy 기준으로 늘어난 정답 수와 Brier 기준으로 회수한 이득이 달랐다.** [환경별 action gap](SCORE_KO.md)

## 6. GPU p90과 물리 비용

Datacenter3의 빈 RTX PRO 6000 Blackwell GPU 한 장에서 24 catalog × 4개 cell, 정책당 96회 paired crossover를 실행했다. GPU 앞뒤로 동기화하여 메모리의 raw clip·waveform 판독부터 후속 선택과 512개 물체의 최종 forecast까지 걸린 경과 시간을 측정했다. 표면 두 개에 대한 32개 clip이 한 호출에 들어간다. Confidence − random GPU p90은 **−13.089 ms** [동시 95% CI −126.278, +32.249], confidence − gate는 **−12.766 ms** [−131.276, +46.449]였다. 세 방법 모두 첫 두 catalog가 특히 느려서 interval이 넓었고, 이 두 catalog를 포함해 보고했다. [GPU 원자료](timing63.json) · [실행·검산](GPU_WORLD_KO.md)

이 대표 timing 표본에서 confidence와 random의 후속 구매는 384개 모두 같았다. Gate와 비교하면 4개가 달랐다. 별도 one-thread CPU에서 판독과 정책 선택까지만 측정한 p90은 confidence **0.610 ms**, random **0.797 ms**, gate **0.001 ms**였다. Random도 k를 계산한 뒤 subset을 추출하므로 이 추가 연산이 포함된다. 대표 GPU p90만으로 드물게 행동이 바뀌는 경우의 비용이나 GPU 속도 우위를 판단하기 어렵다. 물리 예산은 동일하지만 실제 접촉·reset·영상 전송·검증용 strain 센서의 장비 소요 시간과 에너지는 측정하지 않았다. [CPU 비용 분해](reader_cost63.json) · [Board 민감도](gpu_board63.json)

## 7. 논문에서 주장할 수 있는 범위

근거가 가장 분명한 것은 **영상 판독의 개선과 그 판독을 써서 얻는 행동의 개선을 같은 실험에서 분리했다는 점**이다. v62에서는 기존 reader의 유효 판독이 모두 옳아 confidence가 오판을 구별하는지 시험하기 어려웠다. v63에서는 transient fault에서 실제 오판을 줄였고, 그 사실을 독립 sensor로도 확인했다. 그런데도 같은 수의 무작위 보류와 최종 Brier 차이가 없었다. Coherent ghost에서는 그보다 앞선 부호 판독 단계에서 이미 실패했다. 두 실패 원인은 서로 다른 개선을 요구한다. [v62 결과](../exposure_abstention_v62/REPORT_KO.md) · [v63 광학 결과](optical_results63.json) · [행동이 가능한 범위](headroom63.json)

현재 결과로 **새 confidence 방법의 성능 우위**를 논문의 주결과로 삼기는 어렵다. 대신 같은 수를 맞춘 대조, 독립 strain 검증, 가능한 모든 subset에 대한 행동 진단을 함께 제시하면 *판독률이나 부호 정확도만으로 sensing policy를 평가할 때 무엇을 놓치는지*에 관한 기여가 된다. 35%의 발생률과 86–99% ghost 비중은 의도적으로 강한 교란이다. 실제 표면의 prevalence를 측정하지 않았고, pickup과 영상은 같은 latent deformation을 공유하며, 평가에는 기존 두 contact law를 썼다. 전체 현장 성능이나 새로운 contact law로의 일반화를 주장할 근거는 부족하다.

다음 우선순위는 실제 반사 표면에서 같은 impact를 서로 다른 조명 방향으로 촬영하고 뒤쪽 strain pickup과 동기화하는 검증이다. 장비가 작동하는 동안 marker identity를 수작업 또는 독립 공간 기준으로 확인할 수 있어야 한다. 독립 표면·날짜에서 transient와 coherent 빈도, 두 센서의 공동 결측을 먼저 추정하고, 이미 사용한 k와 acquisition cutoff를 고정한 채 **행동을 바꿀 수 있는 pilot 비율**부터 다시 잴 수 있다. 이 비율이 계속 낮으면 같은 k의 confidence threshold를 더 학습하는 것만으로는 Brier 개선 폭이 제한된다. [실물 추적에서의 identity 조건](https://pmc.ncbi.nlm.nih.gov/articles/PMC4788418/) · [동기화된 slip 검증](https://www.mdpi.com/1424-8220/18/2/333)

## 문헌 검토와 심사

### 1. 원문 비교

| 연구 | 질문·방법·표본 | 결과와 결론 | v63에 적용할 때의 한계 |
|---|---|---|---|
| Song & Godøy (2016) | 근접한 passive marker의 identity를 유지하려면 몇 Hz가 필요한가? 모터 회전체에서 세 가지 marker 간격을 바꾸고 속도·frame rate를 조절했다. 두 번째 실험은 15개 측정 session을 사용했다. | 세 배치에서 성공한 최소 frame rate는 각각 78·100·163 Hz였다. 최대 속도와 최소 간격의 비율이 identity 혼동을 설명했다. | 모터를 무반사 재질로 덮어 외부 반사를 줄였다. 이 실험으로 반사 표면의 ghost 발생률을 정할 수 없다. [원문 §Experiments, Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC4788418/) |
| Lückemann et al. (2020) | 강체 marker 간 거리를 기준으로 고속 추적의 uncertainty를 추정했다. 골퍼 한 명이 marker 크기를 바꾼 3일 동안 하루 60회 swing을 수행했다. | 공간 위치·속도와 uncertainty가 관련됐고, marker를 구성하는 camera 조합이 바뀔 때 1 mm가 넘는 오차가 나타났다. | 연구자가 mislabelling을 수동 수정했다. 보고된 위치 정밀도로 일관된 identity 오류를 검출할 능력을 판단할 수 없다. [원문 §§2–4](https://www.mdpi.com/2504-3900/49/1/72) |
| Rigi et al. (2018) | Event-based tactile slip detector를 동기화한 독립 high-speed video와 대조했다. 강체·plastic·고무 등 5개 물체에서 37회 검증했다. | 평균 onset latency는 44.1 ms였고 물체별 sensitivity와 latency가 달랐다. 다른 계측으로 접촉 판독의 성공·실패 시점을 평가할 수 있었다. | Slip onset과 변형 부호는 서로 다른 outcome이다. 계측 agreement가 후속 정책의 Brier 이득으로 이어지는지는 평가하지 않았다. [원문 §§4–5](https://www.mdpi.com/1424-8220/18/2/333) |
| Geifman & El-Yaniv (2017) | CIFAR-10·CIFAR-100·ImageNet의 classifier에서 confidence threshold와 calibration 표본으로 selective risk를 제어했다. | ImageNet 예에서는 약 60% coverage에서 top-5 error에 2% 상한을 제시했다. 정해진 가정 아래 risk–coverage 교환을 정량화했다. | Label 예측을 아예 거절할 수 있는 문제다. v63은 중간 관측 일부를 버려도 모든 물체에 stick 확률을 내므로 보장 대상이 다르다. [원문 §§2–4](https://proceedings.neurips.cc/paper_files/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf) |
| Jones et al. (2021) | Vision·NLP 5개 dataset에서 confidence가 평균과 하위 group의 selective accuracy에 미치는 영향을 비교하고 margin 분포를 분석했다. | 평균 accuracy가 올라도 일부 group의 accuracy는 떨어졌다. Confidence의 순서가 그룹마다 다를 수 있다. | 기존 분류 dataset의 집단은 실험적으로 배정한 optical fault와 다르다. 이 결과는 fault별 비교를 뒷받침하지만 v63의 효과 크기를 알려주지 않는다. [원문 §§3–5](https://openreview.net/pdf?id=N0M_4BkQ05i) |

### 2·3. 주제별 근거와 knowledge map

**Marker 간격·속도·관측 구성 → 추적한 marker의 identity와 위치 → 부호 판독**의 첫 부분에는 실물 계측 근거가 있다. Identity를 정확히 추적하면서 얻은 위치 정밀도와, 처음부터 다른 marker를 따라가며 얻은 일관된 변위는 따로 검사해야 한다. 독립적인 시각·촉각 계측의 짝만 검사하면 두 계측이 동시에 실패한 경우가 분모에서 사라질 수 있다. 마지막 문장은 Rigi의 동기화 대조를 v63에 적용할 때 생기는 검증상의 한계에 대한 추론이다. [Song & Godøy](https://pmc.ncbi.nlm.nih.gov/articles/PMC4788418/) · [Lückemann et al.](https://www.mdpi.com/2504-3900/49/1/72) · [Rigi et al.](https://www.mdpi.com/1424-8220/18/2/333)

**Confidence → 어떤 부호를 남기는가 → 어떤 후속 관측을 사는가 → 최종 Brier**에서, 첫 연결은 selective prediction의 risk–coverage 문제다. 뒤의 두 연결까지 같은 수의 판독과 같은 물리 예산으로 검사하는 것이 이번 실험의 질문이다. 먼저 같은 k개에서 서로 다른 후속 행동이 실제로 가능한지 알아야 sign error가 감소해도 최종 위험 차이가 왜 작을 수 있는지 판단할 수 있다. 이는 선행 두 연구의 classification 결과에서 도출한 실험 설계다. [Geifman & El-Yaniv](https://proceedings.neurips.cc/paper_files/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf) · [Jones et al.](https://openreview.net/pdf?id=N0M_4BkQ05i)

### 4–7. 연구 공백·상충 결과·방법·근거 강도

- **근거가 강한 범위.** 회전체에서 marker 간격·속도·sampling의 관계는 통제 실험과 반복 측정에서 확인됐다. 골프의 자유 운동에서는 위치와 추적 uncertainty가 함께 변했다. 후자는 여러 요인이 함께 변하므로 속도 하나의 인과 효과로 옮기기 어렵다. 반사 표면의 한 접촉에서 어떤 부호가 틀리는지는 이 두 연구 모두 직접 측정하지 않았다. [Song & Godøy](https://pmc.ncbi.nlm.nih.gov/articles/PMC4788418/) · [Lückemann et al.](https://www.mdpi.com/2504-3900/49/1/72)
- **상충처럼 보이는 결과.** 목표 risk를 만족한 Geifman의 ImageNet 결과와 일부 집단에서 보류 후 오히려 accuracy가 떨어진 Jones의 결과는 양립한다. 전자는 calibration sample을 바탕으로 전체 수용 집합의 risk를 평가했고, 후자는 특정 group의 조건부 정확도를 측정했다. 어느 결과도 장치 변경 후의 개별 fault에 새 risk 보장을 주지 않는다. [Geifman & El-Yaniv](https://proceedings.neurips.cc/paper_files/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf) · [Jones et al.](https://openreview.net/pdf?id=N0M_4BkQ05i)
- **간접 근거와 남은 공백.** 독립 영상으로 실제 slip을 검증한 연구는 있으나, 참조 계측 자체가 결측인 작은 변형, 지속되는 marker 오인식, 후속 intervention의 위험을 동시에 검증한 표본은 이 다섯 원문에 없다. 특히 5개 물체와 1명의 골퍼는 광택·속도·마모가 다양한 실물 접촉의 prevalence를 추정할 집단이 아니다. 따라서 설계한 35%를 현장 빈도로 사용할 근거가 없다. [Rigi et al.](https://www.mdpi.com/1424-8220/18/2/333) · [Lückemann et al.](https://www.mdpi.com/2504-3900/49/1/72) · [Song & Godøy](https://pmc.ncbi.nlm.nih.gov/articles/PMC4788418/)

### 8. 다음 연구 질문 10개

| 질문 | 공백과 중요성 | 독립 변수·측정치·접근 |
|---|---|---|
| 1. 실물 표면에서 같은 밝기의 ghost와 진짜 marker를 구별할 수 있는가? | 무반사 처리한 회전체에서 얻은 spacing 기준을 실제 광택 표면에 옮기기 어렵다. [Song & Godøy](https://pmc.ncbi.nlm.nih.gov/articles/PMC4788418/) | 반사율·marker 간격·회전 속도를 교차하고 annotated identity와 부호를 측정한다. |
| 2. 위치 정밀도가 좋아도 identity 오류는 남는가? | Mislabelling을 수동 보정한 위치 오차는 자동 판독 실패를 누락할 수 있다. [Lückemann et al.](https://www.mdpi.com/2504-3900/49/1/72) | 같은 영상을 수동 수정 전후로 비교하고 identity와 displacement error를 따로 집계한다. |
| 3. 두 번째 조명 방향이 일관된 ghost를 깨는가? | 시간축의 sampling 실험은 조명 방향에 대한 ghost의 민감도를 검사하지 않았다. [Song & Godøy](https://pmc.ncbi.nlm.nih.gov/articles/PMC4788418/) | 같은 접촉을 두 각도에서 비추고 독립 strain 부호와 대조한다. |
| 4. 작은 변형에서 참조 센서와 영상의 공동 결측이 늘어나는가? | 독립 영상으로 slip을 검증해도 모든 물체의 관측 조건이 같지 않았다. [Rigi et al.](https://www.mdpi.com/1424-8220/18/2/333) | Load와 surface를 교차하고 기계 기준 displacement로 두 센서의 공동 coverage를 측정한다. |
| 5. 새 board에서도 목표 sign risk가 유지되는가? | 기존 selective risk 보장은 평가한 분류 과제의 calibration에 의존한다. [Geifman & El-Yaniv](https://proceedings.neurips.cc/paper_files/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf) | Board 전체를 hold out하고 source cutoff·소수 calibration cutoff의 risk–coverage를 비교한다. |
| 6. 한 surface에서 frame이 아니라 session 전체에 ghost가 지속되면 어떻게 되는가? | 집단별 confidence 분포가 다를 수 있고 물체별 sensor 성능도 달랐다. [Jones et al.](https://openreview.net/pdf?id=N0M_4BkQ05i) · [Rigi et al.](https://www.mdpi.com/1424-8220/18/2/333) | 평균 빈도를 고정하고 session 상관을 바꿔 tail loss를 측정한다. |
| 7. 서로 다른 surface에서 confidence cutoff 하나로 충분한가? | 전체 risk가 하위 집단 risk를 가릴 수 있다. [Jones et al.](https://openreview.net/pdf?id=N0M_4BkQ05i) | 이름을 숨긴 표면의 광학 관측만으로 cutoff를 조절하고 worst-surface risk를 검증한다. |
| 8. 부호와 marker identity 확률을 함께 전파하면 hard abstention보다 좋은가? | 기존 selective 연구는 accepted classification error를 다뤘다. 후속 측정 선택은 다른 목적이다. [Geifman & El-Yaniv](https://proceedings.neurips.cc/paper_files/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf) | 8 impacts를 고정하고 두 identity의 likelihood를 적분한 정책과 hard gate의 Brier를 비교한다. |
| 9. Action이 실제로 바뀔 수 있는 pilot을 사전에 찾을 수 있는가? | Group별 정확도 개선이 downstream decision 이득을 식별하지 못한다. [Jones et al.](https://openreview.net/pdf?id=N0M_4BkQ05i) | Paid evidence에서 가능한 action set과 source action gap을 계산하고 blind target에서 전환 이득을 측정한다. |
| 10. 실제 impact와 reset까지 포함하면 독립 검증이 비용을 회수하는가? | 기존 실물 연구는 detector onset latency를 측정했다. 이는 최종 행동의 시간·손실과 다른 지표다. [Rigi et al.](https://www.mdpi.com/1424-8220/18/2/333) | 장비 시간을 직접 재고, 검증 센서를 audit 전용 또는 online으로 쓸 때 비용 대비 위험을 비교한다. |

### 9. 심사 의견

Temporal consistency를 이용하는 threshold 자체의 독창성은 이 문헌에서 입증되지 않는다. 논문에서는 같은 k에서 바뀔 수 있는 후속 행동의 비율, 실제로 그 행동이 바뀐 비율, 바뀌었을 때의 Brier 차이를 분리해야 한다. 실물에서 센서와 camera가 동시에 틀리는 상태, 접촉 상태의 실제 prevalence와 board 교체 비용은 별도 표본이 필요하다. 독립 계측을 도입한 선행연구 및 group별 risk 결과가 이 보완의 근거다. [Rigi et al.](https://www.mdpi.com/1424-8220/18/2/333) · [Jones et al.](https://openreview.net/pdf?id=N0M_4BkQ05i)
 
## 실행·재현 자료

서버 경로는 `datacenter3:/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/reflective_tracking_v63`이다. Physics·GPU, optics·strain, policy·statistics를 분담했고 가능한 subset에 대한 행동 하한은 별도로 재구현했다. 세 주요 정책의 평균 Brier는 독립 구현과 10⁻¹⁴ 이내에서 일치했다. Physics solver, Bayesian 계산과 표본 검산은 Slurm compute node에서 CUDA를 숨기고 실행했다. Common forecast와 timing은 사용 중인 GPU process가 없음을 확인하고 같은 할당의 guard 아래 실행했다. [Physics 데이터 checksum](deployment63.sha256) · [Optical·strain checksum](optics_bundle63.sha256) · [Brier checksum](score_results63.sha256) · [Timing checksum](gpu_results63.sha256)
