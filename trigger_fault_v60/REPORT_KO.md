# v60 · 동기화한 camera로 원인을 더 잘 구별하면 새 표면의 정책도 좋아지는가?

**표면별 sensor fault 빈도와 유리한 정책을 독립시킨 주 실험에서, 동기화한 reset + camera는 원인을 더 잘 구별했지만 평균 held-out Brier loss는 줄이지 못했다.** Sham + camera보다 원인 판별률이 0.729 percentage points 올랐다. 같은 8회 impact를 쓴 두 정책의 평균 loss 차이는 불확실했고, 항상 v54 gate를 적용한 정책이 reset + camera보다 24개 catalog 모두에서 낮았다. 반면 source에서의 fault 빈도와 정책의 관계를 유지하면 reset + camera가 두 대조 모두를 이겼다. 이 차이는 원인 추정의 정확도를 deployment 정책의 가치로 해석할 때 어떤 가정이 필요한지 보여준다. [사전 판정 기준](SCORE_PLAN_KO.md) · [전체 결과](score60.json)

## 1. 무엇을 바꿨는가

새 표면에서 이후 사용할 두 행동을 정한다. **Fixed 3R+I**는 세 번의 reference 다음에 항상 impact 방향을 관측한다. **V54 gate**는 세 reference를 읽은 뒤 네 번째를 reference 또는 impact로 결정한다. 이 둘 중 무엇을 쓸지 정하는 v60의 pilot에는 표면당 네 session과 session마다 순서를 무작위화한 서로 다른 두 물체가 있다. 물체마다 2.0 N reference와 impact를 한 번씩 가하고 양쪽의 변형 부호가 다른지를 읽는다. Reset arm은 둘째 물체 직전 무작위로 고른 전자센서 한 대의 local latch를 해제한다. Sham은 같은 센서 ID를 지정하고 그대로 측정한다. 상시 gate 대조에도 같은 pilot 예산을 배정했다. [설계](VISION_PLAN_KO.md) · [비용 장부](cost60.json)

- **물리와 센서의 원인.** P에서는 같은 session 안의 물체에서 실제 변형 방향의 mismatch가 함께 일어나기 쉽다. C에서는 물체의 실제 mismatch는 독립이고 두 전자센서가 같은 upstream fault를 받는다. 환경 A·B의 marginal mismatch는 모두 29%지만, A에서는 대체로 fixed, B에서는 gate의 deployment Brier가 낮다. Source의 C에서 A에는 높은 fault 빈도 29.58%, B에는 낮은 4.08%가 연결돼 있었다. 이 배정을 유지하거나, A/B 각각에서 두 빈도에 절반씩 가중하거나, 뒤집었다. 별도 실험에서는 48개 표면에 독립적인 연속 빈도를 배정했다. [손익 설계](SCORE_PLAN_KO.md) · [연속 빈도 설계](TRANSFER_PLAN_KO.md)
- **Camera의 시간 오차.** 1,000 fps 영상에서 impact clip의 0%, 20%, 100%를 2 ms 늦게 기록했다. 이 배정은 sensor fault와 별도 추첨했다. 같은 raw pixel에 두 reader를 적용했다. 하나는 고정 frame에서 시작하고, 다른 하나는 물리 contact에 맞춰 발광한 LED를 영상에서 찾아 여섯 frame을 읽는다. 표식이 없거나 필요한 frame이 부족하면 결측으로 처리했다. 순서·noise·물체를 공유해 각 차이를 대응 비교했다. [광학 원자료와 구현](VISION_KO.md)
- **정책이 받은 정보.** Frozen v59 likelihood에는 유료 센서·영상의 mismatch, 결측, 순서와 reset 배정만 들어간다. 원인 판별은 P/C의 posterior MAP를 쓴다. 행동 선택에는 fixed를 잘못 쓸 위험을 반영한 사전 odds cutoff 2.016을 썼다. 실제 fault 빈도, 물리 contact 시간, surface 이름, deployment label은 정책에 주지 않았다. [Frozen likelihood](../sensor_reset_v59/route59.py) · [구현](score60.py)

평가는 새 24개 catalog, 48개 surface, 기존 두 rate law, 12,288개 held-out deployment object에서 진행했다. Catalog·surface·가설 조합당 512개의 8-impact pilot을 생성했다. 같은 물체에 변형 방향을 달리 적용해 6,144개 active object 중 2,301개에서 stick/slip label이 바뀌었다. 평균 Brier는 frozen decoder가 낸 stick 확률과 binary outcome의 squared error를 생성 확률에 대해 적분했다. Session loss p90은 독립 binary replay에서 session별 128개 물체의 평균 loss를 구한 뒤 pooled 분포의 90번째 백분위수를 취했다. 대표본처럼 보이는 pilot 반복은 24개 독립 catalog를 대신하지 않는다. [World audit](world60.json) · [분석 코드](score60.py)

## 2. 같은 8회 impact에서 세 정책의 결과

아래 주 평가에서는 fault 빈도가 유리한 정책과 독립이다. 세 지연 비율에는 각각 1/3, P/C와 A/B에는 각각 1/2을 주었다. 이는 stress test를 포함한 설계 가중치다. 현장에서 100% 지연이 얼마나 자주 일어나는지는 측정하지 않았다. **정책 판별률**은 각 표면에서 실제로 Brier가 낮았던 행동에 대한 balanced accuracy다. 사전에 정한 A/B 대응과 실제 행동 순위는 A에서 47/48개, B에서 48/48개 표면에서 일치했다. 예외도 결과에 포함했다. [예외와 판별률 정의](SCORE_KO.md)

| 정책 | 원인 판별률 | 실제 저손실 정책 판별률 | 평균 held-out Brier ↓ | Session loss p90 ↓ | GPU p90, ms/512 objects ↓ |
|---|---:|---:|---:|---:|---:|
| LED 보정 reset + camera | 55.832% | 50.019% | 0.159348796 | 0.188999951 | 480.029 |
| LED 보정 sham + camera | 55.103% | 50.035% | 0.159348781 | 0.188999951 | 480.451 |
| 항상 v54 gate | — | 50.000% | **0.159348430** | 0.188999400 | 479.928 |

상시 gate는 P/C 진단을 출력하지 않는다. 위 GPU p90은 주 평가와 같은 independent 모집단에서 marker 판독부터 GPU forecast 완료까지 걸린 시간이다. [정책 원자료](score60.json) · [Latency 원자료](timing60.json)

원인 판별에서 reset − sham은 **+0.729 pp** [동시 95% CI +0.652, +0.804]였다. 평균 Brier의 reset − sham은 **+0.014 × 10⁻⁶** [−0.026, +0.051], reset − gate는 **+0.366 × 10⁻⁶** [+0.278, +0.458]였다. 양수는 reset의 손해다. Session loss p90 차이는 각각 **0.000 × 10⁻⁶** [−1.043, 0.000], **+0.551 × 10⁻⁶** [−0.358, +1.848]였다. 24개 catalog를 단위로 10,000회 paired bootstrap하고 endpoint별 두 대조를 보정했다. Ultra solver와 두 binary replay에서도 reset의 평균 loss는 gate보다 높았고, sham과의 차이는 0을 포함했다. [대응 차이와 민감도](score60.json)

**왜 더 잘 진단해도 손익이 좋아지지 않았나?** C에서는 독립 배정으로 인해 fault의 빈도가 A/B를 구별할 정보를 주지 않는다. P에서는 실제 두 물체의 연관이 남지만 8회 관측으로 cost-sensitive threshold를 넘는 경우가 드물었다. 보정한 reset은 A의 1.531%, B의 1.515%에서만 fixed를 골랐다. Sham과 행동이 달랐던 pilot도 0.901%였다. Fixed의 A 평균 이득은 39.29 × 10⁻⁶인 반면 B 평균 손해는 89.00 × 10⁻⁶여서, 몇 건의 방향이 틀려도 얻은 이득을 상쇄한다. 비용을 무시한 MAP는 A/B를 51.379% 맞혔지만 평균 Brier는 0.159355682로 더 높았다. [전환 빈도](switch60.json) · [표면별 손익](SCORE_KO.md)

## 3. 동기화가 복구한 관측과 복구할 수 없는 관측

Impact의 20%가 늦은 조건에서 두 reader가 **모두 읽은 같은 223,712건**을 비교했다. 고정 frame reader는 32,163건(14.38%)의 mismatch를 틀렸고, LED reader는 0건을 틀렸다. 오독률 차이의 paired 95% CI는 −14.69~−14.06 pp였다. 모든 active 관측 393,768건을 분모로 하면 보정 후 판독률은 60.91%다. 정시에는 고정 reader도 정확했고, LED를 요구하면 원래 맞았던 20,884건을 포기했다. [판독 가능 비율·음성 대조](VISION_KO.md)

| 2 ms 늦은 impact의 비중 | 미보정 reset 원인 판별률 | 보정 후 | 미보정 → 보정 평균 Brier | 보정으로 감소한 Brier, ×10⁻⁶ |
|---|---:|---:|---:|---:|
| 0% | 56.219% | 55.832% | 0.159348815 → 0.159348796 | 0.020 |
| 20% | 53.534% | 55.832% | 0.159349538 → 0.159348796 | 0.742 |
| 100% | 48.490% | 55.832% | 0.159356407 → 0.159348796 | 7.611 |

이 설계에서는 정시와 +2 ms buffer 모두 동일한 0~5 ms 구간을 담아 보정 후 세 조건에서 같은 pixel을 읽을 수 있다. 따라서 보정 결과가 같다는 것이 세 번의 독립 재현을 뜻하지 않는다. **+4 ms 지연으로 contact가 buffer 밖에 있으면** 완전히 판독한 쌍은 0개였다. **Camera와 함께 LED 자체가 2 ms 늦게 발광하면**, 사후 raw-frame 대조에서 판독한 300쌍 모두를 틀렸다. 실물로 옮길 때 검증해야 할 시점은 발광 신호가 물리 contact를 가리키는 순간이다. [광학 대조 원자료](vision_controls60.json)

![동기화에 따른 광학 오독률과 surface fault 배정에 따른 Brier 손익](trigger_fault60.png)

그림 왼쪽은 판독 가능한 쌍에서의 오류율이다. 오른쪽은 common-mode branch의 source식 fault 배정을 유지할 확률 α를 바꿀 때 나타나는 평균 Brier 차이다. 점선 α=0.5가 주 실험이며, 곡선이 0 아래로 가면 보정한 reset이 해당 대조를 이긴다. [그림 코드](figure60.py) · [교차점 원자료](transfer60.json)

## 4. 어느 표면에서 정책의 손익이 뒤집히는가

| C의 fault 빈도와 A/B 대응 | 보정 reset의 원인 판별률 | 선택 행동과 사전 A/B 대응 | Reset − sham 평균 Brier, ×10⁻⁶ | Reset − gate 평균 Brier, ×10⁻⁶ |
|---|---:|---:|---:|---:|
| Source 관계 유지 | 55.832% | 51.069% | −0.145 [−0.226, −0.085] | −0.312 [−0.395, −0.223] |
| 서로 독립 | 55.832% | 50.008% | +0.014 [−0.039, +0.062] | +0.366 [+0.251, +0.485] |
| Source 관계 반전 | 55.832% | 48.947% | +0.174 [+0.122, +0.234] | +1.044 [+0.859, +1.241] |

이 표의 CI는 3개 지연 × 3개 관계 × 2개 대조 전체 18개 평균 비교에 대해 보정했다. P/C 판별률이 일정한 이유는 A/B가 같은 비중이고, 두 C 빈도의 전체 비중도 같기 때문이다. 달라진 것은 똑같은 센서 관측이 이후 어느 행동의 이득과 연결되는가이다. 절대 빈도를 모든 target에서 높이거나 낮추면 원인 판별률은 각각 59.64%, 52.02%로 달라지지만 두 경우 모두 gate의 평균 loss가 더 낮았다. [9개 cell과 추가 대조](score60.json)

**교차점.** C에서 source 대응을 유지할 확률 α를 0에서 1로 높일 때, 보정 reset과 sham의 Brier 순위가 바뀐 지점은 **0.545** [95% CI 0.440, 0.646], gate와 바뀐 지점은 **0.770** [0.724, 0.812]였다. 두 대조를 모두 이기는 점추정 구간은 α가 약 0.77보다 큰 경우다. 이는 synthetic mixture의 평균 위험이 선형이라는 사실에서 구한 민감도다. 개별 표면의 fault 발생 확률이나 현장 적용 threshold로 사용할 수 없다. CI는 6,000회 catalog bootstrap의 각 비교별 구간이며 모든 추출에서 [0,1] 안에 교차점이 있었다. [사전 명세](TRANSFER_PLAN_KO.md) · [수치](transfer60.json)

**표면마다 다른 연속 빈도.** 48개 표면의 C fault 빈도를 4.35~29.31%로 배정하고, 같은 표면에서는 네 session 각각에서 해당 빈도가 적용되도록 했다. 기존 high/low potential readout을 session 전체 단위로 짝지어 만든 synthetic counterfactual이다. 이때도 frozen reset의 gate 대비 평균 손해는 **+0.351 × 10⁻⁶** [0.260, 0.450]였고 sham 대비 차이는 **+0.016 × 10⁻⁶** [−0.018, +0.051]였다. 부가 실험에서 C의 high/low likelihood를 A/B에 공통으로 적분하면 gate 대비 차이는 **−0.0015 × 10⁻⁶** [−0.0046, +0.0011]로 줄었다. 이 정책은 pilot의 **0.0153%**에서만 fixed를 선택했다. 이는 spurious cue를 덜 믿어 손해를 피한 결과로 해석할 수 있으나, 추가 sensing의 순이익을 입증할 정밀도에는 이르지 못했다. [재배정 코드](transfer60.py) · [수치와 구간](transfer60.json)

## 5. Compute와 sensing 비용

서버의 빈 RTX PRO 6000 Blackwell 한 장(node55, allocation 2744468)에서 frozen forecast와 timing을 끝냈다. 호출당 2개 surface와 512개 deployment object를 처리했다. 두 surface에서 낸 32개 raw clip 판독, 부호 추출과 정책 결정, 공통 decoder, GPU prediction을 포함했으며 CUDA를 앞뒤로 동기화했다. Independent 모집단의 24개 paired block에서 reset − sham GPU p90 차이는 **−0.422 ms** [동시 95% CI −18.078, +0.145], reset − gate는 **+0.100 ms** [−3.842, +0.828]였다. 세 fault 관계를 모두 합친 72개 block의 p90은 reset 480.649, sham 481.030, gate 480.407 ms였다. 두 모집단 모두 속도 우위를 확인하지 못했다. 각 9개 cell의 표본은 8회이므로 cell별 p90은 기술통계로만 제공한다. [계측과 실행 기록](GPU_WORLD_KO.md) · [전체 benchmark](timing60.json)

모든 arm의 표면당 물리 예산은 **8 impacts + 8 references + 16 mechanical resets**다. Reset arm에만 전자센서 reset 4회가 추가된다. Camera arm은 16개 clip, 128 frames, 51,840 bytes를 읽는다. 동기화는 이미 산 영상으로 수행했다. 항상 v54 gate는 실제 운영 시 이 pilot을 생략할 수 있다. 주 비교에서는 같은 물리 예산을 부과하되, 쓰지 않을 영상을 읽는 연산은 생략했다. 독립 배정에서 후속 네 번째 관측으로 impact를 산 비율은 reset 35.90%, sham 35.87%, gate 34.91%였다. 장비별 setup·전자 reset wall time은 실측하지 않았다. [장비·측정 범위](GPU_WORLD_KO.md) · [물리 비용 장부](cost60.json)

## 6. 논문에서 주장할 수 있는 범위

v59는 fault 빈도와 행동의 대응이 유지될 때 loss-weighted reset의 작은 이득을 보였고, 그 대응을 끊으면 손익이 역전될 수 있다는 사후 결과를 남겼다. v60은 새 catalog에서 그 관계를 사전에 독립화하고, 실제로 생성한 frame을 통해 **시간 정렬 → 원인 판별 → 행동 전환 → stick 예측 loss**의 네 단계를 같은 실험에서 추적했다. 시간 정렬은 오독을 막았고 reset은 진단에 정보를 보탰다. 손익의 방향은 fault 빈도와 유리한 정책의 관계에 따라 바뀌었다. 논문의 가장 강한 주장은 **정확한 센서 진단을 더 나은 정책의 대리 지표로 쓸 수 있는 조건을 검증한 결과**다. 범용적으로 우월한 active policy나 새로운 synchronization 알고리즘을 입증했다는 주장은 이 결과로 지지하기 어렵다. [v59 결과](../sensor_reset_v59/REPORT_KO.md) · [v60 비교](SCORE_KO.md)

**출판 판단.** 현재 계측과 decision loss를 연결하는 synthetic 실험 및 반례로는 논문 본문에 넣을 수 있다. 다만 matched에서도 주 정책의 gate 대비 개선은 평균 0.312 × 10⁻⁶이고, 독립 조건에서는 gate가 오히려 우세했다. GPU p90과 물리 비용까지 정당화하는 실용적 효과 크기는 아직 없다. 실물 검증의 첫 비교는 같은 물체·표면을 서로 다른 sensor board와 교차하고, contact switch·독립 photodiode·고속 영상으로 event와 LED 사이의 lag를 별도 기록하는 것이다. 이후 **새 board와 날짜 전체를 hold out**해, 사전에 정한 gate 대비 최소 Brier 개선과 전자 reset·촬영 비용을 함께 검정할 수 있다. 현재 24개 catalog는 같은 generator와 두 기존 rate law를 공유하며 simulated 카메라의 missingness, trigger 두 시점과 100% local reset 성공도 제한이다. [세계 생성과 solver 검산](GPU_WORLD_KO.md) · [광학 구현](VISION_KO.md)

## 문헌 검토와 연구 심사

### 1·6. 논문별 질문, 방법, 표본, 결과와 한계

| 연구 | 질문·방법 | 표본·평가 단위 | 관찰과 결론 | v60으로 옮길 때의 한계 |
|---|---|---|---|---|
| Tschopp et al., *VersaVIS* (2020) | 서로 다른 sensor의 측정 시간을 어떻게 맞추는가? Hardware trigger와 exposure feedback으로 mid-exposure time을 보정하고 LED board로 검증했다. | Timing 시험에서 독립 trigger camera 2대, 1 kHz LED board, 10 Hz 촬영과 1 ms exposure. Visual–inertial 실험도 별도로 수행했다. | 자체 장치에서 1 ms 미만의 synchronization 정확도를 보고했다. Exposure·데이터 전달·host clock은 구분할 필요가 있었다. | 해당 timing 수치를 v60의 impact sign이나 GPU inference latency로 환산할 수 없다. [원문 §2–3](https://pmc.ncbi.nlm.nih.gov/articles/PMC7085520/) |
| Meyer et al., *RocSync* (2026) | 다른 기종의 camera가 같은 clock을 동시에 보지 않아도 정렬할 수 있는가? 광학 LED clock으로 exposure interval을 해독하고 clock drift를 추정했다. | Heterogeneous camera 실험. Ground-truth와의 timing 비교 5회. | 해당 비교에서 평균 RMSE 1.34 ms. Rolling shutter가 residual error의 한 원인으로 제시됐다. | 30 fps 실험에서 sub-frame에 해당하는 1.34 ms도 v60의 1,000 fps에서는 한 frame보다 길다. 본 연구의 짧은 impact에서 별도의 부호 검증이 필요하다. [원문 §3–4, Table 1](https://www.mdpi.com/1424-8220/26/3/1036) |
| Rigi et al., *A Novel Event-Based Incipient Slip Detection Using DAVIS* (2018) | Event-based slip을 독립 영상과 대조할 수 있는가? 다른 각도의 1,000 fps camera와 LED를 이용해 contact area 변화를 정렬했다. | Baxter robot, 5개 물체, 37회 실물 실험. | 독립 영상과 비교한 detector의 평균 latency가 44.1 ms였고 조명 및 물체 조건에 영향을 받았다. | Slip onset 검출 시간과 camera 간 정렬 오차는 다른 수량이다. 실제 변형의 reference–impact 부호나 정책 손익을 측정한 결과는 아니다. [원문 §4.1–4.4](https://www.mdpi.com/1424-8220/18/2/333) |
| Zech et al., *Variable generalization performance … in chest radiographs* (2018) | 획득 기관과 target의 관계에 따라 외부 성능이 달라지는가? 서로 다른 기관에서 학습·검증하고 prevalence를 조정한 cohort도 비교했다. | 3개 기관 158,323개 흉부 X-ray. 환자·기관이 일반화 단위다. | 외부 기관에 대한 5개 비교 중 3개에서 내부 성능이 더 좋았다. 영상에서 기관을 높은 정확도로 식별할 수 있었다. | 획득 장치 또는 장소가 target의 proxy가 될 수 있다는 근거다. Contact fault와 마찰 사이의 인과관계나 Brier 크기의 근거는 아니다. [원문](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002683) |
| Acosta, Yang & Posa, *Validating Robotics Simulators on Real-World Impacts* (2022) | 실물 impact trajectory를 복수의 simulator가 어느 정도 재현하는가? 실측·solver별 parameter fit을 비교했다. | 550개 cube toss trajectory, Cassie의 22개 jump. | 일부 inelastic impact와 전체 운동은 재현했지만 elasticity 및 착지 세부에서 오차가 남았다. | 이 데이터에는 무작위 sensor reset·동기화 표식·표면별 fault와 미래 stick label이 함께 없다. v60의 실물 재현을 대신할 수 없다. [원문 §III–IV](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) |

### 2–3. 주제별 종합과 knowledge map

**Contact 발생 시점 → 표식의 광학 시점 → 보존된 frame 구간 → 변형 sign → 원인 posterior → sensing policy → 별도 deployment의 stick 예측**으로 연결된다. VersaVIS와 RocSync는 앞 세 단계 중 시간 기준과 exposure의 관계를 검증했다. Rigi et al.은 그 관계를 실물 접촉 검증에 사용했다. 표면별 sensor fault는 전자센서의 측정 과정과 환경 판별의 관계를 바꾸는 별도 변수다. 마지막 세 단계에서 그 빈도가 만드는 confounding을 정량화한 것은 이번 연구의 실험이다. [Tschopp et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7085520/) · [Meyer et al.](https://www.mdpi.com/1424-8220/26/3/1036) · [Rigi et al.](https://www.mdpi.com/1424-8220/18/2/333) · [Zech et al.](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002683)

### 4–5·7. 근거 강도, 결과 차이, 연구 공백

- **Timing.** 독립 표식과 exposure를 이용한 시간 정렬에는 실측 근거가 있다. VersaVIS의 sub-ms 결과와 RocSync의 1.34 ms RMSE는 장치, timing reference, sampling rate와 오차 정의가 다르다. 하나의 평균으로 합치거나 방법 순위를 정할 수 없다. 두 연구 모두 이미 촬영 구간 밖으로 사라진 contact 변형을 복구한다고 주장하지 않는다. [Tschopp et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7085520/) · [Meyer et al.](https://www.mdpi.com/1424-8220/26/3/1036)
- **Contact 및 timing의 전이.** 실제 slip에서 high-speed camera를 reference로 쓴 사례와 여러 engine의 물리적 한계가 확인됐다. 같은 trigger 정렬이 모든 재질에서 동일한 sign yield로 이어지는지, 여덟 contact로 실제 policy risk를 낮추는지는 근거가 부족하다. [Rigi et al.](https://www.mdpi.com/1424-8220/18/2/333) · [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf)
- **장치와 target의 관계.** 획득 환경이 target과 연동될 때 그 관계의 변화가 외부 예측에 영향을 줄 수 있다는 경험적 근거가 있다. 단일 synthetic contact task에서 hazard 관계를 명시적으로 바꾸는 실험과 병원의 observational cohort는 서로 다른 설계다. 어느 한쪽의 상관관계가 다른 분야의 실제 원인일 것으로 단정할 수 없다. [Zech et al.](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002683)

### 8. 근거에서 나온 후속 질문 열 가지

| 질문 | 왜 중요한가 | 변수·연구 방법 |
|---|---|---|
| 1. LED 기동 자체가 실제 contact보다 늦으면 어느 지연부터 효과가 사라지는가? | 장치 clock과 물리 사건은 다른 기준이다. [Rigi et al.](https://www.mdpi.com/1424-8220/18/2/333) | Photodiode·contact switch를 별도로 기록해 LED–contact lag만 바꾼다. |
| 2. Pre-trigger buffer를 어느 길이까지 확보해야 하는가? | Exposure가 onset을 포착해야 시간 정렬이 가능하다. [Tschopp et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7085520/) | 같은 충돌 영상을 잘라 buffer 길이별 sign yield와 Brier를 비교한다. |
| 3. 한 frame보다 작은 offset에서도 margin이 작은 변형 부호를 지킬 수 있는가? | Sub-frame 정렬의 물리적 의미가 task마다 다르다. [Meyer et al.](https://www.mdpi.com/1424-8220/26/3/1036) | 연속 위상·속도·exposure를 교차해 sign error를 추정한다. |
| 4. LED와 marker의 영상 행이 다를 때는 어떤가? | Rolling shutter에서는 행별 exposure 시점이 다르다. [Meyer et al.](https://www.mdpi.com/1424-8220/26/3/1036) | Row offset과 움직임 속도를 조절하고 global shutter를 대조한다. |
| 5. 반사 표면에서 표식 누락이 contact 결과에 의존하는가? | 조명이 광학 검증에 영향을 준다. [Rigi et al.](https://www.mdpi.com/1424-8220/18/2/333) | 반사율·가림을 층화하고 missing trial도 독립 gauge로 audit한다. |
| 6. 같은 표면에 다른 acquisition board를 달아도 정책이 유지되는가? | 획득 환경에서 유래한 target proxy를 분리해야 한다. [Zech et al.](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002683) | Surface와 board를 완전 교차하고 board 및 날짜를 hold out한다. |
| 7. Camera와 전자센서가 공유하는 clock drift는 표식 하나로 구별되는가? | 같은 기준에 묶이는 정도가 다르면 보고된 정렬 정확도도 달라진다. [Tschopp et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7085520/) | 독립 oscillator와 별도 diode를 무작위 삽입해 shared drift를 측정한다. |
| 8. 표면 마모가 fault 빈도와 실제 마찰을 함께 바꾸는가? | 실물 impact가 이상적인 simulator와 다를 수 있다. [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf) | 누적 충격 횟수, surface 교체, 전자 fault injection을 독립 교차한다. |
| 9. 추가 reset·촬영 비용까지 고려할 때 필요한 Brier 감소량은 얼마인가? | Detector latency와 장비 setup은 별도 비용이다. [Rigi et al.](https://www.mdpi.com/1424-8220/18/2/333) | 장비 blocking time, LED duty cycle, false-slip 비용을 사전 지정해 expected utility를 검증한다. |
| 10. Source의 장치 빈도를 모르는 독립 site에서 어떤 행동이 가장 안전한가? | In-domain validation만으로 external risk를 보장하기 어렵다. [Zech et al.](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002683) | 장치–표면 연관이 다른 site를 통째로 hold out하고 고정 gate와 blind selector를 비교한다. |

### 9. 독립 심사

광학 동기화 자체와 장치 단서의 외부 일반화 문제는 기존 연구에 선례가 있다. 논문에 기여할 수 있는 지점은 **같은 물체·같은 contact에서의 시간 정렬 성공**, **서로 다른 물리·센서 원인의 식별**, **새 표면에서의 downstream 손익**을 별도 endpoint로 두고, 어느 단계에서 효과가 사라지는지를 보이는 데 있다. 이를 실험 장치 표준이나 실제 material 일반화로 확대하려면 독립 clock으로 관측한 contact 시점, 서로 다른 장치의 fault 빈도, 실제 stick outcome이 함께 필요하다. 이 다섯 원문은 위 질문에 맞춘 목적별 검토이며 분야 전체에 대한 최초성 조사는 아니다. [Meyer et al.](https://www.mdpi.com/1424-8220/26/3/1036) · [Rigi et al.](https://www.mdpi.com/1424-8220/18/2/333) · [Acosta et al.](https://dair.seas.upenn.edu/assets/pdf/Acosta2022.pdf)


## 재현 자료

서버 프로젝트는 `datacenter3:/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/trigger_fault_v60`에 있다. 광학 생성 및 negative control, PyBullet·GPU forecast, 정책 채점과 문헌·전이 검토를 병렬 진행했다. 프로젝트 디렉터리와 동명인 local mirror에 24개 pilot, 광학 fixture, fine·ultra solver, GPU forecast, timing block, JSON 결과와 한글 세부 보고서를 보관했다. Git의 이전 실험 모듈 중 v54 gate, v55 deployment, v59 router를 재사용한다. [광학 checksum](vision_bundle60.sha256) · [Deployment checksum](deployment60.sha256) · [채점 checksum](score_results60.sha256) · [GPU checksum](gpu_results60.sha256) · [추가 표면 분석](transfer60_cells.npz)

채점 과정에서 privileged A/B comparator가 target regime을 유지하도록 재배정 처리를 바로잡고 실제 표면별 저손실 행동에 대한 판별률을 추가했다. Deployable 정책의 행동과 주 비교값은 그대로였다. 별도 구현으로 매칭·독립·반전의 두 대조, 총 여섯 평균 차이를 재계산해 원래 채점과 절대오차 10⁻¹⁴ 이내에서 일치함을 확인했다. [초기 채점 코드](score60_initial.py) · [최종 채점 코드](score60.py) · [보조 검산](transfer60.py)
