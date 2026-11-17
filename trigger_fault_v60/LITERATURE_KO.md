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
