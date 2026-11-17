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
