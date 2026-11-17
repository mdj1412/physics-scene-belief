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
