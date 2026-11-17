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
