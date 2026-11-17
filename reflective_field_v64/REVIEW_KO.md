## 원문 비교와 연구 심사

### 1. 질문·표본·방법·결과의 비교

| 연구 및 연구 질문 | 표본과 측정 방법 | 저자가 보고한 결과와 v64에서의 용도 | 식별할 수 없는 부분 |
|---|---|---|---|
| Shi et al., DiLiGenT. 반사 특성이 달라도 photometric stereo로 표면을 복원할 수 있는가? | 정지 물체 10개, 물체마다 보정한 조명 96방향과 레이저 스캔에서 얻은 normal. 이번에 공식 main archive를 받았다. | 실제 조명이 바뀌어도 카메라와 표면의 상대 위치가 고정되는 benchmark다. v64의 거짓 변위 대조에 사용했다. | 움직이는 접촉, strain, marker identity, stick/slip이 없다. [논문](https://saikit.org/static/projects/PSDataset/PSDataset_TPAMI.pdf) · [공식 데이터](https://sites.google.com/site/photometricstereodata/single) |
| Wang et al., DiLiGenT-Π. 평면에 가까운 물체의 미세 형상이 재질에 따라 어떻게 복원되는가? | 실제 물체 30개. 금속 동전 10개와 enamel badge 10개 등을 회전식 LED로 촬영하고 profilometer로 형상을 측정했다. | 금속·광택 표면에서 조명 방향과 shape truth를 함께 제공하는 후보였다. | 정지 표면의 normal label이 있고, 접촉 시계열과 독립 strain은 기록 범위 밖이다. [원문](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_DiLiGenT-Pi_Photometric_Stereo_for_Planar_Surfaces_with_Rich_Details_-_ICCV_2023_paper.pdf) · [데이터](https://photometricstereo.github.io/diligentpi.html) |
| Wu et al., MSDD. 여러 조명으로 금속 부품 결함을 더 잘 검출하는가? | 138,585개 single-channel 이미지, 9,239개 mixed 이미지와 8종 defect. | 저자들은 다방향 strobe로 명암·반사를 달리한 입력에서 defect detection을 평가했다. 조명 방향을 실제로 조작한 산업 자료다. | Label은 결함 위치와 유형이며 접촉 변형 부호 또는 후속 행동이 아니다. [원문·자료 설명](https://www.nature.com/articles/s41597-025-04454-6) |
| Duran Vergara et al., OPARA-655. 압축 충격 중 영상과 bar strain이 변형 및 파동을 일관되게 설명하는가? | Concrete cylinder 4회 파괴 시험, 초고속 영상과 incident/transmitter bar의 독립 strain gauge. 이번에 archive와 oscilloscope channel을 검사했다. | 저자들은 두 측정법으로 변형을 비교했다. v64에서 저자가 정렬한 최대 압축 강도 시점의 두 strain 부호를 재확인했다. | Gauge는 bar에 있고 영상은 시편을 본다. 조명 방향 배정·국소 shear marker identity·동일 표면의 후속 query가 없다. [원문](https://doi.org/10.1088/1361-6501/adb16f) · [공개 파일](https://doi.org/10.25532/OPARA-655) |
| TaF-VLA. Tactile 영상과 외력의 정렬이 force-aware manipulation에 도움이 되는가? | GelSight, ATI 6축 wrench, 12×12 pressure array의 동기 기록. v64는 marker가 있는 세 물체 sequence에서 첫 8초씩 분석했다. | 실제 접촉 중의 영상과 독립 기계량을 함께 읽을 수 있다. 공개 영상·Parquet에서 timestamp 대응을 검사했다. | Force와 pressure는 해당 물질점의 signed strain과 단위·위치가 다르다. RGB channel만 바꿔 읽으면 조명 방향의 실험적 배정이 되지 않는다. [원문](https://arxiv.org/html/2601.20321v1) · [고정된 공개 버전](https://huggingface.co/datasets/jiamig/taf-dataset/tree/239c8ee8156bf11eb6d6c11c004f9a8721528d9d) |
| Higuera et al., Sparsh / TacBench. Tactile representation이 force·slip을 예측하는가? | GelSight Mini에서 3가지 indenter가 normal load 뒤 2 mm shear를 가하고 ATI로 force를 측정한다. | 배포 자료에 slip label이 있어 최종 outcome 후보로 확인했다. Marker 없는 영상이고 slip은 friction cone에서 간접 산출했다. | 실물 marker 오인식과 대응시키거나 v54가 살 대안 관측을 재생할 수 없다. [논문](https://sparsh-ssl.github.io/sparsh.pdf) · [공식 데이터 카드](https://huggingface.co/datasets/facebook/gelsight-force-estimation) |
| Yu et al., MIT Push. Surface 및 push 조건이 동역학과 마찰의 변동에 미치는 영향은 무엇인가? | 11개 물체·4종 표면, 위치·속도·가속도 등을 바꾼 push. Force, pusher pose, object pose에 timestamp가 있다. | 상이한 초기 조건의 행동과 결과를 연결할 수 있다. 저자들은 각 측정 시간축의 resampling이 필요하다고 명시한다. | 본 자료의 시각화 PNG는 기록에서 렌더링한다. 독립 조명 영상이나 signed strain을 추가해 주지 않는다. [원문](https://arxiv.org/abs/1604.04038) · [공식 schema](https://mcube.mit.edu/push-dataset/index.html) |

### 2·3. 연구 흐름과 knowledge map

물리적 **illumination direction → 영상의 반사와 그림자 → material point 대응 → 부호 판독 → calibration 뒤의 정책 선택 → 실제 stick/slip 확률**로 이어진다. 앞의 두 연결에는 DiLiGenT와 MSDD의 실제 조명 조작 자료가 있다. OPARA는 독립 계측과 시편 변형을, TaF는 표면 영상과 contact wrench를 연결한다. MIT Push는 제어된 action과 후속 운동을 연결한다. 각 데이터가 관측한 연결을 합쳐도 전체 경로의 joint distribution은 정해지지 않는다. 예를 들어 같은 광학 오류율에서도 오류가 policy threshold 바로 옆에 몰리는 경우와 멀리 있는 경우의 행동 전환율은 다르다. 이는 자료를 비교해 도출한 식별상의 결론이다. [DiLiGenT](https://sites.google.com/site/photometricstereodata/single) · [OPARA](https://doi.org/10.1088/1361-6501/adb16f) · [TaF](https://arxiv.org/html/2601.20321v1) · [MIT Push](https://mcube.mit.edu/push-dataset/index.html)

### 4–7. 근거 강도, 방법 차이, 상반된 해석과 공백

- **직접 근거.** 고정된 물체에 실제 조명만 바꿔 다방향 영상을 만들 수 있다. 초고속 충격과 독립 gauge도 한 shot에서 기록할 수 있다. 전자는 geometry가 고정된 광학 대조, 후자는 전역 압축의 기계적 대조를 준다. 두 기법은 실험 단위와 표면 재질이 다르다. [DiLiGenT](https://sites.google.com/site/photometricstereodata/single) · [OPARA](https://doi.org/10.1088/1361-6501/adb16f)
- **겉보기에 상반되는 결과.** 여러 광원을 결합하면 금속 결함 검출이 개선될 수 있지만, 단일 patch를 조명 사이에서 추적하면 위치가 틀릴 수 있다. 전자는 여러 영상의 결함 검출 mAP, 후자는 고정 표면점의 위치 오차다. 같은 성능 주장을 서로 반박하는 실험이 아니다. [MSDD](https://www.nature.com/articles/s41597-025-04454-6) · [DiLiGenT](https://saikit.org/static/projects/PSDataset/PSDataset_TPAMI.pdf)
- **제한된 근거.** 동기화한 force/pressure는 접촉 여부를 확인하는 데 도움이 되지만, 한 점의 부호 있는 strain이나 어떤 반사를 marker로 따라갔는지까지 알려주지 않는다. 서로 다른 기계량 간 상관을 local strain의 정답으로 사용하는 것은 추가 calibration을 전제한다. [TaF](https://arxiv.org/html/2601.20321v1) · [TacBench](https://huggingface.co/datasets/facebook/gelsight-force-estimation) · [OPARA](https://doi.org/10.1088/1361-6501/adb16f)
- **근거 부족.** 조사한 공개 자료에서 반사 표면별 marker 오인식률과 v54 action gap의 공동 분포를 추정할 수 없었다. 정지 동전, 파괴되는 콘크리트, 센서 내부의 탄성체, 평면 push는 서로 다른 모집단이다. 여러 자료의 frame을 같은 물체의 8회 impact로 간주하면 policy 평가 단위가 사라진다. [DiLiGenT-Π](https://photometricstereo.github.io/diligentpi.html) · [OPARA](https://doi.org/10.25532/OPARA-655) · [TaF](https://arxiv.org/html/2601.20321v1) · [MIT Push](https://mcube.mit.edu/push-dataset/index.html)

### 8. 근거에서 나온 다음 연구 질문 10개

| 질문과 공백 | 독립 변수·평가값·검증 방법 |
|---|---|
| 1. 실측 조명 각도가 어느 정도 달라져야 같은 물질점의 추적이 깨지는가? 정지 shape benchmark가 제공하는 방향 정보를 활용할 수 있다. [DiLiGenT](https://sites.google.com/site/photometricstereodata/single) | 조명 간 각도와 재질을 교차하고, 고정 좌표의 이동 오차와 coverage를 함께 측정한다. |
| 2. Polarization이 ghost 억제와 유효 판독률에 어떤 영향을 주는가? 반사 재질은 광원에 따라 외관이 달라진다. [MSDD](https://www.nature.com/articles/s41597-025-04454-6) | 같은 표면과 lamp에서 편광 각도만 바꿔 identity error와 판독률을 비교한다. |
| 3. 고정 표면점의 오등록은 충격 시 실제 marker 혼동을 예측하는가? 정지 benchmark에는 동적 identity가 없다. [DiLiGenT-Π](https://photometricstereo.github.io/diligentpi.html) | 같은 시편의 정지·impact를 짝짓고 식별 가능한 marker로 조건별 예측력을 측정한다. |
| 4. Strain gauge와 optical marker가 떨어진 거리가 부호 불일치에 미치는 영향은 무엇인가? Bar와 시편은 다른 위치를 측정한다. [OPARA](https://doi.org/10.1088/1361-6501/adb16f) | Gauge 위치·방향과 하중을 교차하고, 전파지연 보정 전후의 부호를 평가한다. |
| 5. 파일 time alignment와 실제 노출·센서 지연은 얼마나 차이 나는가? TaF는 동기 신호를 제공한다. [TaF](https://huggingface.co/datasets/jiamig/taf-dataset/blob/main/README.md) | 같은 광학·기계 trigger를 기록하고 frame timestamp, shutter midpoint, ADC edge의 차이를 추정한다. |
| 6. 독립 marker identity 검증이 힘 기반 slip label과 언제 불일치하는가? 공개 slip label은 간접 판정일 수 있다. [TacBench](https://huggingface.co/datasets/facebook/gelsight-force-estimation) | 독립 encoder 또는 texture를 표시한 고속 영상으로 최초 slip을 기록하고 friction-cone 판정과 비교한다. |
| 7. 조명이 confidence를 바꾸더라도 fixed와 gate 사이를 오가는가? 기존 검사 자료는 촬영 후 행동을 사지 않는다. [MSDD](https://www.nature.com/articles/s41597-025-04454-6) | 동일 8회 접촉에서 조명별 frozen policy를 blind replay하고 상위 선택과 실제 네 번째 구매를 따로 센다. |
| 8. 마모가 누적될 때 조명과 측정 순서의 효과를 분리할 수 있는가? 반복 push의 결과도 표면에 따라 달라진다. [MIT Push](https://mcube.mit.edu/push-dataset/index.html) | Surface-day 안에서 조명 순서를 무작위화하고 reset·lamp 순서·접촉 횟수를 기록한다. |
| 9. 물체가 바뀌면 confidence와 동일 판독 수 무작위 보류의 차이가 유지되는가? 정지·tactile 자료들의 대상 물체가 다르다. [DiLiGenT](https://sites.google.com/site/photometricstereodata/single) · [TaF](https://arxiv.org/html/2601.20321v1) | Object family·surface-session 단위 holdout을 쓰고 같은 k에서 최종 Brier와 CI를 비교한다. |
| 10. 추가 조명이 실제 시간과 에너지까지 회수하는가? 실물 sensing과 파일 replay의 비용이 다르다. [TaF](https://arxiv.org/html/2601.20321v1) · [MSDD](https://www.nature.com/articles/s41597-025-04454-6) | Trigger부터 reset까지 장비 시간·LED 에너지를 기록하고 같은 8회 impact에서 loss와 비용을 함께 비교한다. |

### 9. 심사 의견

가장 큰 미검증 주장은 **실물에서 조명 조작 때문에 더 좋은 행동을 샀다**는 것이다. 이를 확인하려면 두 방향을 같은 event에서 시간 분해할 수 있게 촬영하거나 paired 반복의 순서를 무작위화하고 wear·reset을 기록해야 한다. 같은 영상과 다른 원리로 얻은 local strain, 그리고 어느 물질점이 보이는지를 확인할 signed identity 기준을 각각 둔다. 정책에는 검증용 signal과 다른 조명의 영상을 숨긴 채 8회씩 replay한다. 이후 물체에서 선택 가능한 reference·impact 결과를 함께 기록하거나 관측 선택의 propensity와 overlap을 보장하고, 두 방법 모두에 같은 stick/slip outcome을 제공해야 한다. 먼저 조명 탓에 **실제 구매가 달라지는 사건**이 충분한지 집계하고, 미리 정한 실용적 Brier 차이를 검증할 만큼 독립 surface-session이 모였는지 판단하는 것이 순서다. 이 제안은 광학·계측·행동을 각각 검증한 공개 연구의 범위를 연결한 설계다. [DiLiGenT](https://sites.google.com/site/photometricstereodata/single) · [OPARA](https://doi.org/10.1088/1361-6501/adb16f) · [MIT Push](https://mcube.mit.edu/push-dataset/index.html)
