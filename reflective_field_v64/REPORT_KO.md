# v64 · 실제 조명 변화가 반사 표면의 추적과 다음 관측 선택을 바꾸는가?

**실제 정지 표면에서 조명만 바꿔도 유사도가 높은 tracker가 같은 방향으로 잘못된 위치를 고르는 사례를 확인했다. 실제 충격에서 그 현상이 얼마나 자주 일어나고 후속 행동을 바꾸는지는 확인하지 못했다.** 다방향 실측 영상, 독립 strain, 같은 표면의 8회 calibration impact, 후속 stick/slip이 한 실험에 연결된 공개 자료를 이번 조사에서 찾지 못했다. Datacenter3에서 공개 원자료 세 종류를 분석하고, 8회 예산의 정책 대조는 별도로 두 방향의 영상을 새로 생성한 simulation에서 실행했다. [실물 광학 자료](https://sites.google.com/site/photometricstereodata/single) · [실물 impact·strain 자료](https://doi.org/10.25532/OPARA-655) · [실물 tactile 자료](https://huggingface.co/datasets/jiamig/taf-dataset/tree/239c8ee8156bf11eb6d6c11c004f9a8721528d9d)

| 질문 | 실측 자료에서 얻은 답 | 같은 8회 impact의 보조 simulation |
|---|---|---|
| 일관된 marker 오인식이 얼마나 잦은가? | 실제 접촉의 identity 오류율은 미추정. 정지 물체의 다방향 관측에서는 사전 정의한 지속적 수평 오등록이 정방향 9/62개 수용 patch에서 관찰됐다. | 원래 v61 reader에서 강한 ghost와 잘못된 부호가 함께 관측된 비율은 활성 쌍의 0.631–1.811%. Angular width와 lamp에 의존한다. |
| 조명을 바꾸면 후속 행동이 달라지는가? | 같은 현장 pilot을 두 lamp로 재생할 자료가 없어 미추정. | Confidence의 상위 정책이 0.288%, 실제 네 번째 관측 구매가 0.186% 달라졌다. |
| 어떤 정책의 평균 held-out Brier가 낮은가? | 관측 선택과 독립 deployment label을 연결할 수 없어 미추정. | 상시 v54 gate가 가장 낮았다. Confidence–무작위 차이의 interval은 0을 포함했다. |
| GPU에서 얼마나 걸리는가? | 정지 실측 이미지의 patch 재탐색만 GPU p90 **1.502 ms**. 실제 8회 정책 전체의 p90은 미측정. | 512개 물체의 전체 경로에서 confidence **483.406 ms**, 같은 수 무작위 **482.531 ms**, v54 gate **480.780 ms**였다. |

## 1. 어떤 기록이 있어야 질문에 답할 수 있는가?

검색 시점은 2026-11-17이다. Photometric stereo와 reflective marker, 고속 DIC·SHPB와 strain gauge, visuotactile force/slip, planar pushing 네 분야에서 원 논문, Data Availability, 저자 repository와 file schema를 대조했다. 검색어에는 `multi-directional illumination`, `specular marker`, `high-speed impact`, `synchronized strain gauge`, `slip labels`를 조합했다. 이는 목적을 정한 후보 조사이며 공개 데이터 전체의 부재를 입증하는 systematic review는 아니다. 원문별 표본·방법·근거는 아래 문헌 비교에, 기계 및 tactile 후보의 접근 가능 여부는 각 부록에 기록했다. [사전 기준](FIELD_PLAN_KO.md) · [기계 자료 조사](MECHANICS_KO.md) · [Tactile 자료 조사](TACTILE_KO.md)

- **오인식률**의 분모는 사전에 정의한 실물 impact 또는 읽을 수 있었던 marker track이다. 같은 물질점의 identity를 검증할 기준이 있어야 한다. 독립 strain은 변형 부호를 검증하지만, 다른 marker를 따라가도 같은 부호가 나올 수 있다. 두 검증을 따로 기록해야 한다.
- **행동 전환율**에는 같은 8회 pilot의 lamp A/B 원자료와 frozen selector 입력이 필요하다. 상위 선택이 바뀌더라도 v54가 이후 물체에서 같은 관측을 사면 실제 구매는 같으므로, 실제 구매 전환에는 배포 시점의 세 reference도 필요하다. 행동 전환 자체는 stick/slip label 없이 계산할 수 있다.
- **Held-out Brier**에는 선택한 관측으로 얻은 확률과 그 후의 물리 outcome이 함께 있어야 한다. 가능한 관측을 모두 기록한 replay 또는 관측 배정과 propensity가 확인되는 자료에서 세 정책을 공정하게 비교할 수 있다. 서로 다른 연구의 frame·strain·outcome을 이어 붙이면 어느 pilot의 선택이 어느 물체의 예측을 바꿨는지 확인할 수 없다. [원래 정책·평가 정의](../reflective_tracking_v63/SCORE_PLAN_KO.md)

## 2. 실제 방향성 조명 · 움직이지 않은 표면에서의 추적 오류

Shi 등의 공식 DiLiGenT main archive에서 실제 물체 10개의 16-bit RGB 원본을 읽었다. 96개 조명 중 수평 −40°, −15°, +15°, +40°에 가장 가까운 네 방향을 물체별 calibration vector만으로 선택했다. 채널별 보정 광량으로 나눈 뒤 object mask 내부 25 pixel 간격 격자에서 15×15 patch를 선택했다. Tracker는 첫 조명의 patch와 가장 유사한 위치를 이후 세 조명 각각에서 ±8 pixel 안에서 찾았다. 세 normalized cross-correlation (NCC)이 모두 threshold 이상일 때 수용하고, 그 세 위치가 처음 위치보다 **모두 오른쪽 2 pixel 이상 또는 모두 왼쪽 2 pixel 이상**이면 일관된 수평 오등록으로 셌다. 좌표는 항상 첫 이미지 기준이며 단조롭게 움직여야 한다는 조건은 없다. [자료·보정 형식](https://sites.google.com/site/photometricstereodata/single) · [결과 전 고정한 분석 계획](FIELD_PLAN_KO.md) · [분석 코드](static_light64.py)

| 조명 순서 | NCC threshold | 비교 가능 patch | 세 방향 모두 수용 | 일관된 수평 오등록 / 수용 | 전체 비교 가능 patch 대비 |
|---|---:|---:|---:|---:|---:|
| 정방향 | 0.70 | 337 | 102 | 14/102 (13.73%) | 4.15% |
| **정방향·주 분석** | **0.80** | **337** | **62** | **9/62 (14.52%)** | **2.67%** |
| 정방향 | 0.90 | 337 | 16 | 4/16 (25.00%) | 1.19% |
| 역방향 | 0.80 | 337 | 64 | 14/64 (21.88%) | 4.15% |

같은 이미지끼리 재생하면 337개 위치 모두 자기 좌표로 돌아왔다. 정방향 주 분석의 오류 9개는 buddha 1개, cat 4개, cow 1개, pot2 3개에 집중됐다. 가장 엄격한 threshold에서는 전체 오등록 건수와 수용 건수가 함께 줄었다. 10개 물체 속에 모인 patch는 독립 surface-session이 아니므로 이 표에 현장 빈도의 CI를 붙이지 않았다. 이 tracker의 patch에는 사람이 부착한 고유 marker가 없고, 실제의 미세 registration 오차와 텍스처 반복도 기여할 수 있다. 따라서 14.52%를 **실물 impact의 ghost 빈도**로 해석할 수 없다. [물체별 결과](static_tracks64.json) · [좌표별 결과](static_tracks64.csv)

![같은 정지 buddha 표면의 네 조명에서 기준 위치와 tracker의 선택](static_patch64.png)

예시는 사전 기준을 통과한 첫 오등록이다. 파란 십자표가 기준 좌표, 주황 원이 tracker의 선택이다. 물체는 고정돼 있지만, 세 조명에서 수평 offset은 +6·+4·+4 pixel, NCC는 0.979·0.932·0.894였다. 이 사례는 같은 조명에서 시간에 걸쳐 추적한 연속 동영상과 구별된다. [예시 수치](example64.json) · [Shi et al. 원본](https://sites.google.com/site/photometricstereodata/single)

실제 광학 단계도 서버에서 별도 측정했다. 빈 RTX PRO 6000 Blackwell GPU에서 CPU 메모리에 준비된 물체당 8–56개 ROI를 전송하고 3개 조명과의 NCC 계산부터 수용·오등록 건수를 host로 되돌려 받기까지, 물체 10개 × 정·역방향 × 10회, 총 200회였다. GPU/CPU 구현의 수용·오등록 건수가 모든 호출에서 일치했고 p50은 **1.193 ms**, p90은 **1.502 ms**였다. 이미지 decoding, ROI 자르기, strain 동기화, 후속 정책 및 forecast는 이 시간에 포함하지 않았다. [GPU 원자료](real_ncc_gpu64.json) · [GPU 구현](real_ncc_gpu64.py)

## 3. 실제 impact에서 strain 부호와 동기화는 어디까지 검증됐나?

**OPARA-655**가 가장 가까운 기계 자료였다. 4개 concrete cylinder의 파괴 충격에서 test당 180개 raw TIFF와 incident/transmitter strain gauge·optical trigger가 있었다. 시험당 1,600,004개의 80 MHz scope row 전부와 지정한 raw frame 4개씩을 검사했다. 전기 trigger는 네 시험 모두 scope 기준 뒤 61.966–61.979 μs에 검출됐다. 이는 shutter midpoint나 stress wave가 시편에 도달한 시각과는 다른 계측값이다. 저자가 시차를 보정한 공개 곡선에서 최대 압축 강도 시점의 영상·gauge 유래 global strain을 다시 읽으면 각각 0.203/0.257%, 0.304/0.281%, 0.281/0.300%, 0.242/0.255%였다. 압축을 양으로 정한 부호가 4/4에서 같았고, 이 수치는 **저자가 공개한 곡선의 digitization**이다. 국소 marker를 독립 추적하여 얻은 4회 정답으로 세지 않았다. [실물 원문](https://doi.org/10.1088/1361-6501/adb16f) · [실험 파일](https://doi.org/10.25532/OPARA-655) · [검산과 제한](MECHANICS_KO.md)

**TaF**에서는 marker가 있는 세 object sequence의 첫 8초씩, 720 frame을 실제로 decode해 6축 ATI wrench·12×12 pressure와 연결했다. 저장한 video와 Parquet 시간축의 최대 차이는 0.223 μs였다. 동일한 영상에서 channel별 수평 추적 부호가 달랐던 feature-window는 R–G 9/2,951, R–B 8/2,988, G–B 0/3,115였다. 이 결과는 RGB channel에 대한 추적 민감도다. 물체의 변형 부호를 나타내는 독립 local strain, lamp를 따로 켠 촬영, marker identity 정답은 이 파일에서 확인되지 않았다. 저장 timestamp의 일치로 물리적인 동기 정밀도까지 추정하지 않았다. [공개 데이터](https://huggingface.co/datasets/jiamig/taf-dataset/blob/main/README.md) · [실측 파일 및 통계](TACTILE_KO.md)

## 4. 같은 8회 impact의 정책 비교 · simulation

V63의 frozen reader·decoder와 기존 held-out 물체 12,288개를 유지하고, 새 calibration pilot 12,288개에 **−30°와 +30° lamp를 각각 적용**했다. 한 정책이 쓰는 예산은 8 references·8 impacts·16 mechanical resets로 동일하고 두 영상 중 결과를 보고 더 좋은 lamp를 선택하지 않는다. 두 방향은 같은 변형·노이즈·전자센서·strain을 공유한다. 반사 facet의 각도에 따라 ghost의 세기가 변하며 angular width 8°와 26°를 같은 비중으로 뒀다. 잠재 ghost 기회 35%와 transient fault 15%는 설계값이다. DiLiGenT의 오등록과 OPARA의 압축률로 이 상수를 맞추지 않았다. [사전 설계](SIMULATION_PLAN_KO.md) · [렌더링 구현](illumination64.py)

| 정책 | 평균 held-out Brier ↓ | GPU p90, 512개 물체 ↓ |
|---|---:|---:|
| Confidence 보류 | 0.155701540000 | 483.406 ms |
| 같은 수 무작위 보류 | 0.155701541708 | 482.531 ms |
| 상시 v54 gate | **0.155701480487** | 480.780 ms |

무작위 보류는 각 pilot에서 confidence가 남긴 k를 맞췄다. Brier에는 모든 k-subset에서 실제로 고른 행동의 loss를 적분했고, GPU timing에서는 subset을 한 번 추출했다. Confidence − random의 평균 Brier 차이는 **−0.00171 × 10⁻⁶** [동시 95% CI −0.01241, +0.00821], confidence − gate는 **+0.05951 × 10⁻⁶** [+0.01021, +0.11934]였다. 전자는 개선 근거가 불충분하고 후자는 상시 gate가 유리하다. 24 catalog × 8 board에서 복원추출했으며 같은 물리 물체의 counterfactual을 독립 표본으로 세지 않았다. Gate는 기존 deployment와 고정한 정책을 사용하므로 v63과 같은 Brier가 나온다. [점수·분모](sim_score64.json) · [분석 보고서](SIMULATION_KO.md)

- **조명 전환과 행동 전환.** 동일 8회 calibration pilot을 두 lamp로 비교하면 confidence가 fixed 3R+I와 v54 gate 사이에서 선택을 바꾼 비율은 **0.288%** [95% CI 0.076, 0.542]였다. 상위 선택이 달라져도 v54가 이후 세 reference를 보고 같은 impact를 살 수 있다. 실제 네 번째 구매가 reference와 impact 사이에서 바뀐 비율은 **0.186%** [0.049, 0.352]였다. 무작위 정책에 같은 추첨 순위를 사용한 조명 대조에서는 각각 0.291%, 0.188%였고, lamp 입력을 사용하지 않는 v54 gate는 0%였다. [조명별 전환](SIMULATION_KO.md)
- **왜 ghost에 조명이 영향을 주는가?** 두 lamp 중 한 곳 이상에서 기준 v61 reader가 강한 ghost로 오판한 사건 중 양방향에서 모두 오판한 경우가 좁은 lobe에서는 **0/2,601**, 넓은 lobe에서는 **1,757/5,098**이었다. 같은 현장 표면에서 조명 두 방향의 오류가 얼마나 상관되는지가 중요한 실측 대상이다. 이 수치는 가정한 반사 분포에 의존한다. [광학 모형과 기전 검산](SIMULATION_KO.md)
- **GPU p90.** 96개 paired block에서 raw 영상·waveform 판독, 선택, common forecast, CUDA 동기화까지 측정했다. Confidence − random은 +0.876 ms [동시 95% CI −6.475, +14.028], confidence − gate는 +2.627 ms [−0.273, +15.143]였다. 비교에 사용한 대표 384개 deployment session에서 세 정책의 실제 구매는 같았다. 관측된 p90은 전체 경로의 계산 비용으로, 드물게 조명 때문에 구매가 달라지는 경우의 별도 비용이나 하드웨어 접촉 시간을 뜻하지 않는다. [GPU 원자료](sim_latency64.json)

## 5. 논문 기여 판단

가장 방어 가능한 주장은 **실측 광학 영상에서 높은 confidence와 같은 방향의 거짓 변위가 공존할 수 있으며, 그 오류가 행동을 바꾸는 빈도는 별도로 측정해야 한다**는 것이다. 이번 자료는 광학 실패 사례와 기계적 시간축 검증을 제공한다. 실물 장비에서 다방향 조명이나 confidence 보류가 Brier loss를 낮춘다는 기여는 아직 입증되지 않았다. 보조 simulation에서도 같은 수 무작위 대비 우위는 불확실했다.

다음 확인 실험의 첫 목표는 평균 Brier를 다시 추정하기에 앞서 **동일한 실제 8-impact pilot에서 조명에 따라 네 번째 구매가 달라진 사건 수를 확인하는 것**이다. 독립 식별자 또는 두 번째 시점으로 material point를 검증하고, 같은 위치·축의 strain 또는 기하적으로 변환 가능한 strain을 common trigger에 기록해야 한다. Trigger delay·surface 마모·재접촉을 기록하고 sample split은 표면과 session 단위로 고정한다. 다른 조명의 입력이나 검증용 strain은 단일 lamp의 배포 정책에 주지 않는다. 이후의 별도 물체에서 두 관측·outcome을 수집하면 같은 k의 무작위 보류 및 v54 gate와의 Brier를 직접 비교할 수 있다. [실물 계측 사례](https://doi.org/10.1088/1361-6501/adb16f) · [실제 반사 표면 광원 제어](https://www.nature.com/articles/s41597-025-04454-6)

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


## 실행 경로와 재현 파일

서버 작업 디렉터리는 `datacenter3:/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/reflective_field_v64`다. 공개 자료는 `/scratch2/mdj1412/datasets` 아래에 보존했다. DiLiGenT 공식 main archive의 SHA-256은 `d13f54e1bcd3da9cc19e2ececc5ad1a768784448a7fbb9d4428dc64f86c3e9f9`다. 물체·조명·patch 좌표와 추적 결과는 [CSV](static_tracks64.csv)에 있다. 실물 기계 자료, 실물 tactile 자료, 보조 simulation을 분담했고 정지 실물 조명 대조는 CPU·GPU 두 구현으로 검산했다. 두 GPU 실행은 node55의 빈 GPU 한 장을 순차 사용했다. 원자료 분석은 Slurm compute node에서 진행했다. [실물 기계 checksum](mechanics64.sha256) · [실물 tactile checksum](tactile64.sha256) · [보조 simulation checksum](sim_bundle64.sha256) · [원자료 panel checksum](sim_raw64.sha256)
