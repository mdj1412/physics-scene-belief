# v64 공개 impact·strain 자료 검토

**실물에서 독립 strain 측정과 영상을 같은 시간축에 두고 압축 부호를 읽을 수 있는 사례를 확인했다.** 가장 가까운 OPARA 자료는 분리된 조명 방향과 물체별 marker identity를 기록하지 않아, 조명에 따른 오인식과 후속 행동 전환을 식별할 수 없다. [공개 원자료](https://doi.org/10.25532/OPARA-655) · [원문](https://doi.org/10.1088/1361-6501/adb16f)

| 조사한 연구·자료 | 실제 측정과 표본 | 질문과 비교·한계 |
|---|---|---|
| Duran Vergara et al., Longitudinal Concrete Analysis | 4개 C35/45 concrete cylinder의 Split Hopkinson Pressure Bar 압축, 1–2 Mfps 영상, bar의 semiconductor strain gauge, optical trigger | 저자는 stress wave 전파와 변형을 영상·계측으로 비교했다. Monocular camera가 촬영한 spray-painted speckle의 단일 영상 stream이다. Gauge는 시편에서 1.5 m 떨어진 bar에서 측정되며 one-dimensional wave theory로 시편의 global strain을 추정한다. 개별 marker identity와 query-local transverse strain이 없다. [실험·계측 원문](https://doi.org/10.1088/1361-6501/adb16f) · [archive](https://doi.org/10.25532/OPARA-655) |
| JRC SUSHI | 두 종류 고강도 강재의 고속 인장·파단, 직교하는 두 camera 영상, bar strain gauge와 시간 동기화한 displacement·force | 금속 시편의 양쪽 면을 다른 viewing angle에서 본다. 독립 조명 방향을 같은 표면에 적용한 처치는 기술되지 않는다. 기술 보고서는 열람했고, 보고서에 설명된 전체 실험 DB의 파일 공개 링크는 검증하지 못했다. [실험 보고서 §4](https://publications.jrc.ec.europa.eu/repository/bitstream/JRC130148/JRC130148_01.pdf) |
| 304L stainless steel DIC/IR | 7개 형상, 2개 속도, 3개 재료 방향의 인장 및 반복. Stereo-DIC, 온도, force, grip displacement | 재료 구성·형상은 풍부하지만 저자는 specular reflection을 cross-polarization으로 억제했다. DIC에서 계산한 strain과 load cell·온도 자료로 특정 marker의 독립 부호나 조명 전환 후의 오판을 검증할 수 없다. [데이터 논문](https://www.nature.com/articles/s41597-024-03949-y) |
| Baldi et al., DIC noise-induced bias | 노출 3종과 shot averaging 4종의 정밀 rigid translation. 조건당 46개 image와 변위 참조 | 연속 이동의 subpixel noise bias를 다룬다. Exposure가 다르고 설정별 위치도 조금씩 달라 동시의 조명 방향 대조가 아니다. Rigid translation에는 impact로 발생한 국소 변형이 없다. [데이터 논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC9046951/) |
| Reinforced concrete beam impact | Drop-weight impact에서 고속 영상과 독립 load cell·accelerometer. 연구자는 cross-correlation로 시간축을 맞추는 방법을 명시 | LED는 shadow/glare를 최소화해 균일하게 비췄고 strain은 DIC로 추정할 수 있다고 설명한다. 같은 영상의 위치와 strain을 비교하면 공통 tracking error가 남는다. [데이터 논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC12870799/) |

## 서버에서 확인한 OPARA 원자료

공개 archive의 central directory 877개 entry를 읽어 4개 test별 raw TIFF 180장, 총 720장과 80 MHz PicoScope 파일 4개를 확인했다. 각 test에서 frame 0·40·80·120의 native TIFF header를 읽고 해상도 924×768을 확인했다. 각 oscilloscope 파일의 time·incident bar·transmitter bar·optical trigger 네 열을 전부 파싱했고, 모두 1,600,004개 연속 row, 기록 폭 약 20 ms였다. 전체 frame을 새로 tracking하지 않았다. 파일 크기와 archive CRC는 extraction 때 검사했고, gauge 신호와 저자의 검증 그림의 SHA-256도 저장했다. [Archive 색인](opara64_inventory.json) · [검산 코드](audit_opara64.py) · [검산값과 checksum](opara64_results.json)

각 test에서 scope의 trigger 기준 0 ms 뒤 camera의 전기 trigger가 안정된 시점을 계산했다. −0.15~−0.05 ms와 0.12~0.15 ms의 baseline 사이 중간 전압을 넘은 최초 10개 연속 sample의 첫 시각을 사용했다. 4회에서 61.966~61.979 μs였다. 이는 acquisition chain에 기록된 trigger edge다. Camera의 추가 configured delay, stress wave의 bar 전파 시간과 시편에서 strain이 발생한 시점은 별개다. 시편 strain에는 저자가 시차 보정해 공개한 검증 곡선을 사용했다. [Trigger 검산](opara64_triggers.json) · [실험의 timebase와 SHPB 보정](https://doi.org/10.1088/1361-6501/adb16f)

| Impact test | Scope → camera trigger (μs) | 영상 유래 global strain (%) | Gauge 유래 global strain (%) |
|---|---:|---:|---:|
| 01 | 61.966 | 0.203 | 0.257 |
| 02 | 61.979 | 0.304 | 0.281 |
| 03 | 61.975 | 0.281 | 0.300 |
| 04 | 61.970 | 0.242 | 0.255 |

값은 저자의 네 `strain validation.tif`에서 저자가 표시한 ultimate compressive strength 도달 시점에 optical curve와 strain gauge curve를 digitize한 근사값이다. 두 색상 곡선과 시간 표식의 교점을 y축 grid로 환산했다. Compression을 양으로 표시한 네 시편 모두에서 두 곡선의 부호가 일치했다. 압축 실험만 골라 threshold 시점의 global strain을 비교해 얻은 결과라서 4/4를 reader accuracy 또는 marker 오인식 0%로 해석할 수 없다. 부분적인 곡선의 크기는 다르며 이 대조는 독립적으로 raw 영상에서 다시 추정한 strain을 검증한 결과가 아니다. [곡선 추출 코드](audit_opara64.py) · [숫자 원자료](opara64_results.json)

## 진단의 범위와 재현

OPARA는 illumination stability와 image lag를 통제했고 bar의 기계 신호가 별도로 존재한다. 따라서 두 measurement path가 서로 다른 사례는 검증할 수 있다. 그러나 한 impact에 light A/B로 구분된 두 영상 stream, 반사된 영상과 true marker의 identity, 8쌍의 reference·impact calibration과 그 뒤의 stick/slip outcomes는 확인되지 않는다. 720 frame을 720개 impact로 세거나 서로 다른 시편의 frame을 같은 접촉의 가상 대조로 연결할 수 없다. [연구 원문](https://doi.org/10.1088/1361-6501/adb16f) · [데이터 항목](https://doi.org/10.25532/OPARA-655)

원자료는 서버 `/scratch2/mdj1412/datasets/physics_scene_belief_v64/opara655` 아래에 저장했다. CC BY-NC-SA 4.0로 배포된 Duran Vergara, Leicht, Beckmann, Maas의 자료 중 필요한 파일만 추출했다. 보고서에는 원자료에서 읽은 수치와 절차를 담았다. Slurm allocation 2744471의 node56에서 CUDA를 숨기고 수행했다. `fetch_opara64.py`에 두 member manifest의 경로 목록을 전달한 뒤 `audit_opara64.py`와 `check_trigger64.py`를 실행하면 재현된다. [다운로드 코드](fetch_opara64.py) · [Raw member 목록](opara_members64.txt) · [Curve member 목록](opara_strain_plots64.txt) · [저자·라이선스](https://opara.zih.tu-dresden.de/items/8a5c42cf-51d6-48b4-a6dd-8a730c0fdb87/full)
