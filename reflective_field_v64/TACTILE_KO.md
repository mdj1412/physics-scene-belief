# v64 · 공개 tactile 영상에서 실제로 검증한 범위

**TaF는 실제로 같은 frame에 tactile 영상, ATI wrench, pressure map을 제공했다.** 세 markered sequence의 원본을 내려받아 판독했다. 영상에 물리적으로 가한 조명 변경이나 독립적인 local signed strain, marker identity 정답, 이후 stick/slip 결과는 공개 schema에서 확인할 수 없었다. RGB channel 비교는 동일 영상에서 tracker가 어느 channel에 민감한지 보여 준다. 이 자료만으로는 일관된 오인식의 빈도나 조명 변경 후 행동 전환율을 추정할 수 없다. [TaF 원문](https://arxiv.org/html/2601.20321v1) · [고정한 원본 revision](https://huggingface.co/datasets/jiamig/taf-dataset/tree/239c8ee8156bf11eb6d6c11c004f9a8721528d9d)

## 파일 검증

저자는 408개 object/sensor sequence, 3,592개 episode, 10,053,265개 synchronized frame을 기술한다. 이 중 첫 세 markered GelSight Mini ID(1, 2, 3)를 결과 확인 전에 지정했다. 각 ID에서 공개 meta, Parquet, 원본 AV1 video를 서버로 내려받았다. 세 Parquet의 행 수는 각각 27,000, 27,120, 32,554였고 meta의 frame 수와 일치했다. 실제 필드는 pressure matrix, force/torque, timestamp, frame index, episode index, global index, task index였다. 6축 wrench는 물체에 작용하는 합력·합토크이고 국소 변형 방향의 독립 정답을 제공하지 않는다. [Dataset card와 계측 설명](https://huggingface.co/datasets/jiamig/taf-dataset/blob/239c8ee8156bf11eb6d6c11c004f9a8721528d9d/README.md) · [원본 파일 무결성과 검산](tactile64_results.json)

각 ID의 첫 episode 앞 240 frame씩 총 720 frame을 320×240 RGB로 decode했다. 저장된 video PTS와 공개 Parquet timestamp의 최대 차이는 0.223 μs였다. 이는 저장된 두 시간축의 index 정렬을 확인한 것으로, camera shutter와 ATI의 실제 하드웨어 동기 오차를 이 정밀도로 측정했다는 뜻은 아니다. 세 구간에서 pressure array와 6축 wrench에 결측이 없었고 Fz는 공개 축 부호 기준 −26.65에서 +0.101 범위였다. 모든 구간에서 pressure array가 0과 양의 값을 모두 포함했다. [실행 로그](tactile64_run.log) · [분석 코드](tactile64_audit.py)

## 실제 RGB tracker의 channel 민감도

각 8초 구간에서 5 frame 간격인 양 끝을 취하고 다음 window를 6 frame 뒤에 시작해 40개 비중첩 window를 만들었다. 같은 grayscale Shi–Tomasi corner 좌표를 고정한 뒤, R·G·B channel 각각을 사용해 Lucas–Kanade flow를 추정했다. 두 channel 모두 forward–backward closure error 0.5 px 이하이며 추정 수평 변위 절댓값이 양쪽에서 0.2 px 이상인 feature-window만 분모로 썼다. Corner가 항상 material marker 위에 있다는 보장은 없다. [사전 선택과 분모](tactile64_plan.md)

| Object sequence | R–G 부호 불일치 / 유효 feature-window | R–B | G–B |
|---|---:|---:|---:|
| 1 | 5 / 1,863 | 5 / 1,870 | 0 / 1,969 |
| 2 | 1 / 339 | 2 / 363 | 0 / 362 |
| 3 | 3 / 749 | 1 / 755 | 0 / 784 |
| 세 구간 합계 | 9 / 2,951 (0.305%) | 8 / 2,988 (0.268%) | 0 / 3,115 |

움직임이 약하거나 한 channel에서 추적이 실패하면 분모에 들지 않는다. 같은 frame·gel·codec을 공유하므로 세 channel이 모두 잘못된 동일 feature를 추적해도 이 측정은 잡아내지 못한다. RGB에서 한 channel을 선택하는 연산은 다른 조명으로 재촬영한 것이 아니며, 압축된 YUV420P 영상에는 색차 공간 표본화까지 포함되어 있다. 표본은 첫 세 episode의 짧은 구간이므로 feature-window 수를 독립 물체 수로 간주하지 않았다. [Frame별 원계수](tactile64_results.json)

## 비슷한 공개 연구와의 대조

| 원문 | 연구 질문과 관측 | v64 적합성 |
|---|---|---|
| Huang et al., TaF-VLA (2026) | 실제 tactile image와 독립 ATI wrench 및 piezo pressure를 정렬한 대규모 데이터다. 같은 물체 ID는 configuration이 바뀌면 같은 실제 물체를 가리킨다고 보장되지 않는다. | 공개 파일을 실제로 점검했다. Wrench·pressure·timestamp는 있으나 조명 intervention ID·부호가 있는 local strain·후속 stick label은 해당 schema에 없다. [원문](https://arxiv.org/html/2601.20321v1) · [공개 schema](https://huggingface.co/datasets/jiamig/taf-dataset/blob/239c8ee8156bf11eb6d6c11c004f9a8721528d9d/README.md) |
| Ren et al., GelLight (2025) | 10개 형상 benchmark와 lighting configuration에 따른 reconstruction을 simulation으로 비교하고 실물 prototype의 영상과 geometry를 검증했다. | 실물 prototype은 있으나 저자의 data availability는 요청 후 제공이라고 명시한다. 해당 글에서 동일 impact의 조명 crossover·strain·stick/slip raw panel을 확인하지 못했다. [원문, §2 및 Data Availability](https://doi.org/10.1002/aisy.202400596) |
| Wang et al., SpecTac (2022) | UV LED를 켜면 fluorescent marker를 추적하고 끄면 gel 너머 장면을 촬영한다. Force estimation과 triangulation을 별도 과제로 평가했다. | LED on/off는 marker를 볼 수 있는 modality 자체를 바꾼다. 동일 marker가 두 입사각에서 유지되는지와 후속 intervention outcome을 joint 측정한 것이 아니다. [원문 §§III–IV](https://www.duyipai.me/assets/files/Wang2022SpecTac.pdf) |
| Yang et al., Touch and Go (2022) | 사람이 물체를 누르는 동안 GelSight 영상과 외부 RGB를 함께 수집하고 접촉 시점·재질을 표시했다. | 원 배포 자료의 time1·time2는 두 영상 timestamp다. 외부 camera를 독립 strain sensor로 쓸 수 없다. [원문 datasheet](https://papers.nips.cc/paper_files/paper/2022/file/354892587fe39b17c2b727af02abff4a-Supplemental-Datasets_and_Benchmarks.pdf) · [공개 파일 안내](https://github.com/touch-and-go/touch-and-go.github.io/blob/main/index.html) |

## 재현

Dataset은 서버의 `/scratch2/mdj1412/datasets/v64_taf_markered/239c8ee8156bf11eb6d6c11c004f9a8721528d9d` 아래에 보존했다. 분석은 datacenter3의 node56, Slurm allocation 2744471에서 CPU 1개, CUDA 비활성으로 실행했다. Dataset revision과 세 video 및 각 companion file의 SHA-256은 결과 JSON에 저장했다. [다운로드 코드](tactile64_fetch.py) · [분석 코드](tactile64_audit.py) · [실행 결과](tactile64_results.json)
