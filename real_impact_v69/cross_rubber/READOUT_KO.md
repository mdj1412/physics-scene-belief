# Rubber ball과 chalk footprint 한 건의 독립 판독

Rod Cross가 공개한 smooth rubber ball 실험에서는 blackboard에 남은 footprint 사진과 600 fps 동영상을 함께 볼 수 있다. 사진은 접촉한 영역을 시간 전체에 걸쳐 누적한 흔적이다. 동영상에는 입사와 이탈이 모두 보인다. 설명은 두 자료를 같은 예로 연결하지만 event ID나 동일 specimen의 반복실험 목록은 공개하지 않는다. 원문의 nominal condition은 ball diameter 60 mm, 입사 속도 23 m/s, 수평면과 이루는 각도 18.5°이다. [실험자 설명과 footprint](https://www.physics.usyd.edu.au/~cross/HAWKEYE.htm) · [원본 MOV](https://www.physics.usyd.edu.au/~cross/Rubber%20600fps.MOV)

MOV에 포함된 640 × 480 해상도의 22 frames를 판독했다. 재생용 약 30 fps와 실험 촬영 속도 600 fps를 구분했다. orange channel threshold와 정지 배경 차분으로 ball silhouette를 분리한 후 bounding box 중심을 추적했다. 완전히 보이고 지면에서 떨어진 frames의 median image diameter와 실물 60 mm로 길이 단위를 정했다. 중심이 변형의 영향을 받는 contact frames는 속도 적합에서 제외했다. 수평축은 화면 오른쪽, 수직축은 위쪽을 양수로 두었다.

| 판독, 기본 red channel threshold 85/255 | 추정값 |
|---|---:|
| Free-flight diameter | 60.0 px |
| 입사 frames 1–3, 0-based, 수평 / 수직 속도 | +21.90 / −7.50 m/s |
| 입사 속력과 각도 | 23.15 m/s, 하향 18.90° |
| 이탈 frames 10–14, 수평 / 수직 속도 | +10.95 / +5.88 m/s |
| 이탈 수평 속도, 4개 threshold × 3개 이탈 window | 10.68–11.48 m/s |

입사 속력은 반올림 전 성분으로부터 계산하며 원문 값과 1% 내에서 일치한다. 사용한 세 이탈 windows는 frames 9–12, 10–14, 10–21이다. 범위는 한 영상에서 추정 설정을 바꾼 민감도이며 반복 impact의 confidence interval로 해석할 수 없다. Left-to-right side view로부터 추정한 center-of-mass velocity와 투과면에서 얻는 local slip velocity는 서로 다른 측정량이다. 저장된 rubber footprint 이미지에는 세로 기준선 하나만 보인다. 두 선 사이가 100 mm라는 설명을 이 crop에 직접 사용하면 길이와 면적이 식별되지 않는다. 공개된 다른 tennis footprint나 ball 영상의 배율을 이 사진으로 옮기지 않았다.

실물 재현은 원본 MOV를 받은 뒤 `python3 measure.py 'Rubber 600fps.MOV'`로 가능하다. 출력에 SHA-256, 22개 bounding boxes와 모든 sensitivity 결과가 들어 있다. 이 예는 물리적 footprint와 rebound 속도를 함께 재확인하는 용도이다. 같은 ball의 반복성, 순간별 압력 분포, 독립 terminal stick/slip label은 이 자료로 점수화할 수 없다.

원본 410 × 240 pixel의 footprint 이미지를 별도로 분할했다. 입사 방향의 길이/가로 폭은 gray threshold 105/255에서 **332/155 = 2.14**였다. 85, 95, 105, 115의 네 threshold에서 길이 322–332 px, 폭 154–157 px, aspect ratio 2.09–2.14였다. 5 × 5 median filter로 chalk texture를 줄이고, 각 scanline에서 흰색 기준선으로 끊긴 자국 양 끝을 연결했다. 이에 따른 검은 자국의 외곽 면적은 35,026–39,151 px²이다. 이 면적은 사진 속 swept silhouette로만 해석한다. `footprint.py`와 `footprint.json`에 구현·결과·원본 checksum이 있다. [실험자 원본 footprint 이미지](https://www.physics.usyd.edu.au/~cross/HAWKEYE_files/image009.png)
