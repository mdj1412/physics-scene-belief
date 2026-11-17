# Preston Super Anti 영상의 독립 판독

## 1. 촬영 및 표본

저자 설명은 이 시리즈에서 Butterfly Super Anti의 오래된 시편을 사용했다고 밝힌다. 1:10–2:16에는 동일 TSP blade에 rubber 한 면만 부착했고 blade를 수직으로 고정했다고 설명한다. 야외에서 사람이 공을 넣었고 장면 간 camera 거리·각도가 조금 달랐으며 편집 단계에서 비교할 만한 장면을 선별했다. 따라서 각 조건의 입사 상태는 실제로 다시 읽어야 한다. Video MP4의 640×480, 30000/1001 fps, 671.27 s는 재생 사양이다. 원본 camera clock이나 실제 시간으로의 변환율은 해당 영상이나 저자의 설명에서 확인되지 않았다. [촬영자 영상과 설명](https://www.youtube.com/watch?v=43WWRf5yW6M&t=70s)

Image Ref 1–24는 top/back, 두 feed speed, 선행 bounce 유무, blade의 세 높이를 교차한 서로 다른 24개 조건이다. 24개 모두 상단 왼쪽 Anti Spin panel이 채워져 있다. 각 조건에서 처음 보이는 운동을 사용하고 뒤따르는 contact replay는 같은 shot으로 처리했다. title_inventory.csv의 경계는 0.4 s 간격으로 읽은 표본 시각이다. 표지 25–36은 side spin이다. 입·출사 운동 품질을 통과한 10개 조건에서 20개 fit window의 연속 playback-frame 쌍 435개를 다시 검사했다. 8-bit grayscale 차이 16을 초과한 pixel이 10개 이하인 near-static 쌍은 0개였다. 이 검사는 눈에 띄는 hold frame을 찾는다. 재인코딩 전에 부드럽게 적용한 시간 배율 변경까지 식별할 수는 없다. [원본 영상](https://www.youtube.com/watch?v=43WWRf5yW6M&t=142s)

## 2. 독립적인 각도와 접점 상대속도 판독

자동 angular correlation 결과와 별도로, Image Ref 3·15·17에서 화면상 동일한 검은 marker limb의 시작·끝 각도를 육안으로 읽었다. 촬영 방향에서 본 2D planar 회전의 근사다. 각 frame의 ball centre는 해당 clip의 median background를 빼고 얻은 원형에 가까운 밝기 변화 영역의 중심으로 산출했다. 공이 라켓과 겹친 frame은 trajectory fit에서 제외했다. 입·출사 수평 궤적의 직선 교점을 contact time으로 정하고 구간별 2차식으로 contact에서의 수직속도를 추정했다. 화면 아래와 clockwise를 양수로 하여 공의 오른쪽 접점에서 s = v_y + rω_CW를 계산했다. 속도 단위는 px/playback frame이다. 반경 15.5 px를 중심 추정으로 사용했고, 14–17 px, 두 marker 방향 차이 ±12°, trajectory window 끝 ±2 frame을 함께 바꿨다. 표의 범위는 sensitivity이며 확률적 confidence interval과 구분한다.

| Ref. | 위상 | contact에서 v_y | 육안 ω (deg/frame) | contact에서 s | sensitivity |
|---|---|---:|---:|---:|---:|
| 3 · Top | 입사 | +3.413 | +10.92 | +6.368 | [+5.693, +7.049] |
| 3 · Top | 이탈 | +0.284 | −2.00 | −0.257 | [−0.678, +0.077] |
| 15 · Back | 입사 | +3.696 | −15.56 | −0.512 | [−1.315, +0.439] |
| 15 · Back | 이탈 | +4.129 | −13.78 | +0.402 | [−0.811, +1.931] |
| 17 · Back | 입사 | +3.499 | −13.64 | −0.190 | [−0.963, +0.518] |
| 17 · Back | 이탈 | −3.007 | −12.25 | −6.321 | [−7.310, −5.501] |

Ref 3의 자동 contact 추정은 −0.57, Ref 17은 −6.30으로 독립 중앙값의 부호와 일치한다. Ref 3에서는 가속도 때문에 release 이후 interval-average를 그대로 쓰면 +0.459로 부호가 바뀐다. Ref 17은 이탈 sign이 뚜렷한 반면 입사 sign이 불확실하다. 자동 입사값 +0.63과 육안 추정 −0.19의 차이는 두 항이 상쇄되므로 생긴다. Ref 15는 자동 quality rule에서 제외되었고 수동 판독에서도 양쪽 모두 모호했다. 그러므로 절대 release sign을 평가할 때와 sign change를 평가할 때 필요한 신뢰 구간은 다르다. 후자의 분모에는 입사와 이탈 양쪽을 검증한 점만 넣을 수 있다.

## 3. 재현

공개 영상의 동일한 640×480 first-pass track을 확보한 뒤, source_centers.py와 contact_check.py 순서로 세 영상의 centre와 표를 재생성한다. visual 판독 값은 manual_angles.csv, contact fit에 사용한 중심 위치는 centres_ref*.json, 10개 eligible shot의 hold frame 검사는 cadence_check.py와 cadence.csv에 있다. read_manual.py는 관측 window의 평균 속도를 계산하므로 contact velocity와 별도로 사용한다.
