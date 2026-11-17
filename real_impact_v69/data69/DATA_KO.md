# v69 실물 impact data와 판독 가능한 지표

같은 물체에서 footprint와 충돌 직후 속도를 연결한 사례는 Cross의 Wilson tennis ball이다. 충분한 수의 실측으로 solver를 비교할 수 있는 자료는 Biber의 1,023회 golf ball impact이다. 전자는 두 대표 궤적만 공개했고 후자에서는 잔디가 contact를 가린다. 두 자료의 trial은 서로 독립적이다. 이번에 확인한 범위에서, 동일한 실물의 반복 oblique impact 각각에 대해 순간 contact patch mask, 접촉면의 stick/slip 판독, lift-off velocity를 함께 공개한 benchmark는 찾지 못했다. [Cross](https://doi.org/10.1007/s12283-014-0159-x) · [Biber et al.](https://doi.org/10.1007/s12283-023-00442-4)

## Wilson Australian Open tennis ball

Cross는 지름 66.0 mm, 질량 58.1 g의 같은 Wilson ball을 초기 spin 없이 chalked blackboard에 충돌시켰다. 각 조건을 5–6번 반복했다. 실제 footprint를 촬영하고, 별도 side view로 접촉 직전과 직후 ball bottom 좌표를 기록했다. Ball bottom은 자유 비행 중 공의 가장 낮은 기하학적 위치를 가리킨다. 회전하는 특정 material point의 속도를 측정하지는 않는다. 자료는 두 대표 trajectory를 공개한다. 다른 bounce에는 평균 및 범위가 기술되어 있다. Contact 동안 찍힌 흔적 전체의 길이와 너비는 압력을 받는 순간 영역의 면적과 단위도 의미도 다르다. [출판 원문](https://doi.org/10.1007/s12283-014-0159-x) · [저자 원고 §4, Fig. 8](https://www.researchgate.net/publication/271948217_The_footprint_of_a_tennis_ball) · [출판본 Fig. 9 원본](https://media.springernature.com/full/springer-static/image/art%3A10.1007%2Fs12283-014-0159-x/MediaObjects/12283_2014_159_Fig9_HTML.gif)

| example | 입력 vx, vy (m/s) | fps | footprint 길이 × 너비 (mm) | 복원한 vx_out (m/s) | ±1 px에 따른 범위 (m/s) |
|---|---:|---:|---:|---:|---:|
| a, 50° | 5.8, −6.8 | 300 | 42 × 40 | 2.496 | 2.317–2.678 |
| b, 17.7° | 21.1, −6.8 | 600 | 99 × 약 44 | 15.914 | 15.154–16.687 |

공개 raster의 검은 outbound marker 네 점에 각 panel의 frame interval을 부여하고 x의 linear slope를 측정했다. 공의 중심을 따라 이동하는 화면 아래쪽 좌표에서 얻은 자유 비행 초기 translational velocity이다. 운동 방향이 일정할 때 첫 네 프레임 간의 평균으로 lift-off 속도를 근사한다. 양 tick과 모든 marker의 판독이 개별적으로 ±1 px 바뀔 때의 최악 범위를 계산했다. 측정기의 오차나 반복 간 분산을 포함한 confidence interval로 해석할 수 없다. 같은 절차로 inbound vx를 5.767과 21.098 m/s로 복원했고, 입력값은 논문 본문의 5.8과 21.1 m/s를 사용했다. b의 길이 99 mm는 대표 사례에서 직접 읽은 값이며 너비 44±1 mm는 저자가 5회 모두에서 관찰한 값이다. 개별 outbound spin이나 시간별 patch 내부 속도는 공개되지 않았다. [출판 원본](https://media.springernature.com/full/springer-static/image/art%3A10.1007%2Fs12283-014-0159-x/MediaObjects/12283_2014_159_Fig9_HTML.gif)

이 ball의 별도 **normal impact**에서 wire grid로 측정한 최대 contact diameter는 42±1 mm이고 speed 7.7±0.1 m/s에서 contact duration은 4.72±0.01 ms였다. 저자는 이를 바탕으로 effective impact stiffness를 약 26 kN/m로 추산했다. 같은 ball의 다른 실험에서 얻은 독립적인 조건이다. Oblique trial에서 instantaneous patch를 복원하는 직접 측정값으로 결합할 수 없다. [저자 원고 §2–3](https://www.researchgate.net/publication/271948217_The_footprint_of_a_tennis_ball)

해당 normal impact에서 반경이 5·10·15·20 mm가 된 실측 시점을 solver의 조건부 예측과 따로 비교했다. [wire-grid 결과](wire_grid/RESULT_KO.md)

수치 재생성은 `python3 cross_figure_measurements.py`로 할 수 있다. 원본 `600 × 227` raster checksum은 SHA-256 `6d4061999964416f4677ce2197ac242ca8bcf50eeddd9728e65bf138393860e7`이다. [입력 pixel과 계산 코드](cross_figure_measurements.py) · [결과](cross_2014_examples.csv)

## Biber의 artificial turf와 natural turf

영상을 처리한 원저자의 Online Resource 2에는 Campaign A 330회와 Campaign B 693회가 들어 있다. 한 행은 한 영상이다. 열 순서는 inbound vx, vy, angular velocity와 outbound vx, vy, angular velocity이다. 속도는 ball radii/s, angular velocity는 반시계 방향을 양으로 한 rad/s다. 따라서 가장 낮은 점의 scaled tangential velocity는 vx + ω이다. COM의 vx만으로 판정하면 spin을 놓친다. 기존 논문은 inbound/outbound sign change로 충돌 중 zero crossing을 추론했다. 최종 속도가 작다는 threshold label과 contact patch 내부에서 직접 관찰한 slip은 각각 다른 outcome이다. [논문 §2.4](https://doi.org/10.1007/s12283-023-00442-4) · [21 KB 공식 supplement](https://static-content.springer.com/esm/art%3A10.1007%2Fs12283-023-00442-4/MediaObjects/12283_2023_442_MOESM2_ESM.zip)

A는 10,000 fps, B는 5,000 fps로 촬영했다. A는 같은 launch condition을 6번, B는 통상 6번 또는 12번 반복했고 매회 launcher를 옮겨 손상되지 않은 turf에서 시험했다. 6열 CSV에 개별 공 serial ID, launch setting ID, 판독 오차나 contact patch가 없다. 별도 raw video 목록에는 setting과 repeat를 암시하는 파일명이 있지만 CSV 행과 파일명 사이의 대응은 supplement에서 명시하지 않는다. 풀어본 ZIP과 CSV별 checksum을 남겼다. [원영상 목록 및 라이선스](https://data.bris.ac.uk/data/dataset/1cwnj42n63q5w2sc843iww7bbd) · [영상 README](https://data.bris.ac.uk/datasets/1cwnj42n63q5w2sc843iww7bbd/readme.txt)

| File | SHA-256 |
|---|---|
| biber_online_resource_2.zip | 7cb5710d8673857ffa544b720aaf0eea16040d1ff9194b4652abfd02aabe421c |
| ESM_2/data_campaignA.csv | 2bb13a0761f6d813b2d9e3f41e757f6c8ba3e6eafca979dbf53d246538a3dca2 |
| ESM_2/data_campaignB.csv | e7b6c01b505888a94b856ef3627af80141a76e41cc415eb994fe7d55351008f7 |

## Contact patch가 있는 다른 원문

| 논문 | 측정과 sample | 사용 가능한 결론과 한계 |
|---|---|---|
| Arakawa et al. (2005) | 투명 PMMA 뒤에서 관찰한 two-piece golf ball. 대표 impact의 incident speed 31.5 m/s, frame interval 10 µs. Fig. 5의 contact area와 Fig. 8의 ball center·contact center·initial contact point displacement를 같은 nominal speed로 제시 | Fig. 11의 tangential velocity는 접촉 동안의 평균이다. Fig. 2에는 contact 직후 사진이 있지만 연속 자유 비행 raw frames와 반복별 기록은 J-STAGE에 첨부되지 않았다. PMMA 실제 접촉면 영상 판독은 가능하다. [원문 PDF](https://www.jstage.jst.go.jp/article/jjsem/5/4/5_4_356/_pdf/-char/en) |
| Arakawa et al. (2007) | Dry/oiled PMMA 후면 영상으로 접촉시간·면적과 공의 이동·회전을 측정하고 steel과 비교 | 마찰 감소 시 contact area와 duration은 비슷했고 평균 tangential velocity는 증가했다. 이 평균을 즉시 lift-off velocity로 대체할 수 없다. [저자 기관 논문 정보](https://kyushu-u.elsevierpure.com/en/publications/dynamic-contact-behavior-of-a-golf-ball-during-oblique-impact-eff/) |
| Arakawa (2017) | PMMA에서 contact area, steel에서 angular velocity. 28–61 m/s, 연구진이 wear를 줄이기 위해 새 공 사용 | 최대 면적 150 mm²는 32 m/s의 PMMA에서 측정했다. 다른 표면·속도의 steel endpoint와 한 trial로 묶으면 가상의 쌍이 된다. [Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC5213996/) |
| Wang et al. (2022) | 30 mm steel sphere 등의 oblique impact에서 carbon paper의 장단축을 측정하고 각 조건 10회 평균. 얇은 rubber cushion과 bare steel 등의 조건 비교 | Contact trace의 기하로 slip ratio를 추론했다. Table 1의 표면에 남은 흔적은 time-resolved pressure map이나 trial별 outbound velocity가 아니다. [원문](https://doi.org/10.15632/jtam-pl/146621) |

2026년 elliptical rockfall 실험도 확인했다. 16개 series, 70회에서 pre/post kinematics를 측정했다. 이 논문의 contact length는 ellipse가 회전하며 표면에 닿은 둘레의 길이를 전후 orientation으로 계산한 값이다. 면적을 관측하려는 본 과제와 차이가 있으며 trial data는 저자에게 요청하는 방식이다. [Asteriou et al., Methods와 Data Availability](https://www.mdpi.com/2076-3417/16/6/2896)

검색은 golf·tennis·steel·rubber sphere의 oblique rebound, contact area/footprint와 tangential velocity/spin의 조합, Zenodo·Figshare·대학 및 저널 저장소를 대상으로 했다. 가장 가까운 논문들의 보충자료와 선행연구도 추적했다. 공개 문헌 전체에 부합하는 자료가 없다는 증명으로 해석하지 않는다.
