# v69 문헌·공개 자료 검토

## 1. 연구별 비교

조사 기준은 반복한 물체의 identity, 개별 impact의 식별자, contact patch의 측정 시점·방식, 이탈 직후의 tangential velocity, 독립 stick/slip label이었다. Oblique impact, rebound spin, time-resolved contact area, footprint, grip/slip을 golf, tennis, rubber, softball, table tennis, tactile sensing과 조합해 검색하고 원 논문의 Methods, 저자 저장소, 저자가 제공한 영상·데이터를 대조했다. 아래의 부재 판단은 확인한 공개 자료 범위이다.

| 연구와 질문 | 방법·표본 | 결과와 결론 | 독립 검증에 사용할 때의 제약 |
|---|---|---|---|
| Arakawa 외 (2005). 입사 속도와 윤활이 면적과 운동을 어떻게 바꾸는가? | 42.6 mm, 46 g 두 겹 golf ball, PMMA, 30° 입사, 10–50 μs/frame. Fig. 2의 31.5 m/s impact에 Fig. 5와 8의 조건이 대응한다. | 입사 속도가 높으면 peak apparent contact area가 커졌다. 오일 처리는 면적·접촉시간 변화가 작아도 접선 이동과 회전을 바꿨다. | Fig. 11의 속도는 접촉 구간 변위/접촉시간이다. 저자가 공개한 값으로 이탈 직후 속도를 대체하면 시점이 달라진다. [논문 §2–7](https://www.jstage.jst.go.jp/article/jjsem/5/4/5_4_356/_pdf/-char/ja) |
| Arakawa (2017). 면적과 sliding speed를 사용해 회전을 설명할 수 있는가? | 3-piece golf ball, 28–61 m/s. PMMA에서 underside imaging, steel에서 high-speed angular tracking, roughness 통제를 위해 새 ball 사용. | 변하는 contact area를 포함한 friction model이 관측된 angular velocity 변화와 맞았다. | PMMA와 steel은 다른 target이다. Fig. 1의 두 조건을 event-level pair로 조인하거나 동일 specimen 반복으로 셀 수 없다. [Methods·Fig. 1–3](https://pmc.ncbi.nlm.nih.gov/articles/PMC5213996/) |
| Wang 외 (2022). initial spin은 접촉 흔적을 어떻게 바꾸는가? | steel sphere의 forward, backward, no spin, steel·rubber 조건. 얇은 carbon paper에 축 길이와 footprint를 기록. | spin 방향에 따라 기록된 footprint 치수가 달라졌고 rubber cushion이 그 효과를 바꿨다. | 유한 시간의 mark를 한 장에 누적하며 논문의 slip ratio는 footprint에서 유도한다. 각 pixel의 순간 slip 정답과 대응할 때 별도 검증이 필요하다. [논문·방법](https://doi.org/10.15632/jtam-pl/146621) |
| Cross (2002)와 저자 공개 실험 (2014). grip이 있으면 이탈 시까지 유지되는가? | 다섯 ball type의 normal·friction force 및 운동 측정. 2014 공개 예는 별도 실험으로, 60 mm rubber ball의 chalk footprint와 600 fps 영상을 연결한다. | 접촉 중 friction force 방향이 바뀌고 grip과 slip이 이어질 수 있다. 누적 footprint에 미끄러진 구간과 굴러간 구간이 함께 남을 수 있다. | 힘의 부호나 footprint 정지사진만으로 terminal slip을 정의할 수 없다. 2014 예의 속도를 동영상에서 별도로 추적했다. [논문](https://www.physics.sydney.edu.au/~cross/PUBLICATIONS/GripSlip.pdf) · [공개 실험](https://www.physics.usyd.edu.au/~cross/HAWKEYE.htm) · [v69 판독](cross_rubber/READOUT_KO.md) |
| Biber 외 (2023). 착지 전 속도와 spin에서 이탈 상태를 얼마나 정확히 예측하는가? | polyurethane cover의 multi-piece golf ball, 인조 turf와 천연 tee, 합계 1,000건 이상의 impact. 영상의 전후 kinematics와 반복성 평가. | 시험한 rigid law보다 piecewise-affine model의 적합도가 좋았다. 접촉 중 rolling을 거친 것으로 해석한 사례도 이탈 시 slip을 보였다. | 변형하는 turf의 local marker나 시간별 pressure map을 그들의 전후 속도로 역산하면 평가가 순환한다. [논문](https://doi.org/10.1007/s12283-023-00442-4) · [데이터](https://data.bris.ac.uk/data/dataset/1cwnj42n63q5w2sc843iww7bbd) |
| Mabbs·Smith (2026). 질량 분포 및 friction law가 softball impact 예측을 바꾸는가? | 0.2 kg softball, steel plate, 20.1–63.5 m/s 범위. 150 kHz force와 4,700 fps 측면 추적. 대표 sliding/gripping 각 1건을 14,100 fps로 분석. | 시간에 따라 friction을 바꾼 FE가 constant dynamic friction보다 각속도 normalized RMSE를 29% 줄였다. 질량 분포를 교정하면 이탈 각속도의 과대예측도 줄었다. | raw data는 저자 요청 방식이고 수평·회전 측정에 기반한 regime과 local patch label을 구분해야 한다. [Methods·Data availability](https://doi.org/10.1007/s12283-026-00564-5) |
| Gossard (2026, preprint). 서로 다른 racket rubber에서 state-dependent impulse의 예측력은 어떠한가? | 10가지 고정 racket 구성. 50 ms local regression으로 속도, dotted ball로 spin 추정. 저자 저장소는 original 8,184건, fixed train 6,490건, fixed test 1,623건을 명시. | Gaussian process로 추정한 상태별 계수가 고정 계수보다 이탈 속도·spin 오차를 줄였다고 보고한다. | 논문에 수록된 수집 수(8,194건)와 공개 original 수가 다르다. CSV에는 frame-level patch·trial별 ball serial·독립 stick/slip label 열이 없다. [논문](https://arxiv.org/html/2604.11349v1) · [확인한 저장소 revision](https://github.com/gossardt/tt_rackets/tree/159c6f040dd7832570b537b3b5fd00a97fc28c15) |

저자 저장소 30개 CSV의 실제 행 수, 열 이름 및 SHA-256은 [schema audit](gossard_schema_audit.json)에 기록했다.

## 2. 주제별 근거

실제 접촉면을 보려면 투명한 target 뒤에서 경계를 관측하는 방법이 있다. 이때 dimple을 포함한 apparent ellipse와 실제 닿는 면적을 분리할 수 있다. 반면 carbon paper나 chalk는 접촉점이 지나간 모든 위치가 겹쳐 남아, 최대 순간 면적과 누적한 면적이 다를 수 있다. [Arakawa 외](https://www.jstage.jst.go.jp/article/jjsem/5/4/5_4_356/_pdf/-char/ja) · [Wang 외](https://doi.org/10.15632/jtam-pl/146621) · [Cross](https://www.physics.usyd.edu.au/~cross/HAWKEYE.htm)

이탈 운동은 free-flight frames를 사용해 위치와 orientation을 각각 적합할 수 있다. Biber의 turf와 Gossard의 racket은 그 분석을 할 수 있는 공개 예이다. 별도의 TT3D도 130개 trajectory와 200 Hz multi-camera reconstruction을 제공하지만, 공개 명세의 목적은 bounce 위치와 궤적 복원이다. EventTouch는 실제 tactile sensor, camera frames와 FEM node 데이터를 구분해서 배포하며, pose·press depth별 접촉 실험이므로 free rebound 평가로 옮기면 측정 과제가 달라진다. [Biber](https://doi.org/10.1007/s12283-023-00442-4) · [Gossard](https://github.com/gossardt/tt_rackets) · [TT3D 저자 저장소](https://cogsys-tuebingen.github.io/tt3d/) · [EventTouch 공식 명세](https://huggingface.co/datasets/BAAI/EventTouch)

## 3. Knowledge map

입사 속도·spin·ball inertia와 표면 반응 → 접촉 중 deformation, 면적, local stick/slip → 접선·법선 force와 torque의 시간적 적분 → 이탈 시 center-of-mass velocity와 spin → 실험자가 선택한 kinematic event와 Brier score로 연결된다. Arakawa는 앞쪽 면적, Cross와 Mabbs·Smith는 가운데의 힘·회전, Biber와 Gossard는 뒤쪽의 전후 상태를 주로 측정했다. 한 층에서의 높은 예측력이 이웃한 층을 식별하는지는 별도로 시험해야 한다. [Arakawa](https://pmc.ncbi.nlm.nih.gov/articles/PMC5213996/) · [Cross](https://www.physics.sydney.edu.au/~cross/PUBLICATIONS/GripSlip.pdf) · [Mabbs·Smith](https://doi.org/10.1007/s12283-026-00564-5) · [Biber](https://doi.org/10.1007/s12283-023-00442-4) · [Gossard](https://arxiv.org/html/2604.11349v1)

## 4. Research gaps

- **동일 impact에서의 공간·시간 정렬.** PMMA 연구의 평균 속도와 turf 연구의 이탈 속도는 각각 측정 정의가 다르다. 두 사실만으로 시간별 patch와 이탈 오차의 event-level 관계를 알 수 없다. [Arakawa 외](https://www.jstage.jst.go.jp/article/jjsem/5/4/5_4_356/_pdf/-char/ja) · [Biber 외](https://doi.org/10.1007/s12283-023-00442-4)
- **Specimen과 wear.** 매번 새 공을 쓴 연구와 ball serial을 제공하지 않은 배열은 같은 실물의 반복성과 개체 간 변동을 분리하는 자료로 쓰기 어렵다. [Arakawa Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC5213996/) · [Gossard 데이터](https://github.com/gossardt/tt_rackets)
- **Local slip의 독립성.** Force reversal, 이탈 spin 및 흔적 모양은 서로 관련되지만, 한 측정량으로 나머지를 자동 판정할 근거는 약하다. 접촉 중 grip과 말기 재슬립이 함께 일어나는 사례가 있다. [Cross](https://www.physics.sydney.edu.au/~cross/PUBLICATIONS/GripSlip.pdf) · [Biber](https://doi.org/10.1007/s12283-023-00442-4)

## 5. 상충해 보이는 결과

PMMA에서는 윤활에 따라 접선 운동은 달라져도 peak patch의 변화가 작았고, steel/rubber에 initial spin을 가한 실험에서는 footprint 크기가 달라졌다. 조작한 변수(윤활 / spin), 접촉쌍(공·PMMA / steel·cushion), endpoint(순간 광학 면적 / 누적 자국)가 모두 다르므로 어느 하나를 일반 법칙으로 선택할 근거가 부족하다. Softball FE에서 temporal friction을 넣어 각속도 예측이 좋아진 결과와, turf에서 간단한 piecewise model이 선전한 결과 역시 매질·예측 대상·비교군의 차이를 수반한다. [Arakawa 외](https://www.jstage.jst.go.jp/article/jjsem/5/4/5_4_356/_pdf/-char/ja) · [Wang 외](https://doi.org/10.15632/jtam-pl/146621) · [Mabbs·Smith](https://doi.org/10.1007/s12283-026-00564-5) · [Biber](https://doi.org/10.1007/s12283-023-00442-4)

## 6. 방법이 결론에 미치는 영향

| 방법 | 식별하는 물리량 | 남는 오차 |
|---|---|---|
| Under-target high-speed imaging | 특정 시각의 optical contact mask | 광학 threshold, dimple 포함 여부, roughness에 따라 면적 변화. [Arakawa 외](https://www.jstage.jst.go.jp/article/jjsem/5/4/5_4_356/_pdf/-char/ja) |
| Carbon/chalk footprint | 접촉의 누적 범위, 이동 방향 단서 | 접촉 시각과 sliding distance가 중첩. [Wang](https://doi.org/10.15632/jtam-pl/146621) · [Cross](https://www.physics.usyd.edu.au/~cross/HAWKEYE.htm) |
| 두 개 이상의 free-flight frames와 marker tracking | COM velocity, 각속도, 전후 spin 방향 | 감쇠, frame rate, 표면의 움직임 및 rigid-radius 근사. [Biber](https://doi.org/10.1007/s12283-023-00442-4) · [Gossard](https://arxiv.org/html/2604.11349v1) |
| Force plate와 FE | Normal·shear resultant 및 그 시간적 구조 | 같은 합력이 다른 traction 분포에서 생길 수 있다. Mabbs·Smith 연구에서 peak shear timing 오차가 남았다. [Mabbs·Smith](https://doi.org/10.1007/s12283-026-00564-5) |

## 7. Evidence synthesis

**강한 근거**는 여러 실물 시스템에서 이탈 translation과 rotation이 서로 다른 측정량이며, 특정 조건에서 grip 뒤에 다시 slip이 생긴다는 점이다. **중간 근거**는 상태나 재료에 맞춘 friction response가 일부 검증 조건에서 예측 오차를 줄인다는 것이다. **부족한 근거**는 이 결과가 특정 v68 projection의 우위를 결정하거나, 소수의 누적 footprint만으로 새 contour에서 local stick/slip Brier를 예측한다는 주장이다. [Cross](https://www.physics.sydney.edu.au/~cross/PUBLICATIONS/GripSlip.pdf) · [Biber](https://doi.org/10.1007/s12283-023-00442-4) · [Mabbs·Smith](https://doi.org/10.1007/s12283-026-00564-5) · [Gossard](https://arxiv.org/html/2604.11349v1)

## 8. 근거에서 도출한 후속 질문 10개

| 질문 | 중요한 이유·측정 변수 | 식별 설계 및 근거 |
|---|---|---|
| 1. 누적 면적이 순간 최대 면적을 얼마나 과대평가하는가? | footprint와 peak area의 불일치 | 투과 영상 mask를 시간에 걸쳐 합친 면적과 각 frame 최대값을 같은 trial에서 비교. [Arakawa 외](https://www.jstage.jst.go.jp/article/jjsem/5/4/5_4_356/_pdf/-char/ja) |
| 2. 윤활이 같은 면적에서 접선 속도를 바꾸는가? | 면적만 쓰는 predictor의 실패 조건 | 같은 공·입사각·속도로 dry/oil 순서를 무작위화하고 마찰 변화와 separation velocity를 함께 측정. [Arakawa 외](https://www.jstage.jst.go.jp/article/jjsem/5/4/5_4_356/_pdf/-char/ja) |
| 3. Spin 방향이 mark의 이동 길이와 넓이를 각각 바꾸는가? | footprint 축 비율의 물리적 해석 | matched spin magnitude로 forward/backward를 바꾸고 고속 영상의 centroid path와 자국을 조인. [Wang 외](https://doi.org/10.15632/jtam-pl/146621) |
| 4. Terminal low-speed가 mid-contact grip을 얼마나 놓치는가? | label의 시간축 검증 | 동일 impact를 local optical flow track과 이탈 rigid kinematics로 이중 판독. [Cross](https://www.physics.usyd.edu.au/~cross/HAWKEYE.htm) · [Biber](https://doi.org/10.1007/s12283-023-00442-4) |
| 5. 겉면만 교체해도 solver 순위가 바뀌는가? | roughness·compliance 교란 | 같은 blade에서 rubber를 바꾸고 incoming state를 매칭해 velocity MAE를 추적. [Gossard](https://arxiv.org/html/2604.11349v1) |
| 6. 알려진 inertia를 넣으면 속도와 spin 순위가 함께 변하는가? | geometry 외 대안 설명 | 측정 inertia를 고정·교체한 predictive ablation을 같은 test shots에 수행. [Mabbs·Smith](https://doi.org/10.1007/s12283-026-00564-5) |
| 7. Surface-relative velocity가 COM 기준보다 더 안정적인가? | 움직이는 표면에서의 정의 | 표면 marker displacement를 빼기 전후의 event rate와 Brier를 비교. [Biber](https://doi.org/10.1007/s12283-023-00442-4) |
| 8. Rebound velocity fit window가 solver 순위를 바꾸는가? | 운동 추정 자체의 민감도 | 동일 영상을 여러 짧은 window로 처리하고 paired velocity difference를 재계산. [Biber](https://doi.org/10.1007/s12283-023-00442-4) · [Gossard](https://arxiv.org/html/2604.11349v1) |
| 9. 같은 물체의 마모가 specimen 간 차이만큼 큰가? | repeat와 독립 표본의 구분 | object serial, session, impact order를 기록해 variance components 추정. [Arakawa](https://pmc.ncbi.nlm.nih.gov/articles/PMC5213996/) |
| 10. Local patch를 추가 측정할 비용이 예측 개선을 정당화하는가? | v68과 연결할 수 있는 contribution | 서로 독립적으로 기록한 전후 운동·patch·label로 acquisition 전후 Brier를 비교하고 held-out surfaces에서 재평가. [Arakawa 외](https://www.jstage.jst.go.jp/article/jjsem/5/4/5_4_356/_pdf/-char/ja) · [Biber](https://doi.org/10.1007/s12283-023-00442-4) |

## 9. 현재 논문에 대한 심사 의견

논문에는 각 표의 **관측 단위와 label 생성식**을 넣는 편이 좋다. 예컨대 Biber의 velocity benchmark, 이탈 rigid surface speed로 만든 event, local stick/slip을 세 줄로 구분하면 같은 단어가 서로 다른 측정을 가리키는 문제를 피할 수 있다. v68의 contour별 projection 부호가 실물에서도 유지된다고 주장하려면 같은 contour의 반복 trial에서 patch와 local motion을 독립적으로 동기화해야 한다. 현재 자료로는 공개 velocity에 대한 예측 실패, 특정 solver 가정이 남기는 residual, 이탈 event 정의에 대한 민감도가 더 직접적인 기여다. 강한 비교군은 source-only fitted piecewise model이나 rubber-dependent impulse model이며, 물체·표면 전체를 분리한 검증이 뒤따라야 한다. [Biber](https://doi.org/10.1007/s12283-023-00442-4) · [Gossard](https://arxiv.org/html/2604.11349v1)
