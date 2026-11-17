# v73 문헌 검토 · 빠른 anti-spin impact의 이탈 방향을 어느 자료에서 검증할 수 있는가?

## 1. 연구별 질문, 방법, 표본과 한계

| 원 연구·기록 | 질문과 방법 | 표본과 발견 | v73에서 사용할 수 있는 근거와 한계 |
|---|---|---|---|
| Gossard (2026) | 입사 상태에 따라 rebound model의 계수를 추정하면 velocity·spin 예측이 좋아지는가? 고정 racket, launcher, 2D 평면, 370 fps 영상과 SpinDOE를 사용했다. | 10개 configuration에서 Gaussian process로 추정한 state-dependent parameter가 constant model보다 작은 예측오차를 보였다. 논문은 8,194건을 보고하고 공개 저장소는 original CSV 8,184건과 공식 train/test 6,490/1,623건을 명시한다. | Anti-spin은 Dr. Neubauer A-B-S II soft, 2.1 mm, configuration 10 하나다. v71에 포함된 시편의 fast-input 대조에 적합하다. 공개 CSV에서는 개별 frame이나 새 시편을 식별할 수 없다. [원문 §III·V–VII](https://arxiv.org/html/2604.11349v1) · [공식 데이터](https://github.com/gossardt/tt_rackets/tree/159c6f040dd7832570b537b3b5fd00a97fc28c15) |
| Gossard 등 (2023), SpinDOE | 알려진 비대칭 dot pattern에서 orientation과 각속도를 복원할 수 있는가? Stencil로 표시한 공의 연속 노출에서 자세를 추정했다. | 제시된 조건에서는 350 fps에서 최대 175 rps를 측정했다. 공개 implementation은 이미지 파일명에 nanosecond timestamp를 요구한다. | Optical readout의 복원 방법과 시각 저장 형식이 명확하다. 특정 anti-spin 시편이 있는 impact dataset이라는 근거는 이 연구에서 찾지 못했다. [원 논문](https://arxiv.org/abs/2303.03879) · [연구자 코드·데이터 안내](https://github.com/cogsys-tuebingen/spindoe) |
| Saiki & Sakai (2017) | 입사 속도·입사각·spin과 rebound의 관계가 제품별로 달라지는가? 고정 racket에 표시한 공을 발사하고 4,000 fps로 촬영했다. | 2개 회사의 rubber 4개, 속도 4수준과 spin 2수준. 가장 빠른 입사 설정은 65±5 km/h이며 모든 시료에서 속도가 올라갈수록 restitution이 감소했다. | 빠른 입사, metric scale과 영상 표본 간격을 함께 제시한다. 본문 서론에 anti-spin이라는 말이 있지만 Table 2의 시험 재료 네 개는 모두 ‘In’이며 상용 모델을 익명 코드로 표기했다. [원문 §2–3, Table 2, Fig. 4](https://www.jstage.jst.go.jp/article/jsmeshd/2017/0/2017_D-7/_pdf) |
| Ishizuka 등 (2025) | Incoming spin에 따라 racket strain과 이탈 운동이 어떻게 달라지는가? 고정 blade, 2,000 fps side view와 strain gauge를 사용했다. | STIGA DNA pro를 부착한 한 blade에서 weak topspin, strong topspin, backspin을 비교했다. 충돌 전후 시점과 위치에서 속도를 산출하고 strain을 분석했다. | 빠른 입사와 동시 계측 설계의 선례다. 사용한 시편이 명시돼 v73의 신규 anti-spin 후보에서 제외할 수 있다. [원문 §2, Fig. 3](https://www.jstage.jst.go.jp/article/transjsme/91/945/91_25-00035/_pdf) |
| Kreiser 등 (2026), 1000 Rallies | Event camera로 빠른 경기에서 ball-state 추정이 개선되는가? APS와 event camera를 200 Hz trigger로 동기화하고 dot pattern으로 spin을 측정했다. | 주 시설 elite 표본은 선수 7명·1,167 rally·최고 ball speed 25.9 m/s다. 논문은 velocity를 관측에 추가해 bounce-point prediction error가 36% 줄었다고 보고한다. | Racket contact가 수동 표시되며 1 kHz label은 200 Hz 관측에서 보간했다. 논문에서 각 contact의 rubber identity와 사용한 racket face의 연결을 확인하지 못했다. [원문 Table I, §III](https://arxiv.org/html/2606.25620v1) |
| Preston Table Tennis (2010), 제작자 실험 | 서로 다른 rubber에서 top/backspin·입사 속도·타점에 따라 공이 어떻게 나가는가? 실제 시편을 고정하고 표시된 공을 촬영했다. | v72의 Super Anti 본편에서 서로 다른 24개 planar scene 중 10개를 복원했다. Average 9/12, faster 1/12에서만 입·출사 tracking을 모두 통과했다. | 빠른 입사에서 발생하는 판독 누락을 보여준다. 제작자가 시리즈 중 별도 challenge에 표기한 1,000 fps는 본편의 촬영 시각이나 독립 시편을 입증하지 않는다. [제작자 본편](https://www.youtube.com/watch?v=43WWRf5yW6M) · [제작자 시리즈](https://www.tabletennis365.com/Preston/Pages/Rubbers) |

## 2. 주제별 문헌 종합

**접촉 물성과 관측 성능.** Gossard는 같은 실험 계통에서 여러 rubber의 조건부 응답을 비교했고, Saiki–Sakai는 속도·입사각의 영향을 실험적으로 확인했다. 이 결과는 입사 상태를 구분할 이유를 제공한다. 빠른 표본이 영상에서 더 자주 누락되면 retained impact만의 평균은 공급된 모든 공의 평균과 달라질 수 있다. v72의 speed별 coverage가 그 구분이 실제로 필요한 사례다. [Gossard](https://arxiv.org/html/2604.11349v1) · [Saiki–Sakai](https://www.jstage.jst.go.jp/article/jsmeshd/2017/0/2017_D-7/_pdf) · [Preston](https://www.youtube.com/watch?v=43WWRf5yW6M)

**시간과 회전의 계측.** SpinDOE의 파일명에는 촬영 시각이 들어가며, 1000 Rallies는 공통 trigger와 원 관측 빈도를 공개한다. 같은 rubbers가 다른 날의 영상에도 쓰였는지, 편집 후 어느 frame이 어느 노출에 해당하는지는 이와 별도의 문제다. 공개 제품 소개에서 Hallmark Devil Anti 시편과 강한 topspin 상대 타구를 확인할 수 있었지만, 게시된 제작자 소개로는 capture clock과 물리적 입사속도를 검증할 수 없었다. [SpinDOE](https://github.com/cogsys-tuebingen/spindoe) · [1000 Rallies](https://arxiv.org/html/2606.25620v1) · [WRM 제작자 설명·영상](https://rubber.ocnk.net/news-detail/1741)

## 3. Knowledge map

입사 속도·spin과 표면 운동 → 상대 접선속도 → tangential impulse → 이탈 translation·rotation은 물리 경로다. 여기서 이탈 부호를 읽는 경로는 노출 시각·기하 보정 → 공 중심·orientation 추적 → 접촉 시점으로의 외삽 → 두 측정량 차이의 부호다. 첫 경로는 Gossard의 impact model에서, 두 번째는 SpinDOE와 1000 Rallies의 측정·보간 절차에서 근거를 찾을 수 있다. 두 경로를 함께 관측해도 하나의 새 표면에서 제품군 전체의 특성은 정해지지 않는다. [Gossard](https://arxiv.org/html/2604.11349v1) · [SpinDOE](https://arxiv.org/abs/2303.03879) · [1000 Rallies](https://arxiv.org/html/2606.25620v1)

## 4. 데이터 공백과 배포본 검증

- Gossard의 고정 commit에서 60개 파일을 전수 확인했다. 30개 CSV, 25개 Python 파일, 장비 사진 한 장과 문서·설정이다. Configuration 10은 original 791행, train 632행, test 158행이었다. 14개 숫자 열을 multiset으로 대조하면 공식 split에 없는 original 행은 하나뿐이며 header에 specimen·session·frame timestamp가 없다. 원시 운동값, 동일 시편의 새 영상, 독립 시편의 영상은 별개다. [고정 저장소](https://github.com/gossardt/tt_rackets/tree/159c6f040dd7832570b537b3b5fd00a97fc28c15) · [배포 파일·대조 원장](audit/archive_index.json)
- DataCite에서 표제에 ‘table tennis’가 있는 dataset 138개 record를 반환받았다. ‘table tennis’와 ‘anti-spin/antispin’을 동시 요청한 경우 0개, racket/rubber와 spin/impact를 요구하면 동일 forehand MediaPipe dataset의 두 버전이었다. 검색식은 제목·메타데이터가 빈약한 영상까지 포괄하지 않는다. 관련 DOI의 원문을 열면 TTSwing은 대만 선수 93명의 9-axis grip 신호, T3Set은 영상·sensor·코칭 언어의 정렬을 공개했다. 이들 배포 안내에서 rubber별 signed pre/post spin과 충돌 직전·직후 속도는 확인되지 않았다. [질의·전체 반환 목록](audit/archive_index.json) · [TTSwing](https://zenodo.org/records/14524670) · [T3Set](https://zenodo.org/records/15516144)
- HAS-D71을 사용한 DITECT의 자체 실험은 공의 logo로 top/backspin 전후 rpm·km/h를 산출하며 실제 데모 영상이 있다. 제조사 페이지에는 시험한 rubber와 clip별 capture clock이 명시되지 않아 anti-spin임을 검증하지 못했다. 이 경우 제품 신원을 보완하면 되고, 기존 Super Anti 본편에는 촬영 clock을 보완해야 한다. [DITECT 원 실험](https://www.ditect.co.jp/products/soft/case/dipp_motion_v_3d/903.html) · [Preston](https://www.youtube.com/watch?v=43WWRf5yW6M)

## 5. 무엇이 불일치처럼 보이는가?

1 kHz로 제공되는 ball state, 200 Hz로 얻은 실제 frame, Sony sensor의 개별 event 시각은 서로 다른 물리량이다. 1000 Rallies는 이 차이를 명시한다. Saiki–Sakai에서도 §2의 촬영률 4,000 fps와 Fig. 4에 표시한 여러 ball 위치 사이의 간격 2.5 ms는 다르다. 같은 저자 안에서도 게시용 도해의 frame 간격을 카메라 고유 cadence로 대체하면 속도 환산이 달라진다. 단위가 다른 두 연구의 성능 차이를 곧바로 model 차이라고 볼 수 없다. [1000 Rallies §III-B](https://arxiv.org/html/2606.25620v1) · [Saiki–Sakai Fig. 4](https://www.jstage.jst.go.jp/article/jsmeshd/2017/0/2017_D-7/_pdf)

## 6. 방법별 적합성

Known-dot·hardware-timed tracking은 실제 각속도의 부호와 크기를 복원하는 데 적합하고, 고정 racket은 표면 좌표계의 상대속도 계산을 단순화한다. 반면 경기 영상은 넓은 입사 분포를 담아도 ball velocity에서 racket motion을 빼고 사용 면을 식별해야 한다. Rating이나 stroke별 집계는 제품 후보를 찾는 단서가 되지만 K개의 calibration 사건과 이후 held-out 사건을 연결해 주지 않는다. 빠른 입사에서 장면을 거의 읽지 못하면 속도를 높인 조건의 physics와 optical selection이 뒤섞인다. [SpinDOE](https://arxiv.org/abs/2303.03879) · [Gossard](https://arxiv.org/html/2604.11349v1) · [1000 Rallies](https://arxiv.org/html/2606.25620v1) · [Preston](https://www.youtube.com/watch?v=43WWRf5yW6M)

## 7. Evidence synthesis

- **직접 확인**할 수 있는 것은 Gossard의 지정 시편·370 fps 촬영법·공개 측정값, 일본 실험의 지정된 재료·촬영 속도, 정해진 조건에서 dot pattern으로 spin을 읽을 수 있다는 결과다. [Gossard](https://arxiv.org/html/2604.11349v1) · [Ishizuka 등](https://www.jstage.jst.go.jp/article/transjsme/91/945/91_25-00035/_pdf) · [SpinDOE](https://arxiv.org/abs/2303.03879)
- **제한적인 근거**는 한 시편의 retained fast impact에서 조건부 예측 오차를 비교하는 데 적용된다. 새 시편에서의 방향 판독률을 산출하려면 판독 실패까지 포함하는 원영상의 분모가 필요하다. 추정치가 이미 있는 CSV만으로는 이 분모가 복원되지 않는다. [공개 데이터 구조](https://github.com/gossardt/tt_rackets) · [Preston 관측 자료](https://www.youtube.com/watch?v=43WWRf5yW6M)
- **미확인인 주장**은 공개된 해당 독립 시편에서 K=8이 K=4보다 release 방향을 잘 맞춘다는 것, 특정 방법이 빠른 모든 anti-spin impact에서 우월하다는 것이다. 선택된 장면 및 기존 시편에서의 결과만으로 causal material effect를 평가할 수 없다. [Saiki–Sakai](https://www.jstage.jst.go.jp/article/jsmeshd/2017/0/2017_D-7/_pdf) · [Gossard의 한계](https://arxiv.org/html/2604.11349v1)

## 8. 근거가 있는 후속 질문 10개

| 연구 질문 | 변수·중요성과 검증 방법 |
|---|---|
| 1. Fast input에서 판독 불가가 늘어나는 원인은 무엇인가? | 실제 incoming speed × exposure를 교차하고 blind 판독자가 전체 시도에서 center·spin 성공률을 따로 잰다. [Preston](https://www.youtube.com/watch?v=43WWRf5yW6M) |
| 2. 어느 이탈 상대속도에서 sign이 재현되는가? | 두 카메라의 독립 추정으로 0 근처 sign coverage와 재현율을 그린다. [1000 Rallies](https://arxiv.org/html/2606.25620v1) |
| 3. 동일 제품의 시편 교체가 model ranking을 바꾸는가? | A-B-S II soft의 서로 다른 sheet와 생산 lot를 구분하고 leave-one-sheet-out으로 효과를 비교한다. [Gossard의 config 표](https://github.com/gossardt/tt_rackets) |
| 4. 같은 시편에서 시간축만 바꾸면 네 방법의 순위가 달라지는가? | Hardware timestamp를 기준으로 전·후반을 서로 다르게 subsample한 뒤 SI 오차를 다시 계산한다. [SpinDOE 파일 형식](https://github.com/cogsys-tuebingen/spindoe) |
| 5. 정지 racket 결과를 active block에 옮길 수 있는가? | Blade 속도·법선을 동시에 측정한 뒤 같은 시편의 고정/이동 조건에서 racket-frame error를 비교한다. [Gossard](https://arxiv.org/html/2604.11349v1) · [Ishizuka 등](https://www.jstage.jst.go.jp/article/transjsme/91/945/91_25-00035/_pdf) |
| 6. 4번의 동일 속도와 4번의 분산 속도 중 무엇이 유리한가? | 입사 결과를 보기 전에 speed-stratified allocation을 고정하고 같은 test의 spin·velocity를 비교한다. [Saiki–Sakai](https://www.jstage.jst.go.jp/article/jsmeshd/2017/0/2017_D-7/_pdf) |
| 7. 단일 stripe의 방향이 dot-pattern 회전을 얼마나 보존하는가? | 같은 공을 두 표시·시점에서 동시에 계측해 high spin 조건의 aliasing과 sign error를 분리한다. [SpinDOE](https://arxiv.org/abs/2303.03879) · [Preston](https://www.youtube.com/watch?v=43WWRf5yW6M) |
| 8. Impact 위치를 고정하면 빠른 입사에서 모델 차이가 남는가? | Blade 위치와 속도를 교차하고 local strain을 관측한 상태로 held-out MAE를 비교한다. [Ishizuka 등](https://www.jstage.jst.go.jp/article/transjsme/91/945/91_25-00035/_pdf) |
| 9. 이탈 방향 판독과 gate 선택에 같은 수준의 신뢰도가 필요한가? | 동일 calibration으로 전문가가 부여한 branch, 실제 expert 출력 sign, 연속 오차를 세 지표로 기록한다. [Gossard](https://arxiv.org/html/2604.11349v1) · [Preston](https://www.youtube.com/watch?v=43WWRf5yW6M) |
| 10. 기존 시편에서 선택한 판독 기준이 새 시편에서도 같은 오차를 보장하는가? | Source-only로 margin을 동결하고 다른 anti-spin 시편·날짜의 hardware-timed 영상에서 coverage와 conditional accuracy를 같이 평가한다. [SpinDOE](https://arxiv.org/abs/2303.03879) · [Gossard](https://arxiv.org/html/2604.11349v1) |

## 9. 논문 주장에 대한 검토

기존 시편의 K=4/8 결과는 물리 단위가 있는 비교 근거로 가치가 있다. 새로운 주장을 강화하려면 원 영상의 시각, 양쪽 시점의 위치·각도, calibration과 held-out이 공유하지 않는 impact ID, 고유 specimen ID가 하나의 실험 기록에 연결돼야 한다. 특히 retained fast impact가 적으면 읽힌 표본에서의 적중률과 전체 빠른 충돌 중 정확히 예측할 수 있었던 비율이 다르다. 두 값에 각각 분자와 분모를 쓰고, 독립 시편 수와 실험 session 수도 제시해야 한다. 이미 state-dependent physical adaptation과 dot-based spin measurement가 발표됐으므로, 논문의 고유 주장은 이탈 방향 비교가 재현되는 조건과 관측 실패가 바꾸는 결론으로 좁힐 수 있다. [Gossard](https://arxiv.org/html/2604.11349v1) · [SpinDOE](https://arxiv.org/abs/2303.03879) · [1000 Rallies](https://arxiv.org/html/2606.25620v1)
