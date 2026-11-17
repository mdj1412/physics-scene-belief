# v74 문헌 검토 · 작은 운동 오차가 잘못된 이탈 부호와 공존하는 이유

## 1. 연구별 질문·방법·표본·결론 비교

| 연구 | 질문·방법 | 표본과 주요 발견 | v74에서 활용할 근거와 한계 |
|---|---|---|---|
| Gossard (2026) | 입사 상태에 따라 tangential impulse를 바꾸면 이탈 운동을 더 잘 예측하는가? 고정 racket과 370 fps 영상, SpinDOE로 전후 상태를 복원했다. | 6제품·10 configuration 가운데 anti-spin은 A-B-S II soft 한 종류다. Eq. 9는 선속도 변화와 각속도 변화에서 각각 impulse 계수 α를 계산한다. Eq. 10의 sliding–rolling transition은 velocity MAE를 줄이도록 적합했다. | v74에 가장 직접적인 선행이다. §VI의 online adaptation은 inverted 제품군의 5개 configuration에서 수행했고 약 30회 관측 후 velocity 개선을 보고했다. 4·8회 anti-spin 보정으로 이탈 부호까지 고쳤다는 근거는 아니다. [원문 §III, §V-A, §VI](https://arxiv.org/html/2604.11349v1) · [공개 configuration](https://github.com/gossardt/tt_rackets/tree/159c6f040dd7832570b537b3b5fd00a97fc28c15) |
| Rinaldi 등 (2019) | rubber 구조·입사각·spin이 rebound에 미치는 영향을 stroboscopic imaging과 finite element model로 비교했다. | foam과 foam+compact 시편, 14 m/s에서 세 입사각과 세 spin 조건을 조사했다. Backspin 66 rev/s에서는 수직·경사 입사에서 spin ratio와 deviation angle의 부호가 달랐다. | 한 지표만으로 rebound를 기술하기 어렵다는 직접 실험이다. 이 논문의 spin ratio는 두 각속도의 비이고, v74가 판정하는 접촉점의 이탈 상대속도와 다르다. 시험 시편도 다르다. [원문 §2–3, Figs. 4–6](https://www.mdpi.com/2076-3417/9/1/158) |
| Cross (2002) | 접촉 도중 grip·slip이 바뀌는지 force transducer로 시간에 따른 normal·tangential force를 측정했다. | tennis ball, superball, baseball, basketball, golf ball의 경사 충돌을 다뤘다. 일부 조건에서 tangential force가 충돌 도중 방향을 바꿨다. | Phase 전환을 확인하는 데 필요한 관측의 선례다. 시험한 다섯 공에 table-tennis ball은 없고, 충돌 도중 force의 부호와 이탈 후 상대속도 부호도 다르다. [저자 원문 §II–IV](https://www.physics.usyd.edu.au/~cross/PUBLICATIONS/GripSlip.pdf) |
| Gossard 등 (2023), SpinDOE | 알려진 dot pattern의 orientation을 추적해 ball spin을 얼마나 빨리 읽을 수 있는가? | 350 fps 카메라와 저자 조건에서 최대 175 rps를 측정했다. Orientation-to-spin 추정은 연속 frame과 공의 가시 무늬에 의존한다. | Spin 계측 경로의 근거다. 속도 측정과 spin 측정을 뺀 값의 부호는 별도 오차 전파가 필요하다. 논문의 최대 측정 범위만으로 A-B-S II soft의 개별 impact 부호 신뢰구간은 정해지지 않는다. [방법·평가 원문](https://arxiv.org/html/2303.03879) |
| Brynjarsdóttir & O’Hagan (2014) | 물리 simulator의 파라미터와 model discrepancy를 동시에 추정하면 어느 정도 식별되는가? | 참값을 아는 예제로 11·31·61개 관측을 비교했다. Discrepancy를 넣으면 관측 범위 내 예측이 개선됐지만 참 파라미터와의 구별에는 타당한 prior가 중요했다. | 보정 계수의 물리 해석에 적용할 통계적 근거다. Synthetic example의 성능을 anti-spin의 성능으로 이전할 수 없다. [저자 원문 §§2–3](https://www.tonyohagan.co.uk/academic/pdf/simmach.pdf) |
| Guo 등 (2017) | 분류기의 신뢰도가 실제 적중 확률과 일치하도록 조정할 수 있는가? 영상·문서 분류 benchmark에서 post-hoc calibration을 비교했다. | 여러 모델에서 양수 한 개로 logits를 나누는 temperature scaling이 confidence calibration을 개선했다. 같은 logits의 argmax는 유지된다. | 확률의 신뢰도와 방향 결정을 구별할 근거다. Signed margin에 양수만 곱해 결정 경계를 그대로 두면 방향 적중률은 바뀌지 않는다. [원문 §4.2](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf) |
| Geifman & El-Yaniv (2019), SelectiveNet | 예측과 reject option을 같이 학습하면 정해진 coverage에서 risk가 줄어드는가? | 영상 분류 3종과 concrete regression 1종에서 비교했다. 예를 들어 concrete 표본은 1,030개였다. 일부 관측에 응답을 보류함으로써 risk–coverage 관계를 제어했다. | 같은 coverage의 의미를 명시하는 데 쓸 수 있다. v74의 readout mask는 관측된 label에 적용하며, 배포 때 모델이 confidence를 보고 거절하는 selective policy와 역할이 다르다. [원문 §§3, 6](https://proceedings.mlr.press/v97/geifman19a/geifman19a.pdf) |

## 2. 주제별 문헌 종합

**물리 모델이 설명하는 양.** 고정 racket의 접선 좌표에서 이탈 상대속도는 $s'=u'-Rω'$다. Thin-shell 가정에서 $u'=u−αs$, $Rω'=Rω+1.5αs$이므로 두 식을 빼면 $s'=(1−2.5α)s$다. 같은 입사 상태에서도 추정 α가 0.4의 어느 쪽에 있느냐에 따라 이탈 부호가 바뀐다. 이 경계와 두 가지 α 추정 경로는 이미 원 연구에 있다. 이 식은 v74의 제약 모델 안에서 성립하며 실제 시편의 모든 접촉이 이 식을 정확히 따른다고 가정할 근거는 없다. [Gossard Eq. 9, Table I](https://arxiv.org/html/2604.11349v1)

**오차의 원인과 예측의 변경.** 복원한 관측을 기준으로 $ŝ'−s'=e_u−R e_ω$다. 두 component의 절대오차 평균만으로 이 차이의 부호나 두 residual의 공분산은 결정되지 않는다. 한 component를 관측값으로 바꾸는 oracle 진단은 어떤 출력이 판정을 바꿨는지 보여준다. 그것만으로 그 오차를 낳은 마찰, 재료 변형, 계측 오차를 구분할 수 없다. Contact force 이력과 운동을 별도로 측정한 실험, model discrepancy와 물리 파라미터의 confounding이 그 구분의 근거다. [Cross](https://www.physics.usyd.edu.au/~cross/PUBLICATIONS/GripSlip.pdf) · [Brynjarsdóttir & O’Hagan](https://www.tonyohagan.co.uk/academic/pdf/simmach.pdf)

## 3. Knowledge map

입사 translation·spin → 입사 접촉점 상대속도 → tangential impulse → 이탈 translation·spin → 이탈 상대속도·부호로 이어진다. 관측 경로에서는 영상의 center·dot orientation → 복원한 운동 상태 → 판독 mask → 조건부 적중률로 이어진다. Calibration이 바꿀 수 있는 것은 impulse 계수, 연속 예측의 residual, 또는 별도 decision boundary다. 조건부 적중률만으로 이 중 어느 항이 물리적 원인이었는지 식별되지 않는다. [Gossard](https://arxiv.org/html/2604.11349v1) · [SpinDOE](https://arxiv.org/html/2303.03879) · [Model discrepancy](https://www.tonyohagan.co.uk/academic/pdf/simmach.pdf)

## 4. 근거가 지지하는 연구 공백

- **예산과 목적이 동시에 다른 실험.** 가장 가까운 원문은 anti-spin에서 α를 velocity 오차에 맞추고, 온라인 적응은 inverted 설정에서 평가했다. 동일 시편의 K=4·8에서 연속 운동과 이탈 부호를 함께 개선하는지, 그때 어떤 단순 정책이 남는지는 별도 실험이다. [Gossard §V-A, §VI](https://arxiv.org/html/2604.11349v1)
- **오차 원천을 가를 반복 계측.** Dot-based spin의 가시성, 촬영 각도, rubber 구조는 서로 다른 불확실성을 만든다. 같은 충돌의 독립 camera·강체 좌표 검산·접선 force 이력 중 어느 것도 없는 경우, 부호 오답을 특정 material law의 실패로 단정하기 어렵다. [SpinDOE](https://arxiv.org/html/2303.03879) · [Rinaldi 등](https://www.mdpi.com/2076-3417/9/1/158) · [Cross](https://www.physics.usyd.edu.au/~cross/PUBLICATIONS/GripSlip.pdf)
- **독립성의 단위.** 공개 anti-spin은 한 configuration이다. 같은 22개 판독 가능 fast impact에 calibration draw를 여러 번 적용해도 다른 시편과 날짜에서의 성능을 측정한 것은 아니다. [공식 제품 목록](https://github.com/gossardt/tt_rackets/tree/159c6f040dd7832570b537b3b5fd00a97fc28c15) · [v73 판독 분모](../racket_video_v73/gpu/control/metrics.csv)

## 5. 연구 간 불일치를 설명할 수 있는 조건

Gossard가 사용한 α=0.4의 rolling 경계와 Cross가 관찰한 접촉 중 tangential force reversal은 서로 다른 시간적 관측이다. 전자는 rigid-body 형태의 impulse 모델과 충돌 전후 상태를 연결하고, 후자는 다른 다섯 종류의 공에서 충돌 중 force의 이력을 직접 측정했다. Rinaldi 등의 spin inversion 역시 각속도의 입출사 부호에 대한 결과다. 서로 다른 명제이므로 어느 관찰도 A-B-S II soft의 22개 충돌에서 실제 contact patch의 반전 여부를 확정하지 않는다. [Gossard](https://arxiv.org/html/2604.11349v1) · [Cross](https://www.physics.usyd.edu.au/~cross/PUBLICATIONS/GripSlip.pdf) · [Rinaldi 등](https://www.mdpi.com/2076-3417/9/1/158)

## 6. 방법 비교와 검증 기준

| 접근 | 보존하는 정보 | v74에서 확인할 사항 |
|---|---|---|
| Component별 연속 예측·impulse 제약 | 속도와 spin의 크기, 제약된 coupling | 같은 37개 fast impact에서 velocity와 spin 오차를 각각 계산한다. $s'$의 부호 오류는 별도로 센다. [Gossard](https://arxiv.org/html/2604.11349v1) |
| 별도 sign classifier | Calibration에서 읽힌 부호와 입사 정보 | Signed label이 모호한 calibration도 관측 비용에 포함한다. 최종 state와 독립적인 sign만 바꿨다면 두 출력의 불일치율을 공개한다. [Confidence와 decision의 구분](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf) |
| Sign을 반영한 state correction | 최종 state에서 물리 부호를 다시 계산할 수 있음 | 보정한 state에서 두 연속 오차를 다시 계산한다. 제약을 유지한다는 주장에는 그 제약의 residual도 필요하다. [Impulse 연립식](https://arxiv.org/html/2604.11349v1) |
| 고정된 label mask의 조건부 평가 | 모든 방법이 같은 고유 impact를 맞힘 | 모든 observed state에는 continuous loss, 고정된 동일 행에는 sign accuracy와 양·음수 recall을 쓴다. 동일 판독률에 서로 다른 test 행을 남기는 방식은 paired 비교가 아니다. [Coverage의 방법론적 배경](https://proceedings.mlr.press/v97/geifman19a/geifman19a.pdf) |

## 7. Evidence synthesis

근거가 가장 확실한 부분은 고정한 좌표계에서의 residual 항등식, 동일한 impulse로 연결된 두 component가 특정 경계를 사이에 두고 다른 이탈 부호를 낼 수 있다는 점, 양수 temperature만으로 부호 자체를 바꿀 수 없다는 점이다. 여러 관측을 아우르는 중간 수준의 근거는 입사각·spin·재료 및 측정 방식에 따라 rebound 지표가 달라진다는 것이다. 어떤 물리 변형이 개별 오답을 일으켰는지, 독립 rubber에서 K=8 sign correction이 재현되는지는 이 자료만으로 정해지지 않는다. [Gossard](https://arxiv.org/html/2604.11349v1) · [Rinaldi 등](https://www.mdpi.com/2076-3417/9/1/158) · [Guo 등](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf) · [Model discrepancy](https://www.tonyohagan.co.uk/academic/pdf/simmach.pdf)

## 8. 근거에서 도출한 열 가지 후속 연구 질문

| 질문 | 공백·변수·검증 방법 |
|---|---|
| 1. 아슬아슬한 α 오차와 큰 model mismatch를 구별할 수 있는가? | α의 0.4 경계까지 거리, 입사 slip, 부호를 함께 기록한다. 두 출처의 α가 상반되는 영역을 보류하고 별도 분석한다. [Gossard Eq. 9](https://arxiv.org/html/2604.11349v1) |
| 2. 동일 충돌에서 두 α 추정치의 차이가 영상 복원 오차로 설명되는가? | 두 camera의 독립 궤적과 angular velocity를 끼워 맞춰 calibration 전후의 agreement를 측정한다. [SpinDOE](https://arxiv.org/html/2303.03879) · [Gossard](https://arxiv.org/html/2604.11349v1) |
| 3. 빠른 충돌에서 spin과 translation 중 어느 residual이 판정을 바꾸는가? | 두 방향의 oracle component substitution과 둘을 함께 치환한 결과를 paired 비교한다. 실제 힘의 분리와 계산상의 분리를 구별한다. [Gossard](https://arxiv.org/html/2604.11349v1) |
| 4. 같은 속도에서도 타격각이 바뀌면 learned threshold가 이동하는가? | 각도×spin을 교차하고 같은 session의 공통 부호 양측을 평가한다. [Rinaldi 등](https://www.mdpi.com/2076-3417/9/1/158) |
| 5. Spike 형태의 접선 force가 오답과 연결되는가? | 고정 racket에 force sensor와 시각 동기화를 추가한다. 시간 이력, 누적 impulse, 이탈 부호를 각각 검정한다. [Cross](https://www.physics.usyd.edu.au/~cross/PUBLICATIONS/GripSlip.pdf) |
| 6. Source prior의 기준 시편이 바뀌어도 보정 부호는 안정적인가? | Source rubber를 하나씩 제외해 보정량과 paired accuracy 변화를 비교한다. [Gossard §VI](https://arxiv.org/html/2604.11349v1) · [Model discrepancy](https://www.tonyohagan.co.uk/academic/pdf/simmach.pdf) |
| 7. 보정값이 model parameter와 계측 bias 중 무엇을 흡수하는가? | 동일 시편에서 camera 위치와 modulus 조건을 따로 바꾸고 알려진 calibration target을 함께 촬영한다. [Model discrepancy](https://www.tonyohagan.co.uk/academic/pdf/simmach.pdf) |
| 8. 양·음수 prevalence가 바뀌면 다수결 대비 이득은 남는가? | 시편·session별 class recall을 보존하고 여러 prevalence로 가중한 점수를 보고한다. [독립 시험의 필요성을 보이는 단일 anti-spin configuration](https://github.com/gossardt/tt_rackets/tree/159c6f040dd7832570b537b3b5fd00a97fc28c15) |
| 9. 부호에 맞춘 state correction의 크기가 실제 측정 해상도를 넘는가? | $s'$가 0의 양옆에서 극소량만 움직여도 hard label은 바뀐다. 예측 margin을 기록하고 반복 계측으로 practical margin을 정한 뒤 재실험한다. [SpinDOE](https://arxiv.org/html/2303.03879) · [Confidence calibration](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf) |
| 10. 모델이 배포 전에 보류할 수 있다면 같은 관측 판독률에서 의사결정 risk를 줄이는가? | Test label 없이 계산한 confidence로 별도 reject threshold를 고정한다. Label readout coverage와 predicted acceptance를 독립 분모로 보고한다. [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a/geifman19a.pdf) |

## 9. 논문 주장에 대한 검토

안전하게 주장할 수 있는 기여 후보는 4회·8회의 동일 관측 예산에서 **전후 운동을 설명하는 손실과 이탈 부호를 맞히는 손실을 분리하고, 보정한 부호를 운동 상태에 되돌렸을 때 어떤 이득과 비용이 남는지** 수치로 정하는 것이다. 두 정책이 고른 서로 다른 expert의 branch 이름만 비교하면 이 질문에 답할 수 없다. 사전 동결한 기준으로 같은 fast impact에서 majority vote 대비 paired 차이, 원래 elastic predictor 대비 두 연속 오차의 변화, 실질적 예측 margin을 함께 제시해야 한다. 연속 오차 차이가 통계적으로 불확실한 것과 사전에 정한 noninferiority 기준을 통과한 것도 다르다. 기존 한 시편의 반복 평가에서 얻은 양성·음성 결과 모두 그 범위로 제한한다. [가장 가까운 선행과 기존 adaptation](https://arxiv.org/html/2604.11349v1) · [부호 판독 분모](../racket_video_v73/gpu/control/metrics.csv) · [Discrepancy 해석의 한계](https://www.tonyohagan.co.uk/academic/pdf/simmach.pdf)
