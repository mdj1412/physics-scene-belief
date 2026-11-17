# v70 문헌 검토와 contact patch의 식별 가능성

## 1. 원문별 질문, 측정, 결과와 한계

| 연구와 질문 | 방법·표본 | 주요 결과와 결론 | v70에서 확인 가능한 범위 |
|---|---|---|---|
| Gossard (2026, preprint). 서로 다른 rubber의 rebound를 incoming state로 예측할 수 있는가? | 6개 rubber 제품, 3개 blade 종류, 10개 구성. 370 fps planar tracking, 50 ms local regression, dotted ball spin 측정. 공개 original CSV에 8,184개 impact. | 저자 평가에서 상태에 따라 달라지는 impulse parameter가 constant model보다 정확했다. 저자 Gaussian process는 각 racket의 학습자료로 적합했다. | 기존 80/20 split에는 모든 구성이 양쪽에 있다. 새 rubber를 처음 접하는 v70의 비교에는 target 자료를 사용하는 fit과 공개 aggregate score를 그대로 가져올 수 없다. [원문 §III–V](https://arxiv.org/html/2604.11349v1) · [저자 데이터](https://github.com/gossardt/tt_rackets/tree/159c6f040dd7832570b537b3b5fd00a97fc28c15) |
| Nakashima 외. 표면 종류에 따라 다른 tangential impulse를 적용해야 하는가? | 고정 table과 Butterfly racket에 공을 발사하고 900 fps 카메라로 중심과 회전을 추적했다. | Racket rubber가 접선 방향의 탄성 에너지를 돌려준다는 가정에서 입사 접선 속도에 비례하는 impulse를 유도하고 영상 자료와 비교했다. | 단순한 rigid/Coulomb의 물리적 대안이다. 다양한 상용 rubber의 product-level transfer나 patch 판독을 검증한 연구는 아니다. [원문 §II–IV](https://nagoya.repo.nii.ac.jp/record/12580/files/1031.pdf) |
| Rinaldi 외. Foam과 상부 compact layer가 회전하는 공의 충돌을 어떻게 바꾸는가? | 14 m/s, 3개 입사각, 3개 spin 조건, foam 및 foam+compact 표적의 촬영 실험과 3D finite-element model. | 실험에서 restitution, 반사 방향, spin ratio를 구했다. 모형으로 두 층의 국소 변형을 계산했다. | Fig. 9는 **모사한** displacement field다. Fig. 4–5의 광학 측정값은 공 궤적과 표식이다. Fig. 9를 외부 실측 patch로 쓰면 simulator 가정을 다시 평가하는 셈이다. [원문 §2–4 및 Fig. 9](https://www.mdpi.com/2076-3417/9/1/158) |
| Rémond 외 (2022). 얇은 shell의 buckling을 impact 중에 직접 볼 수 있는가? | 지름 40 mm ABS ball, 무회전 normal impact, 고정 유리, 약 1–12 m/s. 20,000 fps로 같은 영상에서 shell 단면과 전후 속도를 판독. | 약 5.5±0.5 m/s에서 접촉 중심이 들리기 시작했고 restitution 변화율도 달라졌다. 외부 직경이 같아도 중심부 접촉 여부가 달라질 수 있다. | 같은 trial에서 geometry와 rebound를 측정하는 방법이 실현 가능하다. 유리·정면 충돌에서 입사 속도를 바꾼 결과이므로 rubber의 oblique impact에서 patch의 독립적 예측 이득을 확인하려면 새 비교가 필요하다. [원문 §II, Fig. 2–6](https://link.aps.org/accepted/10.1103/PhysRevE.106.014207) |
| Zhang·Fan·Kang (2025). Racket에 부착한 sensor로 hit 위치와 회전을 구분할 수 있는가? | 5×5 row-column PVDF array, 10개 전압 channel. S2는 서로 다른 위치의 4 hits, S3는 회전 조건의 3 hits. | 행·열 신호로 타격 위치를 찾고 회전 방식에 따른 신호 차이를 보고했다. 압력 변화에 반응하는 실물 데이터다. | S3의 11열은 시간과 channel voltage이다. 고속 영상의 독립 rebound velocity·angular velocity 또는 실제 접촉 경계가 주어지지 않는다. [원문과 supplement 목록](https://doi.org/10.1371/journal.pone.0333735) · [직접 검사한 S3](https://doi.org/10.1371/journal.pone.0333735.s003) |
| Biber 외 (2023). 두 turf에서 golf ball의 전후 운동은 어떤 모델로 설명되는가? | 인조·천연 turf 2종, 1,023회, 입출력 translation과 spin의 영상 측정. | 저자는 inclined surface와 piecewise affine를 포함한 모델들을 비교했고, 간단한 데이터 기반 fit이 당시 검증 조건에서 물리 model을 앞설 수 있었다. | 골프공의 잔디 충돌과 얇은 탁구공의 rubber 충돌은 inertia·표면 구조·실험 장비가 다르다. v70은 모델 계열의 전이를 검증한다. [측정·모델링 원문](https://doi.org/10.1007/s12283-023-00442-4) |

## 2. 문헌 종합

충돌 전후의 **전체 운동**은 공의 중심과 회전 표식을 추적해 검증할 수 있다. Gossard는 서로 다른 rubber에서 이 측정을 반복했고, Nakashima와 Rinaldi는 에너지를 저장하는 rubber의 접선 응답이 rebound와 spin에 미치는 영향을 모델링했다. 이들 연구가 주는 공통 제약은 잘 맞은 endpoint만으로 힘이 어느 위치에 어떻게 분포했는지 결정할 수 없다는 점이다. [Gossard](https://arxiv.org/html/2604.11349v1) · [Nakashima 외](https://nagoya.repo.nii.ac.jp/record/12580/files/1031.pdf) · [Rinaldi 외](https://www.mdpi.com/2076-3417/9/1/158)

**접촉 경계**는 별도의 관측이다. 투명 유리 아래에서 본 shell 단면에서는 접촉 중심의 이탈을 프레임마다 판독할 수 있었다. 불투명 racket에 붙인 전압 sensor는 전달된 응력에 반응하므로, 시간별 실제 접촉 경계의 바깥에서도 신호가 측정될 수 있다. 두 관측을 같은 변수로 사용하려면 먼저 같은 충돌에서 광학적으로 경계를 판독해 sensor의 spatial response를 보정해야 한다. [Rémond 외](https://link.aps.org/accepted/10.1103/PhysRevE.106.014207) · [Zhang 외](https://doi.org/10.1371/journal.pone.0333735)

## 3. Knowledge map

입사 상태와 장착 조건 → shell·rubber·sponge 변형 → 접촉 경계와 국소 traction → 힘과 토크의 시간 적분 → 이탈 velocity와 spin으로 이어진다. Gossard와 Nakashima의 관측은 주로 양 끝에 있다. Rémond는 중간 단면을 광학적으로 관측했고, Zhang은 지지부에 전달된 stress signal을 기록했다. Rinaldi는 실측 양 끝과 그 사이를 연결하는 FE 해석을 제공한다. **가운데의 서로 다른 변형 history가 같은 endpoint로 합쳐질 수 있다.** 따라서 endpoint residual에서 geometry의 존재 가능성을 제안할 수 있으나 공간 분포를 유일하게 복원할 수는 없다. [Gossard](https://arxiv.org/html/2604.11349v1) · [Rémond](https://link.aps.org/accepted/10.1103/PhysRevE.106.014207) · [Zhang](https://doi.org/10.1371/journal.pone.0333735) · [Rinaldi](https://www.mdpi.com/2076-3417/9/1/158)

## 4. 연구 공백과 실제 파일 감사

- **같은 trial에서 추가 feature를 분리할 수 있는가.** 저자 repository의 revision 159c6f04에서 전체 파일을 조사했다. CSV 30개의 14열은 racket angle, trajectory directory, 전후 spin 세 축, 전후 velocity 세 축이다. 유일한 이미지 파일은 장비 사진 `rackets.jpeg`이다. `traj_dir`은 원 acquisition의 이름이며 그 행에 대응하는 patch 영상이 공개 tree에 없다. 따로 조사한 Zhang의 S3 workbook에도 시간과 전압 10개만 있다. 공개 자료 사이에 동일 impact를 잇는 키가 없다. [원 repository](https://github.com/gossardt/tt_rackets/tree/159c6f040dd7832570b537b3b5fd00a97fc28c15) · [Zhang S3](https://doi.org/10.1371/journal.pone.0333735.s003) · [파일 검사 결과](patch_schema_audit.json)
- **계측과 계산의 구분.** 위치 한 점 $(x,y)$는 racket의 어디를 맞혔는지 나타낸다. `v_s`는 rigid ball의 회전·translation에서 계산한 접점 상대 속도이다. 시간별 면적·경계·압력 분포는 공간과 시간에 걸친 별도 관측이다. Incoming state에서 생성한 simulated patch는 학습 prior의 가치를 시험할 수 있다. Outbound로부터 역산한 patch 또는 impulse coefficient를 같은 행의 입력으로 사용하면 label leakage가 생긴다. [Gossard §II, §V](https://arxiv.org/html/2604.11349v1) · [Rinaldi Fig. 9](https://www.mdpi.com/2076-3417/9/1/158)
- **독립 표면·반복 단위.** 원 논문의 10개 racket 구성에는 동일 rubber 제품을 다른 blade 또는 sponge thickness로 장착한 사례가 있다. 1·3·9, 2·4, 5·6은 각각 같이 holdout해야 새 rubber 제품이 된다. 제품 수가 6개이고 일부 조합만 존재하므로 제품별 성능 차이를 pimple·blade·sponge 중 하나의 인과 효과로 배분할 수 없다. [Gossard Tables II–IV](https://arxiv.org/html/2604.11349v1)

## 5. 서로 달라 보이는 연구 결과

Gossard에서 state-dependent GP가 알려진 racket의 성능을 높였다는 것과 새 rubber에서 flexible model이 실패할 수 있다는 것은 다른 평가 조건이다. 전자에는 같은 racket의 training impacts가 있고, 후자는 target product에 대한 계수 보정 기회가 없다. 또한 rigid table model의 rolling clamp와 rubber 탄성을 담은 impulse model은 회전 반전이 가능한 범위가 다르다. 서로 다른 오차를 설명하려면 같은 held-out shot의 velocity와 spin을 따로 비교하고 두 평가에서 사용할 수 있는 정보를 명시해야 한다. [Gossard §III–V](https://arxiv.org/html/2604.11349v1) · [Nakashima 외](https://nagoya.repo.nii.ac.jp/record/12580/files/1031.pdf)

## 6. 방법 간 비교

| 방법 | 장점 | 주요 취약점과 검정 |
|---|---|---|
| Inclined rigid | 소수의 parameter로 effective normal과 tangential impulse를 함께 제한한다. | 식별된 기울기는 잔차를 줄이는 유효값이다. 카메라에서 측정한 실제 patch normal과 비교하려면 독립 관측이 필요하다. [Biber](https://doi.org/10.1007/s12283-023-00442-4) |
| Two-regime affine | 방향 및 회전 결합의 서로 다른 두 mapping을 구분해 학습할 수 있다. | Gate·expert fit을 source로 제한한다. 소수 regime의 target에서 outlier가 결과를 주도하는지 product별 오차를 본다. [Biber](https://doi.org/10.1007/s12283-023-00442-4) |
| GP 또는 state-dependent impulse | $(v_s, v_z)$에 따라 연속적으로 바뀌는 restitution과 impulse를 학습한다. | 원문 GP의 hyperparameter는 individual racket에서 적합됐다. Source-only로 합친 모델이나 몇 회의 target calibration을 사용한 모델은 별도 비교군이다. [Gossard §IV–V](https://arxiv.org/html/2604.11349v1) |
| Optical profile 및 PVDF array | 전자는 geometry의 시계열, 후자는 backing에 전달되는 stress의 시계열을 읽는다. | 유리의 정상 충돌이나 소수 회전 hit 결과를 6개 rubber에 행 단위로 결합할 수 없다. 시간 해상도·시야·공간 응답·측정 위치가 서로 다르다. [Rémond](https://link.aps.org/accepted/10.1103/PhysRevE.106.014207) · [Zhang](https://doi.org/10.1371/journal.pone.0333735) |

## 7. Evidence synthesis

**강한 근거.** 공개 racket 자료는 동일 shot에서 운동을 측정했으므로 planar rebound velocity와 spin 오차를 직접 채점할 수 있다. 채점 단위는 실제 impact이며 서로 다른 holdout으로 평가하는 단위는 6개 rubber 제품이다. **조건부 근거.** Shell buckling, rubber의 탄성, stress propagation이 contact 과정에 관여한다는 실험 결과가 있지만 운동과 접촉 경계를 동시에 읽은 가장 가까운 연구는 유리의 normal impact이다. **미식별.** 같은 입사 정보와 같은 prediction 시점에 측정 patch를 추가할 때 새 rubber에서의 loss가 얼마나 줄어드는지는 Gossard 자료에서 계산할 수 없다. 정보가 없다는 판정에 수치상 gain=0을 대입하면 안 된다. [Gossard](https://arxiv.org/html/2604.11349v1) · [Rémond](https://link.aps.org/accepted/10.1103/PhysRevE.106.014207) · [Zhang](https://doi.org/10.1371/journal.pone.0333735)

## 8. 근거에서 나온 후속 질문 10개

| 질문 | 왜 필요한가·변수 | 평가 설계와 근거 |
|---|---|---|
| 1. 같은 rubber에서 blade가 바뀐 효과가 제품 교체보다 큰가? | Blade와 rubber의 교란, velocity·spin error | Inverted 제품의 반복 blade 조합에서만 두 holdout을 별도로 평가한다. [Gossard Table IV](https://arxiv.org/html/2604.11349v1) |
| 2. Sponge가 없는 long pips에서 spin을 가장 크게 놓치는가? | 동일 제품 안의 sponge thickness 차이 | 두 long-pips 구성을 동시에 holdout한 예측을 configuration별로 나누고 incoming support를 맞춘다. [Gossard §V-C](https://arxiv.org/html/2604.11349v1) |
| 3. Incoming spin이 비슷한 shot에서도 새 rubber의 uncertainty가 커지는가? | state shift와 material shift 구분 | 공통 입사 범위에서 제품별 paired residual을 비교한다. [Gossard §III–V](https://arxiv.org/html/2604.11349v1) |
| 4. 소수의 target shots로 zero-shot 실패를 회복하는가? | zero-shot과 few-shot의 경계 | 무작위 calibration k회를 분리하고 한 번도 fit에 쓰지 않은 나머지 shot을 평가한다. [Gossard §IV](https://arxiv.org/html/2604.11349v1) |
| 5. 고정 기울기의 이득이 독립 측정한 표면 방향과 일치하는가? | effective parameter의 물리적 의미 | 실제 surface profile을 blind reconstruction해 fitted inclination과 paired 비교한다. [Biber](https://doi.org/10.1007/s12283-023-00442-4) · [Rémond](https://link.aps.org/accepted/10.1103/PhysRevE.106.014207) |
| 6. 같은 outer diameter에서 annular contact가 spin 정보를 더 주는가? | 면적 추정이 놓치는 중심 이탈 | Pre-impact state·외경을 맞춘 뒤 중앙 들뜸과 local flow를 측정하고 held-out spin 오차를 비교한다. [Rémond Fig. 2–5](https://link.aps.org/accepted/10.1103/PhysRevE.106.014207) |
| 7. 격자 센서에서 같은 hit 위치라도 서로 다른 patch를 구별할 수 있는가? | stress propagation과 경계 측정의 혼동 | 동일 landing cell에서 time-aligned optical boundary로 10개 channel readout을 검증한다. [Zhang S2·S3](https://doi.org/10.1371/journal.pone.0333735) |
| 8. Area, centroid path, contact duration 중 어떤 측정이 추가 정보를 주는가? | 더 적은 비용으로 찍을 수 있는 feature 선택 | 같은 shot, 같은 model capacity에서 feature를 하나씩 추가하고 lag와 cost를 함께 기록한다. [Rinaldi](https://www.mdpi.com/2076-3417/9/1/158) · [Rémond](https://link.aps.org/accepted/10.1103/PhysRevE.106.014207) |
| 9. 입사속도만으로 생성한 FE proxy가 실측 patch 효과를 대체하는가? | inductive bias와 독립 관측의 가치 구분 | kinematics only, kinematics+FE, kinematics+measured patch를 같은 source-only fit과 target에 비교한다. [Rinaldi Fig. 9](https://www.mdpi.com/2076-3417/9/1/158) |
| 10. 같은 shot의 접촉 정보가 실제 decision 시점에 도착하는가? | 사전 예측과 충돌 중 nowcast의 시간 차이 | Pre-impact prediction, pre-liftoff prefix, 과거 calibration impact의 patch를 분리하고 sensor latency를 센다. [Gossard §I](https://arxiv.org/html/2604.11349v1) · [Rémond §II](https://link.aps.org/accepted/10.1103/PhysRevE.106.014207) |

## 9. 논문 심사 관점에서 보강할 주장

v70에서 주장할 수 있는 단위는 **측정된 표면 제품을 통째로 제외한 rebound prediction**이다. Shot 개수를 독립적인 surface 개수처럼 보고하면 transfer의 확실성을 과대평가한다. 특정 제품에서 모델이 틀렸다면 target에 알려지지 않은 restitution, sponge 구조, 불충분한 입사 범위, spin 추정 잡음 등도 같은 패턴을 만들 수 있다. 두 모델의 순위만으로 실패 원인이 patch라고 결론내릴 수 없다. [Gossard](https://arxiv.org/html/2604.11349v1) · [Rinaldi](https://www.mdpi.com/2076-3417/9/1/158)

독립적인 patch 이득의 주 비교는 동일 held-out impact의 **입사 상태만 사용했을 때의 loss에서 같은 상태와 실제로 측정한 patch를 사용했을 때의 loss를 뺀 값**이다. 두 방법에 같은 source, decoder capacity, 시간 cutoff를 준다. Outer fold에서 rubber 제품을 분리하고 같은 제품·비슷한 입사 조건 안에서 patch를 shuffle한 negative control을 두면, patch가 단지 rubber ID나 입사 속도를 알려준 효과를 점검할 수 있다. 같은 impact의 접촉 정보는 충돌 시작 이후에만 읽을 수 있으므로 pre-impact deployment를 주장하려면 별도 calibration shot에서 얻은 patch를 쓰고 감지·reset 비용을 포함해야 한다. [Rémond](https://link.aps.org/accepted/10.1103/PhysRevE.106.014207) · [Zhang](https://doi.org/10.1371/journal.pone.0333735) · [Gossard](https://arxiv.org/html/2604.11349v1)
