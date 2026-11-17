# v52 문헌 검토 · 표면 교체 후 어느 보정을 신뢰할 수 있는가

문헌 질문은 두 가지다. 소수의 교체 후 측정으로 이전 calibration의 유효성을 식별할 수 있는가? 변화에 대한 증거가 실제로 다음 impact의 예측을 바꿀 만큼 큰가? 아래 다섯 원 논문을 선정해 원문 방법과 결과를 확인했다. 이 범위 안에서만 근거를 합성했다. v52 결과와 후속 실험 제안은 별도로 표시했다.

## 1. 원 논문 비교

| 연구 | 질문과 방법 | 표본과 비교 단위 | 발견·결론 | v52로 옮길 때의 한계 |
|---|---|---|---|---|
| Adams & MacKay (2007) | 순차 자료에서 마지막 changepoint가 언제였는가? Hazard와 predictive likelihood로 run-length posterior를 갱신했다. | Well-log, Dow Jones 일별 수익률, 탄광 사고 간격의 세 실제 시계열. 평균·분산·사건율의 변화를 각각 모델링했다. | 가정한 generative model에서 마지막 변화점에 대한 exact online inference를 유도했다. | 변화 뒤 parameter가 이전 구간과 독립이라는 가정이 있다. v52는 교체 시점이 알려져 있고 유지율이 유지되거나 반전된다. [원문](https://arxiv.org/pdf/0710.3742) |
| Lipton, Wang & Smola (2018) | Target label이 없어도 label marginal을 추정하고 classifier를 보정할 수 있는가? Black Box Shift Estimation을 유도하고 재가중했다. | MNIST의 Dirichlet shift 실험에서 표본 수를 변화시켰다. Fig. 2의 오차 범위는 BBSE 20회, KMM 5회다. | Class-conditional 관측분포가 유지되고 confusion matrix가 invertible일 때 추정의 일관성을 보였다. | Contact law가 측정·결과의 관계를 바꾸면 이 가정이 깨진다. Unlabeled adaptation의 조건을 비교할 근거다. [원문](https://proceedings.mlr.press/v80/lipton18a/lipton18a.pdf) |
| Fishel & Loeb (2012) | 어떤 촉각 탐색 동작이 texture identity의 불확실성을 빨리 줄이는가? BioTac으로 얻은 신호에서 Bayesian exploration으로 다음 움직임을 고른다. | 117개 texture에서 선택한 3개 동작을 각 5회 측정했다. 별도로 16개 texture에서 얻은 validation 측정으로 117개 후보 중 정답을 찾는 8,000회 virtual exploration을 평가했다. | Median 5회 동작, 정답률 95.4%. 고정된 후보 사전에서 적은 측정으로 분류할 수 있었다. | 8,000회는 새 표면 8,000개가 아니다. 목표가 texture identity이고, 별도 impact prediction의 손실을 측정하지 않았다. [원문](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00004/full) |
| Ma & Rodriguez (2018) | 같은 planar push가 다양한 결과를 내는 이유는 무엇인가? 이미 수집한 trajectory와 anisotropic friction simulation을 비교했다. | Plywood, ABS, Delrin, PU의 4개 표면에서 각 2,000회 연속 push·repositioning을 수행한 자료를 분석했다. | Friction anisotropy, surface non-homogeneity, reset으로 인한 초기 방향 편향이 결과의 구조화된 변동 상당 부분을 설명할 수 있었다. | 반복된 push는 서로 영향을 줄 수 있다. 목표도 planar pose이므로 weak-to-impact sign retention과 같은 물리량으로 해석할 수 없다. [원문](https://arxiv.org/html/1802.10089) |
| Boonstra et al. (2024) | 접촉면 변형에서 slip까지 남은 safety margin을 읽을 수 있는가? 양쪽 tactile image의 marker deformation으로 CNN을 학습하고 force sensor로 검증했다. | Flat·curved object와 세 friction 조건의 96 trials. 4일 뒤 같은 두 물체·두 friction 조건에서 64 trials를 별도 측정했다. | 4개 validation 조건 평균 safety-margin MSE는 0.01821이었다. Low-margin 구간에서는 friction 조건에 따라 오차가 달랐다. | 같은 물체의 후속 날짜에 대한 실물 근거다. Unseen object family의 impact label이나 Brier loss는 측정하지 않았다. [원문](https://pure.tudelft.nl/ws/portalfiles/portal/220701333/Learning_to_estimate_incipient_slip_with_tactile_sensing_to_gently_grasp_objects.pdf) |

## 2–4. 주제별 합성, 지식 지도와 근거가 있는 공백

| 연구 흐름 | 확인된 관계 | v52에 남는 질문 |
|---|---|---|
| 변화 식별 | Hazard·관측모형 → changepoint posterior. 관측분포가 상태별로 구분된다는 조건에서 unlabeled data → shift 추정. | Surface 교체를 기록해도 drift 발생 자체는 모른다. 기존 보정과 새 보정이 같은 witness 분포를 만들면 두 설명이 식별되지 않는다. 이때 예측까지 동일한지 따로 확인해야 한다. [Adams & MacKay](https://arxiv.org/pdf/0710.3742) · [Lipton et al.](https://proceedings.mlr.press/v80/lipton18a/lipton18a.pdf) |
| 관측 설계 → 결과 예측 | 탐색 동작 → diagnostic tactile signal → texture classification. 별도 연구에서 tactile deformation → independently measured safety margin. | 두 단계를 연결한 weak probe → full impact transfer, 특히 transfer 부호가 바뀌었을 때의 보정 중단은 두 실험의 평가 대상 밖에 있다. [Fishel & Loeb](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00004/full) · [Boonstra et al.](https://doi.org/10.1109/ICRA57147.2024.10611517) |
| 수집 절차 → 재현성 | Reset이 다음 push의 초기 방향을 편향시킬 수 있다. Texture 측정에서는 sensor skin 마모를 점검하고 손상이 보이면 교체했다. | 실제 surface 교체에서 drift를 검출하더라도 material, sensor wear, reset pose 중 어느 변화인지 구별할 실험이 필요하다. [Ma & Rodriguez](https://arxiv.org/html/1802.10089) · [Fishel & Loeb](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00004/full) |

## 5–7. 상충하는 해석, 방법 차이, 증거 강도

- **조건이 갖춰졌을 때의 강한 근거.** Online posterior 계산과 label-shift 추정에는 각각 명시적인 모형 가정이 있다. v52에서 유지 가설과 반전 가설이 같은 분포를 만들도록 설계한 κ=0.5는 두 가설을 witness만으로 식별할 수 없는 대조군이다. 관측 수를 늘려도 이 대칭성은 유지된다. [Adams & MacKay](https://arxiv.org/pdf/0710.3742) · [Lipton et al.](https://proceedings.mlr.press/v80/lipton18a/lipton18a.pdf) · [v52 정책](POLICY_KO.md)
- **일부 실물 장치에서의 중간 강도 근거.** 빠른 texture identification과 반복 push의 높은 variability는 언뜻 어긋난다. 전자는 후보가 고정된 분류이고 후자는 연속 pose 의존성이 있는 trajectory다. 표본 단위도 독립 validation texture와 연속 push로 다르다. 두 결과만으로 friction의 일반적인 예측 가능성에 우열을 정할 수 없다. [Fishel & Loeb](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00004/full) · [Ma & Rodriguez](https://arxiv.org/html/1802.10089)
- **결론을 보류할 부분.** Tactile image가 slip margin과 연관된다는 결과는 reference와 impact 사이에서 변형 부호가 보존된다는 인과 근거가 아니다. 또한 95.4% classification accuracy, 0.01821 safety-margin MSE, v52의 binary Brier는 추정 대상·단위가 달라 수치 크기를 비교할 수 없다. 다섯 논문은 모두 v52의 세 보정 action을 같은 probe 예산에서 비교하지 않았다. [Boonstra et al.](https://doi.org/10.1109/ICRA57147.2024.10611517) · [Fishel & Loeb](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00004/full)

## 8. 후속 연구 질문 열 가지

표의 질문과 설계는 위 연구에서 확인된 한계를 토대로 제안했다. 모두 아직 확인하지 않은 가설이다.

| # | 질문·중요성 | 변수와 판별 실험 | 근거 |
|---:|---|---|---|
| 1 | κ가 0.5에서 얼마나 멀어져야 실용적인 drift detection이 가능한가? | κ×K를 세분화하고 고정 false-positive rate에서 sensitivity와 Brier gain을 함께 측정한다. | [v52](POLICY_KO.md) |
| 2 | 반전보다 작은 변화도 세 action으로 처리할 수 있는가? | Post retention을 pre retention과 독립적인 연속 offset으로 생성해 interpolation과 reversal 사이의 prediction regret를 비교한다. | [Adams & MacKay](https://arxiv.org/pdf/0710.3742) · [v52](POLICY_KO.md) |
| 3 | Drift prior의 오차가 probe의 가치를 바꾸는가? | 실제 교체율과 고정 hazard를 교차해 false alarm, keep 비율, Brier를 기록한다. | [Adams & MacKay](https://arxiv.org/pdf/0710.3742) |
| 4 | Reference에서는 drift가 보여도 impact 보정을 끄는 편이 나은가? | Reference-to-impact mismatch와 drift를 독립 조작하고 uncorrected·unsigned·soft를 같은 query에서 비교한다. | [Boonstra et al.](https://doi.org/10.1109/ICRA57147.2024.10611517) · [v52](POLICY_KO.md) |
| 5 | 두 번째 witness의 방향을 바꾸면 동일한 예산에서 더 정확한가? | 고정 방향·직교 방향·posterior 기반 방향을 무작위 배정하고 terminal Brier 및 reset 횟수를 비교한다. | [Fishel & Loeb](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00004/full) |
| 6 | 표면 교체와 센서 마모를 구별할 수 있는가? | Surface와 sensor skin을 독립 교체하는 2×2 crossover에서 anchor material을 함께 측정한다. | [Fishel & Loeb](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2012.00004/full) |
| 7 | Reset이 만든 orientation bias를 실제 drift로 오인하는가? | Reset pose를 무작위화한 군과 반복한 군에서 drift score와 실제 impact를 비교한다. | [Ma & Rodriguez](https://arxiv.org/html/1802.10089) |
| 8 | 같은 surface에서도 witness 위치와 query 위치 간 거리가 transfer를 약화하는가? | 동일 소재 내 위치와 방향을 교차한 position holdout에서 거리별 paired Brier를 추정한다. | [Ma & Rodriguez](https://arxiv.org/html/1802.10089) |
| 9 | Displacement 크기가 sign count 이상의 정보를 주는 sensor regime은 어디인가? | Saturation·noise floor·dead zone를 개별 조작하고 같은 세 action의 likelihood와 sign vote를 비교한다. | [Boonstra et al.](https://doi.org/10.1109/ICRA57147.2024.10611517) · [v52](POLICY_KO.md) |
| 10 | Session p90 개선이 새 실물 조건에서도 남는가? | 새 물체·표면·날짜를 통째로 hold out하고 fresh randomized impact 반복, session-level resampling, 실측 sensing·reset 시간을 사용한다. | [Boonstra et al.](https://doi.org/10.1109/ICRA57147.2024.10611517) · [Ma & Rodriguez](https://arxiv.org/html/1802.10089) |

## 9. 논문 심사 관점에서의 수정

**주장은 구체적인 조건에 묶는다.** “Drift detector가 표면 교체 뒤 slip을 예측한다”는 표현은 교체 시점, 숨은 drift event, 별도 impact outcome을 혼동한다. 본문에 이 세 변수와 정책이 접근할 수 있는 정보를 한 그림에 표시한다. 같은 K에서 matched sign vote와의 차이를 주 비교로 두고, AUROC와 Brier의 paired 차이를 별도 축에 둔다. K=4에서 likelihood와 vote의 AUROC는 각각 0.779와 0.774였고 likelihood의 고정 cutoff sensitivity는 33.20%였다. 이 결과만으로 expensive sensor likelihood의 실용적 우위를 주장할 수 없다. [v52 검출 결과](POLICY_KO.md)

**가장 가까운 대안 설명을 먼저 제거한다.** 세 action을 고르는 구조, 이전 관측의 재사용, displacement 크기 중 어느 요소가 이득을 주는지 확인하려면 동일 witness로 post-only vote, matched three-action vote, full likelihood를 나란히 비교해야 한다. Source risk를 적산한 catalog에서 predictor도 학습한 점은 action value를 낙관적으로 만들 수 있다. 다음 확인 실험은 모델 훈련·action value 추정·평가의 catalog를 각각 분리하고, reference와 실제 impact 사이의 mismatch를 target label을 보지 않고 사전 지정한다. [v52 정책과 source fit](POLICY_KO.md) · [원 논문의 분포 가정](https://proceedings.mlr.press/v80/lipton18a/lipton18a.pdf)

**외부 타당성의 우선순위.** 다섯 논문은 전체 문헌을 대표하는 systematic review가 아니다. 장기 sensor drift compensation 및 비모수적 sequential two-sample detection 분야는 novelty 판정 전에 별도로 대조해야 한다. 실물에서는 서로 다른 날에 독립 surface session을 먼저 확보하고 reset orientation을 기록한다. Witness는 같은 session에서 취하되 impact label은 별도 물체·측정으로 확보한다. 현재 simulator 안의 진단적 기여와 물리 장치로 일반화할 수 있는 범위는 이 설계로 구체적으로 검증할 수 있다. [Ma & Rodriguez](https://arxiv.org/html/1802.10089) · [Boonstra et al.](https://doi.org/10.1109/ICRA57147.2024.10611517)
