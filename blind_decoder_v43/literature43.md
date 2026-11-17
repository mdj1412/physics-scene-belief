## 선행연구에 비추어 본 기여

아래 표는 v43의 두 가설을 반박할 수 있는 원저를 선정한 비교다. 각 논문에서 사용한 표본과 평가 단위가 달라 성능 수치를 합산하지 않았다. 이전 설계와의 직접 대조에는 [v42 연구 보고서](../height_offset_v42/REPORT_KO.md)를 사용했다.

| 질문을 던진 연구 | 방법·표본 | 저자들이 확인한 결과 | v43과의 차이 및 한계 |
|---|---|---|---|
| Guo et al. · 복잡한 모델의 confidence 오차를 단순하게 줄일 수 있는가? | ImageNet, CIFAR, Reuters 등의 분류 모델에서 validation NLL로 scalar temperature를 학습하고 교정법을 비교했다. | 여러 자료에서 temperature가 calibration을 개선했다. Reuters에는 같은 개선이 성립하지 않았다. | Source에서 학습하는 양의 스칼라 하나가 강한 대안이다. 접촉 시점이 바뀌는 binary collision에 그대로 전이하는지는 이 논문이 다루지 않았다. [원문 §4](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf) |
| Ovadia et al. · shift가 커져도 예측 불확실성을 신뢰할 수 있는가? | 이미지 corruption의 종류·강도, text, 광고 클릭 자료 등에서 여러 uncertainty 방법을 대조했다. | IID validation의 temperature 보정이 shift의 강도가 커질 때 calibration을 보장하지 않았다. | Guo et al.과의 차이는 validation과 test의 관계 및 shift의 강도다. 두 결과 모두 새 마찰 법칙에서의 승자나 Brier 효과 크기를 정해 주지 않는다. [원문](https://arxiv.org/abs/1906.02530) |
| Benton et al. · 어느 범위의 변환까지 invariance를 학습해야 하는가? | Augerino가 augmentation 분포와 예측기를 함께 학습했다. Mario–Iggy 회전 분류와 얼굴 각도 예측 등을 사용했다. | 목표 label에 따라 유용한 변환 범위가 달라졌다. | Invariance가 task에 의존한다는 원리는 선례가 있다. 같은 물체의 실제 접촉 위치와 운동 방정식으로 label 전환을 만드는 검증은 다르다. [원문](https://arxiv.org/abs/2010.11882) |
| Zhao et al. · invariant representation이 domain adaptation에 충분한가? | 이론적 반례와 bound, MNIST·USPS·SVHN의 transfer 실험을 사용했다. | Input distribution을 맞춰도 조건과 label 분포가 달라지면 target 성능이 악화될 수 있다. | 이론과 digit transfer는 point impact의 구체적인 sign reversal을 예측하지 않는다. [원문](https://proceedings.mlr.press/v97/zhao19a.html) |
| Meißenhelter et al. · 위치 오차가 접촉 이후 상태로 어떻게 전파되는가? | 서로 다른 3D mesh에서 위치와 contact normal의 불확실성을 근사하고 simulation sampling을 참조 분포로 비교했다. | 해당 예제에서 선형화보다 분포 근사가 나았고 대량 sampling보다 빠르게 계산했다. | 위치 불확실성의 물리 전파 자체는 알려져 있다. Decoder가 변형 축을 버릴 때 downstream Brier가 어떻게 변하는지는 그 논문의 지표와 다르다. [원문](https://cgvr.cs.uni-bremen.de/papers/icra40_24/ICRA40_2024_UncertainPhysics.pdf) |
| Fazeli et al. · impact solver가 실물 충돌을 어디까지 설명하는가? | 실험 장치의 2,000회 planar drop 중 품질 검사를 통과한 1,718회로 여섯 rigid-body contact model을 비교했다. | 일부 실측 impulse를 모형이 표현하기 어렵고, 최적으로 맞춘 parameter의 물리 해석에 한계가 있었다. | 한 solver에서 family를 바꿔도 이 외부 타당성 문제는 남는다. [원문](https://arxiv.org/pdf/1710.04979) |

**근거 지도.** Sensor와 contour가 만드는 확률 오차 → source에서 fit한 confidence 보정 → 미래 물체의 Brier라는 경로와, 접촉 높이 → local tangent → 마찰 한계 → 실제 stick event라는 경로가 만난다. 앞 경로에는 분류 calibration에 대한 실증 근거가 있고, 뒤 경로에는 접촉 위치 불확실성과 실물 solver 오차에 대한 독립 근거가 있다. 실물에서 두 경로의 상호작용이 재현되는지는 별도 가설이다. 같은 형상에서 label이 달라지는 simulator intervention은 모형 안의 원인을 검증한다. 서로 다른 독립 family에서의 평균 상관만으로 같은 인과 해석을 할 수 없다. [Guo et al.](https://proceedings.mlr.press/v70/guo17a.html) · [Meißenhelter et al.](https://cgvr.cs.uni-bremen.de/papers/icra40_24/ICRA40_2024_UncertainPhysics.pdf) · [Fazeli et al.](https://arxiv.org/pdf/1710.04979)

### 남은 질문과 검증 방법

| 연구 질문 | 공백·중요성 | 조작할 변수와 검증 |
|---|---|---|
| 1. Confidence를 조정한 다음에도 환경별 signed correction이 남는가? | 단일 T로 설명할 수 있는 개선과 마찰 cue의 독립 기여가 겹칠 수 있다. [Guo et al.](https://proceedings.mlr.press/v70/guo17a.html) | 별도 source에서 T를 먼저 fit하고 residual correction을 추가해 새 family Brier를 비교한다. |
| 2. 언제 source temperature 자체가 부적절한가? | Shift가 커질 때 calibration이 악화될 수 있다. [Ovadia et al.](https://arxiv.org/abs/1906.02530) | Label prevalence·위치오차 분포를 각각 변경하고 Brier와 reliability curve를 얻는다. |
| 3. 어떤 공간 해상도부터 기하 정보를 보존할 수 있는가? | 센서에서 보이지 않는 normal과 decoder가 놓친 normal은 대응이 다르다. [Meißenhelter et al.](https://cgvr.cs.uni-bremen.de/papers/icra40_24/ICRA40_2024_UncertainPhysics.pdf) | 7점·고밀도 profile × quadratic·local spline을 교차하고 slope error와 event 확률을 채점한다. |
| 4. 어떤 변형 축을 얼마나 약화해야 하는가? | Hard invariance가 task에 따라 불필요한 정보를 제거할 수 있다. [Benton et al.](https://arxiv.org/abs/2010.11882) | 같은 prior·decoder에서 hard, finite-variance, 원래 likelihood를 비교한다. |
| 5. 접촉 높이를 직접 재면 반전이 얼마나 줄어드는가? | 위치 uncertainty가 contact normal에 전파된다. [Meißenhelter et al.](https://cgvr.cs.uni-bremen.de/papers/icra40_24/ICRA40_2024_UncertainPhysics.pdf) | 절대오차를 아는 위치 측정과 geometry 해상도를 독립 조작한다. |
| 6. 새로운 법칙인가 새 표면인가, 어느 변화에서 정책이 깨지는가? | 두 요인을 동시에 홀드아웃하면 실제 배포 시험은 되지만 개별 요인의 기여는 분리되지 않는다. [Zhao et al.](https://proceedings.mlr.press/v97/zhao19a.html) | 동일 object family에서 single-axis 및 joint holdout을 비교한다. |
| 7. 다른 solver와 finite contact patch에서도 같은 부호인가? | 일부 실측 impulse를 point impact model이 설명하지 못했다. [Fazeli et al.](https://arxiv.org/pdf/1710.04979) | Tip radius·compliance·속도 threshold를 고정한 별도 solver에서 signed effect를 사전 예측한다. |
| 8. 몇 개 target label을 사면 고정보다 나은가? | V42에서 별도 물체의 target outcome을 준 경우와 네 번의 calibration만 준 경우 사이에 큰 차이가 있었다. [이전 검산](../height_offset_v42/SELECTOR_AUDIT_KO.md) | 환경별 0·4·16·64개 donor를 무작위 배정해 다른 물체의 비용 대비 Brier를 평가한다. |
| 9. 네 번째 물리 접촉을 구매하는 것이 유리한가? | 같은 네 번을 지불한 대조는 구매 여부의 효과를 식별하지 못한다. [기존 비용 분석](../adaptive_stopping_v29/REPORT_KO.md) | 같은 prefix에서 stop·buy를 무작위 배정해 reset·대기 시간을 포함한 break-even을 낸다. |
| 10. 실물에서 surface session이 달라져도 geometry 신호가 남는가? | 실측 planar pushing에는 위치·힘·형상 변이가 기록됐지만 v43의 단발 impact와 시간척도가 다르다. [Yu et al.](https://arxiv.org/pdf/1604.04038) | 공개 trajectory에서는 tangential velocity를 먼저 검증하고, 실제 impact에서는 다른 날짜·물체·표면 session별 paired probability를 평가한다. |
