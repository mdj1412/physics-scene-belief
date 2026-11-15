## 속도만 바꾼 실험

환경마다 서로 다른 물체 네 개를 한 번씩 접촉한다. 첫 세 probe는 높이 0, tangential speed ±1.10에서 교대로 수행하고 첫 부호를 block 안에서 균형화했다. 네 번째는 동일한 물체에서 **높이 23 mm, 방향 +**를 고정하고 tangential speed를 +0.50, +0.75, +1.10으로 바꿨다. 예측할 별도 충돌은 **25 mm, +0.425**로 고정했다. 여기서 속도 수치는 기준 좌표의 incoming x velocity를 1로 둔 simulator의 단위다. 접촉점의 local normal component는 contour에 따라 달라진다. 실물 장비의 m/s로 환산하지 않았다.

세 속도에서 네 번째 물체의 latent geometry, 처음 세 접촉, factual observations와 두 성분의 Gaussian velocity noise(각 표준편차 0.027)를 공유한다. Gate가 관찰하는 값은 noisy outgoing velocity다. Slip 상태와 깨끗한 물리 응답은 평가할 때만 열었다. 새로운 deployment 물체는 환경당 128개다. 각 속도의 acquisition이 그중 13개에 probe, 나머지에 query view를 배정한다. 높이와 속도가 다른 관측을 잘못된 action으로 해석하지 않도록 해당 속도의 Coulomb likelihood를 사용한다. 같은 속도의 두 방법은 동일한 선택과 같은 physics probability p를 받는다.

- **고정 32.5% shrinkage**는 첫 세 관측에서 추정한 signed friction cue z의 32.5%를 항상 적용한다. 최종 예측은 sigmoid(logit(p)+0.325βz)이다. 0.325는 v36 source에서 가정한 네 near-field multiplier 1, 0.3, 0, 0의 균등 평균이다.
- **Hard gate**는 마지막 측정으로 각 후보 surface model의 posterior를 갱신한다. Source에서 기록한 expected Brier gain이 양수면 sigmoid(logit(p)+βz)를, 아니면 p를 반환한다. 두 방법에 같은 β=0.8289760825845313과 같은 z를 사용한다. +0.50에서 v36 모델을 그대로 불러오고, 다른 속도에서는 v36과 동일한 source draws로 네 번째 측정 likelihood만 다시 계산했다. Gate threshold, prior와 gain lookup은 속도별로 다시 맞추지 않았다. [고정된 모델의 기록](freeze37.json) · [Inference](gate37.py)
- **Primary target**은 strong, weak, local null이 동일 비율로 존재하는 조건이다. 세 regime은 높이 0에서 같은 방향성을 보이지만, query 부근에서 원래 방향 효과의 100%, 30%, 0%를 갖는다. 별도 global-null 표면은 모든 높이에서 방향성이 없다. 이 조건을 primary 평균에 넣지 않고 자체 결과와 네 regime을 균등하게 혼합한 결과를 공개한다.
- **새 표본**은 v36과 동일한 두 contour와 두 target friction law에서 독립된 seed로 생성했다. 네 cell에 8개 독립 block씩, 각 block에 8개 환경과 환경당 128개 deployment 물체가 있어 unique latent deployment objects는 32,768개다. 각 물체의 32개 observation noise draws, orientation mirror, regime와 속도는 같은 block 안에 둬 paired interval을 계산했다. 각 cell 안에서 첫 probe 부호별 독립 block은 네 개씩이다. 구간은 이 네 고정 cell 내부의 변동에 대한 bootstrap 근사다. [설계와 seed](design37.json) · [결과 전 분석 계획](ANALYSIS_PLAN_KO.md)

Source에서는 방향성이 없는 query의 gain을 근사할 때 v34의 기존 directional deployment observation을 유지하고 outcome만 isotropic으로 바꾼 부분이 있다. v37 global-null target은 calibration과 deployment 모두 처음부터 isotropic으로 생성했다. 두 방법의 공통 base physics decoder는 population proxy coupling을 안다는 기존 가정을 유지한다. 이 결과를 실제 새 표면 또는 다른 sensor noise 범위로 옮길 때에는 이 두 조건을 별도로 검증해야 한다.
