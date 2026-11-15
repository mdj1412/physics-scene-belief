# 독립 source-law reference · 사전 계획

V32의 model lock을 열기 전에 정했다. Coulomb의 과거 v31 finite-pool panel 6 blocks에서 같은 8개 population 조건을 쓴다. 환경당 1,024개 factual rows와 이미 지불한 2개 또는 4개의 probe ID 및 측정값으로 v30의 15개 features를 재구성한다. Detector seed 3003913, Sobol bank 8192, CPU float64로 고정한다. 당시 저장한 v30 score 및 posterior friction SD와의 절대차를 1e−8 이하로 확인한다. Features를 별도 hash로 봉인한 뒤 v32 모델의 예측을 계산한다. Label은 그 이후 평가에만 읽는다.

각 stage에서 독립적인 v31 calibration futures 4개와 deployment batches 4개의 crossed gain 평균을 목표로 삼는다. Source에서 한 번 동결한 각 v32 arm의 intercept를 그 arm의 intercept-only comparator로 둔다. 환경 48개 전체에 같은 가중치를 주어 MSE 및 Pearson correlation을 계산한다. 두 정책의 MSE 차이(repeated−broad), 각 정책 MSE에서 그 정책의 intercept-only MSE를 뺀 차이, 상관계수 차이를 paired block bootstrap 10,000회(seed 32093151) pointwise 95% interval로 기술한다. 두 stage와 모든 예측을 같은 6개 block index로 재표집한다. 음의 MSE 차이는 앞 항이 유리하다. 상수 예측의 Pearson correlation은 정의되지 않는다. 이 자료는 이전 연구에서 관찰했지만 v32 fit·예산 설계 비교에는 입력하지 않으며, 추가 model selection에 쓰지 않는다. 작은 6-block 표본이고 여러 탐색적 대비가 있으므로 별도 가설검정이나 승자 선언을 하지 않는다.

4×4 평균도 유한 표본의 label이다. MSE의 절대 크기에는 label noise가 남고 Pearson correlation은 약화될 수 있다. 동일한 독립 목표에 대해 두 고정 predictor 간 paired MSE 차이를 구하면 label noise의 제곱 항은 상쇄되지만 유한 표본의 cross term은 남는다. 모든 환경의 4회 prefix에 대한 결과와 실제 선별을 통과한 환경에서의 4→8 판단을 구별한다.
