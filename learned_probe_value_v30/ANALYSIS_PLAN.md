# v30 사전 분석 계획

2026-11-14. Unseen-law test 결과를 열기 전 고정했다. 분석의 모든 부호는 Brier가 작아질수록 좋다는 기준을 따른다.

- 한 environment는 동일한 population 내의 calibration cohort와 별개의 deployment batch다. 같은 8개 환경에 총 32 calibration probes를 나눈 block이 관측 단위다. 128개 worlds와 32회 observation noise 반복은 각각의 독립 block처럼 추론하지 않는다.
- Primary endpoint는 환경당 128개 중 13개를 probe하는 batch quota에서 frozen source-law common decoder로 계산한 held-out Brier다. 두 target contact law에 동일 가중치를 주고 learned − fixed 4, learned − uncertainty-only 두 대비를 simultaneous 95% interval로 평가한다. 실질적 개선 기준은 v29와 같은 절대 0.003이다. 두 상한 모두 0 미만일 때에만 두 기준선보다 낫다고 판단한다. 양의 차이는 learned의 손해다. 두 특정 물리 법칙에 조건부인 interval이다.
- Independent numerical bank/seed block을 단위로 paired bootstrap 10,000회. Law가 같은 block seed를 공유하면 두 law의 해당 block을 함께 재표집하고, disjoint seeds이면 law마다 따로 resample한다. Bootstrap 표준편차로 표준화한 두 차이의 최대 절댓값에서 simultaneous interval을 얻는다. Law별, law-aware decoder 및 streaming 결과는 별도의 descriptive pointwise interval을 보고한다. 다른 학습 결과를 보고 가장 유리한 endpoint를 선택하지 않는다.
- Uncertainty-only는 같은 2/4/8회 histogram을 사용한다. Fixed 4도 block별 total calibration count와 quota test contacts가 같아야 한다. Calibration 대상의 hidden cohort ID, law ID, test labels, 무료 future probes가 score에 유입되는지 구현을 검토한다. Uniform shuffled histogram 기대값은 관측된 각 환경 prefix별 potential loss와 배정 확률(1/2,1/4,1/4)에서 계산한다.
- 직접 평가할 scorer diagnostics는 예측한 절대 Brier gain의 평균·실제 평균·MAE, 0 예측 및 학습 평균과의 비교(기록 시), block 내 pairwise gain concordance(동률 0.5), 실제 선택이 바뀐 world의 비율, 두 번째 stage 생존 환경이다. 동일 test population의 signed paired potential loss 차이를 평가용으로만 연다. 같은 slope가 모든 source/test에 적용된다는 보장은 두지 않는다.
- GPU p90은 동일 common decoder로 warm streaming 1건 및 batch 128개 전체를 구분한다. Hash/예측 일치, sample count, device 명, setup 포함/제외, 비교 정책별 센서 횟수, 표본 추출 단위를 보고한다. Source 데이터를 만드는 과거 simulator 비용, 모델 fit, 초기화·fit·acquisition 비용과 센서/물체 reset 시간을 구분한다.
