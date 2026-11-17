# v70 추가 검증 계획

각 fold는 같은 rubber model의 모든 blade와 sponge thickness를 통째로 제외한다. Target-free RBF baseline은 source의 공식 train만 사용한다. 입력 순서는 (vx, vz, 0.02 wy), 출력은 (vx', vz', 0.02 wy')다. 64개 고정 lattice center, bandwidth {0.55, 1.05}, ridge penalty {0.3, 3.0}를 사용한다. 각 source rubber를 한 번씩 inner holdout한 평균 3성분 squared loss(모두 m/s 단위)로 네 후보 중 하나를 고른다. 표준화와 parameter fitting은 매 inner split에서 새로 한다.

- Primary comparison은 unseen rubber의 official test rows 전부이며 지표는 XZ velocity vector의 mean Euclidean error, Y-axis spin의 MAE다. 여섯 rubber를 동등 가중한 macro와 shot 수로 가중한 micro를 모두 낸다.
- Incoming-state support는 fold의 source train에서 standardization한 뒤, 자기 rubber 전체를 뺀 source nearest-neighbor distance의 95th percentile을 threshold로 쓴다. Target row는 prediction 전에 incoming state만으로 분류한다. 가깝다는 것만으로 모든 surface response가 같다고 볼 수 없다.
- 별도 진단으로 동일 RBF model을 **target rubber의 official train**으로 학습해 official test를 예측한다. Hyperparameter는 source-only 선택값을 그대로 쓴다. 같은 수의 source rows를 deterministic random subsampling한 matched-count baseline도 학습한다. 이 진단은 target label의 정보 가치를 간접 점검하며 fixed rubber에 대한 측정된 patch의 효과를 추정하지 않는다. Held-out rubber score에 섞지 않는다.
- 차이는 모든 방법이 공통으로 예측한 동일 impact에서 계산한다. 6개 surface에 대한 방향과 한 개씩 뺐을 때 macro의 범위, descriptive exact sign test를 확인한다. 두 metric의 joint superiority나 새 surface 모집단 신뢰성은 별도로 판단한다.
