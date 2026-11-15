# v33: 전체 stopping policy의 사전 평가 계획
최초 target Brier를 계산하기 전에 작성했다. 대상은 새 synthetic object profile 두 종류 × surface response 두 종류 × friction orientation 두 개다. 같은 object/surface/seed block의 두 orientation은 동일 latent draws와 noise를 공유하는 paired intervention으로 평가한다. 네 object/surface 조합은 고정한 네 조건이며 실제 표면 모집단의 무작위 표본이 아니다. 각 조합에서 독립 block 8개, block당 환경 8개를 한 단위로 다룬다. Block을 넘나드는 tuning을 하지 않는다.
 
V32의 Coulomb source에서 이미 같은 시간으로 학습한 세 쌍의 15-feature ridge를 고정한다. 새 target에서 다시 학습하지 않는다. Source model을 재사용하면 training set이 달라져 생긴 효과와 물체·표면 변화가 섞이지 않는다. Brier의 기본 단위는 개별 seed나 noise draw가 아닌 cluster다. Primary estimand는 네 조합과 두 orientation에 동일 가중치를 준, 세 고정 training pairs의 평균 full-policy Brier다. 다섯 차이(2×2−1×1, 1×1−fixed 4, 2×2−fixed 4, 1×1−uncertainty, 2×2−uncertainty)의 simultaneous 95% interval을 같은 cluster bootstrap 표본의 최대 절대 표준화 편차로 구한다. 각 조합 안에서 cluster 8개를 복원 추출하며, 두 direction과 세 models에 같은 indices를 적용한다. Training seed pair도 복원추출하는 crossed interval은 세 쌍에 한정된 sensitivity로 제시한다. Practical margin은 v32와 동일한 absolute Brier 0.003이다.
 
V32의 독립 재검증: 기본 + orientation에서 2×2−1×1 차이가 양수인지, orientation을 반전시키면 그 차이가 변하는지 확인한다. 새로운 target의 pooled plus, minus, paired plus-minus interaction 세 값에 하나의 simultaneous family를 사용한다. 각 family나 seed 결과를 따로 선택해 주 결과를 바꾸지 않는다. Direction 자체는 정책 입력이 아니다.
 
GPU timing은 첫 training pair에서 4가지 정책을 같은 하드웨어에서 직접 실행한다. 시작 전 warm-up을 제외하고 trial 순서를 섞어 각 condition에서 2 blocks ×3 repeats를 수행한다. 1,024개 물체를 처리하는 full-call p90과 deployment-only p90을 구분한다. 모든 방법의 timing을 같은 paired block에서 재표집하며 전체 8개 독립 timed cluster뿐이므로 interval은 descriptive다. Seed 2·3의 latency를 재측정한 것으로 세지 않는다. 실제 센서 대기 시간은 측정하지 않는다.
 
Gain 진단은 별도로 정한 prefix plan을 사용한다. Full-policy evaluation에서 우연히 gain이 큰 prefix를 고른 뒤 이를 독립 검증으로 부르지 않는다. 모든 정책의 paid allocations, 결정 시점에 접근한 feature, 13/128 deployment quota, saved probabilities와 Brier, model 및 결과 hash를 검산한다.
