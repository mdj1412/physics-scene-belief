# v32 사전 통계 계획

## 평가 단위와 가설

두 학습 정책은 동일한 15개 paid-prefix feature, 같은 두 stage, training feature의 mean과 population SD를 사용한다. 각 stage의 signed Brier gain을 standardized ridge로 예측하며 penalty는 v30에서 두 stage 모두 선택된 λ=3으로 고정한다. Broad는 1×1, repeated는 2×2 label의 평균을 사용한다. 각 policy는 별도 source environments에서 측정한 540초 예산으로 한 번 학습한다. 두 arm의 synchronized label-generation wall time과 stage별 GPU fit을 합한 realized cost는 작은 값으로 나눈 상대 차이 2% 이내인지 판정한다. 양쪽의 source load/hash time을 포함한 값도 공개한다. Training budget 초과 여부와 두 arm 간 차이는 별개로 기재한다. 모든 평가 결과는 이번에 고정한 두 training realization에 조건을 둔다.

동일한 세 target laws(split_static, speed_weak, directional)에 같은 비중을 준다. Law당 16 independent blocks, block당 8 environments, environment당 common world 128개와 sensor noise 32개를 공유한다. Deployment의 observation quota는 각 environment에서 probe 13개다. Calibration budget은 block당 32개다. 8 environments와 모든 policy를 통째로 묶어, 각 law 안에서 block 16개를 복원추출한다. Bootstrap 10,000회, seed 32071023을 고정한다.

한 block의 Brier는 각 method가 고른 observation의 확률과 binary label 간 squared error를 noise, world, environment 순서로 평균한 값이다. ΔBrier는 앞 method에서 뒤 method를 뺀 값으로, 음수일 때 앞 method가 우수하다. Primary family는 repeated−broad, broad−fixed4, repeated−fixed4, broad−uncertainty, repeated−uncertainty 다섯 개다. 추정치 θ, bootstrap SD s에 대하여 resample별 다섯 |(θ*−θ)/s|의 최댓값 중 95% quantile c를 구한다. θ±cs를 simultaneous 95% interval로 사용한다. 0을 포함하면 우열은 미결정이다. 해당 interval 전체가 ±0.003 안에 들 때 이 설정에서 practical difference가 작다고 평가한다. 실용적 우월성은 upper bound ≤−0.003이어야 한다. 사전 설정한 각 law와 이전에 사용한 두 law의 평균은 pointwise interval과 함께 secondary 결과로 표시한다. 이 block bootstrap은 새로운 training realization 또는 임의의 target law까지의 불확실성을 반영하지 않는다.

## 시간과 검사

각 law의 첫 6 blocks에서 randomized interleaved repeated calls 3회씩 측정한다. Full-call p90은 세 law의 54개 측정값에서 구한다. Timed block 단위로 같은 policy와 repeats를 함께 복원추출하여 full-call p90 ratio의 pointwise 95% percentile interval을 계산한다. Warm deployment-only p90도 같은 방식으로 별도 보고한다. Calibration sensing time, detector 준비 등 제외된 작업은 evaluation design에 따른다. Latency는 secondary endpoint이며 isolated kernel time과 sensor-to-result wall time을 구분한다.

독립적인 NumPy 계산으로 두 GPU ridge fit, allocation ranking, quota, 선택된 predicted probability에서 계산한 Brier, stage별 Brier와 실제 선택된 counts의 일치를 확인한다. Training block·bank·prefix·deployment seed와 held-out block seed의 중복을 검사한다. 동일 compute budget에서는 다양한 source prefixes를 보는 효과와 한 prefix를 반복하는 효과가 함께 변화한다. Training seed 반복 없이 두 학습 절차 전반의 우월성을 주장할 수 없다.

## Source-fit sensitivity

각 arm의 source block을 생성 순서의 index modulo 2 또는 3으로 분할한다. 매 분할에서 양쪽 arm에 같은 residue를 적용해 disjoint subset 쌍을 만들고, 원래 λ·feature·stage를 유지해 coefficients를 다시 계산한다. 각 subset의 실제 label time과 sample 수를 함께 기재한다. Modulo-2와 modulo-3의 결과는 서로 중첩된다. Training data를 줄인 subset sensitivity는 full-budget 학습을 독립적으로 반복한 실험으로 간주할 수 없다. Full fit을 포함한 모든 후속 policy replay는 저장된 stage별 held-out risk를 이용한다. 모든 subgroup 결과는 secondary이며 우수한 subset을 선택해 주 결과를 교체하지 않는다.
