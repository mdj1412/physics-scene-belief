# v60 · Raw buffered pixels에서 deployment forecast까지의 사전 측정 계획

Primary 비교는 같은 여덟 impact와 2개 material당 pilot 한 개씩에서 marker-corrected loss-weighted reset과 (i) corrected loss-weighted sham, (ii) 항상 v54 gate의 latency 차이다. 네 번째 sensor에서 v54를 쓰는 부분까지 실행한다. 독립 baseline은 uncorrected loss-weighted reset, uncorrected loss-weighted sham이다. 여섯 번째 행은 corrected reset MAP다. 모두 같은 512개 물체의 실제 frozen joint GPU forecast까지 수행한다. Gate의 liability는 동일한 paid pilot이며 online gate는 불필요한 영상 판독을 생략한다.

24개 fresh catalog에서 6개 policy를 한 번씩 warmup하고 catalog마다 3개 measured block에서 모든 policy를 Williams crossover로 순환한다. 72회/정책으로 각 위치와 직전 정책을 균형화한다. 3가지 late-clip 빈도와 3가지 surface fault 관계의 각 조합을 8개 matched block에 배정한다. 같은 catalog 안에서는 세 delay와 세 fault 관계를 한 번씩 사용한다. Raw fixture를 준비하고 GPU를 synchronize한 뒤 marker 검출·frame 정렬, camera 및 전자 sensor 부호의 encoding, likelihood와 loss-weighted action, 공통 terminal과 GPU 예측까지 host wall clock을 잰다. 각 정책에서 실제 readout·route·terminal의 독립 저장 결과와 일치하는지 확인한다. Camera capture, 물리 접촉 및 장비 reset, raw archive decompression, model initialization은 이 시간에서 제외한다.

Primary GPU p90은 72개 block을 합쳐 비교한다. Catalog 24개를 paired bootstrap 16,000회 재추출하고 두 primary contrast에 대해 simultaneous 95% CI를 계산한다. Secondary contrast는 별도 표시한다. Cell당 8개 block의 p90은 표본이 적어 기술통계로 제시한다. 물리 시간은 별도 sensing ledger로 기록한다. Forecast가 끝난 GPU의 memory, utilization 및 다른 process 명단을 재검사하고 같은 node55/2744468과 guard를 사용한다.
