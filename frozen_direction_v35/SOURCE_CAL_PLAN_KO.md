# v35 score-only 보조 비교의 사전 고정

v35 target archives의 수치에 접근하기 전에 v34 ridged_coarse의 fold 0 학습 자료만으로 source-only offset 및 temperature를 고정했다. [source hashes와 계수](source_cal35_frozen.json)

동일한 네 번의 paid probes 및 같은 observation을 사용하고 다음 세 가지 paired Brier 차이를 시험한다.

1. Frozen signed correction − constant offset.
2. Frozen signed correction − temperature.
3. Temperature 뒤에 같은 v34 estimated direction을 더한 correction − temperature. 두 확률은 sigmoid(logit(p)/T + βz)와 sigmoid(logit(p)/T)다. T=1.7905158513994117, β=0.8289760825845313, z는 v34 fold 0 schedule별 k=4 reader 출력 2P(plus|prefix)−1이다. 계수·연산 순서·sensor 선택을 고정한다.

Repeated sensor draws를 유지하며 family와 시작 방향을 층화한 paired block bootstrap 12,000회로 pooled 및 cell별 차이를 계산한다. 4개 directional family와 pooled × 두 schedule × 세 contrast에 대한 30개 simultaneous 95% interval, 두 isotropic control과 pooled에 대해서는 별도 18개 simultaneous interval을 작성한다. 이 추가 비교는 secondary이며 원 primary 판정을 바꾸지 않는다. [v35 primary 계획](AUDIT_PLAN_KO.md)
