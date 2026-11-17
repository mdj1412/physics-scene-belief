# v56 · GPU 및 sensing 비용의 사전 측정 계획

빈 Slurm allocation을 직접 확인했다. 예측 생성에는 node55 2744468, 배포 GPU p90 측정에는 node56 2744471을 사용하며, 시작 직전 다른 compute PID와 GPU 사용률을 다시 확인하고 도중에도 guard를 사용한다. v55 수치 재사용 없이 v56 미사용 catalog의 물체에서 실행한다.

주 GPU 실험은 사전 지정한 두 인접 환경 w=0.35, 0.52를 균형화한다. 각 catalog의 네 environment에 session 한 개씩 배정하고 128개 물체씩 한 batch(512개)로 묶는다. w, pilot repeat, deployment prefix 인덱스, 방향 유지율 및 e 배정은 seed로 고정한다. K=8 pilot에서 각 짝은 해당 session에 맞춘 2.0 N reference와 impact 방향 측정으로 구성된다. Material당 pilot 8쌍을 세 policy 모두 지불하며 두 law의 배포 session이 그 결정을 공유한다. Ledger는 하나의 pilot 뒤 1회 session을 운영하는 경우와 여러 session에 나눠 부담하는 경우를 분리한다. Switch는 pilot에서 mismatch를 다시 세어 v54 gate와 fixed I 중 하나를 고른다. v54 gate를 쓰게 정한 session만 첫 세 weak/reference 쌍으로 네 번째 sensor를 결정한다. 두 단일-policy 대조는 pilot을 구입하지만 그 판독값을 판단에 사용하지 않는다. 세 조건 모두 동일한 frozen continuous joint decoder에서 같은 binary physics forecast를 만든다.

CPU에서 판독이 끝난 pilot과 세 prefix 쌍을 받아 route를 정하는 연산부터, 구매한 네 번째 readout으로 terminal을 갱신하고 512개 물체의 physics forecast를 완료하는 마지막 CUDA synchronization까지 시간에 포함한다. Sensor capture, reset과 offline label/geometry generation은 별도다. Policy당 1회 warmup 이후 catalog마다 세 정책의 6개 순열을 무작위 순서로 실행한다. 24개 catalog인 경우 정책당 144회 측정한다. 쌍을 보존한 16,000회 catalog bootstrap으로 switch 대 fixed I, switch 대 v54 gate의 두 차이에 simultaneous 95% interval을 제시한다. Gate-only CPU p90도 따로 기록한다. Timing 반복은 sample size로 합치지 않는다.

물리 예산은 K번의 reference+impact pilot과 교체 후 네 weak+sensor 쌍을 모두 세며, 실제 elapsed와 물리 비용은 구분한다. Impact를 선택하지 않아도 같은 시각에 다음 session을 시작하도록 할 때는 reference+reset과 impact+reset 중 더 긴 시간으로 네 번째 slot을 고정해 세 조건에 동일 wall-clock cap을 준다. T=0.52, 1.04, 2.08초의 cost sensitivity, 공통 pilot 및 적게 calibration하고 여러 deployment session에 재사용할 때의 변화를 제시한다. 실제 장비 T는 현재 이용할 수 없다.
