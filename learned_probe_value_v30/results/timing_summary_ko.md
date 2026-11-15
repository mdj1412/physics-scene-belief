## Common decoder의 GPU p90과 calibration 비용

| 측정 구간 | 정책당 측정 횟수 | Learned (ms) | Fixed 4 (ms) | Uncertainty-only (ms) | Learned / Fixed 4 [95%] | Learned / Uncertainty [95%] |
|---|---:|---:|---:|---:|---:|---:|
| Streaming, 1 world | 384 | 3.563 | 3.561 | 3.557 | 1.001 [0.998, 1.003] | 1.002 [0.999, 1.003] |
| Batch quota, 128 worlds 전체 | 48 | 428.363 | 427.925 | 428.396 | 1.001 [0.998, 1.003] | 1.000 [0.997, 1.002] |
| Streaming, setup을 128건에 균등 배분 | 384 | 4.220 | 4.019 | 4.177 | 1.050 [1.039, 1.173] | 1.010 [0.997, 1.135] |
| Batch quota, setup 전액 포함 | 48 | 512.072 | 488.094 | 509.020 | 1.049 [1.041, 1.187] | 1.006 [0.998, 1.143] |

각 target law의 첫 8개 independent blocks에서 환경 한 곳씩 사전 추출했다. 각 환경에서 batch당 128개 world를 3회, streaming에서는 12개 서로 다른 world를 2회씩 측정했다. 정책 실행 순서를 섞고 개별 실행 전후 CUDA synchronization을 수행했다. 측정한 것은 CPU·GPU 전송과 predictor 호출을 포함한 host-to-host wall time이다. Runner에서 각 측정의 관측 선택과 첫 noise draw의 stick probability를 동결된 선택 및 common decoder의 예측과 대조했다(확률의 절대오차 허용치 1e−8). 비율의 pointwise 95% interval은 law별로 block을 따로 10,000회 재표집했다. 같은 block에서 나온 세 정책의 측정과 모든 반복을 함께 유지했다.

환경별 setup은 factual·paid prefix posterior fit, CPU scoring과 정렬, selector bank 생성을 포함한다. 배포 전 한 번만 실행하는 detector 초기화 및 source model 학습과 별도로 산정했다. 동일한 common decoder 생성, 실물 sensing, 충돌·reset 대기 시간은 제외했다. Streaming 상각값은 비용 배분값이고 한 건의 응답시간은 첫 행에서 읽는다.

Source 데이터 로딩, nested cross-validation, 최종 ridge fit을 포함한 학습은 2.13초였다. 공유 detector 초기화는 159.19 ms였다. Source target은 이전 v29 실행 기록을 재사용했으므로 과거 simulator 계산은 학습 시간에 포함되지 않았다.
