### 실제 probe response의 source 분포 이탈

Ridge scorer의 입력에 포함되는 probe velocity 두 성분의 prefix 평균을 원래 단위로 확인했다. v2는 scorer가 probe_tan_mean이라 부르는 두 번째 좌표다. Local surface normal로 회전시킨 새로운 변수는 만들지 않았다. 비교 기준은 같은 횟수의 Coulomb training 환경에서 계산한 5·95 percentile이다. 아래 백분율은 해당 구간을 벗어난 환경 비율이다.

| Law | Prefix | 환경 수 | v2 평균 [source 5%, 95%] | v2 이탈 | v1 이탈 | 같은 물체의 factual v1 이탈 | 실제 평균 Brier gain |
|---|---|---:|---|---:|---:|---:|---:|
| Coulomb (source) | 2→4 | 192 | -0.6899 [-0.8717, -0.4823] | 10.4% | 10.4% | 10.4% | -0.00030 |
| Coulomb (source) | 4→8 | 192 | -0.6810 [-0.8132, -0.5364] | 10.4% | 10.4% | 10.4% | +0.00071 |
| split_static | 2→4 | 128 | -0.7493 [-0.8717, -0.4823] | 10.9% | 13.3% | 9.4% | +0.00032 |
| split_static | 4→8 | 128 | -0.7484 [-0.8132, -0.5364] | 21.9% | 10.9% | 8.6% | +0.00064 |
| speed_weak | 2→4 | 128 | -0.8153 [-0.8717, -0.4823] | 26.6% | 7.8% | 1.6% | +0.00082 |
| speed_weak | 4→8 | 128 | -0.8109 [-0.8132, -0.5364] | 49.2% | 4.7% | 10.9% | -0.00006 |

Source의 각 192개 환경, 두 target law 각각의 128개 환경을 모두 사용했다. 4회 prefix는 중단한 곳에도 가상으로 공개한 사후 평가다. 실제 gain은 같은 128개 worlds에서 두 prefix의 common-decoder quota Brier를 빼 계산했다. 과거와 현재의 평균 차이에는 contact law와 새로 추출한 유한 표본의 차이가 함께 들어간다. 이탈은 training support 밖이라는 판정이 아니라 source 중앙 90%를 벗어났다는 뜻이다. 각 변수의 주변분포를 따로 보므로 이 표로 joint shift의 크기, 성능 하락의 원인, 배포용 alarm의 정확도를 추정할 수는 없다. [감사 수치](results/shift_audit.json)

Frozen linear scorer에서 source 평균 예측과 target 평균 예측의 차이는 계수 × 입력 평균 변화의 합으로 정확히 나뉜다. Probe velocity 평균·표준편차와 factual/probe covariance 등 다섯 sensor features의 합, 나머지 posterior features의 합을 원래 Brier gain 단위로 표시했다. 서로 상관된 feature가 함께 달라지므로 이 분해는 계수에 따른 산술적 기여다.

| Law | Prefix | 평균 예측 변화 | Sensor features 기여 | Posterior features 기여 |
|---|---|---:|---:|---:|
| split_static | 2→4 | +0.00032 | +0.00007 | +0.00025 |
| split_static | 4→8 | -0.00026 | -0.00027 | +0.00001 |
| speed_weak | 2→4 | +0.00028 | +0.00006 | +0.00022 |
| speed_weak | 4→8 | -0.00044 | -0.00044 | +0.00000 |
