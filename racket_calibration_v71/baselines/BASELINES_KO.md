# v71 · Calibration을 허용한 두 baseline과 prior 비교

v70에서 고정했던 model family에 새 rubber의 paid impact를 같은 수로 제공했다. 여섯 제품, 제품당 256개 nested draw에서 calibration의 순서까지 core 실험과 동일했다. 두 draw artifact의 모든 NumPy array가 byte-for-byte 일치했다. K=0에서 두 model family의 제품별 test 오차는 v70과 10⁻⁸ 이내로 일치했다.

각 칸은 제품별로 256회 평균을 낸 뒤 여섯 제품에 같은 비중을 준 XZ velocity error (m/s) / spin MAE (rad/s)다.

| Calibration impacts | Inclined rigid | Single affine · ridge | Single affine · empirical Bayes | Gate + two experts · empirical Bayes |
|---:|---:|---:|---:|---:|
| 0 | 1.087 / 63.57 | 1.030 / 62.56 | 1.030 / 62.56 | 1.065 / 67.16 |
| 1 | 1.057 / 64.14 | 0.897 / 55.11 | **0.451 / 23.20** | 0.548 / 32.34 |
| 2 | 1.021 / 63.75 | 0.771 / 46.68 | **0.401 / 20.51** | 0.465 / 25.23 |
| 4 | 0.988 / 63.44 | 0.556 / 30.71 | **0.373 / 19.53** | 0.399 / 20.40 |
| 8 | 0.934 / 61.82 | 0.406 / 20.89 | **0.348** / 18.16 | 0.359 / **17.94** |
| 16 | 0.880 / 60.18 | 0.347 / 17.82 | 0.329 / 17.12 | **0.327 / 16.02** |
| 32 | 0.865 / 59.79 | 0.323 / 16.75 | 0.317 / 16.52 | **0.306 / 14.89** |

## 같은 한 번의 측정이 무엇을 바꾸는가

- **Inclined rigid**는 v70 source fit에서 시작한다. 매 prefix에서 기울기·restitution·friction 후보를 다시 평가해 실측으로 업데이트한다. Source incoming state 128개에서 기존 예측에서 벗어나는 정도에 penalty를 줬다. K=0만 source 식을 그대로 쓴다. Contact law가 허용하는 범위가 좁아 32회에서도 spin 오차 59.79 rad/s가 남았다. [설계](DESIGN_KO.md)
- **Single affine ridge**는 source-standardized 세 입력과 절편을 쓰는 한 affine head에서 target residual을 학습한다. 12개 coefficient의 변화에 같은 ridge penalty를 주었다. 네 번 실측에서 spin 오차가 62.56에서 30.71 rad/s로 줄었다. [구현](compare.py)
- **Single affine empirical Bayes**는 다른 다섯 source rubber의 coefficient들이 함께 움직인 방향과 각 제품 안의 measurement residual covariance를 추정한다. Paid impact 한 번은 velocity 두 성분과 spin 한 성분을 함께 제공한다. 특정 방향의 response를 확인하면 같은 source에서 연동되던 다른 coefficient도 업데이트할 수 있어, 작은 예산에서 ridge보다 유리했다. 여섯 제품에서 하나씩 빼고 비교한 실험이며 실제 material parameter 식별, 모든 상용 rubber에 공통인 prior 혹은 한 번 측정하면 매 draw에서 종전보다 낫다는 보장은 아니다. [Fold별 결과](summary.json)
- **Gate + two experts empirical Bayes**는 single-head prior의 이점과 architecture 차이를 분리하기 위해 추가한 exploratory control이다. 같은 source 선택·정규화·inner validation·covariance shrinkage·hyperparameter grid를 사용하고 두 branch의 coefficient를 함께 모델링한다. 16회에서는 이 방법이 두 지표에서 macro 기준으로 처음 single-head EB를 앞섰다. 제품별로 나누면 Hexer·Grass·Blowfish에서 모두 개선됐고 Power·Diamant·A-B-S에서는 모두 악화됐다. 32회에서도 anti-spin spin 오차는 single-head보다 2.26 rad/s 컸다. [추가 설계](GATED_PLAN_KO.md) · [제품별 결과](gated_eb.json)

## Fitting과 평가의 경계

각 outer fold는 예산을 소비하지 않고 접근 가능한 다른 제품의 official train만으로 prior를 만들었다. 다른 다섯 source 제품에서 하나씩 제외한 inner split을 사용해 penalty를 정했다. Inner calibration 및 평가 샘플은 pseudo-target의 train 내부에서 분리했다. 24개 calibration draw와 192개 validation shot, 여섯 nonzero budget의 source-only squared error (vx, vz, Rωy)를 사용했다. 새 제품에서 쓸 λ는 한 번 결정하고 모든 budget에 고정했다. Outer target의 calibration과 평가는 각각 official train과 official test로 갈랐다. [선택한 penalty와 source-only 오차](fit.json)

Empirical Bayes의 prior는 다섯 제품에서 추정했으므로 full covariance 자체의 rank가 작다. Single-head와 two-expert control 양쪽에 full matrix 85%와 diagonal 15%를 사용했다. 같은 제품의 open-loop paid shots을 반복해서 뽑은 offline replay다. Reset·실시간 tracking latency·새 sensor의 재현성은 측정하지 않았다.

서버 node56의 Slurm allocation 2744471에서 RTX PRO 6000의 memory 0 MiB, utilization 0%, compute PID 없음 확인 후 float64로 실행했다. Primary baseline fitting·evaluation의 script wall time은 6.44초이다. 이 값은 dataloader·inner tuning·모든 fold를 포함한다. 실시간 inference latency로 사용하면 안 된다. [GPU 사용 전 기록](gpu_before.txt) · [Draw별 결과](draw_metrics.csv) · [Matched EB draw별 결과](gated_eb_draw_metrics.csv)
