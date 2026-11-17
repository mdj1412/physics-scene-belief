# v71 release orientation과 expert의 분리

Grass D.TecS에서는 실측 signed release와 v70 source gate의 부호가 반대인 경우가 많다. Prefix의 불일치 표가 일치 표보다 많으면 gate를 뒤집는 간단한 규칙으로 4회 249/256, 8회 255/256, 16회 256/256번 반전을 선택했다. 두 값을 구분해야 한다. **반전을 선택한 draw 비율**과 **새 impact에서 release 부호를 맞힌 비율**은 다르다. 후자는 k=8에서 90.32%였으며, 16회부터는 이 frozen gate를 항상 뒤집었을 때의 90.64%에 도달했다.

| Grass calibration 횟수 | Majority flip 비율 | 단측 검출 비율 | Gate-only velocity / spin | Gate+expert velocity / spin | Frozen routing에 맞춰 expert 학습 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 1.584 / 119.10 | 1.584 / 119.10 | 1.584 / 119.10 |
| 1 | 92.58% | 0 | 0.736 / 46.41 | 0.664 / 41.73 | 1.345 / 97.81 |
| 2 | 86.72% | 0 | 0.790 / 51.01 | 0.668 / 41.94 | 1.166 / 81.18 |
| 4 | 97.27% | 0 | 0.694 / 42.73 | 0.532 / 30.20 | 0.879 / 55.10 |
| 8 | 99.61% | 88.67% | 0.672 / 40.89 | 0.455 / 23.70 | 0.632 / 35.80 |
| 16 | 100% | 98.83% | 0.668 / 40.58 | 0.388 / 19.96 | 0.426 / 22.61 |
| 32 | 100% | 100% | 0.668 / 40.58 | 0.346 / 17.93 | 0.354 / 18.63 |

오차는 공식 test의 mean planar velocity vector error (m/s)와 Y-axis spin MAE (rad/s)다. 256 draws는 각 제품의 동일한 target-train pool에서 복원 없이 다시 뽑되, draw들 사이의 표본 재사용은 허용했다. 두 번째 shot에서 첫 shot의 판단을 무효화할 수 있고, 1표 대 1표인 경우 원래 routing으로 돌아간다. 따라서 예산이 1에서 2가 될 때 detection이 잠시 낮아진다. 단측 검출은 각 고정 예산에서 Binomial(k,0.5)의 상측 tail ≤0.05인 횟수이며 4회의 결과가 모두 불일치해도 p=.0625다. 여러 prefix를 계속 보며 멈추는 검정의 false-alarm rate는 따로 검증해야 한다.

A-B-S II soft에서는 k=8에서 34.77%가 뒤집혔고 단측 검출은 3.52%였다. 어느 방향을 선택해도 해당 제품의 평균 release 부호 정확도는 약 50%다. 나머지 네 제품에서는 k=8에 256 draw 모두 원래 gate를 유지했다. 하나의 제품에 강한 반전이 있다는 관찰과 반전되는 표면의 전체 빈도를 추정하는 문제는 서로 다르다.

| 예산 | Gate-only | Expert-only, 관측 class로 fit | Gate+expert, 관측 class로 fit | Expert-only, 고정 배포 route로 fit |
|---:|---:|---:|---:|---:|
| 0 | 1.065 / 67.16 | 1.065 / 67.16 | 1.065 / 67.16 | 1.065 / 67.16 |
| 1 | 0.879 / 50.65 | 0.978 / 62.02 | 0.793 / 46.92 | 0.930 / 58.37 |
| 2 | 0.905 / 53.20 | 0.907 / 57.50 | 0.753 / 45.36 | 0.824 / 51.07 |
| 4 | 0.886 / 51.52 | 0.789 / 49.32 | 0.640 / 37.96 | 0.665 / 39.63 |
| 8 | 0.878 / 50.76 | 0.618 / 36.81 | 0.505 / 28.48 | 0.499 / 27.81 |
| 16 | 0.875 / 50.52 | 0.465 / 25.78 | 0.379 / 19.52 | 0.363 / 18.23 |
| 32 | 0.872 / 50.22 | 0.386 / 20.63 | 0.319 / 15.61 | 0.299 / 14.01 |

위 표는 제품별 평가값을 동일 가중한 macro 평균이다. 관측 class를 기준으로 fit하는 두 expert 방식은 동일 coefficient로 예측하고 routing만 바꿨다. 배포 route로 직접 fit하는 대안에서는 calibration부터 frozen route를 이용한다. 이때 고정된 partition 안에서 target의 연속 response를 배울 수 있다. 이 대안의 8/16/32회 macro 오차가 gate+expert보다 작았으므로 모든 새 rubber에서 명시적인 flip이 필요하다고 주장하기 어렵다.

Source fit은 v70에서 target rubber 전체를 제외해 얻은 coefficient를 사용했다. Hyperparameter는 나머지 다섯 rubber에서 한 제품씩 pseudo-target으로 지정해 선택했다. Inner pseudo-target의 train 내부에서 calibration과 검증을 나누고, outer evaluation에 들어가기 전에 fit.json을 먼저 기록했다. Primary에서 여섯 fold 모두 후보 0.25/1/4/16 중 0.25가 선택됐다. [사전 계획](PLAN_KO.md) · [source 검증 및 서버 기록](fit.json) · [draw별 오차](draw_metrics.csv) · [검출 기록](detection.csv) · [실제 사용한 순서](draw_indices.npz) · [첫 draw의 예측](draw0_predictions.csv) · [공유한 test shot별 평균 오차](shot_mean_errors.csv)

Primary에서 후보의 최솟값이 선택됐으므로, 별도 Slurm CPU step에서 λ=0.015625/0.0625/0.25를 비교했다. Source-only inner risk를 처음부터 재계산해 원래 λ=0.25에서의 값과 절대차 10⁻¹⁰ 이하로 맞았는지 검사했다. 여섯 outer fold 모두 0.25를 유지했고, Grass의 inner risk는 위 후보 순서대로 0.555/0.528/0.518이었다. [추가 grid 및 전체 재검증](sensitivity.json)
