# v35 독립 계산 기록

양 방법은 동일한 4회의 paid calibration, 같은 deployment action, 같은 sensor draw를 사용한다. 이 표에서 Δ는 corrected − original이며 음수일 때 개선이다. 전체 평균은 각 object × surface cell, friction orientation, schedule에 동일 가중치를 준다. 32개의 독립 seed block 안에서 mirror 및 반복 측정을 묶었다.

| 평가 | Original Brier | Corrected Brier | Δ [95% CI] |
|---|---:|---:|---:|
| pooled | 0.11860 | 0.11277 | -0.00583 [-0.00637, -0.00532] |
| same | 0.11885 | 0.11284 | -0.00601 [-0.00688, -0.00514] |
| alternating | 0.11835 | 0.11270 | -0.00565 [-0.00647, -0.00484] |

Isotropic control (16개 독립 seed block)의 평균 Δ는 0.00607 [0.00449, 0.00786]이다. 이 control에서는 true orientation이 정의되지 않아 direction accuracy를 계산하지 않았다.

## 효과가 발생한 조건

| 조건 | Same Δ | Alternating Δ |
|---|---:|---:|
| sinusoid_atan | -0.00359 | -0.00419 |
| sinusoid_algebraic | -0.00737 | -0.00663 |
| quartic_atan | -0.00806 | -0.00653 |
| quartic_algebraic | -0.00502 | -0.00527 |
| sinusoid | -0.00548 | -0.00541 |
| quartic | -0.00654 | -0.00590 |
| atan | -0.00582 | -0.00536 |
| algebraic | -0.00620 | -0.00595 |
| start_negative | -0.00551 | -0.00529 |
| start_positive | -0.00651 | -0.00602 |
| plus | -0.00676 | -0.00565 |
| minus | -0.00526 | -0.00566 |

각 조건의 값은 descriptive estimate다. Simultaneous interval은 두 schedule의 primary comparison에 적용한다.

## Direction readout과 Brier 변화

| Schedule | Direction accuracy | Direction Brier | Mean squared correction | Residual alignment term | Reversed cue Δ | True cue Δ |
|---|---:|---:|---:|---:|---:|---:|
| same | 0.89648 | 0.07984 | 0.00959 | -0.01560 | 0.02371 | -0.00905 |
| alternating | 0.89453 | 0.07497 | 0.01032 | -0.01598 | 0.02470 | -0.00902 |

Mean squared correction = E[(p′−p)²]이고 residual alignment term = 2E[(p′−p)(p−y)]다. 두 항의 합은 Δ와 같다. 방향을 뒤집은 cue와 true cue에는 v34의 동일 estimated β를 사용했다. 두 값은 추정한 부호가 실제 예측 오차와 정렬되는지를 진단한다.

## Compute 비용과 불확실성

| Schedule | Original p90 (ms) | Corrected p90 (ms) | p90 ratio [동시 95% CI] |
|---|---:|---:|---:|
| same | 1418.09 | 1420.87 | 1.00196 [0.99722, 1.00671] |
| alternating | 1421.19 | 1420.88 | 0.99978 [0.99770, 1.00187] |

각 방법·schedule당 48회, 8개 paired timing block을 사용했다. 한 번의 call은 8 environments × 128 worlds에 대해 실제 관측 하나를 사용한다. 서로 다른 family에서는 seed block을 독립적으로 복원 추출하되, 네 개의 고정된 cell 자체는 다시 뽑지 않았다. 한 장치, 한 실행 session의 p90이므로 다른 GPU나 물리적 probing 대기 시간까지 일반화할 수 없다.

## Physics 및 information flow 검증

- 160개 raw archive의 checksum과 20개 실행 source의 checksum이 일치했다. CPU가 독립 구현한 reader와 GPU worker가 기록한 cue의 최대 편차는 0.00e+00, corrected probability의 최대 편차는 0.00e+00였다.
- 저장된 latent geometry와 contact law로 stick label을 별도 계산해 모든 archive에서 일치했다. 각 calibration의 네 paid object가 서로 다르고, 각 환경에서 128개 중 13개에 deployment probe를 배정한다. 각 양방향 schedule과 mirror에 같은 seed를 배정했고, 독립 block 사이에서 calibration/deployment/bank seeds가 겹치지 않았다.
- v34 reader와 β는 ridged_coarse source 자료에서 정한 저장 계수다. 각 schedule에 따라 reader를 선택하며 새 family 이름, target label 또는 target orientation은 correction 계산에 들어가지 않는다. 두 arm 모두 simulator의 proxy correlation hyperparameter를 공통으로 사용한다.
- Sinusoid 및 quartic profile을 유한 차분하여 local slope와의 일치를 확인했다. Query contact의 factual·probe sticking/sliding 조건과 잔여 normal impulse를 별도 NumPy 식으로 검산했다. 두 새 friction law가 v34와 공유하는 부호·단조성은 검증 범위의 제약이다.

[Statistical audit](audit35.json) · [Physics audit](physics_audit35.json) · [사전 계획](AUDIT_PLAN_KO.md)
