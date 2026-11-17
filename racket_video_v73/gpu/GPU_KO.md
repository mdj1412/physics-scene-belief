# v73 · 370 fps 기존 anti-spin 대조군에서 빠른 입사와 release 방향

**Calibration에 빠른 입사를 절반씩 배정한 기존 A-B-S II soft 대조군**에서, 빠른 held-out 37회의 속도·spin 오차는 4회와 8회 모두 elastic impulse가 가장 작았다. 그 예측에서 8회 보정 후 release 방향은 판독 가능한 22회에 대해 평균 35.2% 맞았다. 속도·spin 정확도와 부호 적중률은 서로 다른 지표이며, 한 값으로 다른 값을 대신할 수 없다. [서버 측정표](control/metrics.csv)

## 1. 대조군과 분할

Gossard의 연구는 고정한 racket, 370 fps 측면 영상 및 dotted ball을 사용해 충돌 전후의 상태를 추정했다. A-B-S II soft는 #10 configuration으로 allround blade와 2.1 mm sponge를 썼다. 공개된 원자료는 SI 상태이고, 촬영을 시도한 모든 장면의 원본 영상·frame timestamp·개별 판독 오차는 확보하지 못했다. 아래 판독률의 분모는 궤적을 복원해 공개한 test 행이다. 실패한 촬영을 포함한 end-to-end 판독률은 알 수 없다. 같은 시편의 기존 v70–v72 기록이므로 이 비교는 독립 시편 재현을 판단하지 않는다. [원 논문 §III와 Table IV](https://arxiv.org/html/2604.11349v1) · [저자 데이터](https://github.com/gossardt/tt_rackets/tree/159c6f040dd7832570b537b3b5fd00a97fc28c15)

A-B-S II soft를 제외한 다섯 제품의 공식 train 5,858개로 전문가와 prior를 학습했다. 여기서 구한 입사 normal speed의 75 percentile인 **8.079 m/s**를 fast 기준으로 고정했다. Target의 공식 train 632개 중 fast가 136개여서 fast·other에 4회는 2+2, 8회는 4+4를 배정했다. 각 층에서 무복원으로 뽑아 256개 prefix를 만들고 모든 방법에 동일하게 제공했다. 두 예산에서 같은 공식 held-out 158개를 사용했고 8회 prefix의 처음 4개를 4회 비교에 썼다. 빠른 입사 37개는 8.088–10.400 m/s였다. 256개 draw는 한 시편의 같은 impact를 되쓰므로 고유 시편이나 평가 대상이 늘어나지 않는다. [실행 전 규약](PROTOCOL_KO.md) · [속도 기준과 고유 식별자](control/run.json) · [행별 분할](control/splits.npz)

Original 파일에서 공식 train 및 test와 겹치지 않는 행은 791개 중 1개(angle 20, trajectory 296)였다. V70 original replay에서 해당 파일 전체가 평가됐으며, 한 건의 공식 split 제외 사유도 알 수 없다. 이 행은 별도 raw_only 표에만 남겼고 외부 시편 수 또는 공식 158회에 더하지 않았다. [V70 원본 평가](../../racket_transfer_v70/solver/original_metrics.json)

## 2. 판독률과 적중률을 분리한 결과

Racket에 상대적인 이탈 접선속도 s′ = v′ₓ − Rω′가 0에서 충분히 떨어진 경우 부호를 표시했다. R = 0.020 m다. 0.20 m/s는 v72에 사용한 민감도 기준으로, 센서가 보장한 수치가 아니다. Margin을 넘은 행만 majority vote에 참여하고 그 수가 동률이면 기존 routing을 유지한다. 연속량 오차에는 모호한 행도 들어간다.

| 최소 release margin | 전체 판독 | Fast 판독 | 나머지 판독 |
|---|---:|---:|---:|
| 0 m/s | 158/158 (100%) | 37/37 (100%) | 121/121 (100%) |
| 0.20 m/s | 106/158 (67.1%) | 22/37 (59.5%) | 84/121 (69.4%) |
| 0.50 m/s | 60/158 (38.0%) | 8/37 (21.6%) | 52/121 (43.0%) |

0.20 m/s에서 전체 106개는 양수 55·음수 51개였고, fast 22개는 양수 7·음수 15개였다. 전체에 음수만 출력한다면 fast의 사후 적중은 15/22(68.2%)다. 선택된 fast subset의 class 구성만으로 높은 적중률이 생길 수 있다. Margin을 0.5 m/s로 높이면 fast가 양수 4·음수 4가 된다. 이 때 8회 majority의 conditional accuracy는 26.9%, elastic impulse는 34.1%다. Margin은 calibration의 가용 표, test의 구성 둘을 함께 바꾼다. [전체 margin 결과](control/metrics.csv) · [유효 calibration 표 수](control/paid.csv)

**빠른 입사.** Velocity는 held-out planar vector error의 평균(m/s), spin은 angular velocity의 MAE(rad/s)이다. 각 연속 오차는 fast 37개 전부를 사용하고 부호 적중률은 판독된 같은 22개의 최종 출력 부호를 사용한다. 백분율은 256개 보정 draw에 걸쳐 평균했다.

| 방법 | 4회 velocity / spin | 4회 적중률 | 8회 velocity / spin | 8회 적중률 |
|---|---:|---:|---:|---:|
| Majority vote | 1.701 / 65.93 | 64.0% | 1.677 / 63.17 | 58.9% |
| Continuous-loss gate | 1.437 / 36.38 | 9.1% | 1.437 / 36.38 | 9.1% |
| Single affine EB | 0.500 / 28.47 | 38.7% | 0.483 / 28.63 | 28.7% |
| Elastic impulse | **0.441 / 21.15** | 37.5% | **0.378 / 17.76** | 35.2% |

Continuous-loss gate는 두 예산의 모든 draw에서 source routing을 뒤집어 expert를 골랐다. Majority는 4회에서 74/256회, 8회에서 91/256회 뒤집었다. 4회를 지불해도 유효 표의 평균은 2.17개였고 10개 draw에는 하나도 없었다. 8회에서는 평균 4.43개, 유효 표가 없는 draw는 0개였다. Prediction의 두 성분이 서로 가까우면 작은 오차에도 차이의 부호가 바뀐다. 예를 들어 fast 8회 elastic의 velocity·spin이 continuous gate보다 각각 1.059 m/s·18.62 rad/s 작으나, fast에서 elastic 방식이 맞힌 방향의 비율도 35.2%에 머문다. [개별 held impact](control/impacts.csv)

**전체 입사.** 전체 158개에 대해 연속 오차, 판독 가능한 106개에 대해 release sign 적중률이다.

| 방법 | 4회 velocity / spin | 4회 적중률 | 8회 velocity / spin | 8회 적중률 |
|---|---:|---:|---:|---:|
| Majority vote | 1.956 / 98.71 | 36.7% | 1.915 / 94.85 | 34.9% |
| Continuous-loss gate | 1.517 / 57.42 | 17.9% | 1.517 / 57.42 | 17.9% |
| Single affine EB | **0.581** / 38.54 | 54.2% | 0.537 / 36.14 | 55.7% |
| Elastic impulse | 0.616 / **32.67** | 52.0% | **0.473 / 24.72** | 59.5% |

## 3. 재현과 논문 주장

Node56에서 사용량·메모리·compute PID를 확인한 NVIDIA RTX PRO 6000 Blackwell Server Edition 한 장에서 PyTorch 2.7.1+cu128, job 2744471 step 319로 실행했다. 비교가 원치 않게 바뀌지 않았는지 v72와 같은 uniform calibration, margin 0, 같은 공식 158개에서도 2,048개 draw·방법·예산 조합을 재실행했다. Velocity, spin, output sign accuracy의 저장된 값과의 최대 차이는 모두 0이었다. [실행 metadata](control/run.json) · [코드](evaluate.py)

재현 명령은 프로젝트 root와 기존 v70–v72 source 코드를 포함한 환경에서 다음과 같다.

~~~sh
python racket_video_v73/gpu/evaluate.py --output racket_video_v73/gpu/control_replay
~~~

새로운 외부 시편이 특정 제품이고 20 mm radius ball, racket-relative 축, SI 단위와 clock을 확인했을 때, 같은 코드에 별도 split의 calibration.csv와 test.csv를 담은 디렉터리를 --external에 전달할 수 있다. 각 층당 서로 다른 calibration impact가 최소 4개 필요하다. Threshold가 source data에 따라 정해지므로 external mode에는 기존 여섯 제품을 source로 쓰고 cutoff를 다시 산출한다. Raw video의 고유 장면과 판독 실패는 별도 provenance로 보존해야 한다.

이 대조군에서 응집력 있는 결과는 같은 비용을 써도 빠른 입사의 연속 오차와 release sign 적중률에서 방법의 순위가 달라진다는 점이다. 이를 독립 anti-spin 전반에 적용하려면 다른 표면 시편과 분리된 session, acquisition clock, 고속 충돌에서 판독이 실패한 기록이 필요하다.
