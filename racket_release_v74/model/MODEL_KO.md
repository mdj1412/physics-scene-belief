# v74 · 동일 관측 예산의 release sign calibration

빠른 A-B-S II soft 입사에서 4회와 8회의 paid sign을 사용한 보정은 v73 majority vote보다 높은 평균 부호 적중률을 기록했고, 보정된 velocity와 spin도 elastic impulse가 얻은 오차 감소를 유지했다. 이 비교만 보면 target 부호가 필수 정보처럼 보일 수 있다. 같은 자료에서 새 시편의 부호 투표 없이 입사 접촉점 slip의 방향을 뒤집은 더 단순한 예측도 같거나 높은 적중률을 냈다. [구현](run.py) · [설계](DESIGN_KO.md) · [재검산](result/reversal_audit.json)

## 무엇을 학습하고 무엇을 고정했는가?

시편을 A-B-S II soft로 바꾸기 전에 기존 5개 source 제품을 하나씩 pseudo-target으로 보류했다. Source train만 읽는 독립 GPU step에서 두 입사 feature의 kernel bandwidth 3개와 source pseudo-count 3개를 비교했다. 빠른 입사의 판독 가능 표본에서 제품과 K를 동등하게 가중한 평균 적중률은 최종 선택에서 99.36%였고 근접한 다른 후보(global bandwidth, pseudo-count 0)도 99.10%였다. 이 수치는 source 제품에 대한 내부 모델 선택값이다. 선택된 모델의 bandwidth는 1 표준화 단위이고 pseudo-count는 0이다. 따라서 판독한 target vote가 있으면 그 정보를 사용하고, 합이 0이거나 수치적 기준 아래일 때 source prior로 fallback한다. 모든 모델은 동일한 target calibration 행을 받는다. File hash를 기록한 뒤 A-B-S II soft target을 계산했다. 공식 test는 v73에서 이미 점검한 이력이 있다. [Source 선택 결과](source_validation.csv) · [동결 기록](frozen.json)

각 impact의 입력 상태에서 s=v_x−Rω를 구해 이탈 부호가 입사 slip 방향과 같은지 학습한다. Kernel에서 |s|/|v_z|와 v_x sign(s)/|v_z|가 가까울수록 paid vote에 높은 가중치를 준다. 이탈 부호는 sign(s)를 다시 곱해 복원한다. 보정 부호가 elastic impulse의 출력과 다르면 ±0.20 m/s를 목표로 Δu=0.4Δs, Δ(Rω)=−0.6Δs만큼 조정한다. Normal velocity와 thin-shell translation/rotation impulse identity를 보존한다. 따라서 아래 적중률은 실제로 출력한 velocity와 spin의 부호 적중률이기도 하다. 보정 부호만 따로 출력하고 연속 예측을 그대로 유지한다면 K=8에서 공식 fast의 45.8%가 두 출력 사이에 모순을 만든다. [평가 기록](result/official/metrics.csv)

## 두 split의 빠른 입사

괄호 안 값은 velocity error(m/s)·spin MAE(rad/s)다. 공식 test의 연속 오차 분모는 37, 적중률 분모는 22다. 추가 split은 632개 공식 train만 fast/other별 절반으로 나눠 316개를 보류했고, fast의 분모는 각각 68과 33이다. 같은 256개 calibration draw에서 4개는 8개의 prefix다. 추가 split이 연구진에게 전혀 새로운 시편·날짜라는 근거는 없다.

| 방법 | 공식 4회 | 공식 8회 | 추가 split 4회 | 추가 split 8회 |
|---|---:|---:|---:|---:|
| v73 majority vote | 64.0% (1.701·65.93) | 58.9% (1.677·63.17) | 56.8% (1.781·72.21) | 59.8% (1.801·74.32) |
| Single affine EB | 38.7% (0.500·28.47) | 28.7% (0.483·28.63) | 43.7% (0.511·27.18) | 37.7% (0.482·27.94) |
| Elastic impulse | 37.5% (0.441·21.15) | 35.2% (0.378·17.76) | 44.5% (0.446·21.80) | 42.4% (0.392·18.98) |
| **Paid sign 보정 + elastic impulse** | **68.0% (0.405·16.48)** | **75.9% (0.343·13.18)** | **66.0% (0.418·17.37)** | **72.3% (0.365·14.13)** |
| Source-only sign + 같은 elastic impulse | 86.4% (0.367·11.84) | 86.4% (0.337·11.77) | 72.7% (0.399·14.12) | 72.7% (0.371·13.75) |

Source-only sign은 K에 상관없이 고정된다. 여기에 결합한 elastic impulse는 여전히 4/8회 calibration의 이탈 연속 상태를 사용하므로 이 행도 전체적으로 유료 관측을 소비한다. Source-only sign은 모든 draw와 impact에서 단순 규칙 −sign(s_in)과 정확히 일치했다. Fast에서 관측된 반전은 공식 19/22, 추가 24/33인 반면 나머지 속도 조건에서는 각각 28/84, 31/146이다. 이는 공 중심의 이동이나 공 자체의 spin의 반전과 구별되는 **접촉점 slip 부호** 반전이다. [독립 검산 코드](reversal_audit.py) · [추가 split 결과](result/secondary/metrics.csv)

두 source 좌표의 local matching도 paid vote를 단순히 모은 경우와 비교했다. K=8 공식 fast에서 paid global 상수 부호의 적중률은 45.7%, paid fast/other 상수는 49.4%, 입사 부호로 정렬한 뒤 모든 거리를 같게 둔 경우는 60.5%였다. 추가 split에서는 46.2%, 48.4%, 63.5%였다. 각 상수 baseline은 판독 표가 없거나 동률이면 원래 elastic 부호를 유지한다. 주 모델과의 차이는 입력을 반영한 routing의 근거다. 더 단순한 target-sign-free reversal이 이미 높은 수치를 내므로, 이 결과가 적은 target sign으로 새로운 시편 방향을 알아내는 방법의 독자적 우위를 보여 주지는 않는다.

Fast 이외의 결과도 달랐다. K=8에서 공식 nonfast는 원래 elastic 65.8%·0.502 m/s·26.85 rad/s, 보정 후 68.8%·0.517 m/s·27.25 rad/s였다. 추가 split의 nonfast는 71.5%·0.571·30.51에서 67.8%·0.606·32.68로 바뀌었다. 보정은 동일한 impulse identity를 유지하지만 기존 α∈[0,0.8]을 보장하지는 않는다. 공식 fast에서는 범위를 벗어난 보정이 없었고, 추가 split fast에서는 4회 약 0.6%, 8회 약 0.4%였다. `alpha_outside`는 모델의 출력에 대해 입력과의 tangential velocity 차이로 계산한 진단이며 affine·majority의 물리 적합성을 보장하지 않는다.

## 재현

Datacenter3 node56의 빈 NVIDIA RTX PRO 6000 Blackwell Server Edition(GPU-f211…)에서 PyTorch 2.7.1+cu128, Slurm 2744471의 step 324와 325로 실행했다. 기존 v73 공식 예측과 동일한 세 방법을 두 예산에서 tensor 단위로 검산해 최대 차이 0을 확인했다. 각 예측 NPZ는 입력, target, 제품별 impact IDs, calibration index와 float64 예측을 포함한다. [실행 기록](result/run.json)

~~~sh
python model/run.py source
python model/run.py target
python model/reversal_audit.py
~~~
