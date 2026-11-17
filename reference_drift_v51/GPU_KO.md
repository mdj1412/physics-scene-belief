# v51 · 같은 calibration 측정에서 정책별 GPU latency

새 benchmark는 4개 환경의 물체 128개씩, 총 512개 물체의 simulation inference를 한 batch에 넣었다. 각 환경에서 **배포 surface와 다른 surface, 같은 contact law**에 속한 calibration specimen의 관측 네 쌍을 가져왔다. 각 쌍은 한 specimen에서 따로 얻은 0.4 N weak displacement와 2.0 N reference displacement다. Deployment object에서는 세 정책 모두 query 위치에서 얻은 동일한 signed displacement 하나와 동일한 factual acquisition을 사용했다. 입력은 보관한 simulator readout이며 실제 로봇 계측은 수행하지 않았다.

Likelihood soft는 새 환경의 네 쌍에서 매번 posterior mean을 계산한다. Soft vote는 동일한 두 dead zone을 적용한 뒤, 읽을 수 있는 쌍에서 부호가 일치한 수를 세어 (1+일치 수)/(2+읽힌 쌍 수)를 구한다. 두 값 모두 p(d)와 p(−d)를 섞는 가중치다. Unsigned fit은 같은 센서를 제공받되 reference를 사용하지 않고 source에서 학습한 |d| 보정식을 적용한다. 같은 vote를 이진화하는 hard vote도 측정했다. 어느 정책에도 impact label, reference–impact disagreement 또는 배포 후 κ를 전달하지 않는다.

시작 전에 GPU를 synchronize한 뒤, 필요한 host-side estimator, factual action, 구매한 observation 포장, original posterior, terminal calibration과 마지막 GPU synchronization까지의 wall time을 쟀다. 각 정책의 calibration 계산은 매번 새로 했다. Policy 밖의 물리 동작, simulation label 생성, 관측 배열을 생성하는 시간은 제외했다. 네 조건을 동일한 배치에 적용했으며, catalog마다 warmup 1회 후 무작위로 재배열한 두 번의 4×4 Williams design을 실행했다. 한 cycle 안에서 각 정책은 각 실행 위치에 한 번씩 들어가고, 서로 다른 두 정책이 연속되는 각 방향의 조합이 한 번씩 등장한다. 열두 catalog에서 정책당 96개의 512-object batch를 측정했다.

결과는 catalog를 단위로 16,000회 paired bootstrap하여 구한 p90이다. 같은 계산을 실행하더라도 policy가 보는 정보만으로는 mismatch 또는 drift 조건을 구분할 수 없으므로 전체 grid에서 공통으로 쓰는 배포 경로의 latency를 보고한다. 각 grid cell에서 따로 측정한 latency로 해석해서는 안 된다. [Benchmark](bench51.py) · [실행 script](run_bench51.sh)

| Policy | GPU end-to-end p90, ms/512 [pointwise 95% CI] |
|---|---:|
| Likelihood soft | 465.59 [464.08, 467.03] |
| Simple vote soft | 466.34 [464.31, 466.88] |
| Unsigned fit | 466.47 [464.66, 467.54] |
| Simple vote hard | 465.50 [464.14, 466.98] |

Likelihood soft − vote soft는 **−0.75 ms [−1.69, +0.70]**, likelihood soft − unsigned는 **−0.88 ms [−1.61, +0.20]**였다. 열두 catalog에 대한 paired bootstrap의 두 primary timing contrast에 Bonferroni를 적용한 simultaneous 95% CI다. 확실한 latency 개선을 확인하지 못했다. 네 환경의 입력 네 쌍씩에서 estimator만 다시 계산한 host p90은 likelihood 0.04548 ms, simple vote 0.00833 ms였다. 전체 경로에서 GPU의 original decoder를 실제로 매번 한 번 실행했다. 예측을 독립적인 저장 forecast 및 원래 vote 구현과 대조했을 때 최대 확률 오차는 4.44×10⁻¹⁶이었다. Node56의 RTX PRO 6000 Blackwell, Torch 2.7.1+cu128에서 실행했고 시작 시 0 MiB·0%와 compute process 부재를 확인했다. 실행 중 GPU guard가 감지한 다른 compute process는 없었다. [측정 수치](timing51.json) · [실행 순서와 batch별 시간](timing51) · [완료 기록](timing_guard51/results/worker_exit.json)
