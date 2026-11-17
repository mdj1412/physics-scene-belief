# v41 실행과 검산

## 준비

실제 실험 디렉터리는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/estimated_projection_v41`다. 상위 디렉터리의 v18~v40 코드와 frozen source artifact, `continuous_v2/.venv`를 사용한다. Primary freeze는 `freeze41.sha256`과 `frozen_utc.txt`에서 확인할 수 있다. 그때 서버의 `results` 디렉터리는 아직 존재하지 않았다. 하위 archive에 기록된 전체 dependency hash는 `results/run.json`에 있다. 추가로 비교한 uncorrected decoder latency의 source는 `latency_freeze41.sha256`과 `estimated_zero_freeze41.sha256`에서 검증한다. 두 번째 보조 timing은 primary outcome 확인 후에 설계했다.

## GPU 실행

RTX PRO 6000 Blackwell Server Edition 한 장에 primary와 두 차례의 supplemental crossover를 순서대로 배정했다. 명령에서 `JOB_ID`를 사용자의 유효한 idle Slurm allocation 번호로 바꾼다. Shell의 idle 검사와 v18 guard가 compute process를 감시한다.

```bash
srun --overlap --jobid=JOB_ID -n1 bash run41.sh
srun --overlap --jobid=JOB_ID -n1 bash run_uncorrected41.sh
srun --overlap --jobid=JOB_ID -n1 bash run_estimated_zero41.sh
```

모든 decoder는 v38에서 쓰던 환경별 두 proxy alignment parameter를 공통으로 입력받아 source bank를 구성한다. 이 두 값은 offset–curvature와 restitution–friction의 연관 관계를 지정한다. K256 acquisition은 factual evidence와 첫 세 번의 paid response에서 배운 detector를 쓰고, decoder는 알려진 query height를 받는다.

Primary worker는 각 archive와 그에 대응하는 SHA를 기록한다. 재시작 시 dependency fingerprint와 파일 hash가 모두 맞는 archive부터 이어간다. Supplemental timing은 새로운 GPU guard를 사용하고 동일한 12개 panel의 같은 1,024개 대상과 첫 observation noise를 복원한다. Primary의 9가지 방법과 supplemental의 3가지 방법에서 cluster마다 warm-up 한 번과 유효 반복 세 번을 실시한다. 각 반복에서 method 순서를 무작위로 정한다.

## 각 비용의 범위

- `full_ms`는 방향·후보 template이 준비된 경우의 online 비용이다. 같은 factual evidence에서 K256으로 구입할 observation을 결정하고, host observation을 GPU로 옮겨 16,384개 가설의 posterior를 업데이트한다. 이후 CPU로 복귀해 해당 방법의 calibration과 선택·보정을 마치고 CUDA를 동기화한다.
- `setup_ms`는 이미 생성한 8개 source bank를 현재 query 높이에 맞춰 template과 label로 바꾸는 데 드는 비용이다. Estimated에서는 이미 등록한 32쌍을 읽어 방향을 추정하는 NumPy SVD와, 8개 bank의 centered coordinate projection까지 이 구간에 포함한다. Oracle에서는 알려진 단위 벡터가 준비되어 있다.
- `fit_ms`는 setup 중 방향 벡터를 준비한 host 구간이다. `cold_ms`는 같은 반복의 setup과 online을 합쳐 기록한다. CUDA context 시작, 8개 source bank 최초 구축, source table 학습, 센서가 실제 장면에 접근하고 관측치를 수집하는 물리적 시간은 이 조건부 latency 밖에 있다.
- 32회 observation noise는 loss 추정을 위한 반복이다. 각 selector 호출은 한 물체에서 실제 선택된 observation 1회의 데이터만 받는다. Warm timing은 환경 8개 × 물체 128개에서 정확히 이 단일 realization을 사용한다.

## CPU 검산

CPU Slurm step에서 `CUDA_VISIBLE_DEVICES=`, `BELIEF_DEVICE=cpu`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`을 지정하고 `../continuous_v2/.venv/bin/python`으로 아래 파일을 실행한다.

| 파일 | 확인 항목 |
| --- | --- |
| `audit41.py` | 576개 archive의 SHA·dependency fingerprint와 9개 방법의 Brier loss, paid prefix에서 복원한 v38 weight, 동일 contact에서 contour·projection invariance |
| `decoder_audit41.py` | 단일 bank의 19·25·31 mm 물리 event와 template, NumPy squared-distance likelihood, legacy 25 mm decoder 재현 |
| `direction_audit41.py` | 독립 paired profile과 shuffled pair에 대한 방향 추정. 전체 검산은 `--panels 12 --check-archives`를 사용한다 |
| `physics_audit41.py` | catalog 분리, 실제 profile·속도·stick label 및 모든 paid observation의 독립 물리 검산 |
| `stats41.py` | 12개 panel의 paired Brier 대조, 동시 신뢰구간, paired timing bootstrap |

`results/<cell>/catalog_XX/block_YY.npz`가 raw archive다. `p_base`의 축은 decoder × sign × environment × deployment object × observation noise다. `risk`의 축은 sign × near-field regime × decoder × correction policy × environment다. `chosen_optimizer`는 sign × near-field regime × decoder × environment × observation noise다. `results/directions`는 panel · shape · height별 passive profile 32쌍, estimated/oracle unit vector를 담는다. `latency_uncorrected41.npz`에는 original/uncorrected, original/fixed, estimated/fixed가 담긴다. `latency_estimated_zero41.npz`에는 estimated/uncorrected, estimated/fixed, estimated/selector의 동일 반복별 시간이 담긴다.
