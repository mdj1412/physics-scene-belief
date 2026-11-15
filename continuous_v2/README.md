# Continuous Geometry and Contact Ambiguity

2026-11-04에 원격 서버에서 실행한 두 번째 physics experiment다. 이전 four-world pilot은 상위 directory에 그대로 보존했다. 설명과 결과는 `REPORT.ko.md`에 있다. 최종 분석은 `precision3840` 아래의 timestep 1/3840초 결과를 기준으로 하며, root의 1/960초 결과는 numerical sensitivity 비교용으로 보존했다.

```text
/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/continuous_v2
```

## 실험 범위

Continuous wall offset과 restitution을 sampling하고, 같은 `translate_wall_y(delta)` query type 안에서 numeric delta를 바꾼다. 모든 truth outcome은 PyBullet의 직접 rollout으로 생성한다. Inference는 별도의 shared physics lookup과 RGB centroid observation model을 이용한다. Truth는 inference grid에 맞추지 않는다.

128개 family를 development 32, validation 32, test 64로 나눴다. Family마다 32개 continuous world와 12개 query를 사용한다. 모든 방법은 같은 truth와 matched observation noise를 받는다. Query-type rule은 validation 결과로만 선택한다.

Main suite는 17개 실행 조건이며, 257×65 joint grid를 별도 numerical reference로 추가했다. `particles512`는 resampling 없는 importance-particle approximation이다. Learned SMC나 neural policy의 공식 재현으로 부르지 않는다. 초기 prior와 observation likelihood가 geometry·restitution별로 factorize하므로 이 실험의 joint belief는 두 uncertainty의 공동 propagation을 뜻한다.

## 원격 실행

최종 high-precision physics pipeline은 `bash run_precision3840.sh`로 실행했다. 이 launcher는 `precision3840/data`에 새로운 direct truth와 lookup을 생성하고 같은 seed·method·budget으로 `precision3840/results`에 GPU 비교 및 분석 결과를 저장한다. Source는 상위 extension의 동일한 파일을 사용하며 `precision3840.py`만 physics timestep을 1/3840초로 설정한다. `precision3840/data/precision_*.json`에 wrapper와 상위 physics source hash를 기록한다. 아래는 먼저 실행한 1/960초 pipeline의 단계별 command다.

실제 실행에 사용한 allocation은 `2642474`, `2642475`, `2642476`, `2642473`이다. 실행 당시 GPU 4개 모두 process 없이 idle임을 확인했다. 해당 ID는 만료 후 재사용할 수 없으므로 재실행 시 현재 사용자 allocation을 확인하고 launcher의 `jobs`를 변경해야 한다.

```bash
cd /scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/continuous_v2
bash run_prepare.sh
srun --jobid=2642474 --overlap --exact -N1 -n1 -c1 --gres=gpu:1 \
  env CUDA_VISIBLE_DEVICES= ../.venv/bin/python prepare.py merge
srun --jobid=2642474 --overlap --exact -N1 -n1 -c1 --gres=gpu:1 \
  .venv/bin/python validate.py
bash run_gpu.sh
bash run_refine.sh
srun --jobid=2642474 --overlap --exact -N1 -n1 -c1 --gres=gpu:1 \
  env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python analyze.py
srun --jobid=2642474 --overlap --exact -N1 -n1 -c1 --gres=gpu:1 \
  env CUDA_VISIBLE_DEVICES= .venv/bin/python audit_timestep.py
```

위 command는 원래 결과 경로에 쓰므로 원본 재실행 전 별도의 output directory 또는 checkout을 사용한다. `--overlap`은 기존 allocation의 workload가 비어 있음을 확인한 상황에서 사용했다. 사용자 작업을 취소하거나 preempt하지 않았다. Physics와 analysis는 compute node의 CPU에서, posterior와 acquisition 계산은 CUDA에서 실행했다.

새 environment에서는 프로젝트 전용 `.venv`에 `requirements.lock.txt`를 설치한다. PyTorch는 `torch==2.7.1+cu128`이며 공식 CUDA 12.8 wheel index를 사용했다. 기존 Python environment는 바꾸지 않는다. `prepare.py`와 `audit_timestep.py`는 상위 directory의 `physics.py`에서 검증된 World와 renderer를 import한다.

## 코드와 artifact

| 파일/경로 | 내용 |
|---|---|
| `prepare.py` | Direct PyBullet truth, physics response lookup, RGB sensor lookup |
| `belief.py` | Continuous quadrature/grid/particle approximation, CUDA Bayes update, EVSI |
| `run.py` | 17개 baseline, paired noise, budgets 0·1·2, metric·runtime 기록 |
| `refine.py` | Evaluation loop를 재사용하는 257×65 reference. Main source·결과는 변경하지 않음 |
| `validate.py` | Off-grid truth, lookup mismatch, integration/grid convergence 검증 |
| `audit_timestep.py` | 1/960·1/3840·1/7680초의 별도 grazing-contact audit |
| `precision3840.py`, `run_precision3840.sh` | 동일한 continuous world를 1/3840초로 다시 생성하고 전체 실험을 독립 경로에서 재실행 |
| `analyze.py` | Validation-only query rule 선택, paired family bootstrap, figure |
| `data/family_*.npz`, `.json` | Direct truth, observation mean, prior metadata, numeric query |
| `data/physics_table.npz` | 모든 방법이 공유한 simulator lookup |
| `results/main` | Main metrics, test details, GPU별 runtime·source hash |
| `results/refined` | Finer-grid reference 결과 |
| `results/analysis` | Test aggregate, paired contrast, same-information contrast, figure |
| `results/timestep_audit` | Physics timestep diagnostic. Main data를 덮어쓰지 않음 |
| `logs` | Physics, GPU, analysis, 검증 log |
| `requirements.lock.txt` | 실제 설치 environment |

`precision3840` 아래에도 동일한 `data`, `results/main`, `results/refined`, `results/analysis`, `logs` 구조가 있다. 최종 보고서의 primary 결과를 확인할 때는 이 경로를 사용한다.

Primary metric은 normalized outgoing x velocity MSE다. Observation cost는 view/probe/repeat 각각 1이고, EVSI의 cost penalty는 0.005다. Budget 2 policy는 greedy myopic이며 non-myopic optimum이 아니다. Design regret은 realized truth를 사후에 보는 oracle이 아니라 candidate model 아래의 expected one-step utility를 기준으로 한다.

Camera, object identity, primitive shape, prior hyperparameter와 sensor model은 알려 준다. Real-video perception이나 correlated geometry–physics posterior를 학습한 결과가 아니다. Simulator lookup과 centroid lookup의 오차 및 timestep artifact는 별도 검증 결과와 함께 해석한다.
