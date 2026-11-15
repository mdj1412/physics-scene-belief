# Decision-loss audit v7

새로운 worlds에서 tempered bridge가 factual-bank SNIS보다 observation selection의 실제 loss를 낮추는지 평가한다. 설명은 Korean, technical terms는 English로 작성한다. 사전 조건은 [PROTOCOL.md](PROTOCOL.md)에 있다.

## 실행 환경과 data boundary

- Server project: `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/decision_loss_v7`.
- Python: `../continuous_v2/.venv/bin/python`. Allocated compute node에서만 inference를 실행한다.
- `results/worlds/case_###/public.json`, `bank.npz`, `generator.npz`, `outer.npz`: acquisition inputs.
- `results/worlds/case_###/private/`: evaluation truth와 실제 post-observation measurements. `run_group`의 acquisition path는 이 경로를 열지 않는다.
- `results/plans/final_policy.json`: actual evaluation 이전에 고정하는 모든 action choices.
- `results/analysis/`: 완료된 결과에 대한 local statistical analysis. `analyze.py`가 이 단계에서만 true counterfactual을 읽는다.

## 재실행

기존 결과를 덮어쓰지 말고 새 experiment directory에 source와 calibration을 복사한다. GPU allocation과 compute process가 비어 있는지 확인한 뒤 아래 command의 JOB을 현재 idle allocation으로 바꾼다.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 srun --jobid=JOB --overlap --exact -N1 -n1 -c1 --gres=gpu:1 ../continuous_v2/.venv/bin/python -u resume_worker.py validate.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 srun --jobid=JOB --overlap --exact -N1 -n1 -c1 --gres=gpu:1 ../continuous_v2/.venv/bin/python -u resume_worker.py
```

GPU마다 worker 하나만 시작한다. `resume_worker.py`는 다른 process group의 GPU compute process가 나타나면 자기 worker만 종료한다. Validation subprocess는 같은 process group으로 식별한다. Existing user processes/allocations를 취소하지 않는다. `pipeline.py`는 atomic claims로 case를 나누며 global phase barrier와 file lock으로 policy freeze를 보장한다.

`python3 status.py`로 완료 case 수와 worker log를 확인한다. Foreign process로 종료된 unfinished claims만 자동으로 해제한다. 다른 오류나 allocation 만료 후 stale claim은 owner가 더 이상 실행되지 않는지 확인한 뒤 해당 claim을 archive해서 재개해야 한다. 완료 `.json` marker 없이 있는 partial outputs를 완료 결과로 취급하지 않는다.

World/input seed는 case별로 고정돼 있다. Batched SMC seed는 phase, case group, condition, method, repeat, budget에서 생성한다. Worker scheduling에 따라 group이 달라질 수 있으므로 exact replay에는 각 result JSON의 `batch_cases`, `seed`, `h`, `n`, `moves`도 사용한다. GPU floating-point/library 차이까지 bitwise reproducibility를 보장하지 않는다.

## 분석

모든 phases가 완료된 뒤 results를 local mirror에 복사한다. NumPy/Matplotlib 환경에서:

```bash
../tempered_bridge_v6/.venv-analysis/bin/python validate_analysis.py
../tempered_bridge_v6/.venv-analysis/bin/python analyze.py
../tempered_bridge_v6/.venv-analysis/bin/python plot.py
../tempered_bridge_v6/.venv-analysis/bin/python make_report.py
```

Figures는 local macOS의 AppleGothic과 DejaVu Sans로 rendering했다. 다른 OS에서 plots를 다시 생성하려면 Korean glyph를 지원하는 font를 설정해야 한다. Saved PNG는 font 설치 없이 열 수 있다.

`analyze.py`는 completion, frozen choices, independent banks, finite arrays, final tempering beta=1, stage ESS, input sizes를 확인한다. `validate_analysis.py`는 action/query indexing 및 repeated observations를 world 수로 오인하지 않는지 확인한다. 결과의 통계적 불확실성은 48 worlds를 기준으로 계산한다. 6 queries와 32 noise repeats를 독립 sample로 세지 않는다.

## 해석 범위

Primary는 Gaussian continuous-proxy observation이다. `precise`는 view standard deviation을 0.25배로 줄인 synthetic condition이다. Raster audit도 synthetic RGB에서 추가 observation만 바꾸며 실제 video 검증이 아니다. Full SMC reference는 finite approximation이다. 이 implementation은 새 SMC algorithm을 주장하지 않는다.
