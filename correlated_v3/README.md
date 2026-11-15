# Factual grazing contact / correlated scene belief

실험 결과와 연구 가치 재검토는 `REPORT.ko.md`, 선행연구 검토만 보려면 `literature_review.ko.md`를 읽는다.

- `core.py`: Exact frictionless impact, factual-conditioned posterior, fixed grid / importance particles / conditional quadrature / tempered SMC.
- `design.py`: CUDA float64 EVSI integration, matched-observation predictive evaluation.
- `run.py`, `launch.sh`: 96 family, 4 factual noise levels, 2 branch regimes, 8 queries, 18 method/seed configurations.
- `analyze.py`: Validation-only fixed rule, held-out family bootstrap, figures. 실행 경로는 이 폴더다.
- `audit_physics.py`, `audit_margin.py`, `audit_reference.py`: PyBullet timestep, collision margin, independent uniform-grid audits.
- `repair_diagnostics.py`: Histogram MI underflow 수정. Original CSV 보존, prediction metrics 불변.
- `provenance/core_at_main_run.py`: 최초 main run 당시 source. 현재 `core.py`와의 차이는 MI diagnostic의 안정적인 log 계산뿐이다.
- `build_report.py`: JSON 결과와 literature review를 결합한 한글 보고서 생성.
- `finalize.py`: Raw row count, finite values, original source hash, GPU manifest 확인; `case_specs.json`과 artifact hash 생성.

환경은 상위 `continuous_v2/.venv`와 `continuous_v2/requirements.lock.txt`를 재사용한다. 기존 실험이나 환경을 수정하지 않는다. Main inference는 compute node에서 CUDA로, audits와 analysis는 compute node에서 CPU로 실행했다. Local mirror는 검토와 보고서 전달용이다.

`launch.sh`의 Slurm job IDs는 2026-11-04 실행 당시 값이다. 재실행 전 현재 allocation과 GPU process를 확인해야 한다. 새 실행은 `--out results/new_run`처럼 별도 output으로 저장하고 `analyze.py`의 ROOT를 맞춘다. 최종 결과는 `results/main`, `results/analysis`, `results/audit`다.

이번 synthetic experiment는 calibrated scalar sensors를 사용한다. Learned RGB perception, robot hardware, expensive general scene simulator에서의 우위는 아직 검증하지 않았다. SMC와 conditional quadrature는 strong baseline이며 새 method로 주장하지 않는다. `runtime` 집계의 reference timer는 cached reference 재사용 시간이라 method speed comparison에 사용하면 안 된다.
