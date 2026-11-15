# Independent corner replication v8

설명은 한글, technical terms는 English로 작성했다. 사전 비교 조건은 `PROTOCOL.md`, 최종 해석은 `REPORT_KO.md`를 참고한다.

## 위치와 실행

Server: `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/decision_replication_v8`

`../continuous_v2/.venv/bin/python`을 현재 사용자에게 할당된 idle GPU의 Slurm step에서만 실행한다. `resume_worker.py`는 foreign GPU process를 발견하면 자기 worker만 종료한다. 기존 사용자 allocation을 변경하지 않는다.

1. `resume_worker.py experiment.py 240`: atomic claims로 최대 4분간 worlds preparation.
2. `resume_worker.py validate.py`: Gaussian quadrature, streaming SNIS, cross-product utility 검증.
3. `resume_worker.py`: 64 worlds의 acquisition을 모두 고정한 다음 actual evaluation과 numerical audits 실행.
4. Primary worker 종료 후 `resume_worker.py nuisance.py`: supplementary analytic nuisance control. `postrun.py`는 primary 완료와 GPU 해제를 확인한 뒤 이 단계를 한 번 실행하는 orchestration이다. `NUISANCE_ADDENDUM.md`를 참고한다.
5. Local NumPy 환경에서 `validate_analysis.py`, `analyze.py`, `plot.py`, `make_report.py` 실행.

재실행은 새 output directory에서 수행한다. 기존 완료 결과를 덮어쓰지 않는다. 중단된 incomplete case는 process가 종료됐는지 먼저 확인해야 한다. 이미 완료된 acquisition의 policy와 checksum은 actual evaluation 전에 검증한다.

## Data boundary와 재현 범위

`worlds/case_###/public.json`와 `outer.npz`만 acquisition에 전달한다. `private/truth.json`과 `private/actual.npz`는 evaluation/analysis 전용이다. Source hashes, seed, beta/ESS diagnostics, compute timing, budget checkpoints와 frozen policy를 저장한다.

Budget baseline은 같은 GPU에서 측정한 wall-time ceiling 아래 가장 큰 완료 checkpoint다. Chunk granularity와 timing variation 때문에 정확히 동일한 elapsed time이 되지는 않는다. 제한을 넘은 최종 chunk는 primary bank에서 제외한다. 생성할 수 있는 최적 SNIS implementation이라는 의미도 아니다. 공통 outer generator/decoder와 disk I/O는 comparative acquisition budget에서 제외한다.

Cross-fitting은 independent factual SMC streams 사이의 cross-product를 사용한다. Self-normalization bias, shared finite current mean error, 각 SMC population의 genealogy correlation을 모두 제거한 unbiased estimator라는 주장이 아니다.

Synthetic 2D known-family corner collision과 Gaussian precise-view proxy를 다룬다. 실제 camera나 real-world physics 검증이 아니다. Figure의 한글 font는 macOS AppleGothic을 사용했다.
