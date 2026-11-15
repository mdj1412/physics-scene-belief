# v33 재현 순서

학습은 [v32의 독립적인 source model 3쌍](../compute_matched_stopping_v32/retrain_roots.json)을 그대로 사용한다. 해당 model lock과 [GPU ledger](results_eval/gpu_ledger33.json)의 SHA256이 같아야 이번 결과와 비교할 수 있다. v32 폴더와 v25·v27–v31의 Python 모듈을 같은 프로젝트 아래 유지한다. 완료된 raw archives로 통계만 재현할 때는 GPU가 필요 없다.

서버 프로젝트 내부에서 Slurm이 배정한 GPU의 memory, active processes, utilization을 함께 확인한다. 새로운 GPU 실험은 결과 폴더를 비워 버리는 대신 새 프로젝트 사본에서 아래 순서로 실행한다. `run_policy33.sh`와 `run_prefix33.sh`는 할당된 노드에서만 실행하며 이미 점유된 GPU를 사용하지 않도록 검사한다. 완료된 결과에 실행할 때는 fingerprint와 checksum을 검사해 이미 끝난 block을 건너뛴다.

1. `bash run_policy33.sh` — 세 source model 중 첫 pair의 네 full policies, common decoder, sensor choices와 확률, randomized timing. `results_eval/completion.json`이 64개 signed block의 완료를 기록한다.
2. `bash run_prefix33.sh` — 독립 A의 24개 후보에서 winner와 threshold를 기록하고, 새 seed로 B의 signed block을 생성한다. `results_prefix/discovery_lock.json`과 `results_prefix/completion.json`을 확인한다.
3. Compute node에서 CPU만 사용해 `policy_stats33.py`, `mechanism33.py`, `headroom33.py`, `transfer33.py`, `shadow33.py`, `prefix33_stats.py`를 실행한다. 모든 정책의 block·training seeds, quota, 최종 확률에서 재계산한 Brier, 방향 pairing을 검사한다. 결과에 hidden labels를 사용하므로 이 분석 스크립트를 policy input에 연결하지 않는다.
4. 실물 자료를 별도로 검산하려면 `pypdf==6.21.4`로 `physical33.py /scratch2/mdj1412/datasets/tramsen2018_friction_supplement.pdf`를 실행한다. [원저자 보충자료](https://doi.org/10.6084/m9.figshare.c.3950773.v2).
5. `discussion33.py`, `fig33.py`, `write_v33.py` 순으로 보고서를 만든다. [평가 설계](design_eval.json) · [prefix 확인 기준](PREFIX_PLAN_KO.md) · [compute 및 timing 조건](GPU_PROTOCOL_KO.md).
