# v35 재현

## 실행

서버 프로젝트 경로는 `/scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief`이다. v34 저장소와 동일한 Python environment, 동결된 v34 artifact, 기존 v25–v34 module을 사용한다. 데이터 생성과 실험에는 외부 다운로드가 필요 없다.

```bash
cd /scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/frozen_direction_v35
srun --overlap --jobid="$ALLOCATION" -n1 bash ./run35.sh
```

`ALLOCATION`은 본인에게 할당된 빈 GPU 한 장의 Slurm job ID이다. 시작 시 `nvidia-smi`로 GPU 개수, process, memory, utilization을 확인한다. `guard.py`는 실행 중 다른 GPU process를 감지하면 이번 실험 worker만 중단한다. `monitor/results/worker_*.log`에 진행 상황을 기록한다. 동일 명령 재실행 시 완성된 cell의 code fingerprint와 archive checksum을 검사하고 다음 cell에서 계속한다. 단일 GPU에 두 worker를 동시 실행하지 않는다.

## 입력과 측정

- `design35.json`은 평가 전 고정한 비교, seed, 표본, timing rule을 담고 있다. `target35.py`는 기준 simulator에서 공유하는 parameter, sensor 및 noise lane을 불러오며, 새 contour와 contact law를 구현한다.
- `run35.py`는 v34의 Python 함수를 직접 호출해 두 방법에 동일한 `fixed 4`, signed likelihood, K256 acquisition, 13/128 deployment probe quota 및 physics decoder를 적용한다. Decoder는 양 arm에서 같은 실제 population proxy coupling `alpha`를 받는다. Orientation, target law 및 label은 decoder 입력에서 제외한다.
- 각 방법의 한 timing call은 8개 environment의 1,024개 deployment prediction을 계산한다. CUDA를 동기화한 wall-clock time은 factual pool 및 paid record 처리, K256 acquisition, physics decoding을 포함한다. Corrected arm에서는 같은 구간 안에 CPU에서 실행되는 frozen direction classifier와 anchored odds update도 포함한다. Simulator가 sensor를 생성하는 시간, 물리적 probe 수집 시간, 공통 decoder bank를 최초 구성하는 시간, model artifact를 디스크에서 읽는 시간은 제외한다.
- 평가는 각 world당 독립 observation noise 32개로 Brier loss를 평균한다. Timing은 그중 첫 draw에서 oracle-label-free online path만 반복한다. 각 호출에서 실제 action이 원본과 정확히 같고 두 방법의 output이 저장된 첫 draw와 absolute tolerance (10^{-9}) 안에 있는지 검사한다.
- `results/<object>_<surface>_<sign>/block_NNN/{same,alternating}.npz`에 correction 전·후 확률 `p0,p1`, ground truth `labels,theta`, 네 frozen classifier의 signed cue `z`, paid record `paid_x,paid_y,paid_direction,paid_ids`, `actions,alpha,betas,full_ms,timing_order`를 저장한다. `full_ms` 열 순서는 original, corrected이고 `timing_order`의 index는 schedule index × 2 + method index이다. Untimed cell에는 빈 timing array가 있다. `meta.json`에는 seed, starting sign과 checksum을 저장한다.
- Isotropic control은 실제 orientation이 없으므로 파일명에 쓰인 `plus`는 저장용 공통 이름이다. Primary에서 제외한다. 한 family의 sign과 schedule은 같은 geometry, deployment worlds, sensor noise 및 bank seed를 공유한다.

## 분석 순서

1. 결과가 모두 기록되어 `results/completion.json`이 생겼는지 확인한 뒤 thread 수를 1로 고정한 NumPy로 `audit35.py`와 `physics_audit35.py`를 실행한다. Source별 hash, 예측, latent query label, 같은 paid record와 deployment quota를 검산한다.
2. 같은 디렉터리에서 `python transfer_stress35.py results`로 query만 바꾸는 대조를 실행한다. `../directional_prefix_v33/physical33.json`은 실제 pulling coefficient의 공개 S1을 변환한 값이다. 시뮬레이션을 다시 fitting하지 않는다.
3. `python fit_source_cal35.py`로 v34 source archive가 frozen reference를 재현하는지 확인하고, `python score_source_cal35.py`로 direction-free 및 combined model의 secondary 비교를 계산한다. Frozen source 파일이 다르면 그대로 덮어쓰지 않는다.
4. GPU 측정이 끝난 후 별도 CPU step에서 `CUDA_VISIBLE_DEVICES= BELIEF_DEVICE=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python head_cost35.py`를 실행한다. 이 결과는 보정 단독의 host time이다.
5. `python figure35.py`로 figure를 만든다. `REPORT_KO.md`는 서버의 정본이고, 로컬 workspace의 `outputs/`에는 relative links를 조정한 같은 내용의 보고서가 있다.
