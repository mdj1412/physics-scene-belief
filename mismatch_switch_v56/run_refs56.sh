#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
test -n "${SLURM_JOB_ID:-}"
sha256sum --quiet -c generation_freeze56.sha256
export BELIEF_DEVICE=cpu CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
py=../continuous_v2/.venv/bin/python
if [[ "$1" = solver ]]; then
  pids=()
  for offset in 0 1 2 3; do
    "$py" generate56.py solver $(seq "$offset" 4 23) > "solver_batch_${offset}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
else
  "$py" generate56.py panel
  "$py" generate56.py audit
fi
