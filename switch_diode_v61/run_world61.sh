#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
sha256sum --quiet -c world_plan61.sha256
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
py=../continuous_v2/.venv/bin/python
if [[ "$1" == forecast ]]; then
  test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, 'NR!=1 || $1+0>=50 || $2+0>0 {bad=1} END{exit(NR!=1 || bad)}'
  export BELIEF_DEVICE=cuda
  mkdir -p generation_guard61
  cd generation_guard61
  exec ../../continuous_v2/.venv/bin/python ../../adaptive_budget_v18/guard.py ../generate61.py forecast
fi
export CUDA_VISIBLE_DEVICES="" BELIEF_DEVICE=cpu
if [[ "$1" == solver ]]; then
  pids=()
  for step in 0 1 2 3; do
    "$py" generate61.py solver $(seq "$step" 4 23) > "solver61_$step.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
else
  exec "$py" generate61.py "$1"
fi
