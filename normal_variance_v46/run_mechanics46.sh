#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
mkdir -p physics_log46
pids=()
for ((i=0; i<$2; i++)); do
  ../continuous_v2/.venv/bin/python -u mechanics46.py "$1" > "physics_log46/${1}_${3}_${i}.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done
exit "$rc"
