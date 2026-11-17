#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
for i in 0 1 2 3; do
  ../continuous_v2/.venv/bin/python illumination64.py $(seq "$i" 4 23) > "illumination64_$i.log" 2>&1 &
  jobs[$i]=$!
done
for job in "${jobs[@]}"; do wait "$job"; done
