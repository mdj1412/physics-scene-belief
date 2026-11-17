#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for j in 0 1; do
  ../continuous_v2/.venv/bin/python optics63.py $(seq "$j" 2 23) > "optics63_$j.log" 2>&1 &
  ids[$j]=$!
done
for p in "${ids[@]}"; do wait "$p"; done
