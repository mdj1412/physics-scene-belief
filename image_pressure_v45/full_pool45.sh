#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
pythonbin=../continuous_v2/.venv/bin/python
count=$1
id=$2
for ((i=0;i<count;i++)); do
    "$pythonbin" -u bullet_full45.py >"pressure_log/pool${id}_${i}.log" 2>&1 &
done
wait
