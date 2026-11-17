#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
workers=${1:-3}
sha256sum -c freeze.sha256
for ((k=0;k<workers;k++)); do
    ../continuous_v2/.venv/bin/python sham67.py $(seq "$k" "$workers" 23) > "panel_worker_$k.log" 2>&1 &
    children[$k]=$!
done
for kid in "${children[@]}"; do wait "$kid"; done
../continuous_v2/.venv/bin/python sham67.py report
