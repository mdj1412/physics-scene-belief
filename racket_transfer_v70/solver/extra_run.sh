#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
nvidia-smi --query-gpu=name,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits > extra_gpu.txt
nvidia-smi --query-compute-apps=pid --format=csv,noheader > extra_processes.txt
test ! -s extra_processes.txt
awk -F, 'NR!=1 || $3+0>50 || $4+0>0 {bad=1} END{exit (NR!=1 || bad)}' extra_gpu.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec ../../continuous_v2/.venv/bin/python -u diagnostics.py
