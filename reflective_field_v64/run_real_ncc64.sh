#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, 'NR!=1 || $1+0>=50 || $2+0>0 {bad=1} END {exit(NR!=1 || bad)}'
mkdir -p real_ncc_guard64
cd real_ncc_guard64
exec ../../continuous_v2/.venv/bin/python ../../adaptive_budget_v18/guard.py ../real_ncc_gpu64.py
