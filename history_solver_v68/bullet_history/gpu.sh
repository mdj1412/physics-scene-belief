#!/bin/bash
set -eu
cd "$(dirname "$0")"
nvidia-smi --query-gpu=name,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits > gpu_before.txt
nvidia-smi --query-compute-apps=pid --format=csv,noheader > procs_before.txt
test ! -s procs_before.txt
awk -F, 'NR!=1 || $3+0>50 || $4+0>0 {bad=1} END{exit (NR!=1 || bad)}' gpu_before.txt
export BELIEF_DEVICE=cuda OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec ../../continuous_v2/.venv/bin/python -u forecast.py
