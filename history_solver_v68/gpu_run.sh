#!/bin/bash
set -eu
cd "$(dirname "$0")"
: "${SLURM_JOB_ID:?}"
export BELIEF_DEVICE=cuda OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
state=$(nvidia-smi --query-gpu=name,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
echo "$state" > gpu_start.txt
procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[ -z "$procs" ]
awk -F, 'NR!=1 || $3+0>50 || $4+0>0 {bad=1} END{exit (NR!=1 || bad)}' gpu_start.txt
sha256sum -c forecast_plan.sha256
../continuous_v2/.venv/bin/python -u predict68.py
