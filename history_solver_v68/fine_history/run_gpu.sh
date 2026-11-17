#!/bin/bash
set -eu
cd "$(dirname "$0")"
: "${SLURM_JOB_ID:?}"
export BELIEF_DEVICE=cuda OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
nvidia-smi --query-gpu=name,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits >gpu_state.txt
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
awk -F, 'NR!=1 || $3+0>50 || $4+0>0 {bad=1} END{exit (NR!=1 || bad)}' gpu_state.txt
sha256sum -c fine_plan.sha256
../../continuous_v2/.venv/bin/python -u fine_predict.py
