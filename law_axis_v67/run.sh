#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
: "${SLURM_JOB_ID:?}"
export BELIEF_DEVICE=cuda OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
nvidia-smi --query-gpu=name,uuid,memory.used,utilization.gpu --format=csv > gpu_start.txt
procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[ -z "$procs" ]
sha256sum -c plan.sha256
../continuous_v2/.venv/bin/python -u forecast67.py
../continuous_v2/.venv/bin/python -u score67.py
sha256sum axes.json fit.json forecast_seal.json score.json > result.sha256
