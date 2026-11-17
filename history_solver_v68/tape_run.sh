#!/bin/bash
set -eu
cd "$(dirname "$0")"
: "${SLURM_JOB_ID:?}"
export CUDA_VISIBLE_DEVICES= BELIEF_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
../continuous_v2/.venv/bin/python -u tapes68.py "$@"
