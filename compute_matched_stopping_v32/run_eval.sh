#!/usr/bin/env bash
# Run from an already assigned Slurm compute node with one visible GPU.
set -euo pipefail
: "${SLURM_JOB_ID:?Run this script inside an existing Slurm allocation}"
cd "$(dirname "$0")"
gpu_state=$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
processes=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
printf '%s\n' "$gpu_state" "$processes"
if [[ -n "$processes" ]] || ! awk -F, 'NR!=1 || $2+0>50 || $3+0>0 {bad=1} END{exit (NR!=1 || bad)}' <<< "$gpu_state"; then
  echo 'The GPU is occupied. Leaving all jobs untouched.' >&2
  exit 3
fi
mkdir -p gpu_eval
cd gpu_eval
export BELIEF_DEVICE=cuda OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec ../../continuous_v2/.venv/bin/python ../../adaptive_budget_v18/guard.py ../evaluate32.py
