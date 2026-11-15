#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Use your GPU allocation}"
cd "$(dirname "$0")"
gpu_state=$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
processes=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
if [[ -n "$processes" ]] || ! awk -F, 'NR!=1 || $2+0>50 || $3+0>0 {bad=1} END{exit (NR!=1 || bad)}' <<< "$gpu_state"; then
  echo "Occupied GPU in allocation; did not start."
  exit 3
fi
mkdir -p gpu_prefix33
cd gpu_prefix33
export BELIEF_DEVICE=cuda OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec ../../continuous_v2/.venv/bin/python ../../adaptive_budget_v18/guard.py ../prefix33.py
