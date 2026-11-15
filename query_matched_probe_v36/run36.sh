#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run within the allocation}"
cd "$(dirname "$0")"
gpu=$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
processes=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
printf '%s\n' "$gpu" "$processes"
if [[ -n "$processes" ]] || ! awk -F, 'NR!=1 || $2+0>50 || $3+0>0 {bad=1} END{exit (NR!=1 || bad)}' <<< "$gpu"; then exit 3; fi
mkdir -p monitor
cd monitor
export BELIEF_DEVICE=cuda OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec ../../continuous_v2/.venv/bin/python ../../adaptive_budget_v18/guard.py ../run36.py
