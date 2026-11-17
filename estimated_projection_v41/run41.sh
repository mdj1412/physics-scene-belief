#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside a user GPU allocation}"
cd "$(dirname "$0")"
sha256sum -c freeze41.sha256
device_state=$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
other_processes=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader)
printf '%s\n' "$device_state" "$other_processes"
if [[ -n "$other_processes" ]] || ! awk -F, 'NR!=1 || $2+0>50 || $3+0>0 {bad=1} END{exit (NR!=1 || bad)}' <<< "$device_state"; then
    exit 3
fi
mkdir -p monitor
cd monitor
export BELIEF_DEVICE=cuda OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec ../../continuous_v2/.venv/bin/python ../../adaptive_budget_v18/guard.py ../runner41.py
