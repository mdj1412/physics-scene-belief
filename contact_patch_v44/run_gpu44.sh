#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?A current GPU allocation is required}"
cd "$(dirname "$0")"
mode="$1"
case "$mode" in even|odd|timing) ;; *) exit 2 ;; esac
sha256sum -c gpu_freeze44.sha256
state=$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
processes=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader)
printf '%s\n%s\n' "$state" "$processes"
if [[ -n "$processes" ]] || ! awk -F, 'NR!=1 || $2+0>50 || $3+0>0 {bad=1} END{exit (NR!=1 || bad)}' <<< "$state"; then
    exit 3
fi
mkdir -p "monitor_$mode"
cd "monitor_$mode"
export BELIEF_DEVICE=cuda OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec ../../continuous_v2/.venv/bin/python ../../adaptive_budget_v18/guard.py ../runner44.py "$mode"
