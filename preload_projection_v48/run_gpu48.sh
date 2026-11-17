#!/usr/bin/env bash
set -eu -o pipefail
cd "$(dirname "$0")"
mode="$1"
case "$mode" in even|odd|timing) ;; *) exit 2 ;; esac
sha256sum --quiet -c freeze48.sha256
state=$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
procs=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader)
printf '%s\n%s\n' "$state" "$procs"
if [[ -n "$procs" ]] || ! awk -F, 'NR!=1 || $2+0>50 || $3+0>0 {bad=1} END{exit (NR!=1 || bad)}' <<< "$state"; then exit 3; fi
mkdir -p "monitor_$mode"
cd "monitor_$mode"
export BELIEF_DEVICE=cuda OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec ../../continuous_v2/.venv/bin/python ../../adaptive_budget_v18/guard.py ../runner48.py "$mode"
