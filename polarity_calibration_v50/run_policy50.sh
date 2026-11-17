#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mode="$1"
case "$mode" in
0|1) task=policy50.py ;;
bench) task=bench50.py ;;
*) exit 2 ;;
esac
sha256sum --quiet -c policy_freeze50.sha256
if [[ "$mode" == "bench" ]]; then sha256sum --quiet -c bench_freeze50.sha256; fi
nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits
state="$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
usage="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader)"
if [[ -n "$usage" ]] || ! awk -F, 'NR!=1 || $1+0>50 || $2+0>0 {bad=1} END{exit (NR!=1 || bad)}' <<< "$state"; then exit 3; fi
mkdir -p "jobs50/$mode"
cd "jobs50/$mode"
export BELIEF_DEVICE=cuda OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
args=()
if [[ "$mode" != bench ]]; then args+=("$mode"); fi
exec ../../../continuous_v2/.venv/bin/python ../../../adaptive_budget_v18/guard.py "../../$task" "${args[@]}"
