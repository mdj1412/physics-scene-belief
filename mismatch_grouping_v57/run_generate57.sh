#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
sha256sum --quiet -c generation_freeze57.sha256
state="$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
usage="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
if [[ -n "$usage" ]] || ! awk -F, 'NR!=1 || $1+0>=50 || $2+0>0 {bad=1} END{exit(NR!=1 || bad)}' <<< "$state"; then exit 3; fi
mkdir -p generation_guard
cd generation_guard
export BELIEF_DEVICE=cuda OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec ../../continuous_v2/.venv/bin/python ../../adaptive_budget_v18/guard.py ../generate57.py forecasts
