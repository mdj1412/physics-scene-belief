#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
sha256sum --quiet -c timing_freeze55.sha256
sha256sum --quiet -c timing_plan.sha256
nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, 'NR!=1 || $1+0>=50 || $2+0>0 {bad=1} END {exit (NR!=1 || bad)}'
mkdir -p timing_guard55
cd timing_guard55
export BELIEF_DEVICE=cuda OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec ../../continuous_v2/.venv/bin/python ../../adaptive_budget_v18/guard.py ../bench55.py
