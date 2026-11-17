#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
nvidia-smi --query-gpu=name,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits > gpu_before.txt
nvidia-smi --query-compute-apps=pid --format=csv,noheader > compute_before.txt
test ! -s compute_before.txt
awk -F, 'NR!=1 || $3+0>50 || $4+0>0 {bad=1} END {exit (NR!=1 || bad)}' gpu_before.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
sha256sum PLAN_KO.md common.py adapt.py study.py > design.sha256
date -u > design.utc
exec ../../continuous_v2/.venv/bin/python -u study.py
