#!/bin/bash
set -eu
cd "$(dirname "$0")"
base="$PWD"
while [ ! -s gpu_worker/results/worker_exit.json ]; do sleep 5; done
../continuous_v2/.venv/bin/python -c 'import json; assert json.load(open("gpu_worker/results/worker_exit.json"))["code"]==0'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BELIEF_DEVICE=cuda
for pair in power:power28 controls:controls28
do
    worker="${pair%:*}_worker"
    script="${pair#*:}.py"
    mkdir -p "$base/$worker"
    cd "$base/$worker"
    nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv
    ../../continuous_v2/.venv/bin/python ../../adaptive_budget_v18/guard.py "../$script"
done
