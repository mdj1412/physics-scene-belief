#!/usr/bin/env bash
set -euo pipefail
group=$1
py=../continuous_v2/.venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
mkdir -p solver_log
sha256sum -c solver44.sha256
for sub in 0 1 2; do
    idx=$((group*3+sub))
    (
        for mode in primary halfdt halfmesh; do
            "$py" solver44.py --shard "$idx" --workers 6 --sensitivity "$mode"
        done
    ) >"solver_log/shard_$idx.log" 2>&1 &
    pids[$sub]=$!
done
for pid in "${pids[@]}"; do wait "$pid"; done
