#!/usr/bin/env bash
set -euo pipefail
cd /scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/correlated_v3
mkdir -p logs
jobs=(2642474 2642475 2642476 2642473)
pids=()
for shard in 0 1 2 3; do
  srun --jobid="${jobs[$shard]}" --overlap --exact -N1 -n1 -c1 --gres=gpu:1 --time=00:30:00 \
    env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ../continuous_v2/.venv/bin/python -u run.py --shard "$shard" > "logs/main_${shard}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
