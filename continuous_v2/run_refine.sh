#!/usr/bin/env bash
set -euo pipefail
cd /scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/continuous_v2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
jobs=(2642474 2642475 2642476 2642473)
pids=()
for shard in 0 1 2 3; do
  srun --jobid="${jobs[$shard]}" --overlap --exact -N1 -n1 -c1 --gres=gpu:1 --time=01:30:00 \
    .venv/bin/python refine.py --shard "$shard" > "logs/refine_${shard}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
