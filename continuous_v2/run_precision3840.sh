#!/usr/bin/env bash
set -euo pipefail
cd /scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief/continuous_v2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p precision3840/logs
for source in prepare.py belief.py run.py refine.py validate.py analyze.py; do
  test -e "precision3840/$source" || ln -s "../$source" "precision3840/$source"
done
jobs=(2642474 2642475 2642476 2642473)
pids=()
for shard in 0 1 2 3; do
  srun --jobid="${jobs[$shard]}" --overlap --exact -N1 -n1 -c1 --gres=gpu:1 --time=01:30:00 \
    env CUDA_VISIBLE_DEVICES= bash -c \
    '.venv/bin/python precision3840.py table --shard "$1" && .venv/bin/python precision3840.py data --shard "$1"' _ "$shard" \
    > "precision3840/logs/prepare_${shard}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
test "$status" = 0
srun --jobid=2642474 --overlap --exact -N1 -n1 -c1 --gres=gpu:1 --time=00:15:00 \
  env CUDA_VISIBLE_DEVICES= .venv/bin/python precision3840.py merge > precision3840/logs/merge.log 2>&1
srun --jobid=2642474 --overlap --exact -N1 -n1 -c1 --gres=gpu:1 --time=00:10:00 \
  bash -c 'cd precision3840 && ../.venv/bin/python ../validate.py' > precision3840/logs/validation.log 2>&1
pids=()
for shard in 0 1 2 3; do
  srun --jobid="${jobs[$shard]}" --overlap --exact -N1 -n1 -c1 --gres=gpu:1 --time=01:00:00 \
    bash -c 'cd precision3840 && ../.venv/bin/python ../run.py --shard "$1" && ../.venv/bin/python ../refine.py --shard "$1"' _ "$shard" \
    > "precision3840/logs/gpu_${shard}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
test "$status" = 0
srun --jobid=2642474 --overlap --exact -N1 -n1 -c1 --gres=gpu:1 --time=00:15:00 \
  env CUDA_VISIBLE_DEVICES= bash -c 'cd precision3840 && ../.venv/bin/python ../analyze.py' > precision3840/logs/analysis.log 2>&1
