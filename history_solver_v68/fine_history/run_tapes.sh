#!/bin/bash
set -eu
cd "$(dirname "$0")"
: "${SLURM_JOB_ID:?}"
export CUDA_VISIBLE_DEVICES= BELIEF_DEVICE=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
ids=()
for i in 0 1 2; do
    ../../continuous_v2/.venv/bin/python -u fine_tapes.py "$i" "$((i+3))" "$((i+6))" "$((i+9))" > "tape_$i.log" 2>&1 &
    ids+=("$!")
done
rc=0;for id in "${ids[@]}"; do wait "$id" || rc=1; done; exit "$rc"
