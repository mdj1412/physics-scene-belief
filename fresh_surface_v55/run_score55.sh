#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
test -n "${SLURM_JOB_ID:-}"
sha256sum --quiet -c score_code55.sha256
sha256sum --quiet -c score_plan55.sha256
export CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python=../continuous_v2/.venv/bin/python
for panel in {0..23}; do
  name=$(printf 'panel_%02d.npz' "$panel")
  while [[ ! -f "panels/$name" ]]; do sleep 5; done
  "$python" score55.py --panel "$panel"
  echo "scored $name"
done
"$python" score55.py --summary
