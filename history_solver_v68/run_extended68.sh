#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
test -n "${SLURM_JOB_ID:-}"
test -f FORECASTS_SEALED
sha256sum -c solver_plan.sha256
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
p=../continuous_v2/.venv/bin/python
ids=()
for s in 0 1 2; do
  "$p" extended68.py "$s" "$((s+3))" "$((s+6))" "$((s+9))" > "label_worker_$s.log" 2>&1 &
  ids+=("$!")
done
rc=0
for job in "${ids[@]}"; do wait "$job" || rc=1; done
test "$rc" -eq 0
find labels_compliant labels_prescribed -name '*.npz' | sort | xargs sha256sum > solver_labels.sha256
test "$(wc -l < solver_labels.sha256)" -eq 1152
date -u '+%FT%TZ' > LABELS_READY
