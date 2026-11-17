#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
sha256sum -c refinement_plan.sha256
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
ids=()
for i in 0 1 2;do
  ../continuous_v2/.venv/bin/python refined68.py "$i" "$((i+3))" "$((i+6))" "$((i+9))" > "refined_worker_$i.log" 2>&1 &
  ids+=("$!")
done
rc=0; for id in "${ids[@]}"; do wait "$id" || rc=1; done
test "$rc" -eq 0
find labels_compliant_refined -name '*.npz' | sort | xargs sha256sum > refined_labels.sha256
test "$(wc -l < refined_labels.sha256)" -eq 576
date -u '+%FT%TZ' > REFINED_READY
