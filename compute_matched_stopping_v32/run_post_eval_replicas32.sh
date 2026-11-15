#!/usr/bin/env bash
# This monitor runs on the login host (file checks only). Training runs in Slurm.
set -euo pipefail
cd "$(dirname "$0")"
primary=$(pwd)
project=$(dirname "$primary")
allocation=${1:?Supply the original evaluation allocation}
: "${WAIT_FOR_EVAL:=1}"
if [[ "$WAIT_FOR_EVAL" == 1 ]]; then
  for result in gpu_eval/results/worker_exit.json results_eval/completion.json; do
    until [[ -s "$primary/$result" ]]; do sleep 20; done
  done
  python3 -c 'import json; from pathlib import Path; g=json.loads(Path("gpu_eval/results/worker_exit.json").read_text()); a=json.loads(Path("results_eval/completion.json").read_text()); assert g["code"]==0 and not g["foreign"] and a["complete_blocks"]==48'
fi
for rep in 2 3; do
  folder="$project/compute_matched_stopping_v32_rep$rep"
  if [[ -s "$folder/training_worker/results/worker_exit.json" && -s "$folder/results_train/model_lock.json" ]]; then
    python3 -c 'import sys,json; from pathlib import Path; a=json.loads((Path(sys.argv[1])/"training_worker/results/worker_exit.json").read_text());assert a["code"]==0 and not a["foreign"]' "$folder"
    continue
  fi
  srun --jobid="$allocation" --overlap -n1 --gres=gpu:1 bash "$folder/run_training.sh"
done
