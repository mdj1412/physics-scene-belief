#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
kind=$1
parity=$2
args=()
if [[ $kind == fine ]]; then args+=(--fine); fi
worker=pressure45.py
if [[ $kind == bullet ]]; then worker=bullet45.py; fi
if [[ $parity == even ]]; then ids=(0 2 4 6 8 10); else ids=(1 3 5 7 9 11); fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec ../continuous_v2/.venv/bin/python -u "$worker" "${args[@]}" --panels "${ids[@]}"
