#!/usr/bin/env bash
set -euo pipefail
cd /scratch2/mdj1412/temporarily_dir/vlm-playground/physics-scene-belief
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs results
.venv/bin/python -m unittest -v test_experiment 2>&1 | tee logs/tests.log
.venv/bin/python physics.py --families 60 --out data/pilot 2>&1 | tee logs/physics.log
.venv/bin/python experiment.py --families 60 --repeats 32 --order 32 2>&1 | tee logs/experiment.log
