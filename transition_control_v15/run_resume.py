"""Operational restart only; no acquisition, data, or statistical settings changed."""
import os,time,platform,subprocess
from common import *
require()
import torch
torch.set_num_threads(2)
import generate,validate,run_policy,evaluate,benchmark
write(R/f'environment_{int(time.time())}.json',dict(job=os.environ['SLURM_JOB_ID'],host=platform.node(),python=platform.python_version(),torch=torch.__version__,gpu=subprocess.check_output(['nvidia-smi'],text=True),time=time.time()))
if not (R/'data/test/private.npz').exists():generate.generate(8,4000101,'pilot');generate.generate()
if not (R/'validation.json').exists():validate.run()
if not (R/'policy_done.json').exists():run_policy.run()
if not (R/'evaluation_done.json').exists():evaluate.run()
if not (R/'timing_gpu_done.json').exists():benchmark.run()
write(R/'run_done.json',dict(time=time.time()));print('GPU phases completed',flush=True)
