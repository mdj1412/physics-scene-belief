import os,time,platform,subprocess
from common import *
require()
import torch
torch.set_num_threads(2)
write(R/'environment.json',dict(job=os.environ['SLURM_JOB_ID'],host=platform.node(),python=platform.python_version(),torch=torch.__version__,gpu=subprocess.check_output(['nvidia-smi'],text=True),time=time.time()))
import generate,validate,run_policy,evaluate,benchmark
generate.generate(8,4000101,'pilot');generate.generate();validate.run();run_policy.run();evaluate.run();benchmark.run()
write(R/'run_done.json',dict(time=time.time()));print('GPU phases completed',flush=True)
