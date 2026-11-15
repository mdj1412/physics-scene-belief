import os,time,platform,subprocess
from common import *
require()
import torch
torch.set_num_threads(2)
import generate,validate,run_policy,evaluate,timing
write(R/'environment.json',dict(time=time.time(),job=os.environ['SLURM_JOB_ID'],host=platform.node(),python=platform.python_version(),torch=torch.__version__,gpu=subprocess.check_output(['nvidia-smi'],text=True)))
generate.generate(8,5000101,'pilot');generate.generate(512,5100101,'test');validate.run();run_policy.run();evaluate.run();timing.run();write(R/'run_done.json',dict(time=time.time()))
