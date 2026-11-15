import os
import torch
from run import prepare,pilot
assert os.environ.get('SLURM_JOB_ID')
torch.set_num_threads(2)
prepare()
pilot()
