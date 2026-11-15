import time
import torch
from common import *
require();torch.set_num_threads(2)
from generate import generate
generate(8,6000101,'pilot');generate(512,6100101,'test');print('Generated fresh worlds',flush=True)
from validate import run as validate
validate()
from run_policy import run as policy
policy()
from evaluate import run as evaluate
evaluate()
from timing import run as timing
timing()
write(R/'run_done.json',dict(time=time.time()))
