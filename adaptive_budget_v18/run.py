import torch,time
from common import *
require();torch.set_num_threads(2)
from generate import generate
generate(8,7000101,'pilot');generate(512,7100101,'test');print('Generated',flush=True)
from validate import run as validate
validate()
from run_policy import run as policy
policy()
from evaluate import run as evaluate
evaluate()
from timing import run as timing
timing()
write(R/'run_done.json',dict(time=time.time()))
