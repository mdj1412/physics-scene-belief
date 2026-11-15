import os,json,time
import torch
from run import acquire,freeze,evaluate,audit,R,write,sha
from pathlib import Path
from rules import fit,apply
assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(2)
started=time.time()
write(R/'source_main.json',{p.name:sha(p) for p in Path('.').glob('*') if p.is_file()})
for split in ['train','validation']:acquire(split,41)
fit()
for split in ['test','misspec']:
    acquire(split,41);apply(split);freeze(split)
for split in ['test','misspec']:evaluate(split,61)
audit(61,1024,32)
write(R/'pipeline_done.json',dict(started=started,ended=time.time(),seconds=time.time()-started))
print('DONE',time.time()-started,flush=True)
