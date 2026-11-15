"""Resume missing worlds with an atomic shared work queue on the same two GPUs."""
import argparse,json,os,platform,time
from pathlib import Path
import torch
from run import run_case

p=argparse.ArgumentParser();p.add_argument('--worker',type=int,required=True);a=p.parse_args()
out=Path('results/main');locks=Path('results/queue_locks');locks.mkdir(exist_ok=True)
meta=dict(worker=a.worker,host=platform.node(),job=os.environ.get('SLURM_JOB_ID'),gpu=torch.cuda.get_device_name(),start=time.time(),completed=[])
runtime=out/f'runtime_queue_{a.worker}.json'
runtime.write_text(json.dumps(meta,indent=2))
while True:
    pending=[i for i in range(72) if not (out/f'case_{i:03d}.json').exists()]
    if not pending:break
    worked=False
    for case in pending:
        lock=locks/f'case_{case:03d}'
        try:lock.mkdir()
        except FileExistsError:continue
        try:
            if (out/f'case_{case:03d}.json').exists():continue
            meta['active_case']=case;runtime.write_text(json.dumps(meta,indent=2))
            run_case(case,out)
            meta['completed'].append(case);meta.pop('active_case',None);runtime.write_text(json.dumps(meta,indent=2));worked=True
        finally:lock.rmdir()
        break
    if not worked:time.sleep(2)
meta.update(end=time.time(),peak_memory=torch.cuda.max_memory_allocated());runtime.write_text(json.dumps(meta,indent=2));print('queue complete',a.worker,flush=True)
