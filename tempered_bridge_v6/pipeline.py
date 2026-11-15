"""Finish outer refinement and independent budget audit after main barrier.

Two allocated workers share atomic case claims. All decisions exclude outcomes.
"""
import json,os,socket,subprocess,sys,time
from pathlib import Path
from run import run_case,save_json
from grouped import run_group

def wait_for(predicate):
    start=time.time()
    while not predicate():
        if time.time()-start>18000:raise RuntimeError('pipeline barrier timeout')
        time.sleep(10)

def phase(out,cases,h,n=1024,moves=12):
    out=Path(out);out.mkdir(exist_ok=True)
    idle_start=time.time()
    while not all((out/f'case_{c:03d}'/'complete.json').exists() for c in cases):
        claimed=False
        for family in range(3):
            pending=[]
            for case in [c for c in cases if c//24==family]:
                if (out/f'case_{case:03d}'/'complete.json').exists():continue
                lock=out/f'claim_{case:03d}'
                try:lock.mkdir()
                except FileExistsError:continue
                save_json(lock/'worker.json',dict(pid=os.getpid(),host=socket.gethostname(),job=os.environ.get('SLURM_JOB_ID')))
                pending.append(case);claimed=True
                if len(pending)==4:run_group(pending,out,h,n,moves);pending=[]
            if pending:run_group(pending,out,h,n,moves)
        if claimed:idle_start=time.time()
        else:
            if time.time()-idle_start>18000:raise RuntimeError('pipeline barrier timeout')
            time.sleep(10)
        # Re-scan released claims if a different worker yields its GPU.

wait_for(lambda:len(list(Path('results/main').glob('case_*/complete.json')))==72)
previous='results/main';planroot=Path('results/adaptive_plans');planroot.mkdir(exist_ok=True)
for h in [128,256,512,1024]:
    ready=planroot/f'outer{h}.ready';lock=planroot/f'outer{h}.lock'
    try:
        lock.mkdir()
        subprocess.run([sys.executable,'adaptive_plan.py','--previous',previous,'--next-h',str(h)],check=True)
        ready.write_text('complete')
    except FileExistsError:pass
    wait_for(ready.exists)
    plan=json.loads((planroot/f'outer{h}.json').read_text());cases=plan['cases']
    print(json.dumps(dict(phase='plan',h=h,cases=cases)),flush=True)
    if h==1024 or not cases:break
    phase(f'results/outer{h}',cases,h)
    previous=f'results/outer{h}'
phase('results/budget_audit',[1,25,49,66,68],128,2048,18)
print('PIPELINE_COMPLETE',flush=True)
