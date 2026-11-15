"""Atomic claims and barriers; independent allocated GPU workers."""
import fcntl,json,os,socket,subprocess,sys,time
from pathlib import Path
from config import N_WORLDS,AUDIT,RASTER
from experiment import ROOT,make_world,run_group,save_json

def phase(name,cases,h=0,n=0,moves=0,reps=0,method='bridge'):
    out=ROOT/name;out.mkdir(exist_ok=True);idle=time.time()
    while not all((out/f'case_{c:03d}'/'complete.json').exists() for c in cases):
        claimed=False
        for family in range(3):
            group=[]
            for c in [c for c in cases if c//16==family]:
                if (out/f'case_{c:03d}'/'complete.json').exists():continue
                lock=out/f'claim_{c:03d}'
                try:lock.mkdir()
                except FileExistsError:continue
                save_json(lock/'worker.json',dict(pid=os.getpid(),job=os.environ.get('SLURM_JOB_ID'),host=socket.gethostname()))
                group.append(c);claimed=True
                if name=='worlds':make_world(c);group=[]
                elif len(group)==4:run_group(group,name,h,n,moves,reps,method);group=[]
            if group:run_group(group,name,h,n,moves,reps,method)
        if claimed:idle=time.time()
        else:
            if time.time()-idle>18000:raise RuntimeError('phase barrier timeout '+name)
            time.sleep(10)

def plan(final=False):
    root=ROOT/'plans';root.mkdir(exist_ok=True);name='final_policy' if final else 'refine512';ready=root/(name+'.ready')
    with (root/(name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not ready.exists():
            subprocess.run([sys.executable,'decision.py']+(['--final'] if final else []),check=True)
            ready.write_text('complete')
    return json.loads((root/(name+'.json')).read_text())

phase('worlds',list(range(N_WORLDS)))
phase('acq256',list(range(N_WORLDS)),256,1024,12,2)
p=plan();phase('acq512',p['cases'],512,1024,12,2);plan(True)
phase('full_audit',AUDIT,128,2048,18,2,'full')
phase('eval',list(range(N_WORLDS)),32,2048,18,1,'full')
phase('decoder_audit',AUDIT,32,4096,18,1,'full')
phase('raster_eval',RASTER,8,2048,18,1,'full')
print('PIPELINE_COMPLETE',flush=True)
