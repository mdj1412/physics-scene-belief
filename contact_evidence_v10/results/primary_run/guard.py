"""Only stop our own worker if another GPU workload appears."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
if not os.environ.get('SLURM_JOB_ID'):raise SystemExit('No allocation')
def processes():
    r=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True,capture_output=True,check=True)
    return [x for x in r.stdout.splitlines() if x.strip()]
existing=processes()
if existing:raise SystemExit('GPU occupied: '+str(existing))
Path('results').mkdir(exist_ok=True)
log=Path('results')/('worker_'+str(int(time.time()))+'.log')
with log.open('w') as f:
    child=subprocess.Popen([sys.executable,'-u','run.py']+sys.argv[1:],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    Path('results/worker.json').write_text(json.dumps(dict(pid=child.pid,job=os.environ['SLURM_JOB_ID'],log=str(log),started=time.time())))
    foreign=[]
    while child.poll() is None:
        time.sleep(10)
        if child.poll() is not None:break
        for line in processes():
            try:owned=os.getpgid(int(line.split(',')[0]))==child.pid
            except ProcessLookupError:continue
            if not owned:foreign.append(line)
        if foreign:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
            break
    rc=child.wait()
Path('results/worker_exit.json').write_text(json.dumps(dict(code=rc,foreign=foreign,time=time.time())))
print(json.dumps(dict(code=rc,foreign=foreign,log=str(log))),flush=True)
sys.exit(rc)
