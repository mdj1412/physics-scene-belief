"""Explicit resume with an occupied-GPU guard throughout this compute step."""
import fcntl,json,os,signal,socket,subprocess,sys,time
from pathlib import Path

job=os.environ.get('SLURM_JOB_ID')
if not job:raise SystemExit('Run inside an allocated Slurm compute step, not the login node.')
check=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True)
if check.stdout.strip():raise SystemExit('STOP: GPU already has compute processes.\n'+check.stdout)
r=Path('results');stamp=int(time.time());log=f'pipeline_{job}_resume_{stamp}.log';command=sys.argv[1:] or ['pipeline.py']
def update_state(fn):
    with (r/'.resource_state.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=r/'resource_state.json';state=json.loads(path.read_text()) if path.exists() else {};fn(state)
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(path)
with (r/log).open('a') as stream:
    child=subprocess.Popen([sys.executable,'-u']+command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    def started(s):
        s['active_jobs']=sorted(set(s.get('active_jobs',[])+[int(job)]))
        s.setdefault('worker_pids',{})[job]=child.pid;s.setdefault('worker_logs',{})[job]=log
        s.setdefault('worker_steps',{})[job]=job+'.'+os.environ.get('SLURM_STEP_ID','unknown')
        s['status']='running';s.setdefault('resumes',[]).append(dict(job=job,pid=child.pid,host=socket.gethostname(),time=stamp,log=log,guard_seconds=15,command=command))
    update_state(started);print(json.dumps(dict(job=job,pid=child.pid,log=str(r/log))),flush=True)
    foreign=[]
    while child.poll() is None:
        time.sleep(15)
        if child.poll() is not None:break
        check=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True)
        if check.returncode:
            foreign=['GPU monitoring failed: '+check.stderr.strip()]
        else:
            foreign=[]
            for line in check.stdout.splitlines():
                if not line.strip():continue
                pid=int(line.split(',')[0])
                try:owned=os.getpgid(pid)==child.pid
                except ProcessLookupError:continue
                if not owned:foreign.append(line)
        if foreign:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            break
    code=child.wait()
    if foreign:
        archive=r/'interrupted_claims'/f'guard_{job}_{int(time.time())}';archive.mkdir(parents=True,exist_ok=True)
        for phase in ['worlds','acq256','acq512','full_audit','eval','decoder_audit','raster_eval']:
            for claim in (r/phase).glob('claim_*'):
                case=int(claim.name.split('_')[-1])
                if (r/phase/f'case_{case:03d}'/'complete.json').exists():continue
                try:owner=json.loads((claim/'worker.json').read_text())
                except (FileNotFoundError,json.JSONDecodeError):continue
                if owner['pid']==child.pid:claim.rename(archive/(phase+'_'+claim.name))
    def ended(s):
        s['active_jobs']=[j for j in s.get('active_jobs',[]) if int(j)!=int(job)]
        s.setdefault('worker_exits',[]).append(dict(job=job,pid=child.pid,code=code,time=time.time(),foreign=foreign))
        if foreign:s['status']='running' if s['active_jobs'] else 'paused_all_gpus_external_training'
        elif code==0:s['status']='running' if s['active_jobs'] else ('inference_complete' if command==['pipeline.py'] else 'validation_complete')
        else:s['status']='worker_error';s['last_error']=dict(job=job,code=code,log=log)
    update_state(ended)
print(json.dumps(dict(job=job,code=code,foreign=foreign)),flush=True)
raise SystemExit(75 if foreign else code)
