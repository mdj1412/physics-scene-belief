"""Read-only experiment progress; no simulation or CUDA imports."""
import datetime,json,subprocess,sys
from pathlib import Path
r=Path('results');counts={k:len(list((r/k).glob('case_*/complete.json'))) for k in ['main','outer128','outer256','outer512','budget_audit']}
plans={}
for h in [128,256,512,1024]:
    p=r/'adaptive_plans'/f'outer{h}.json'
    if p.exists():
        try:plans[str(h)]=len(json.loads(p.read_text())['cases'])
        except json.JSONDecodeError:pass
resource=json.loads((r/'resource_state.json').read_text()) if (r/'resource_state.json').exists() else dict(active_jobs=[2676323,2676324])
logs={};done=bool(resource['active_jobs']) or resource.get('status')=='inference_complete';errors=[]
if resource.get('status')=='worker_error':errors.append(resource.get('last_error',{}))
for job in resource['active_jobs']:
    p=r/resource.get('worker_logs',{}).get(str(job),f'pipeline_{job}_grouped.log');s=p.read_text() if p.exists() else ''
    lines=s.splitlines();logs[str(job)]=lines[-1] if lines else 'initializing';done&='PIPELINE_COMPLETE' in s
    if any(t in '\n'.join(lines[-25:]) for t in ['Traceback','Error:','CANCELLED','TIME LIMIT']):errors.append(str(job))
gpu={};foreign={}
if '--gpu' in sys.argv:
    for job in resource['active_jobs']:
        p=subprocess.run(['srun',f'--jobid={job}','--overlap','--exact','-N1','-n1','-c1','--gres=gpu:1','--time=00:01:00','nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True,capture_output=True)
        gpu[str(job)]=p.stdout.strip();own=resource.get('worker_pids',{}).get(str(job))
        if p.returncode:errors.append(p.stderr.strip())
        if own is not None:
            foreign[str(job)]=[line for line in p.stdout.splitlines() if line.strip() and int(line.split(',')[0])!=own]
print(json.dumps(dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),counts=counts,plans=plans,workers=logs,done=done,resource_status=resource.get('status','running'),errors=errors,gpu_processes=gpu,foreign_processes=foreign)))
