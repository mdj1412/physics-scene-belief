"""Read-only Slurm/GPU checks after inference; never cancel user allocations."""
import datetime,json,subprocess
from pathlib import Path

root=Path(__file__).resolve().parent
state=json.loads((root/'results/resource_state.json').read_text())
assert state['status']=='inference_complete' and not state['active_jobs']
checks={}
for job,pid in sorted(state['worker_pids'].items()):
    exit_record=next(x for x in reversed(state['worker_exits']) if x['job']==job and x['pid']==pid)
    assert exit_record['code']==0 and not exit_record['foreign']
    prefix=['srun','--jobid='+job,'--overlap','--exact','-N1','-n1','-c1','--gres=gpu:1','--time=00:01:00','nvidia-smi']
    def query(fields):
        return subprocess.run(prefix+[fields,'--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=45).stdout.strip()
    processes=query('--query-compute-apps=pid,process_name,used_memory')
    assert all(int(line.split(',')[0])!=pid for line in processes.splitlines() if line.strip()),(job,pid,'worker still has a GPU context')
    checks[job]=dict(worker_pid=pid,worker_exit_code=0,compute_processes=processes,gpu=query('--query-gpu=uuid,memory.used,utilization.gpu'))
result=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='our_workers_exited',allocations_cancelled=False,checks=checks)
(root/'results/validation/final_resource_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
