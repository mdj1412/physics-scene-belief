import json,os,socket,subprocess,time
from pathlib import Path
def query(q):return subprocess.run(['nvidia-smi',q,'--format=csv,noheader'],check=True,capture_output=True,text=True).stdout.strip()
d=dict(time=time.time(),host=socket.gethostname(),job=os.environ.get('SLURM_JOB_ID'),gpu=query('--query-gpu=uuid,name,memory.used,utilization.gpu'),compute_processes=query('--query-compute-apps=gpu_uuid,pid,used_memory'))
Path('results/gpu_final.json').write_text(json.dumps(d,indent=2));print(json.dumps(d))
