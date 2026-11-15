"""One-shot continuation in this active experiment, not a scheduled automation."""
import json,subprocess,sys,time,os
from pathlib import Path
end=time.monotonic()+3600
while True:
 state=json.loads(Path('results/resource_state.json').read_text())
 if state.get('status')=='worker_error':raise RuntimeError('Primary worker failed; inspect before proceeding')
 done=all((Path('results')/p/f'case_{c:03d}'/'complete.json').exists() for p,cs in [('evaluation',range(64)),('reference',[0,1,2,3,8,9,10,11]),('decoder_audit',range(4))] for c in cs)
 if done and not state['active_jobs']:break
 if time.monotonic()>end:raise TimeoutError('Primary completion not observed')
 time.sleep(10)
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
print('PRIMARY_COMPLETE_START_NUISANCE',flush=True)
subprocess.run(['srun','--jobid=2680491','--overlap','--exact','-N1','-n1','-c1','--gres=gpu:1',sys.executable,'-u','resume_worker.py','nuisance.py'],env=env,check=True)
print('POSTRUN_COMPLETE',flush=True)
