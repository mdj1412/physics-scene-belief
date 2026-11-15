import os,json,time,hashlib,ast
from pathlib import Path
import numpy as np
assert os.environ.get('SLURM_JOB_ID')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
R=Path('results');checks={}
for snapshot in ['source_compact.json','source_banks.json']:
    for name,h in read(R/snapshot).items():assert sha(name)==h,(snapshot,name)
checks['source_snapshots_unchanged']=True
for p in (R/'freezes').glob('*.json'):
    d=read(p);assert sha(R/'policies'/f'{p.stem}.npz')==d['policy'] and sha(R/'data/test/public.npz')==d['public']
checks['policies_verified']=len(list((R/'freezes').glob('*.json')))
assert read(R/'compact_policy_freeze.json')['time']<read(R/'compact_done.json')['time']<read(R/'bank_import.json')['time']
assert read(R/'compact_done.json')['no_bank_import'];checks['standalone_compact_before_banks']=True
for name in ['compact.py','flexible.py','acquisition.py','run_compact.py']:
    for n in ast.walk(ast.parse(Path(name).read_text())):
        if isinstance(n,ast.Import):assert 'bank' not in [a.name for a in n.names]
        if isinstance(n,ast.ImportFrom):assert n.module!='bank'
f=read(R/'decision_freeze.json');assert sha(R/'decision_predictions.npz')==f['predictions'] and sha(R/'decision_actions.npz')==f['actions'] and not f['targets_read']
truth=np.load(R/'data/test/private.npz');target=truth['target'].reshape(-1,2);mask=truth['theta'][:,1]==0;pub=np.load(R/'data/test/public.npz')['factual'];pred=np.load(R/'decision_predictions.npz');loss=np.load(R/'world_losses.npz')
for k,v in pred.items():
    assert v.shape==(1536,16,2) and np.isfinite(v).all()
    actual=(.5*((v-target[:,None,:])**2).sum(-1)).mean(1).reshape(256,6);assert np.allclose(actual,loss[k],atol=1e-14,rtol=0)
for j in range(1,6):
    assert np.array_equal(pub[mask,0],pub[mask,j])
    for key in ['target','obs1','obs2','obs3']:assert np.array_equal(truth[key][mask,0],truth[key][mask,j])
checks['decision_arrays_and_negative_controls_verified']=True;checks['negative_control_worlds']=int(mask.sum())
for p in ['compact_worker_exit.json','baseline_worker_exit.json','diagnostics_worker_exit.json']:
    v=read(R/p);assert v['code']==0 and v['foreign']==[]
checks['all_gpu_workers_completed_cleanly']=True
for filename in ['timing_gpu_done.json','timing_cpu_done.json']:
    d=read(R/filename);assert sha('benchmark.py')==d['source'] and sha('runtime.py')==d['runtime_source']
checks['online_timing_code_verified']=True
root=Path('../mismatch_belief_v13');manifest=read(root/'MANIFEST.json')
for p,h in manifest.items():assert sha(root/p)==h,('v13 modified',p)
checks['v13_sealed_files_unchanged']=len(manifest);checks['completed']=time.time()
Path('INTEGRITY.json').write_text(json.dumps(checks,indent=2))
paths=[p for p in Path('.').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='MANIFEST.json']
Path('MANIFEST.json').write_text(json.dumps({str(p):sha(p) for p in sorted(paths)},indent=2));print(dict(checks=checks,files=len(paths)))
