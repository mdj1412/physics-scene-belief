import os,sys,json,hashlib,time,ast
from pathlib import Path
import numpy as np
assert os.environ.get('SLURM_JOB_ID')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks={}
for folder in ['results','results_actionwise']:
    r=Path(folder)
    for name,h in read(r/'source_freeze.json').items():assert sha(name)==h,(folder,name)
    for p in (r/'freezes').glob('*.json'):
        d=read(p);assert sha(r/'policies'/f'{p.stem}.npz')==d['policy'];assert sha(r/'data/test/public.npz')==d['public'];assert sha(r/'thresholds.json')==d['thresholds']
    assert read(r/'compact_policy_freeze.json')['time']<read(r/'compact_done.json')['time']<read(r/'bank_import.json')['time']
    assert read(r/'compact_done.json')['no_bank_import']
    f=read(r/'decision_freeze.json');assert sha(r/'decision_predictions.npz')==f['predictions'] and not f['targets_read']
    pred=np.load(r/'decision_predictions.npz');truth=np.load(r/'data/test/private.npz');target=truth['target'].reshape(-1,2);loss=np.load(r/'world_losses.npz')
    for nm,v in pred.items():
        assert v.shape==(1536,16,2) and np.isfinite(v).all()
        vv=(.5*((v-target[:,None,:])**2).sum(-1)).mean(1).reshape(256,6);assert np.allclose(vv,loss[nm],atol=1e-14,rtol=0)
    mask=truth['theta'][:,1]==0;pub=np.load(r/'data/test/public.npz')['factual']
    for a in range(1,6):
        assert np.array_equal(pub[mask,0],pub[mask,a])
        for key in ['target','obs1','obs2','obs3']:assert np.array_equal(truth[key][mask,0],truth[key][mask,a])
    checks[folder]=dict(source_and_freezes_verified=True,standalone_before_qmc=True,predictions_and_losses_verified=True,negative_controls=int(mask.sum()),workers=[]) 
    workerfiles=['compact_worker_exit.json','baseline_worker_exit.json','branch_worker_exit.json'] if folder=='results' else ['worker_exit.json','timing_worker_exit.json','baseline_timing_worker_exit.json']
    for nm in workerfiles:
        d=read(r/nm);assert d['code']==0 and d['foreign']==[];checks[folder]['workers'].append(nm)
assert read('results/validation.json')['linear_closed_form_max_error']<1e-10
root=Path('../compact_belief_v12');original=read(root/'MANIFEST.json')
for name,h in original.items():assert sha(root/name)==h,('v12 modified',name)
checks['v12_sealed_files_unchanged']=len(original)
# Additional physical consistency check, CPU only, no fitting or policy changes.
import physics
physics.DEV='cpu'
import torch
from worlds import truth_velocity,CONDITIONS
from flexible import velocity as candidate_velocity
torch.set_num_threads(2)
data=np.load('results/data/test/private.npz');maximum=-np.inf
for th,pars in zip(data['theta'],data['parameters']):
    for condition in CONDITIONS:
        for y,vin in [(0,(1,.35)),(0,(1,-.7)),(.025,(1,.9))]:
            v=truth_velocity(physics.t(th),y,vin,condition,pars);maximum=max(maximum,float((v*v).sum())-sum(x*x for x in vin))
assert maximum<1e-12;checks['test_truth_max_energy_excess']=maximum
checks['completed']=time.time();Path('INTEGRITY.json').write_text(json.dumps(checks,indent=2))
files=[p for p in Path('.').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='MANIFEST.json']
Path('MANIFEST.json').write_text(json.dumps({str(p):sha(p) for p in sorted(files)},indent=2))
print(json.dumps(dict(checks=checks,manifest_files=len(files)),indent=2))
