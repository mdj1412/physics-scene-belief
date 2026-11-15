"""Integrity checks and SHA256 manifest; run after all experiment artifacts finish."""
import os,ast,json,time,hashlib
from pathlib import Path
import numpy as np

assert os.environ.get('SLURM_JOB_ID')
R=Path('results')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks={}
for snap in ['source_main.json','source_baselines.json']:
    for p,h in read(R/snap).items():assert sha(p)==h,(snap,p)
checks['original_source_snapshots_unchanged']=True
for p in (R/'freezes').glob('*.json'):
    d=read(p);assert sha(R/'policies'/f'{p.stem}.npz')==d['policy_sha256']
    assert sha(R/'data/test/public.npz')==d['public_sha256']
checks['frozen_policies_verified']=len(list((R/'freezes').glob('*.json')))
for name in ['compact.py','run_compact.py','common.py','physics.py']:
    mods=[]
    for n in ast.walk(ast.parse(Path(name).read_text())):
        if isinstance(n,ast.Import):mods.extend(a.name for a in n.names)
        elif isinstance(n,ast.ImportFrom):mods.append(n.module)
    assert not set(mods)&{'bank','run_baselines','benchmark'}
f=read(R/'compact_policy_freeze_done.json');c=read(R/'compact_stage_done.json');b=read(R/'reference_import.json')
assert f['no_bank_import'] and c['no_bank_import'] and f['time']<c['time']<b['time']
checks['standalone_compact_precedes_reference_import']=True
truth=np.load(R/'data/test/private.npz');pub=np.load(R/'data/test/public.npz')['factual']
th=truth['theta'];mask=th[:,0,1]==0
assert np.array_equal(th[:,0],th[:,2]) and np.array_equal(th[:,0,1:],th[:,1,1:])
assert np.array_equal(pub[mask,0],pub[mask,2])
for k in ['target','obs1','obs2','obs3']:assert np.array_equal(truth[k][mask,0],truth[k][mask,2]),k
checks['contact_negative_control_worlds']=int(mask.sum())
checks['contact_changed_worlds']=int((~mask).sum())
sel=read(R/'development.json')['selected']['name'];target=truth['target'].reshape(-1,2)
summary=read(R/'analysis/summary.json');losses=np.load(R/'analysis/world_losses.npz')
for p in (R/'evaluations').glob('*.npz'):
    d=np.load(p);pred=d['predictive'];assert pred.shape==(768,4,16,2) and np.isfinite(pred).all()
    a=np.load(R/'policies'/f'{p.stem}.npz')['actions']
    v=(.5*((pred-target[:,None,None,:])**2).sum(-1))[np.arange(768),a].mean(1).reshape(256,3)
    assert np.allclose(v,losses[p.stem],atol=1e-14,rtol=0)
    for j,key in enumerate(summary['conditions']):assert abs(v[:,j].mean()-summary['conditions_results'][key]['means'][p.stem])<1e-14
checks['evaluation_arrays_and_report_means_verified']=True
for p in (R/'policies').glob('compact*.npz'):
    z=np.load(p);assert np.linalg.eigvalsh(z['cov']).min()>0
    assert abs(np.exp(z['logp']).sum(1)-1).max()<1e-12
checks['all_saved_compact_factual_covariances_psd']=True
for v in read(R/'linear_validation.json'):
    assert max(v['mean_error'],v['cov_error'],v['probability_error'])<1e-12
checks['closed_form_linear_gaussian_validation']=True
for p in ['compact_worker_exit.json','baseline_worker_exit.json']:
    v=read(R/p);assert v['code']==0 and v['foreign']==[]
checks['gpu_workers_completed_without_foreign_processes']=True
cpu=read(R/'benchmark_cpu.json');assert read(R/'benchmark_cpu_protocol.json')['post_primary_supplement']
checks['max_cpu_gpu_prediction_difference']=max(v['max_prediction_difference_from_gpu'] for v in cpu.values())
assert checks['max_cpu_gpu_prediction_difference']<1e-10
checks['completed']=time.time()
(R/'integrity.json').write_text(json.dumps(checks,indent=2))
paths=[p for p in Path('.').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='MANIFEST.json']
manifest={str(p):sha(p) for p in sorted(paths)}
Path('MANIFEST.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(dict(checks=checks,manifest_files=len(manifest)),indent=2))
