"""Verify policy/result integrity, archive provenance, then hash completed run."""
import hashlib,json,os
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert os.environ.get('SLURM_JOB_ID'),'Verification on server compute node'
r=Path('results');s=json.loads((r/'analysis/summary.json').read_text());f=json.loads((r/'test/freeze.json').read_text())
assert f['count']==128 and len(list((r/'test/evaluation_refined').glob('*.npz')))==128
assert sha(r/'rules.json')==f['rules_sha']
checks=[]
for name,digest in f['policies'].items():
    p=r/'test/policies'/name;assert sha(p)==digest
    pol=json.loads(p.read_text());assert sha(r/'test/public'/name)==pol['public_sha']
    v=np.load(r/'test/evaluation_refined'/name.replace('.json','.npz'))
    assert np.isfinite(v['losses']).all() and v['losses'].shape==(4,16,6)
    choice=pol['choices']['boundary_cf32768']
    checks.append(np.mean([v['losses'][a,:,q].mean() for q,a in enumerate(choice)]))
assert abs(np.mean(checks)-s['means']['boundary_cf32768'])<1e-14
original=json.loads((r/'source_hashes.json').read_text())
for name in ['model.py','run.py']:assert sha(Path(name))==original[name]
# Retain launcher source as it existed during the primary run; only the later
# script-dispatch branch changed to permit refine.py under the same GPU guard.
guard=Path('guard.py').read_text()
block="    args=sys.argv[1:]\n    command=args if args and args[0].endswith('.py') else ['run.py']+args\n    child=subprocess.Popen([sys.executable,'-u']+command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)"
old="    child=subprocess.Popen([sys.executable,'-u','run.py']+sys.argv[1:],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)"
archive=r/'primary_run'/'guard.py';archive.write_text(guard.replace(block,old));assert sha(archive)==original['guard.py']
validation=dict(policies_verified=128,public_data_verified=128,finite_results=128,mean_recomputed=float(np.mean(checks)),
                original_model_run_source_unchanged=True,primary_guard_source_archived=True)
(r/'integrity.json').write_text(json.dumps(validation,indent=2))
files=[]
for p in sorted(Path('.').rglob('*')):
    if p.is_file() and '__pycache__' not in p.parts and p.name not in ['MANIFEST.json','MANIFEST.sha256']:
        files.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
manifest=Path('MANIFEST.json');manifest.write_text(json.dumps(dict(files=files,file_count=len(files),total_bytes=sum(x['bytes'] for x in files)),indent=2))
Path('MANIFEST.sha256').write_text(sha(manifest)+'  MANIFEST.json\n')
print(json.dumps(dict(**validation,file_count=len(files),manifest_sha256=sha(manifest))))
