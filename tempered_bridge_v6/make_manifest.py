"""Create an auditable source/result manifest after all experiment phases finish."""
import datetime,hashlib,json,os,platform,socket
from pathlib import Path

root=Path(__file__).resolve().parent;files=[]
state=json.loads((root/'results/resource_state.json').read_text())
assert state['status']=='inference_complete' and not state['active_jobs'],'Do not seal an active or incomplete run.'
assert len(list((root/'results/main').glob('case_*/complete.json')))==72
assert len(list((root/'results/budget_audit').glob('case_*/complete.json')))==5
for h in [128,256,512]:
    plan=root/'results/adaptive_plans'/f'outer{h}.json'
    if plan.exists():
        assert all((root/f'results/outer{h}'/f'case_{c:03d}/complete.json').exists() for c in json.loads(plan.read_text())['cases']),f'outer{h} incomplete'
old=root.parent/'boundary_belief_v5';sealed=json.loads((old/'server_run_manifest.json').read_text());old_index={r['path']:r['sha256'] for r in sealed['files']}
inputs=[]
patterns=['results/main/case_*.json','results/main/factual_*.npz','results/pixel_only/case_*.json','results/conditional_refined/case_*.json','results/conditional_failure_audit/case_*.json']
paths=[p for pattern in patterns for p in old.glob(pattern)]+[old/'results/calibration'/f'{q}.json' for q in ['high','pixel_only']]+[old/n for n in ['engine.py','belief.py','base_inference.py']]
for p in sorted(set(paths)):
    rel=str(p.relative_to(old));sha=hashlib.sha256(p.read_bytes()).hexdigest();assert old_index.get(rel)==sha,('v5 input differs from sealed manifest',rel)
    inputs.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha,verified_against_v5_manifest=True))
for p in sorted(root.rglob('*')):
    if not p.is_file():continue
    rel=p.relative_to(root)
    if any(part in {'.venv-analysis','__pycache__','mpl_cache'} for part in rel.parts):continue
    if p.name in {'server_run_manifest.json','review_bundle.zip'} or p.suffix=='.tmp':continue
    files.append(dict(path=str(rel),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
complete={str(folder.relative_to(root)):len(list(folder.glob('case_*/complete.json'))) for folder in [root/'results'/n for n in ['main','outer128','outer256','outer512','budget_audit']] if folder.exists()}
result=dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=socket.gethostname(),root=str(root),completion_counts=complete,
    allocations={'2676323':'node56 / GPU-2bb9939f-05dd-8d17-248d-270e7aade000','2676324':'node55 / GPU-d29002d2-3b37-1b3c-24cd-54a0b5820faf','2676322':'node55 / GPU-58f3772e-d982-09e0-d365-c31544173046','2676325':'node55 / GPU-acff86bb-b0ee-515f-5139-502744220e28'},
    main_config=dict(outer=64,particles=1024,moves=12,repeats=2,ess_fraction=.75),resource_state=json.loads((root/'results/resource_state.json').read_text()),inputs=inputs,
    v5_manifest_sha256=hashlib.sha256((old/'server_run_manifest.json').read_bytes()).hexdigest(),files=files)
(root/'server_run_manifest.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(files=len(files),completion_counts=complete,total_bytes=sum(f['bytes'] for f in files))))
