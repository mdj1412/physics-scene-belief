"""Read-only artifact hashing after inference/report completion."""
import hashlib,json,platform,time
from pathlib import Path
root=Path('.');state=json.loads((root/'results/resource_state.json').read_text())
assert state['status']=='inference_complete' and not state['active_jobs']
assert all(x['code']==0 and not x['foreign'] for x in state['worker_exits'])
assert json.loads((root/'results/analysis/integrity.json').read_text())['passed']
assert (root/'REPORT_KO.md').exists()
for name,h in json.loads((root/'inference_source_hashes.json').read_text()).items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,('inference source changed',name)
rows=[]
for p in sorted(root.rglob('*')):
 if not p.is_file() or '__pycache__' in p.parts or any(x.startswith('claim_') for x in p.parts) or p.name in ['server_run_manifest.json'] or p.name.endswith('.lock'):continue
 data=p.read_bytes();rows.append(dict(path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
d=dict(sealed_at_unix=time.time(),root=str(root.resolve()),host=platform.node(),files=rows,total_bytes=sum(x['bytes'] for x in rows),file_count=len(rows))
p=root/'server_run_manifest.json';p.write_text(json.dumps(d,indent=2));print(json.dumps(dict(manifest=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),file_count=len(rows),total_bytes=d['total_bytes'])))
