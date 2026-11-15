import hashlib,json,time
from pathlib import Path
root=Path('.');s=json.loads((root/'results/analysis/summary.json').read_text());assert s['integrity']['passed']
state=json.loads((root/'results/resource_state.json').read_text());assert not state['active_jobs'];assert all(x['code']==0 for x in state['worker_exits'])
for name,h in json.loads((root/'source_hashes.json').read_text()).items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
assert (root/'REPORT_KO.md').exists()
files=[]
for p in sorted(root.rglob('*')):
 if not p.is_file() or '__pycache__' in p.parts or p.name=='server_run_manifest.json' or p.name.endswith('.lock'):continue
 files.append(dict(path=p.as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
m=dict(created=time.time(),files=files,file_count=len(files),total_bytes=sum(p['bytes'] for p in files));path=root/'server_run_manifest.json';path.write_text(json.dumps(m,indent=2));print(json.dumps(dict(file_count=m['file_count'],total_bytes=m['total_bytes'],manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
