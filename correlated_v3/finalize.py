import csv,hashlib,json,math,platform
from pathlib import Path
from datetime import datetime,timezone
from run import case_spec

p=Path(__file__).resolve().parent
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def finite(value):
    if isinstance(value,float):return math.isfinite(value)
    if isinstance(value,dict):return all(finite(x) for x in value.values())
    if isinstance(value,list):return all(finite(x) for x in value)
    return True

checks={};runtimes=[json.loads(x.read_text()) for x in sorted((p/'results/main').glob('runtime_*.json'))]
assert len(runtimes)==4
checks['jobs']=[x['job'] for x in runtimes]
checks['four_distinct_gpus']=len(set(x['gpu'] for x in runtimes))==4
assert checks['four_distinct_gpus']
checks['main_core_source_verified']=all(x['source_sha256']['core.py']==digest(p/'provenance/core_at_main_run.py') for x in runtimes)
checks['main_run_source_verified']=all(x['source_sha256']['run.py']==digest(p/'run.py') for x in runtimes)
checks['main_design_source_verified']=all(x['source_sha256']['design.py']==digest(p/'design.py') for x in runtimes)
assert checks['main_core_source_verified'] and checks['main_run_source_verified'] and checks['main_design_source_verified']
count=0;families=set()
for path in (p/'results/main').glob('metrics_*.csv'):
    for row in csv.DictReader(path.open()):
        assert all(math.isfinite(float(v)) for k,v in row.items() if k not in ['method','split'])
        assert float(row['regret'])>=-1e-12
        families.add(int(row['family']));count+=1
assert count==110592 and len(families)==96
checks['metric_rows']=count;checks['families']=len(families)
for path in (p/'results/analysis').glob('*.json'):
    assert finite(json.loads(path.read_text())),path
checks['analysis_json_finite']=True
repair=json.loads((p/'results/main/diagnostics_repair.json').read_text())
assert repair['non_mi_max_change']==0
checks['diagnostics_repair']=repair
(p/'case_specs.json').write_text(json.dumps([case_spec(i) for i in range(96)],indent=2))
checks['date_utc']=datetime.now(timezone.utc).isoformat();checks['host']=platform.node()
(p/'final_verification.json').write_text(json.dumps(checks,indent=2))
files={str(f.relative_to(p)):dict(bytes=f.stat().st_size,sha256=digest(f)) for f in p.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.name!='artifact_manifest.json'}
(p/'artifact_manifest.json').write_text(json.dumps(dict(created_utc=checks['date_utc'],files=files),indent=2))
print(json.dumps(checks,indent=2))
