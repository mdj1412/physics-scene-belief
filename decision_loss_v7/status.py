import json
from pathlib import Path
r=Path('results')
s=json.loads((r/'resource_state.json').read_text()) if (r/'resource_state.json').exists() else {}
print(json.dumps({k:s.get(k) for k in ['status','active_jobs','last_error','worker_exits']},ensure_ascii=False))
for phase in ['worlds','acq256','acq512','full_audit','eval','decoder_audit','raster_eval']:
 p=r/phase
 print(phase,'complete',len(list(p.glob('case_*/complete.json'))),'batches',len(list(p.glob('case_*/*_0.json'))))
for name in ['refine512','final_policy']:
 p=r/'plans'/(name+'.json')
 if p.exists():
  d=json.loads(p.read_text());print(name,len(d['cases']),d['cases'])
for job in s.get('active_jobs',[]):
 p=r/s['worker_logs'][str(job)]
 print('LOG',job,'\n'+'\n'.join(p.read_text().splitlines()[-2:]))
