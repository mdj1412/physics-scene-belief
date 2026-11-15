import json
from pathlib import Path
r=Path('results');phases=['worlds','acquisition','evaluation','reference','decoder_audit']
counts={p:len(list((r/p).glob('case_*/complete.json'))) for p in phases}
state=json.loads((r/'resource_state.json').read_text()) if (r/'resource_state.json').exists() else {}
print(json.dumps(dict(counts=counts,status=state.get('status'),active_jobs=state.get('active_jobs'),last_error=state.get('last_error'))))
for log in sorted(r.glob('pipeline*.log')):
 lines=log.read_text().splitlines()
 if lines:print(log.name+': '+lines[-1])
