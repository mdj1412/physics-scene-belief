import json
from pathlib import Path
import numpy as np
from belief import Belief
from bridge import sample,normalize,physics_likelihood,summaries
from engine import tensor
from run import ROOT,save_json

out=Path('results/failure_audit');out.mkdir(exist_ok=True)
for case in [66,68]:
    d=json.loads((ROOT/'main'/f'case_{case:03d}.json').read_text())
    old=json.loads((ROOT/'conditional_failure_audit'/f'case_{case:03d}.json').read_text())
    bank=np.load(ROOT/'main'/f'factual_{case:03d}.npz');x0=normalize(tensor(bank['theta']),tensor(bank['rounding']))
    yy=tensor([r['observation'] for r in old['rows']]);model=Belief(d['family'])
    records=[]
    for method in ['bridge','full']:
        for rep in range(2):
            x,z,stat=sample(physics_likelihood(model,d['observations']['factual'],'view',yy),len(yy),4096,method,2900000+case*10+rep+(0 if method=='bridge' else 2000),initial=x0,moves=18)
            summary=summaries(x,d['family'])
            records.append(dict(method=method,rep=rep,stat=stat,summary={k:v.tolist() for k,v in summary.items()}))
            print(json.dumps(dict(case=case,method=method,rep=rep,seconds=stat['seconds'])),flush=True)
    save_json(out/f'case_{case:03d}.json',dict(case=case,hypotheses=[r['hypothesis'] for r in old['rows']],records=records))
