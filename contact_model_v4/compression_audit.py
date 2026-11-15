"""Audit whether contact branches were actually lost, using saved posteriors."""
import json
from pathlib import Path
import numpy as np
from engine import tensor
from inference import predict,compress

rows=[]
for path in sorted(Path('results/main').glob('case_*.json')):
    d=json.loads(path.read_text());case=d['case']
    p=np.load(Path('results/main')/f'posterior_{case:03d}.npz')
    atoms=tuple(tensor(p[k]) for k in ['theta','rounding','weight']);pred=predict(atoms,d['family'])
    sig=pred['code'].T.cpu().numpy();unique,counts=np.unique(sig,axis=0,return_counts=True)
    masses={tuple(k):v/len(sig) for k,v in zip(unique,counts)}
    row=dict(case=case,family=d['family'],split=d['split'],branches=len(unique),records=[])
    for size in [64,128]:
        for method in ['systematic','branch']:
            for rep in range(3):
                _,small=compress(atoms,pred,size,method,50100+case*100+10+rep)
                retained=set(map(tuple,small['code'].T.cpu().numpy()))
                missing=set(masses)-retained
                row['records'].append(dict(size=size,method=method,replicate=rep,missing_branches=len(missing),
                    missing_mass=float(sum(masses[k] for k in missing)),fallback=method=='branch' and len(unique)>size))
    rows.append(row)
out=Path('results/audit/compression_branches.json');out.write_text(json.dumps(rows,indent=2))
test=[r for r in rows if r['split']=='test']
print('test branches min/max',min(r['branches'] for r in test),max(r['branches'] for r in test))
for size in [64,128]:
    for method in ['systematic','branch']:
        rs=[r for d in test for r in d['records'] if r['size']==size and r['method']==method]
        print(size,method,'missing mass',np.mean([r['missing_mass'] for r in rs]),'fallback',sum(r['fallback'] for r in rs))
