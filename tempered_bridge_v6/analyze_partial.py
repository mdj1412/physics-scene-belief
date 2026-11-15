"""Descriptive numerical diagnostics only for fully completed outer128 worlds."""
import json,collections
from pathlib import Path
import numpy as np
from analyze import load,diagnostics,stability

folders=sorted(Path('results/outer128').glob('case_*/complete.json'));rows=[]
for p in folders:
    case=int(p.parent.name.split('_')[-1]);meta=json.loads(p.read_text())
    for q in ['high','pixel_only']:
        rr=dict(case=case,family=meta['family'],quality=q)
        for label,folder in [('before',Path('results/main')/p.parent.name),('after',p.parent)]:
            b=load(folder,q,'bridge');f=load(folder,q,'full')
            rr[label]=dict(comparison=diagnostics(b,f),bridge=stability(b,90000+case),full=stability(f,90000+case))
        rows.append(rr)
summary={}
for q in ['high','pixel_only']:
    ds=[r for r in rows if r['quality']==q];summary[q]={}
    for stage in ['before','after']:
        mask=np.array([r[stage]['comparison']['active'] for r in ds])
        dd=dict(active_queries=int(mask.sum()),method_agreement=float(np.array([r[stage]['comparison']['agreement'] for r in ds])[mask].mean()))
        for method in ['bridge','full']:
            dd[method+'_practical_confidence']=float(np.array([r[stage][method]['practical_confidence'] for r in ds])[mask].mean())
        summary[q][stage]=dd
result=dict(status='partial, not population-wide stabilized-policy evaluation',completed=len(folders),planned=57,
    cases=sorted({r['case'] for r in rows}),families=dict(collections.Counter(json.loads(p.read_text())['family'] for p in folders)),summary=summary,rows=rows)
Path('results/analysis/partial_refinement.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
