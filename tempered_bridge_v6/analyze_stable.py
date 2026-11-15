"""Posthoc conditional policy audit; never used to choose budgets or policies."""
import json
from pathlib import Path
import numpy as np
from analyze import oldcase, scores, ci

root=Path('results')
summary=json.loads((root/'analysis_adaptive/summary.json').read_text())
policies=json.loads((root/'analysis_adaptive/per_case.json').read_text())
result={}
names=['bridge','full','value_rule','type_rule','always_view','random','snis']
for quality,d in summary.items():
    groups={name:[] for name in ['practically_stable','exact_decisive','unresolved_active']}
    for row in policies[quality]:
        case=row['case'];dg=next(x for x in d['diagnostics'] if x['case']==case)
        st={m:next(x for x in d['stability'][m] if x['case']==case) for m in ['bridge','full']}
        full_gain=np.array(dg['full_gain']);active=np.array(dg['active'])
        stable=active & np.array(st['bridge']['practical_confidence']) & np.array(st['full']['practical_confidence'])
        stable &= np.array(dg['regret'])<=np.maximum(1e-4,.05*full_gain.max(1))
        decisive=stable & np.array(st['bridge']['decisive']) & np.array(st['full']['decisive'])
        masks=dict(practically_stable=stable,exact_decisive=decisive,unresolved_active=active & ~stable)
        sc=scores(oldcase(case),quality)['mse'];h=d['budgets'][str(case)]
        bank=np.load(root/'one_step'/f'case_{case:03d}.npz')
        conds=[0 if quality=='high' else 1,2,3,4]
        snis=bank['gain'][conds,:h].mean(1).T.argmax(1)
        values={name:sc[np.arange(6),row['choices'][name]] for name in names if name not in ['random','snis']}
        values['random']=sc.mean(1);values['snis']=sc[np.arange(6),snis]
        for group,mask in masks.items():
            if mask.any():groups[group].append(dict(case=case,queries=np.where(mask)[0].tolist(),mse={n:float(v[mask].mean()) for n,v in values.items()}))
    result[quality]={}
    for group,rows in groups.items():
        result[quality][group]=dict(n_worlds=len(rows),n_queries=sum(len(r['queries']) for r in rows),
            mse={n:float(np.mean([r['mse'][n] for r in rows])) for n in names} if rows else {},
            paired_ci={n:ci([r['mse']['bridge']-r['mse'][n] for r in rows]) for n in names if n!='bridge'} if rows else {},per_case=rows)
(root/'analysis/stable_subset.json').write_text(json.dumps(result,indent=2))
print(json.dumps({q:{g:{k:v for k,v in d.items() if k!='per_case'} for g,d in gs.items()} for q,gs in result.items()},indent=2))
