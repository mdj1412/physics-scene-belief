"""Posthoc, label-free pooling diagnostic for identical inference problems.

Never multiply repeated likelihoods: independent worlds merely have the same
family and observed factual x-velocity, so their Bayesian targets are identical.
"""
import json,collections,itertools,argparse
from pathlib import Path
import numpy as np
from analyze import load,oldcase,scores,selected,ci

cases=[oldcase(i) for i in range(72)];groups=collections.defaultdict(list)
for d in cases:groups[(d['family'],float(d['observations']['factual'][0]))].append(d['case'])
p=argparse.ArgumentParser();p.add_argument('--main-only',action='store_true');args=p.parse_args()
if not args.main_only:assert Path('results/analysis_adaptive/summary.json').exists(),'Final adaptive analysis is not ready'
out={}
for phase in (['main'] if args.main_only else ['main','adaptive']):
    result={}
    for q in ['high','pixel_only']:
        ds={}
        for i in range(72):
            folder=Path('results/main')/f'case_{i:03d}'
            if phase=='adaptive':
                for h in [128,256,512]:
                    p=Path(f'results/outer{h}')/f'case_{i:03d}'
                    if (p/'complete.json').exists():folder=p
            ds[i]=load(folder,q,'bridge')
        choices={};rows=[];mismatch=total=0
        for (family,y),ids in groups.items():
            current=np.mean([ds[i]['current'] for i in ids],axis=0)
            means=np.concatenate([ds[i]['means'] for i in ids],axis=1)
            utility=((means[0]-current[:,:,None])*(means[1]-current[:,:,None])).sum(2).mean(0)
            choice=utility.argmax(1)
            for i in ids:choices[i]=choice
            active=utility.max(1)>1e-4
            for i,j in itertools.combinations(ids,2):
                mismatch+=int(np.sum((ds[i]['samples'].mean(0).argmax(1)!=ds[j]['samples'].mean(0).argmax(1))&active));total+=int(active.sum())
            rows.append(dict(family=family,factual_x=y,cases=ids,pooled_outer=means.shape[1],choice=choice.tolist(),gain=utility.tolist()))
        tests=[d for d in cases if d['split']=='test'];bycase=[]
        for d in tests:
            sc=scores(d,q);pooled=selected(sc,choices[d['case']]);base=selected(sc,ds[d['case']]['samples'].mean(0).argmax(1))
            bycase.append(dict(case=d['case'],pooled_mse=float(np.mean(pooled['mse'])),unpooled_mse=float(np.mean(base['mse']))))
        result[q]=dict(groups=rows,same_evidence_disagreement=mismatch,same_evidence_active_pairs=total,
            pooled_mse=float(np.mean([d['pooled_mse'] for d in bycase])),unpooled_mse=float(np.mean([d['unpooled_mse'] for d in bycase])),
            paired_ci=ci([d['pooled_mse']-d['unpooled_mse'] for d in bycase]),per_case=bycase)
    out[phase]=result
Path('results/analysis/same_evidence_main.json' if args.main_only else 'results/analysis/same_evidence.json').write_text(json.dumps(out,indent=2))
print(json.dumps({p:{q:{k:v for k,v in d.items() if k not in ['groups','per_case']} for q,d in r.items()} for p,r in out.items()},indent=2))
