"""Outcome-free utility/ranking analysis. No private data access."""
import json
from pathlib import Path
import numpy as np
from config import ACTIONS,COSTS,N_WORLDS

def load(folder,precision,method='bridge',prefix=None):
    gains=[];means=[];esses=[]
    for action in ACTIONS:
        name=f'{precision if action=="view" else "native"}_{action}'
        if method.startswith('snis'):
            with np.load(folder/f'{name}_{method}.npz') as d:
                gains.append(d['gain'][:prefix]);means.append(d['mean'][:prefix]);esses.append(d['ess'][:prefix])
        else:
            runs=[np.load(folder/f'{name}_{method}_{rep}.npz') for rep in [0,1]]
            a=runs[0]['mean'][:prefix]-runs[0]['current'];b=runs[1]['mean'][:prefix]-runs[1]['current']
            gains.append((a*b).sum(-1)/2);means.append(np.stack([d['mean'][:prefix] for d in runs]).mean(0))
            for d in runs:d.close()
    return dict(samples=np.stack(gains,-1),means=np.stack(means,-1),ess=np.stack(esses,-1) if esses else None)

def stability(samples,cost,seed):
    h=len(samples);gross=samples.mean(0);net=gross-np.array(cost);choice=net.argmax(1);active=gross.max(1)>5e-5
    boot=samples[np.random.default_rng(seed).integers(h,size=(1000,h))].mean(1)-np.array(cost)
    alternatives=boot.copy();alternatives[:,np.arange(6),choice]=-np.inf
    gap=boot[:,np.arange(6),choice]-alternatives.max(2);regret=boot.max(2)-boot[:,np.arange(6),choice]
    upper=np.quantile(regret,.95,axis=0);tol=np.maximum(5e-5,.05*gross.max(1))
    first=(samples[:h//2].mean(0)-cost).argmax(1);second=(samples[h//2:].mean(0)-cost).argmax(1)
    return dict(choice=choice.tolist(),gross=gross.tolist(),active=active.tolist(),upper_regret=upper.tolist(),tolerance=tol.tolist(),
        practical=(upper<=tol).tolist(),decisive=(np.quantile(gap,.025,axis=0)>0).tolist(),half_agreement=(first==second).tolist(),
        half_regret=(net.max(1)-net[np.arange(6),first]).tolist(),winner_probability=(boot.argmax(2)==choice).mean(0).tolist())

def create_plan(final=False):
    rows=[];selected=[]
    for case in range(N_WORLDS):
        folder=Path('results/acq256')/f'case_{case:03d}'
        if final and (Path('results/acq512')/folder.name/'complete.json').exists():folder=Path('results/acq512')/folder.name
        assert (folder/'complete.json').exists()
        bad=False;record=dict(case=case,phase=folder.parent.name,conditions={})
        for precision in ['native','precise']:
            d=load(folder,precision);record['conditions'][precision]={}
            sg={m:load(folder,precision,m)['samples'].mean(0) for m in ['snis4096','snis16384']} if final else {}
            for label,cost in COSTS.items():
                st=stability(d['samples'],cost,770000+case)
                bad|=bool(np.any(np.array(st['active']) & ~np.array(st['practical'])))
                record['conditions'][precision][label]=dict(bridge=st,snis_choices={m:(g-np.array(cost)).argmax(1).tolist() for m,g in sg.items()})
        record['unresolved']=bad;rows.append(record)
        if bad:selected.append(case)
    out=Path('results/plans');out.mkdir(exist_ok=True)
    path=out/('final_policy.json' if final else 'refine512.json');tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(cases=selected,rows=rows,scope='numerical uncertainty only; no actual observations/outcomes read'),indent=2));tmp.replace(path)
    print(json.dumps(dict(plan=str(path),cases=selected)),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args();create_plan(a.final)
