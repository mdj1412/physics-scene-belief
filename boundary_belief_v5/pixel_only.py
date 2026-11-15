"""Secondary sensitivity experiment: remove pre-render g/d localization jitter.

Reuse factual and dynamic measurements/posteriors exactly. Only view changes.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from engine import tensor
from belief import Belief,predict,metrics,parameter_summary
from geometry import render,extract
from run import plain,geometry_score

p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);p.add_argument('--wait-inputs',action='store_true');args=p.parse_args()
out=Path('results/pixel_only');out.mkdir(parents=True,exist_ok=True)
for case in range(args.worker,72,2):
    dest=out/f'case_{case:03d}.json'
    if dest.exists():continue
    source=Path('results/main')/dest.name
    while args.wait_inputs and not source.exists():time.sleep(2)
    if not source.exists():raise RuntimeError(f'main case not ready: {case}')
    d=json.loads(source.read_text());f=d['family'];theta=d['theta'];b=d['rounding'];truth={k:np.array(v) for k,v in d['truth'].items()}
    im=render(theta[0],theta[3],b,f,'pixel_only',np.random.default_rng(202611088000+case));y=extract(im,f,'pixel_only')
    im.save(Path('results/main/images')/f'{case:03d}'/'pixel_only.png')
    old=np.load(Path('results/main')/f'factual_{case:03d}.npz');initial=tuple(tensor(old[k]) for k in ['theta','rounding','weight']);pred=predict(initial,f)
    records=[];policies={}
    for mode in ['bbox','boundary']:
        model=Belief(f,'pixel_only',mode);g,e=model.acquisition(initial,pred,seed=170100+case)
        atoms,cost=model.smc([('factual',d['observations']['factual']),('view',y)],seed=160100+case)
        param=parameter_summary(atoms)
        records.append(dict(name='view',quality='pixel_only',mode=mode,action='view',metrics=metrics(atoms,predict(atoms,f),truth),parameters=param,geometry=geometry_score(param,theta,b,f),cost=cost))
        policies[mode]=dict(gains=g,eigs=e)
    dest.write_text(json.dumps(plain(dict(case=case,view=y,records=records,policies=policies)),allow_nan=False));print('pixel_only',case,flush=True)
