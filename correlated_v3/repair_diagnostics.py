"""Recompute only a histogram MI diagnostic after underflow correction.

Original diagnostics are preserved. Predictive/decision results are untouched.
"""
import csv,json,shutil
from pathlib import Path
from core import *
from run import case_spec

root=Path('results/main');archive=root/'original_diagnostics';archive.mkdir(exist_ok=True)
maxdiff=0.;count=0
for shard in range(4):
    path=root/f'diagnostics_{shard}.csv'
    original=archive/path.name
    if not original.exists():shutil.copy2(path,original)
    rows=list(csv.DictReader(original.open()))
    refs={}
    for row in rows:
        f=int(float(row['family']));signed=bool(int(float(row['signed'])));sigma=float(row['sigma']);y=float(row['y']);rep=int(float(row['seed']));name=row['method']
        key=(f,signed,sigma)
        if key not in refs:refs[key]=conditional_quad(y,sigma,signed)[0]
        if name=='reference':atoms=refs[key]
        elif name.startswith('grid'):
            ng,ne=map(int,name[4:].split('x'));atoms=fixed_grid(y,sigma,signed,ng,ne)[0]
        elif name=='conditional_quad512':atoms=conditional_quad(y,sigma,signed,16 if signed else 32,16)[0]
        elif name=='factorized_marginals':atoms=factorized(refs[key])[0]
        else:
            rng=np.random.default_rng([SEED,f,NOISES.index(sigma),int(signed),rep,471])
            if name=='prior512':atoms=prior_particles(y,sigma,signed,rng,512)[0]
            elif name=='prior2048':atoms=prior_particles(y,sigma,signed,rng,2048)[0]
            elif name=='posterior512':atoms=posterior_particles(y,sigma,signed,rng,512)[0]
            elif name=='smc512':atoms=smc(y,sigma,signed,rng,512)[0]
        d=diagnostic(atoms,y,sigma)
        for k,v in d.items():
            if k!='mi_hist':maxdiff=max(maxdiff,abs(v-float(row[k])))
            row[k]=v
        assert np.isfinite(list(d.values())).all()
        count+=1
    with path.open('w') as h:
        wr=csv.DictWriter(h,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    print('diagnostics shard',shard,'verified',flush=True)
assert maxdiff<1e-10,maxdiff
(root/'diagnostics_repair.json').write_text(json.dumps(dict(rows=count,non_mi_max_change=maxdiff,reason='log marginal probabilities prevent histogram MI underflow; raw prediction metrics unchanged'),indent=2))
