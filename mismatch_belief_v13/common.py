import json,hashlib
from pathlib import Path
import numpy as np
R=Path('results')
def write(p,d):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(d,indent=2))
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def public(split):return np.load(R/'data'/split/'public.npz')['factual'].reshape(-1,11)
def observations(split):
    z=np.load(R/'data'/split/'private.npz');return {a:z[f'obs{a}'].reshape(-1,*z[f'obs{a}'].shape[2:]) for a in [1,2,3]}
def snapshot(name):write(R/name,{str(p):sha(p) for p in sorted(Path('.').glob('*')) if p.is_file()})
def freeze(name):write(R/'freezes'/f'{name}.json',dict(policy=sha(R/'policies'/f'{name}.npz'),public=sha(R/'data/test/public.npz'),thresholds=sha(R/'thresholds.json')))
def check(name):
    d=read(R/'freezes'/f'{name}.json');assert d==dict(policy=sha(R/'policies'/f'{name}.npz'),public=sha(R/'data/test/public.npz'),thresholds=sha(R/'thresholds.json'))
