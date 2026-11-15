import json,hashlib
from pathlib import Path
import numpy as np
R=Path('results')
def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(d,indent=2))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def snapshot(name):write(R/name,{str(p):sha(p) for p in sorted(Path('.').glob('*')) if p.is_file()})
def public(split):return np.load(R/'data'/split/'public.npz')['factual'].reshape(-1,11)
def obs(split):
    z=np.load(R/'data'/split/'private.npz');return {a:z[f'obs{a}'].reshape(-1,*z[f'obs{a}'].shape[2:]) for a in [1,2,3]}
def freeze(name):write(R/'freezes'/f'{name}.json',dict(policy=sha(R/'policies'/f'{name}.npz'),public=sha(R/'data/test/public.npz')))
def check(name):
    d=read(R/'freezes'/f'{name}.json');assert sha(R/'policies'/f'{name}.npz')==d['policy'] and sha(R/'data/test/public.npz')==d['public']
