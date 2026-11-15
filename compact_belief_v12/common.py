import os,json,hashlib
from pathlib import Path
import numpy as np
from physics import FY,L
R=Path('results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2))
def public(split):
    a=np.load(R/'data'/split/'public.npz')['factual'];return a.reshape(-1,11)
def private(split):
    f=np.load(R/'data'/split/'private.npz');return {k:f[k].reshape(-1,*f[k].shape[2:]) if k!='lam' else f[k] for k in f.files}
def freeze(name):
    p=R/'policies'/f'{name}.npz'
    write(R/'freezes'/f'{name}.json',dict(policy_sha256=sha(p),public_sha256=sha(R/'data/test/public.npz')))
def check(name):
    q=json.loads((R/'freezes'/f'{name}.json').read_text())
    assert sha(R/'policies'/f'{name}.npz')==q['policy_sha256'] and sha(R/'data/test/public.npz')==q['public_sha256']
def tree_choice(factual):
    rule=np.load('frozen_rule/rules.npz');tree=json.loads(Path('frozen_rule/tree.json').read_text())
    z=FY/L;Q,_=np.linalg.qr(np.column_stack([z,z*z,z**4]));feat=np.concatenate([factual[:,:9]@Q,factual[:,9:]],1)
    x=(feat-rule['mean'])/rule['sd']
    def pred(node,v):
        if 'j' not in node:return np.tile(node['mean'],(len(v),1))
        out=np.empty((len(v),4));m=v[:,node['j']]<=node['v'];out[m]=pred(node['left'],v[m]);out[~m]=pred(node['right'],v[~m]);return out
    return pred(tree,x).argmin(1)

def source_snapshot(name):
    write(R/name,{str(p):sha(p) for p in sorted(Path('.').glob('*')) if p.is_file()})
