import numpy as np
from analyze import take,interval
# Unequal action/query losses expose accidental axis swaps in policy evaluation.
a=np.arange(3*6*4).reshape(3,6,4).astype(float);choice=np.array([3,0,1,2,3,1])
assert np.array_equal(take(a,choice),np.array([[a[r,q,choice[q]] for q in range(6)] for r in range(3)]))
# Repeating identical measurements must not manufacture independent worlds.
x=np.linspace(-.01,.02,48);y=np.tile(x[:,None,None],(1,32,6));one=interval(y[:,:1].mean((1,2)));many=interval(y.mean((1,2)))
assert np.allclose(one['ci95'],many['ci95']) and one['n_worlds']==many['n_worlds']==48
z=interval(np.zeros(48));assert z['mean']==0 and z['ci95']==[0,0]
assert interval([])['mean'] is None
from pathlib import Path
import json,platform
Path('results/analysis/validation.json').write_text(json.dumps(dict(passed=True,python=platform.python_version(),numpy=np.__version__,checks=['query/action indexing','world-cluster uncertainty under duplicate repeats','exact zero difference','empty subgroup'])))
print('PASS: query/action indexing, world-cluster uncertainty, exact zero difference, empty subgroup')
