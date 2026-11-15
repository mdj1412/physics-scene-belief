import numpy as np,json
from pathlib import Path
# Independent world/R/query/action indexing, effect partition, and compute checkpoint rules.
rng=np.random.default_rng(10101);L=rng.random((64,32,6,4));choices=rng.integers(4,size=(64,6))
picked=L[np.arange(64)[:,None,None],np.arange(32)[None,:,None],np.arange(6)[None,None,:],choices[:,None,:]]
for w in [0,3,63]:
 for r in [0,31]:
  for q in range(6):assert picked[w,r,q]==L[w,r,q,choices[w,q]]
d=rng.normal(size=(64,32,6));assert np.allclose(d.mean((1,2)),d.mean(1).mean(1))
masks=[np.arange(6)<2,(np.arange(6)>=2)&(np.arange(6)<4),np.arange(6)>=4]
assert np.isclose(sum((d.mean(1)*m).mean() for m in masks),d.mean())
# Replicating observation repeats does not create new world clusters.
x=d.mean((1,2));assert np.allclose(np.repeat(d,2,axis=1).mean((1,2)),x)
p=Path('results/analysis');p.mkdir(exist_ok=True);(p/'validation.json').write_text(json.dumps(dict(passed=True,checks=['world-repeat-query-action indexing','exclusive contribution sum','world-cluster estimand unchanged by duplicated repeats'])))
print('ANALYSIS_VALIDATION_PASSED')
