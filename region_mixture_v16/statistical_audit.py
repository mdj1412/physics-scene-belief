"""Post-analysis numerical audit; no policy or primary criterion changes."""
import os
os.environ['BELIEF_DEVICE']='cpu'
from common import *
from analyze import bootstrap
require()
L=np.load(R/'world_losses.npz');a=L['matched_nominal__region32'];out={}
# Primary upper CI is extremely close to zero: audit bootstrap Monte Carlo error.
d=a[:,4]-L['matched_nominal__native'][:,4]
checks=[]
for seed in [140101,140102,140103,140104,140105]:
    rng=np.random.default_rng(seed);v=[]
    for i in range(100):
        ix=rng.integers(len(d),size=(1000,len(d)));v.append(d[ix].mean(1))
    v=np.concatenate(v)
    checks.append(dict(seed=seed,n=100000,low=float(np.quantile(v,.05/24)),high=float(np.quantile(v,1-.05/24))))
out['primary_stick_native_bootstrap_MC_audit']=checks
out['secondary_near_budget_comparisons']={}
for k,task in [(2,'contact_event'),(4,'slip_event')]:
    for m in ['nested128','qmc64_native']:
        out['secondary_near_budget_comparisons'][task+'/'+m]=bootstrap(a[:,k]-L['matched_nominal__'+m][:,k],n=30000)
write(R/'statistical_audit.json',out);print(json.dumps(out,indent=2))
