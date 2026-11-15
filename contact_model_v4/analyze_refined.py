"""Analyze full SMC refits without changing factual-only acquisition choices."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('results/analysis');OUT.mkdir(parents=True,exist_ok=True)
ACTIONS=['view','left','right','center']
FIELDS=['mse','crps','coverage90','width90','brier','ess']
cases=[json.loads(p.read_text()) for p in sorted(Path('results/main').glob('case_*.json'))]
ref={int(p.stem.split('_')[1]):json.loads(p.read_text()) for p in Path('results/refined').glob('case_*.json')}
assert len(cases)==len(ref)==72,(len(cases),len(ref))
continuous={int(p.stem.split('_')[1]):json.loads(p.read_text()) for p in Path('results/continuous_proxy').glob('case_*.json')}
assert len(continuous)==72,len(continuous)
base=json.loads((OUT/'summary.json').read_text())
test=[d for d in cases if d['split']=='test'];val=[d for d in cases if d['split']=='validation']
def rec(d,action):return next(r for r in ref[d['case']]['records'] if r['action']==action and r['name']=='broad_refit_action')['metrics']
def selection(d,choices):
    rows=[rec(d,a) for a in ACTIONS]
    return {k:np.array([rows[int(a)][k][q] for q,a in enumerate(choices)]) for k in FIELDS}
def ci(v):
    x=np.array(v);rng=np.random.default_rng(512)
    means=x[rng.integers(len(x),size=(10000,len(x)))].mean(1)
    return dict(mean=float(x.mean()),low=float(np.quantile(means,.025)),high=float(np.quantile(means,.975)),n=len(x))
rules={};value_rules={};validation_risks={}
for f in ['corner','gate','deflection']:
    ds=[d for d in val if d['family']==f]
    risks=np.array([np.mean([rec(d,a)['mse'] for d in ds],axis=0) for a in ACTIONS])
    rules[f]=int(risks.mean(1).argmin());value_rules[f]=risks.argmin(0).tolist();validation_risks[f]=risks.tolist()
allmetrics={};percase=[]
for d in test:
    metrics={
        'task_EVSI':selection(d,np.array(d['gains']).argmax(1)),
        'parameter_EIG':selection(d,[int(np.argmax(d['eigs']))]*6),
        'query_type_rule':selection(d,[rules[d['family']]]*6),
        'query_value_rule':selection(d,value_rules[d['family']]),
        'query_value_rule_original':selection(d,base['value_rules'][d['family']]),
        'uniform_random':{k:np.mean([rec(d,a)[k] for a in ACTIONS],axis=0) for k in FIELDS},
    }
    for name in ['broad','known_proxy']:
        r=next(r for r in ref[d['case']]['records'] if r['name']==name)
        metrics['three_obs_'+name]=r['metrics']
    metrics['three_obs_continuous']=continuous[d['case']]['metrics']
    for name,m in metrics.items():allmetrics.setdefault(name,[]).append({k:np.asarray(m[k]) for k in FIELDS})
    percase.append(dict(case=d['case'],family=d['family'],regime=d['regime'],metrics={n:{k:float(np.mean(m[k])) for k in FIELDS} for n,m in metrics.items()}))
def average(ms):return {k:float(np.mean([m[k] for m in ms])) for k in FIELDS}
summary={n:average(ms) for n,ms in allmetrics.items()}
pairs={}
for a,b in [('task_EVSI','query_type_rule'),('task_EVSI','query_value_rule'),('task_EVSI','parameter_EIG'),('task_EVSI','uniform_random'),('three_obs_broad','three_obs_known_proxy'),('three_obs_continuous','three_obs_broad'),('three_obs_continuous','three_obs_known_proxy')]:
    pairs[a+' minus '+b]={k:ci([np.mean(x[k])-np.mean(y[k]) for x,y in zip(allmetrics[a],allmetrics[b])]) for k in ['mse','crps','coverage90']}
groups={}
for key in ['family','regime']:
    groups[key]={}
    for value in sorted(set(d[key] for d in test)):
        ids=[i for i,d in enumerate(test) if d[key]==value]
        groups[key][value]={n:average([ms[i] for i in ids]) for n,ms in allmetrics.items()}
convergence=json.loads(Path('results/audit/posterior_convergence.json').read_text())
assert len(convergence)==36,len(convergence)
cv=[]
for case in sorted(set(d['case'] for d in convergence)):
    ds=[d for d in convergence if d['case']==case]
    means={n:np.mean([d['mean'] for d in ds if d['n']==n],axis=0) for n in [768,3072]}
    cv.append(dict(case=case,mean_difference_rms=float(np.sqrt(np.mean((means[768]-means[3072])**2))),
        seed_rms_3072=float(np.sqrt(np.mean((np.array(ds[2]['mean'])-np.array(ds[3]['mean']))**2))),
        choices_3072=[d['choice'] for d in ds if d['n']==3072],
        useful_choices_3072=[[c if max(g)>1e-4 else -1 for c,g in zip(d['choice'],d['gains'])] for d in ds if d['n']==3072]))
costs={}
for name in ['broad_refit_action','broad','known_proxy']:
    rs=[r['cost'] for d in ref.values() for r in d['records'] if r['name']==name]
    costs[name]={k:float(np.mean([r[k] for r in rs])) for k in ['calls','seconds','acceptance']}
costs['continuous']={k:float(np.mean([r['cost'][k] for r in continuous.values()])) for k in ['calls','seconds','acceptance']}
result=dict(n_test=len(test),summary=summary,paired_ci=pairs,groups=groups,rules=rules,value_rules=value_rules,validation_risks=validation_risks,
    posterior_convergence=cv,costs=costs,notes=['All factual acquisition choices frozen before outcome is revealed.',
    'Rules are selected using validation outcomes only, with the same refined estimator used at test.',
    'One-observation methods use full SMC-1536 refit with 12 rejuvenation moves.',
    'Known proxy baseline receives privileged true collision rounding, and is not a feasible method.',
    'Continuous proxy was added post-hoc to audit discretization, not presented as preregistered evidence.',
    'Wall time was measured under overlapping GPU workloads, not a speed benchmark.',
    'Bootstrap intervals are exploratory, world-clustered, and not multiplicity-corrected.'])
(OUT/'refined_summary.json').write_text(json.dumps(result,indent=2));(OUT/'refined_per_case.json').write_text(json.dumps(percase,indent=2))
fig,axs=plt.subplots(1,2,figsize=(11,4))
names=['query_type_rule','query_value_rule','parameter_EIG','uniform_random','task_EVSI']
axs[0].bar(range(len(names)),[summary[n]['mse'] for n in names],color=['#8b9bab']*4+['#237d94'])
axs[0].set_xticks(range(len(names)),names,rotation=27,ha='right');axs[0].set_ylabel('Counterfactual velocity MSE');axs[0].set_title('Full SMC refit after one observation')
names=['sharp_refit','noise_inflation','adaptive_expand','broad_refit']
axs[1].bar(range(len(names)),[base['summary'][n]['coverage90'] for n in names],color=['#8b9bab']*3+['#237d94'])
axs[1].axhline(.9,color='black',linestyle='--',linewidth=1);axs[1].set_xticks(range(len(names)),names,rotation=27,ha='right');axs[1].set_ylabel('Marginal 90% interval coverage');axs[1].set_title('Same data and particle count (SMC-768)')
fig.tight_layout();fig.savefig(OUT/'refined_comparison.png',dpi=180);plt.close(fig)
print(json.dumps(dict(summary=summary,paired_ci=pairs),indent=2))
