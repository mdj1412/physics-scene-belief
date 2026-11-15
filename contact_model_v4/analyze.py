import json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('results/main');OUT=Path('results/analysis');OUT.mkdir(parents=True,exist_ok=True)
ACTIONS=['view','left','right','center'];FIELDS=['mse','crps','coverage90','width90','brier','ess']
cases=[json.loads(p.read_text()) for p in sorted(ROOT.glob('case_*.json'))]
assert len(cases)==72,len(cases)
test=[d for d in cases if d['split']=='test'];val=[d for d in cases if d['split']=='validation']

def record(d,name,action=None,replicate=None):
    return next(r for r in d['records'] if r['name']==name and (action is None or r['action']==action) and (replicate is None or r.get('replicate')==replicate))
def selected(d,name,choice,replicate=None):
    rows=[record(d,name,a,replicate)['metrics'] for a in ACTIONS]
    return {k:np.array([rows[int(a)][k][q] for q,a in enumerate(choice)]) for k in FIELDS}
def average(items):return {k:float(np.mean([m[k] for m in items])) for k in FIELDS}
def ci(values):
    x=np.array(values);rng=np.random.default_rng(512);means=x[rng.integers(len(x),size=(10000,len(x)))].mean(1)
    return dict(mean=float(x.mean()),low=float(np.quantile(means,.025)),high=float(np.quantile(means,.975)),n=len(x))

rules={};value_rules={}
for family in ['corner','gate','deflection']:
    ds=[d for d in val if d['family']==family]
    risks=[np.mean([record(d,'broad',a)['metrics']['mse'] for d in ds]) for a in ACTIONS]
    rules[family]=int(np.argmin(risks))
    value_rules[family]=np.array([np.mean([record(d,'broad',a)['metrics']['mse'] for d in ds],axis=0) for a in ACTIONS]).argmin(0).tolist()
cal=[json.loads(p.read_text()) for p in Path('results/calibration').glob('*.json')]
thresholds={}
for family in rules:
    scores=sorted(d['score'] for d in cal if d['family']==family)
    assert len(scores)==64,(family,len(scores))
    # Split conformal upper order statistic, nominal 95% null acceptance.
    thresholds[family]=scores[min(len(scores)-1,math.ceil((len(scores)+1)*.95)-1)]
allmetrics={};percase=[];choicecounts=np.zeros(4,int);useful_counts=np.zeros(4,int);diversity=0;useful_diversity=0;det=[];costs=[]
for d in test:
    choices=np.array(d['gains']).argmax(1);choicecounts+=np.bincount(choices,minlength=4);diversity+=len(set(choices))>1
    useful=np.max(np.array(d['gains']),axis=1)>1e-4
    useful_counts+=np.bincount(choices[useful],minlength=4);useful_diversity+=len(set(choices[useful]))>1
    methods={'task_EVSI':selected(d,'broad',choices),
        'parameter_EIG':selected(d,'broad',[int(np.argmax(d['eigs']))]*6),
        'query_type_rule':selected(d,'broad',[rules[d['family']]]*6),
        'query_value_rule':selected(d,'broad',value_rules[d['family']]),
        'uniform_random':{k:np.mean([record(d,'broad',a)['metrics'][k] for a in ACTIONS],axis=0) for k in FIELDS},
        'no_observation':record(d,'broad','stop')['metrics'],
        'joint_MAP':record(d,'joint_MAP')['metrics']}
    for size in [64,128]:
        for method in ['systematic','branch']:
            name=f'{method}_{size}';ms=[]
            for rep in range(3):
                choice=record(d,name,'view',rep)['choice'];ms.append(selected(d,name,choice,rep))
            methods[name]={k:np.mean([m[k] for m in ms],axis=0) for k in FIELDS}
    triggered=d['check_score']>thresholds[d['family']]
    for name in ['sharp_initial','sharp_refit','broad_refit','noise_inflation']:methods[name]=record(d,name)['metrics']
    methods['adaptive_expand']=methods['broad_refit' if triggered else 'sharp_refit']
    det.append(dict(case=d['case'],family=d['family'],regime=d['regime'],triggered=bool(triggered),score=d['check_score']))
    c={x['name']:x for x in d['costs']}
    costs.append(dict(case=d['case'],broad=c['broad_refit']['calls'],sharp=c['sharp_refit']['calls'],
        adaptive=c['sharp_initial']['calls']+c['broad_refit' if triggered else 'sharp_refit']['calls'],
        # 'Always broad' can fit all collected observations directly and needs no initial sharp fit.
        initial_sharp=c['sharp_initial']['calls']))
    for name,m in methods.items():allmetrics.setdefault(name,[]).append({k:np.array(m[k]) for k in FIELDS})
    percase.append(dict(case=d['case'],family=d['family'],regime=d['regime'],metrics={name:{k:float(np.mean(m[k])) for k in FIELDS} for name,m in methods.items()}))
summary={name:average(ms) for name,ms in allmetrics.items()}
pairs={}
for a,b in [('task_EVSI','query_type_rule'),('task_EVSI','query_value_rule'),('task_EVSI','parameter_EIG'),('branch_64','systematic_64'),('branch_128','systematic_128'),('broad_refit','sharp_refit'),('adaptive_expand','broad_refit'),('noise_inflation','sharp_refit')]:
    pairs[f'{a} minus {b}']={k:ci([float(np.mean(x[k])-np.mean(y[k])) for x,y in zip(allmetrics[a],allmetrics[b])]) for k in ['mse','crps','coverage90']}
groups={}
for key in ['family','regime']:
    groups[key]={}
    for value in sorted(set(d[key] for d in test)):
        ids=[i for i,d in enumerate(test) if d[key]==value]
        groups[key][value]={name:average([ms[i] for i in ids]) for name,ms in allmetrics.items()}
result=dict(n_cases=len(cases),n_test=len(test),rules=rules,value_rules=value_rules,thresholds=thresholds,summary=summary,paired_ci=pairs,
    groups=groups,action_counts=choicecounts.tolist(),useful_action_counts=useful_counts.tolist(),multi_action_worlds=diversity,useful_multi_action_worlds=useful_diversity,detection=det,costs=costs,
    notes=['CI unit is world, six queries and compression seeds averaged inside each world.','SMC-768 is a strong practical baseline, not an exact posterior oracle.','Compression comparisons share upstream SMC; no end-to-end speedup is implied.'])
(OUT/'summary.json').write_text(json.dumps(result,indent=2));(OUT/'per_case.json').write_text(json.dumps(percase,indent=2))
names=['no_observation','query_type_rule','parameter_EIG','task_EVSI','systematic_64','branch_64']
fig,ax=plt.subplots(1,2,figsize=(12,4));ax[0].bar(range(len(names)),[summary[n]['mse'] for n in names]);ax[0].set_xticks(range(len(names)),names,rotation=30,ha='right');ax[0].set_ylabel('Counterfactual velocity MSE');ax[0].set_title('One additional observation')
names2=['sharp_refit','broad_refit','noise_inflation','adaptive_expand'];ax[1].bar(range(4),[summary[n]['coverage90'] for n in names2]);ax[1].axhline(.9,color='black',linestyle='--');ax[1].set_xticks(range(4),names2,rotation=30,ha='right');ax[1].set_ylabel('Marginal 90% interval coverage');ax[1].set_title('Same three observations');fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=170);plt.close(fig)
print(json.dumps(dict(summary=summary,paired_ci=pairs,thresholds=thresholds,action_counts=choicecounts.tolist(),multi_action_worlds=diversity),indent=2))
