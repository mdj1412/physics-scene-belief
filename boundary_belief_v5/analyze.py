import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ACTIONS=['view','left','right','center'];FIELDS=['mse','crps','coverage90','width90','brier']
cases=[json.loads(p.read_text()) for p in sorted(Path('results/main').glob('case_*.json'))]
assert len(cases)==72,len(cases)
assert all(d.get('measurement_model')=='finite_frame_50ms' for d in cases)
for d in cases:
    p=json.loads((Path('results/pixel_only')/f"case_{d['case']:03d}.json").read_text())
    d['records'].extend(p['records']);d['policies']['pixel_only']=p['policies']
    d['coarse_conditional']=json.loads((Path('results/conditional_view')/f"case_{d['case']:03d}.json").read_text())['records']
    d['conditional']=json.loads((Path('results/conditional_refined')/f"case_{d['case']:03d}.json").read_text())['records']
val=[d for d in cases if d['split']=='validation'];test=[d for d in cases if d['split']=='test']
OUT=Path('results/analysis');OUT.mkdir(parents=True,exist_ok=True)

def row(d,q,m,a):
    if a=='view':return next(r for r in d['records'] if r['name']=='view' and r['quality']==q and r['mode']==m)
    return next(r for r in d['records'] if r['name']=='dynamic' and r['action']==a)
def selected(d,q,m,choice):
    rs=[row(d,q,m,a)['metrics'] for a in ACTIONS]
    return {k:np.array([rs[int(a)][k][i] for i,a in enumerate(choice)]) for k in FIELDS}
def avg(ms):return {k:float(np.mean([m[k] for m in ms])) for k in FIELDS}
def ci(vals):
    x=np.array(vals);rng=np.random.default_rng(8512);means=x[rng.integers(len(x),size=(10000,len(x)))].mean(1)
    return dict(mean=float(x.mean()),low=float(np.quantile(means,.025)),high=float(np.quantile(means,.975)),n=len(x),median=float(np.median(x)),negative_count=int(np.sum(x<0)),positive_count=int(np.sum(x>0)))

allout={};allcase={}
for quality in ['high','low','pixel_only']:
    rules={};value_rules={}
    for mode in ['bbox','boundary']:
        rules[mode]={};value_rules[mode]={}
        for family in ['corner','gate','deflection']:
            ds=[d for d in val if d['family']==family]
            risks=np.array([np.mean([row(d,quality,mode,a)['metrics']['mse'] for d in ds],axis=0) for a in ACTIONS])
            rules[mode][family]=int(risks.mean(1).argmin());value_rules[mode][family]=risks.argmin(0).tolist()
    metrics={};percase=[];actioncounts={m:np.zeros(4,int) for m in ['bbox','boundary']};useful={m:np.zeros(4,int) for m in ['bbox','boundary']};viewgeom={m:[] for m in ['bbox','boundary']};switches=0;usefulswitches=0
    conditional_counts={m:np.zeros(4,int) for m in ['bbox','boundary']};conditional_diagnostics={m:[] for m in ['bbox','boundary']}
    conditional_by_query={m:np.zeros((6,4),int) for m in ['bbox','boundary']};conditional_switches=0
    no_observation=[]
    for d in test:
        methods={};choices={}
        no_observation.append(next(r for r in d['records'] if r['name']=='no_observation')['metrics'])
        for mode in ['bbox','boundary']:
            p=d['policies'][quality][mode];gain=np.array(p['gains']);choice=gain.argmax(1);choices[mode]=choice
            actioncounts[mode]+=np.bincount(choice,minlength=4);useful[mode]+=np.bincount(choice[gain.max(1)>1e-4],minlength=4)
            methods[mode+'_EVSI']=selected(d,quality,mode,choice)
            methods[mode+'_EIG']=selected(d,quality,mode,[int(np.argmax(p['eigs']))]*6)
            methods[mode+'_type_rule']=selected(d,quality,mode,[rules[mode][d['family']]]*6)
            methods[mode+'_value_rule']=selected(d,quality,mode,value_rules[mode][d['family']])
            methods[mode+'_always_view']=selected(d,quality,mode,[0]*6)
            methods[mode+'_random']={k:np.mean([row(d,quality,mode,a)['metrics'][k] for a in ACTIONS],axis=0) for k in FIELDS}
            methods[mode+'_probe_EVSI']=selected(d,quality,mode,gain[:,1:].argmax(1)+1)
            viewgeom[mode].append(row(d,quality,mode,'view')['geometry'])
            cond=d['conditional'][quality][mode];cg=gain.copy();cg[:,0]=cond['gain'];ce=np.array(p['eigs']);ce[0]=cond['eig']
            cc=cg.argmax(1);choices[mode+'_conditional']=cc;conditional_counts[mode]+=np.bincount(cc,minlength=4)
            conditional_by_query[mode][np.arange(6),cc]+=1
            methods[mode+'_conditional_EVSI']=selected(d,quality,mode,cc)
            coarse=gain.copy();coarse[:,0]=d['coarse_conditional'][quality][mode]['gain'];methods[mode+'_conditional_coarse_EVSI']=selected(d,quality,mode,coarse.argmax(1))
            methods[mode+'_conditional_EIG']=selected(d,quality,mode,[int(ce.argmax())]*6)
            raw=cg.copy();raw[:,0]=cond['raw_gain'];methods[mode+'_conditional_raw_EVSI']=selected(d,quality,mode,raw.argmax(1))
            conditional_diagnostics[mode].append({k:cond[k] for k in ['ess_median','ess_min','seconds','trajectory_rollouts']})
        methods['boundary_frozen_bbox_policy']=selected(d,quality,'boundary',choices['bbox'])
        methods['boundary_conditional_frozen_bbox_policy']=selected(d,quality,'boundary',choices['bbox_conditional'])
        conditional_switches+=int(np.sum(choices['bbox_conditional']!=choices['boundary_conditional']))
        switches+=int(np.sum(choices['bbox']!=choices['boundary']))
        active=np.maximum(np.max(d['policies'][quality]['bbox']['gains'],axis=1),np.max(d['policies'][quality]['boundary']['gains'],axis=1))>1e-4
        usefulswitches+=int(np.sum((choices['bbox']!=choices['boundary'])&active))
        for n,m in methods.items():metrics.setdefault(n,[]).append(m)
        percase.append(dict(case=d['case'],family=d['family'],rounding=d['rounding'],metrics={n:{k:float(np.mean(m[k])) for k in FIELDS} for n,m in methods.items()}))
    pairs={}
    comparisons=[('boundary_EVSI','boundary_value_rule'),('boundary_EVSI','boundary_type_rule'),('boundary_EVSI','boundary_EIG'),('boundary_EVSI','boundary_always_view'),('boundary_EVSI','boundary_probe_EVSI'),('boundary_EVSI','bbox_EVSI'),('boundary_EVSI','boundary_frozen_bbox_policy'),('boundary_always_view','bbox_always_view')]
    comparisons += [('boundary_conditional_EVSI',b) for b in ['boundary_value_rule','boundary_always_view','boundary_conditional_EIG','bbox_conditional_EVSI','boundary_conditional_frozen_bbox_policy','boundary_probe_EVSI','boundary_EVSI','boundary_conditional_raw_EVSI','boundary_conditional_coarse_EVSI']]
    comparisons += [('boundary_conditional_frozen_bbox_policy','bbox_conditional_EVSI')]
    for a,b in comparisons:
        pairs[a+' minus '+b]={k:ci([np.mean(x[k])-np.mean(y[k]) for x,y in zip(metrics[a],metrics[b])]) for k in ['mse','crps','coverage90']}
    allout[quality]=dict(summary={n:avg(ms) for n,ms in metrics.items()},paired_ci=pairs,rules=rules,value_rules=value_rules,
        action_counts={m:a.tolist() for m,a in actioncounts.items()},useful_action_counts={m:a.tolist() for m,a in useful.items()},switches=switches,useful_switches=usefulswitches,
        conditional_action_counts={m:a.tolist() for m,a in conditional_counts.items()},conditional_diagnostics=conditional_diagnostics,
        conditional_by_query={m:a.tolist() for m,a in conditional_by_query.items()},conditional_switches=conditional_switches,no_observation=avg(no_observation),
        geometry={m:{k:float(np.mean([d[k] for d in rs])) for k in rs[0]} for m,rs in viewgeom.items()},
        geometry_ci={k:ci([a[k]-b[k] for a,b in zip(viewgeom['boundary'],viewgeom['bbox'])]) for k in ['rounding_abs_error_mm','boundary_rmse_mm']},
        family={f:{n:avg([ms[i] for i,d in enumerate(test) if d['family']==f]) for n,ms in metrics.items()} for f in ['corner','gate','deflection']})
    allcase[quality]=percase
calibration={}
for q in ['high','low','pixel_only']:
    rs=json.loads((Path('results/calibration')/f'holdout_{q}.json').read_text())
    calibration[q]=dict(n=len(rs),mae_mm=float(np.mean([abs(d['estimate']-d['b'])*1e3 for d in rs])),coverage90=float(np.mean([d['lo']<=d['b']<=d['hi'] for d in rs])),width_mm=float(np.mean([(d['hi']-d['lo'])*1e3 for d in rs])),bins={})
    for lo in [0,.002,.004]:
        ds=[d for d in rs if lo<=d['b']<lo+.002]
        calibration[q]['bins'][str(lo)]=dict(n=len(ds),mae_mm=float(np.mean([abs(d['estimate']-d['b'])*1e3 for d in ds])),coverage90=float(np.mean([d['lo']<=d['b']<=d['hi'] for d in ds])))
result=dict(n_test=len(test),quality=allout,calibration=calibration,
    cross_quality={n:ci([a['metrics'][n]['mse']-b['metrics'][n]['mse'] for a,b in zip(allcase['high'],allcase['low'])]) for n in ['bbox_EVSI','boundary_EVSI','boundary_always_view']})
(OUT/'summary.json').write_text(json.dumps(result,indent=2));(OUT/'per_case.json').write_text(json.dumps(allcase,indent=2))
names=['bbox_conditional_EVSI','boundary_conditional_EVSI','boundary_value_rule','boundary_conditional_EIG','boundary_always_view','boundary_probe_EVSI']
fig,axs=plt.subplots(3,1,figsize=(9,11),sharey=True)
for ax,q in zip(axs,['high','low','pixel_only']):
    cis=[ci([d['metrics'][n]['mse'] for d in allcase[q]]) for n in names];means=np.array([d['mean'] for d in cis]);err=np.array([[d['mean']-d['low'] for d in cis],[d['high']-d['mean'] for d in cis]])
    ax.bar(range(len(names)),means,yerr=err,capsize=3,color=['#99a9b9','#187c8d','#8494be','#8494be','#bd9b72','#8494be'])
    ax.set_xticks(range(len(names)),[n.replace('conditional_','').replace('boundary_','full\n').replace('bbox_','bbox\n') for n in names],rotation=15,ha='right');ax.set_title(q);ax.set_ylabel('Counterfactual velocity MSE')
fig.tight_layout();fig.savefig(OUT/'policy_comparison.png',dpi=180);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10,4))
for ax,q in zip(axs,['high','low']):
    rs=json.loads((Path('results/calibration')/f'holdout_{q}.json').read_text());ax.scatter([d['b']*1e3 for d in rs],[d['estimate']*1e3 for d in rs],s=10,alpha=.5);ax.plot([0,6],[0,6],'k--');ax.set(xlabel='True rounding [mm]',ylabel='Image-derived posterior mean [mm]',title=q+' resolution')
fig.tight_layout();fig.savefig(OUT/'curvature_calibration.png',dpi=180);plt.close(fig)
print(json.dumps(result,indent=2))
