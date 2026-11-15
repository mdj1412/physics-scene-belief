import argparse,json
import os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(Path('results/mpl_cache').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OLD=Path('../boundary_belief_v5/results');OUT=Path('results/analysis')
ACTIONS=['view','left','right','center'];FIELDS=['mse','crps','coverage90','width90','brier']

def ci(x):
    x=np.asarray(x);rng=np.random.default_rng(8512)
    m=x[rng.integers(len(x),size=(10000,len(x)))].mean(1)
    return dict(mean=float(x.mean()),low=float(np.quantile(m,.025)),high=float(np.quantile(m,.975)),n=len(x))

def load(folder,quality,method):
    samples=[];eigs=[];stats=[];repmeans=[];contacts=[]
    case=int(folder.name.split('_')[-1]);rng=np.random.default_rng(85000+case)
    rng.integers(4096,size=1024);noise=rng.normal(size=(1024,3))
    for action in ACTIONS:
        q=quality if action=='view' else 'high'
        rs=[np.load(folder/f'{q}_{action}_{method}_{rep}.npz') for rep in [0,1]]
        ss=[json.loads((folder/f'{q}_{action}_{method}_{rep}.json').read_text()) for rep in [0,1]]
        current=rs[0]['current'];a=rs[0]['mean']-current;b=rs[1]['mean']-current
        samples.append((a*b).sum(-1)) # H,Q
        dims=3 if action=='view' else 2;h=len(a)
        eigs.append(-.5*(noise[:h,:dims]**2).sum(1)-(rs[0]['logz']+rs[1]['logz'])/2)
        repmeans.append(np.stack([r['mean'] for r in rs]));contacts.append(np.stack([r['contact'] for r in rs]))
        stats.extend(ss)
    return dict(samples=np.stack(samples,-1),eigs=np.stack(eigs,-1),stats=stats,current=current,
        means=np.stack(repmeans,-1),contacts=np.stack(contacts,-1))

def diagnostics(b,f):
    bg=b['samples'].mean(0);fg=f['samples'].mean(0);active=fg.max(1)>1e-4
    bc=bg.argmax(1);fc=fg.argmax(1)
    regret=fg.max(1)-fg[np.arange(6),bc]
    return dict(active=active.tolist(),bridge_gain=bg.tolist(),full_gain=fg.tolist(),
        bridge_choice=bc.tolist(),full_choice=fc.tolist(),agreement=(bc==fc).tolist(),regret=regret.tolist(),
        predictive_rmse=float(np.sqrt(np.mean((b['means'].mean(0)-f['means'].mean(0))**2))),
        bridge_rep_rmse=float(np.sqrt(np.mean((b['means'][0]-b['means'][1])**2))),
        full_rep_rmse=float(np.sqrt(np.mean((f['means'][0]-f['means'][1])**2))),
        contact_mae=float(np.mean(np.abs(b['contacts'].mean(0)-f['contacts'].mean(0)))),
        utility_mae=float(np.mean(np.abs(bg-fg))))

def stability(d,seed):
    s=d['samples'];h=len(s);g=s.mean(0);choice=g.argmax(1)
    active=g.max(1)>1e-4;half=h//2
    rng=np.random.default_rng(seed);boot=s[rng.integers(h,size=(1000,h))].mean(1)
    confidence=(boot.argmax(-1)==choice[None,:]).mean(0)
    alternatives=boot.copy();alternatives[:,np.arange(6),choice]=-np.inf
    gaps=boot[:,np.arange(6),choice]-alternatives.max(2)
    decisive=np.quantile(gaps,.025,axis=0)>0
    upper_regret=np.quantile(boot.max(2)-boot[:,np.arange(6),choice],.95,axis=0)
    tolerance=np.maximum(1e-4,.05*g.max(1))
    a=s[:half].mean(0).argmax(1);b=s[half:].mean(0).argmax(1)
    return dict(active=active.tolist(),confidence=confidence.tolist(),decisive=decisive.tolist(),
        upper_regret95=upper_regret.tolist(),tolerance=tolerance.tolist(),practical_confidence=(upper_regret<=tolerance).tolist(),
        half_agreement=(a==b).tolist(),prefix_agreement={str(k):(s[:k].mean(0).argmax(1)==choice).tolist() for k in [16,32] if k<h},
        half_regret=(g.max(1)-g[np.arange(6),a]).tolist(),
        practical_half_stability=((g.max(1)-g[np.arange(6),a])<=np.maximum(1e-4,.05*g.max(1))).tolist())

def oldcase(case):
    d=json.loads((OLD/'main'/f'case_{case:03d}.json').read_text())
    p=json.loads((OLD/'pixel_only'/f'case_{case:03d}.json').read_text());d['records'].extend(p['records']);d['policies']['pixel_only']=p['policies']
    d['conditional']=json.loads((OLD/'conditional_refined'/f'case_{case:03d}.json').read_text())['records']
    return d

def scores(d,quality):
    rows=[]
    for a in ACTIONS:
        if a=='view':r=next(r for r in d['records'] if r['name']=='view' and r['quality']==quality and r['mode']=='boundary')
        else:r=next(r for r in d['records'] if r['name']=='dynamic' and r['action']==a)
        rows.append(r['metrics'])
    return {k:np.array([r[k] for r in rows]).T for k in FIELDS}

def selected(sc,c):return {k:sc[k][np.arange(6),c] for k in FIELDS}

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--adaptive',action='store_true');args=parser.parse_args()
    if args.adaptive:
        assert len(list(Path('results/budget_audit').glob('case_*/complete.json')))==5,'Adaptive pipeline/audit incomplete; do not label partial results final.'
        for h in [128,256,512]:
            plan=Path(f'results/adaptive_plans/outer{h}.json')
            if plan.exists():
                assert all((Path(f'results/outer{h}')/f'case_{c:03d}'/'complete.json').exists() for c in json.loads(plan.read_text())['cases']),f'outer{h} incomplete'
        OUT=Path('results/analysis_adaptive')
    OUT.mkdir(exist_ok=True);folders=sorted(Path('results/main').glob('case_*/complete.json'));assert len(folders)==72,len(folders)
    cases=[oldcase(i) for i in range(72)];val=[d for d in cases if d['split']=='validation'];test=[d for d in cases if d['split']=='test']
    result={};percase={}
    for quality in ['high','pixel_only']:
        rules={};types={}
        for family in ['corner','gate','deflection']:
            risk=np.mean([scores(d,quality)['mse'] for d in val if d['family']==family],axis=0)
            rules[family]=risk.argmin(1);types[family]=int(risk.mean(0).argmin())
        metrics={};diag=[];stab={m:[] for m in ['bridge','full']};counts={};rows=[];costs={m:[] for m in ['bridge','full']};budgets={}
        for d in cases:
            case=d['case'];folder=Path('results/main')/f'case_{case:03d}'
            if args.adaptive:
                for h in [128,256,512]:
                    candidate=Path(f'results/outer{h}')/f'case_{case:03d}'
                    if (candidate/'complete.json').exists():folder=candidate
            bd=load(folder,quality,'bridge');fd=load(folder,quality,'full')
            budgets[str(case)]=len(bd['samples'])
            dg=diagnostics(bd,fd);dg.update(case=case,split=d['split'],family=d['family']);diag.append(dg)
            for m,dd in [('bridge',bd),('full',fd)]:
                st=stability(dd,90000+case);st.update(case=case,split=d['split']);stab[m].append(st);costs[m].extend(dd['stats'])
            if d['split']!='test':continue
            sc=scores(d,quality);choices={};ms={}
            for m,dd in [('bridge',bd),('full',fd)]:
                gain=dd['samples'].mean(0)
                choices[m]=gain.argmax(1);choices[m+'_EIG']=np.full(6,dd['eigs'].mean(0).argmax())
                choices[m+'_probe_only']=gain[:,1:].argmax(1)+1
                for prefix in [16,32]:choices[m+f'_outer{prefix}']=dd['samples'][:prefix].mean(0).argmax(1)
                raw=np.mean((dd['means'][0]-np.load(folder/f'{quality}_view_{m}_0.npz')['current'][:,:,None])**2,axis=2).mean(0)*2
                choices[m+'_single_run']=raw.argmax(1)
            choices['value_rule']=rules[d['family']];choices['type_rule']=np.full(6,types[d['family']]);choices['always_view']=np.zeros(6,dtype=int)
            gain=np.array(d['policies'][quality]['boundary']['gains']);choices['v5_bank']=gain.argmax(1)
            gain[:,0]=d['conditional'][quality]['boundary']['gain'];choices['v5_conditional']=gain.argmax(1)
            mixed=gain.copy();mixed[:,0]=bd['samples'].mean(0)[:,0];choices['v6_view_v5_probes']=mixed.argmax(1)
            mixed=bd['samples'].mean(0).copy();mixed[:,0]=gain[:,0];choices['v5_view_v6_probes']=mixed.argmax(1)
            for name,c in choices.items():
                ms[name]=selected(sc,c);counts.setdefault(name,np.zeros(4,int));counts[name]+=np.bincount(c,minlength=4)
            ms['random']={k:v.mean(1) for k,v in sc.items()}
            ms['no_observation']=next(r for r in d['records'] if r['name']=='no_observation')['metrics']
            for name,v in ms.items():metrics.setdefault(name,[]).append({k:float(np.mean(v[k])) for k in FIELDS})
            rows.append(dict(case=case,family=d['family'],choices={k:v.tolist() for k,v in choices.items()},metrics={name:{k:float(np.mean(v[k])) for k in FIELDS} for name,v in ms.items()}))
        comparisons={}
        for method in ['bridge','full']:
            for baseline in ['value_rule','type_rule','always_view','random','no_observation','bridge_EIG','full_EIG','bridge_probe_only','full_probe_only','v5_bank','v5_conditional','v6_view_v5_probes','v5_view_v6_probes','bridge','full',method+'_outer32',method+'_single_run']:
                if method==baseline:continue
                comparisons[method+' minus '+baseline]={k:ci([a[k]-b[k] for a,b in zip(metrics[method],metrics[baseline])]) for k in ['mse','crps','coverage90']}
        diagnostics_summary={}
        for split in ['validation','test']:
            ds=[r for r in diag if r['split']==split];mask=np.array([r['active'] for r in ds]);agree=np.array([r['agreement'] for r in ds]);regret=np.array([r['regret'] for r in ds])
            dsout=dict(n_active=int(mask.sum()),agreement_all=float(agree.mean()),agreement_active=float(agree[mask].mean()),regret_active=float(regret[mask].mean()),
                predictive_rmse_median=float(np.median([r['predictive_rmse'] for r in ds])),predictive_rmse_max=float(max(r['predictive_rmse'] for r in ds)),
                full_rep_rmse_median=float(np.median([r['full_rep_rmse'] for r in ds])),bridge_rep_rmse_median=float(np.median([r['bridge_rep_rmse'] for r in ds])),
                contact_mae=float(np.mean([r['contact_mae'] for r in ds])))
            for m in ['bridge','full']:
                ss=[r for r in stab[m] if r['split']==split];mm=mask;dsout[m]={}
                for field in ['decisive','half_agreement','practical_half_stability','practical_confidence']:
                    v=np.array([r[field] for r in ss]);dsout[m][field+'_active']=float(v[mm].mean())
                for p in ['16','32']:
                    v=np.array([r['prefix_agreement'][p] for r in ss]);dsout[m]['prefix'+p+'_agreement_active']=float(v[mm].mean())
            diagnostics_summary[split]=dsout
        cost_summary={}
        for m,ss in costs.items():
            cost_summary[m]=dict(seconds=sum(s['seconds'] for s in ss),rollouts=sum(s['trajectory_rollouts'] for s in ss),
                stages_mean=float(np.mean([x for s in ss for x in s['stages']])),
                one_step_ess_median=float(np.median([x for s in ss for x in s['one_step_ess']])),
                one_step_ess_min=float(min(x for s in ss for x in s['one_step_ess'])),
                stage_ess_min=float(min(x for s in ss for x in s['min_stage_ess'])),
                acceptance_mean=float(np.mean([x for s in ss for x in s['acceptance']])))
        result[quality]=dict(summary={name:{k:float(np.mean([r[k] for r in rs])) for k in FIELDS} for name,rs in metrics.items()},
            paired_ci=comparisons,action_counts={k:v.tolist() for k,v in counts.items()},
            value_rules={k:v.tolist() for k,v in rules.items()},type_rules=types,diagnostics_summary=diagnostics_summary,diagnostics=diag,stability=stab,cost=cost_summary,budgets=budgets)
        percase[quality]=rows
    (OUT/'summary.json').write_text(json.dumps(result,indent=2));(OUT/'per_case.json').write_text(json.dumps(percase,indent=2))
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    names=['bridge','full','value_rule','v5_conditional','always_view','bridge_probe_only']
    for ax,q in zip(axs,['high','pixel_only']):
        ds=percase[q];cs=[ci([d['metrics'][n]['mse'] for d in ds]) for n in names];mm=np.array([c['mean'] for c in cs]);err=np.array([[c['mean']-c['low'] for c in cs],[c['high']-c['mean'] for c in cs]])
        ax.bar(range(len(names)),mm,yerr=err,capsize=3,color=['#168b87','#73b9b4','#8494be','#bd9b72','#b3b3b3','#8494be']);ax.set_xticks(range(len(names)),names,rotation=25,ha='right');ax.set_title(q);ax.set_ylabel('Counterfactual velocity MSE')
    fig.tight_layout();fig.savefig(OUT/'policy_comparison.png',dpi=180);plt.close(fig)
    print(json.dumps({q:dict(summary=result[q]['summary'],diagnostics=result[q]['diagnostics_summary']) for q in result},indent=2))

if __name__=='__main__':main()
