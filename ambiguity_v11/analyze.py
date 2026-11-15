"""World-cluster analysis of frozen policies; no policy updates."""
import json,os,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path('results');OUT=R/'analysis';OUT.mkdir(exist_ok=True)
def interval(d,level=.95):
    rng=np.random.default_rng(20261112);samples=[]
    for _ in range(30):samples.append(d[rng.integers(len(d),size=(1000,len(d)))].mean(1))
    q=np.quantile(np.concatenate(samples),[(1-level)/2,(1+level)/2])
    return dict(mean=float(d.mean()),low=float(q[0]),high=float(q[1]),level=level)
def load(split,order=61):
    policies=[json.loads(p.read_text()) for p in sorted((R/split/'policies').glob('*.json'))]
    records=[np.load(p) for p in sorted((R/split/f'evaluation_{order}').glob('*.npz'))]
    assert len(policies)==len(records)>0
    actions={k:np.array([p['actions'][k] for p in policies]) for k in policies[0]['actions']}
    extras=[json.loads(p.read_text()) for p in sorted((R/split/'extra_policies').glob('*.json'))]
    assert len(extras)==len(policies)
    actions['lmmse']=np.array([p['action'] for p in extras])
    arrays={k:np.stack([r[k+'_loss'] for r in records]) for k in ['joint','plugin','mixture','factorized']}
    losses={}
    for k,choice in actions.items():
        decoder={'plugin_aware':'plugin','mixture_aware':'mixture','factorized':'factorized'}.get(k,'joint')
        losses[k]=arrays[decoder][np.arange(len(choice)),choice].mean(1)
        losses[k+'_common']=arrays['joint'][np.arange(len(choice)),choice].mean(1)
    for name in ['plugin','mixture','factorized']:
        for policy in ['joint','rule_best','rule_fixed']:
            losses[name+'_at_'+policy]=arrays[name][np.arange(len(policies)),actions[policy]].mean(1)
        losses[name+'_none']=arrays[name][:,0].mean(1)
    losses['random_expected']=arrays['joint'][:,1:].mean((1,2))
    for a,name in enumerate(['none','query_view','factual_view','probe']):losses['fixed_'+name]=arrays['joint'][:,a].mean(1)
    return policies,actions,arrays,losses

def main():
    assert os.environ.get('SLURM_JOB_ID');summary={};allloss={}
    for split in ['test','misspec']:
        p,a,arrays,l=load(split);allloss[split]=l
        primary={k:interval(l['joint']-l[k],.99) for k in ['none','rule_best','plugin_aware','mixture_aware','factorized']}
        descriptive={k:interval(v-l['joint']) for k,v in l.items() if k!='joint'}
        diag=np.array([[q['diagnostics']['mutual_information'],q['diagnostics']['s_e_correlation'],q['diagnostics']['p_contact'][1]] for q in p])
        cases=[]
        for i in range(len(p)):
            risk=np.array(p[i]['joint_risk']);order=np.argsort(risk)
            cases.append(dict(world=i,action=int(a['joint'][i]),risk=risk.tolist(),margin=float(risk[order[1]]-risk[order[0]]),diagnostics=p[i]['diagnostics']))
        entry=dict(worlds=len(p),means={k:float(v.mean()) for k,v in l.items()},primary_joint_minus=primary,descriptive_minus_joint=descriptive,
                   actions={k:np.bincount(v,minlength=4).tolist() for k,v in a.items()},agreement={k:float((v==a['joint']).mean()) for k,v in a.items()},
                   MI_mean=float(diag[:,0].mean()),MI_median=float(np.median(diag[:,0])),MI_range=[float(diag[:,0].min()),float(diag[:,0].max())],
                   abs_s_e_corr_mean=float(abs(diag[:,1]).mean()),snis_ess_median=np.median([q['snis_ess'] for q in p],0).tolist(),
                   acquisition_seconds_mean=float(np.mean([q['seconds'] for q in p])),cases=cases)
        entry['posthoc_same_action_strata']={str(action):dict(worlds=int((a['joint']==action).sum()),
            factorized_minus_joint=interval((l['factorized_at_joint']-l['joint'])[a['joint']==action]),
            plugin_minus_joint=interval((l['plugin_at_joint']-l['joint'])[a['joint']==action])) for action in [1,3] if (a['joint']==action).any()}
        summary[split]=entry
        np.savez_compressed(OUT/(split+'_world_losses.npz'),**l)
        if split=='test':
            fig,ax=plt.subplots(figsize=(10,5));keys=['none','rule_fixed','rule_best','lmmse','plugin_aware','mixture_aware','factorized','cf_snis','joint']
            labels=['No observation','Best fixed rule','Learned rule + joint decoder','LMMSE + joint decoder','Updated hard plug-in','Updated mixture plug-in','Factorized factual belief','Cross-fitted SNIS + joint decoder','Joint risk selector + joint decoder']
            values=[l[k].mean() for k in keys];cis=[interval(l[k]) for k in keys]
            ax.barh(labels,values,color=['#a5aeb6']*7+['#63a4c9','#1b5880']);ax.errorbar(values,np.arange(len(keys)),xerr=np.array([[v-c['low'],c['high']-v] for v,c in zip(values,cis)]).T,fmt='none',color='black',capsize=3)
            ax.invert_yaxis();ax.set_xlabel('Decision MSE (m²/s²); descriptive 95% world-cluster CI');fig.tight_layout();fig.savefig(OUT/'decision_loss.png',dpi=180);plt.close(fig)
            fig,ax=plt.subplots(figsize=(8,5));colors=['gray','#187f9e','#f8b62b','#be3a43'];names=['none','query-view','factual-view','probe']
            public=[json.loads(q.read_text()) for q in sorted((R/split/'public').glob('*.json'))]
            ys=np.linspace(-.01,.01,9);slope=np.array([np.dot(q['profile'],ys)/np.dot(ys,ys) for q in public])
            for action in range(4):
                m=a['joint']==action
                if m.any():ax.scatter(slope[m],diag[m,2],s=26,c=colors[action],label=f'{names[action]} (n={m.sum()})',alpha=.75)
            ax.set_xlabel('Factual boundary slope estimate');ax.set_ylabel('Posterior probability of frictional contact');ax.set_title('Identical visibility and query in every world');ax.legend();fig.tight_layout();fig.savefig(OUT/'action_context.png',dpi=180);plt.close(fig)
    audit=json.loads((R/'ranking_audit.json').read_text());agreements=[];regrets=[];margins=[];repeatagreements=[];reference_risks={k:[] for k in ['joint','rule_best','rule_fixed','lmmse','cf_snis']}
    for record in audit:
        i=record['world'];p=json.loads((R/'test/policies'/f'{i:04d}.json').read_text());ar=np.array([q['risk'] for q in record['repeats']]);average=ar.mean(0)
        choice=p['actions']['joint'];agreements.append(choice==average.argmin());regrets.append(average[choice]-average.min());margins.append(np.sort(average)[1]-average.min());repeatagreements.append(ar[0].argmin()==ar[1].argmin())
        for k in reference_risks:
            a=p['actions'][k] if k!='lmmse' else json.loads((R/'test/extra_policies'/f'{i:04d}.json').read_text())['action']
            reference_risks[k].append(float(average[a]))
    summary['ranking_audit']=dict(worlds=len(audit),H_per_repeat=1024,order=61,main_agreement=float(np.mean(agreements)),repeat_agreement=float(np.mean(repeatagreements)),mean_reference_regret=float(np.mean(regrets)),max_reference_regret=float(np.max(regrets)),median_reference_margin=float(np.median(margins)))
    summary['reference_expected_risks_first32']={k:float(np.mean(v)) for k,v in reference_risks.items()}
    summary['rule_fit']=json.loads((R/'rule_fit.json').read_text());summary['validation']=json.loads((R/'validation.json').read_text())
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({s:{k:summary[s][k] for k in ['worlds','means','primary_joint_minus','actions','agreement','MI_mean','abs_s_e_corr_mean']} for s in ['test','misspec']},indent=2))
    print('ranking',summary['ranking_audit'])

if __name__=='__main__':main()
