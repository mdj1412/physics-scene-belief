import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('results/analysis')
METRICS=['mse','vy_mse','penalized_mse','cost','brier','crps','coverage90','width90',
         'posterior_risk','ess','design_regret','action_agreement',
         'first_view','first_probe','first_repeat','first_stop']


def read(path):
    with path.open() as f: rows=list(csv.DictReader(f))
    return [{k:(v if k in ['method','split'] else float(v)) for k,v in r.items()} for r in rows]


def region(r):
    distance=abs(r['residual_mean'])
    if abs(distance-.05)<.015:return 'boundary'
    if distance<.025:return 'head_on'
    if distance>.075:return 'clear_miss'
    return 'transition'


def by_family(rows,metric):
    d=defaultdict(list)
    for r in rows:d[int(r['family'])].append(r[metric])
    return {f:float(np.mean(v)) for f,v in d.items()}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=sum([read(Path(f'results/main/family_metrics_{s}.csv')) for s in range(4)],[])
    assert len(rows)==128*12*17*3
    references=sum([read(Path(f'results/main/design_reference_{s}.csv')) for s in range(4)],[])
    selection={}
    for budget,candidates in [(1,['static_vv','static_pp','static_ss']),
                              (2,['static_vv','static_vp','static_pv','static_pp','static_vs','static_ps','static_ss'])]:
        losses={m:float(np.mean([r['penalized_mse'] for r in rows if r['split']=='validation'
                                and r['method']==m and r['budget']==budget])) for m in candidates}
        selection[budget]=dict(selected=min(losses,key=losses.get),validation_penalized_mse=losses)
    (OUT/'query_type_rule_selection.json').write_text(json.dumps(selection,indent=2))
    aliases=[]
    for r in rows:
        b=int(r['budget']);source='static_ss' if b==0 else selection[b]['selected']
        if r['method']==source:aliases.append(dict(r,method='query_type_rule'))
    rows+=aliases
    has_fine=all(Path(f'results/refined/family_metrics_{s}.csv').exists() for s in range(4))
    fine_refs=[]
    if has_fine:
        rows+=sum([read(Path(f'results/refined/family_metrics_{s}.csv')) for s in range(4)],[])
        fine_refs=sum([read(Path(f'results/refined/design_reference_{s}.csv')) for s in range(4)],[])
        fmap={(r['family'],r['q']):r for r in fine_refs}
        for r in rows:
            ref=fmap[r['family'],r['q']]
            values=np.array([ref[k+'_gain']-.005 for k in ['view','probe','repeat']])
            expected=sum(r['first_'+k]*values[i] for i,k in enumerate(['view','probe','repeat']))
            r['fine_design_regret']=float(max(0,values.max())-expected) if r['budget'] else 0.
        METRICS.append('fine_design_regret')
    for r in rows:
        assert all(np.isfinite(r[m]) for m in METRICS)
        assert r['mse']>=0 and r['crps']>-1e-8
        r['region']=region(r)
    methods=sorted(set(r['method'] for r in rows))
    groups=defaultdict(list)
    for r in rows:
        if r['split']=='test':
            groups[(r['method'],int(r['budget']),'all')].append(r)
            groups[(r['method'],int(r['budget']),r['region'])].append(r)
    summaries=[];lookup={}
    for (method,budget,reg),rr in sorted(groups.items()):
        rec=dict(method=method,budget=budget,region=reg,families=len(set(r['family'] for r in rr)),
                 queries=len(rr),atoms=rr[0]['atoms'])
        rec.update({m:float(np.mean(list(by_family(rr,m).values()))) for m in METRICS})
        summaries.append(rec);lookup[method,budget,reg]=rec
    (OUT/'summary.json').write_text(json.dumps(summaries,indent=2))
    rng=np.random.default_rng(271828)
    bootstrap=rng.integers(0,64,(10000,64))
    contrasts=[]
    for reg in ['all','boundary']:
        for b in [1,2]:
            for m in methods:
                if m=='joint_evsi':continue
                for metric in ['mse','penalized_mse','brier','crps','design_regret']:
                    base=by_family(groups[m,b,reg],metric)
                    target=by_family(groups['joint_evsi',b,reg],metric)
                    fam=sorted(base.keys()&target.keys());assert len(fam)==64
                    delta=np.array([base[f]-target[f] for f in fam])
                    ci=np.quantile(delta[bootstrap].mean(-1),[.025,.975])
                    contrasts.append(dict(method=m,reference='joint_evsi',budget=b,region=reg,
                        metric=metric,mean=float(delta.mean()),ci95=ci.tolist(),family_n=64))
    (OUT/'paired_contrasts.json').write_text(json.dumps(contrasts,indent=2))
    equal_information=[]
    for reg in ['all','boundary']:
        for b in [1,2]:
            for metric in ['mse','brier','crps','coverage90']:
                plug=by_family(groups['plugin_static_vp',b,reg],metric)
                joint=by_family(groups['static_vp',b,reg],metric)
                delta=np.array([plug[f]-joint[f] for f in sorted(plug)])
                ci=np.quantile(delta[bootstrap].mean(-1),[.025,.975])
                equal_information.append(dict(region=reg,budget=b,metric=metric,
                    contrast='plugin_static_vp minus static_vp',mean=float(delta.mean()),ci95=ci.tolist()))
    (OUT/'equal_information.json').write_text(json.dumps(equal_information,indent=2))

    refs=[r for r in references if r['split']=='test']
    family_actions=defaultdict(set)
    for r in refs:family_actions[int(r['family'])].add(int(r['reference_action']))
    validation=json.loads(Path('results/development_validation.json').read_text())
    checks=dict(test_reference_action_counts={str(a):sum(r['reference_action']==a for r in refs) for a in [-1,0,1,2]},
        test_families_with_at_least_two_actions=sum(len(a)>=2 for a in family_actions.values()),
        test_families_with_all_three_actions=sum(a>={-1,0,1} for a in family_actions.values()),
        family_count=64,query_type_count=1,main_metric_rows=128*12*17*3,
        test_episode_count=64*32*12,all_episode_count=128*32*12,
        fine_reference_completed=has_fine,development_validation=validation)
    if has_fine:
        fm={(r['family'],r['q']):r for r in fine_refs}
        checks['test_dense_fine_action_agreement']=float(np.mean([r['reference_action']==fm[r['family'],r['q']]['reference_action'] for r in refs]))
        checks['test_dense_fine_max_gain_difference']=max(abs(r[k]-fm[r['family'],r['q']][k]) for r in refs for k in ['view_gain','probe_gain'])
        checks['test_fine_reference_action_counts']={str(a):sum(r['reference_action']==a for r in fine_refs if r['split']=='test') for a in [-1,0,1,2]}
    (OUT/'checks.json').write_text(json.dumps(checks,indent=2))

    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,4))
    for a,label,color in [(-1,'Stop','#999999'),(0,'View','#387fbb'),(1,'Probe','#e59738')]:
        rr=[r for r in refs if r['reference_action']==a]
        ax.scatter([r['residual_mean']*1000 for r in rr],[r['geometry_sd']*1000 for r in rr],
                   s=24,alpha=.65,label=label,color=color)
    for x in [-50,50]:ax.axvline(x,color='black',ls='--',alpha=.45)
    for x in [-30,30]:ax.axvline(x,color='black',ls=':',alpha=.25)
    ax.set(xlabel='Prior mean wall offset after the same translation query (mm)',
           ylabel='Geometry prior standard deviation (mm)',title='One query type, different Bayes-design actions (held-out families)')
    ax.legend(ncol=3,loc='upper center');fig.tight_layout();fig.savefig(figdir/'action_map.png',dpi=180);plt.close(fig)

    selected=['joint_evsi','grid33_evsi','quadrature256_evsi','particles512_evsi','joint_variance_rule',
              'query_type_rule','plugin_refit_evsi','plugin_explore25']
    labels=['Joint 4257','Grid 561','Quadrature 256','Particles 512','Variance rule','Query-type rule','Plug-in','Plug-in + exploration']
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for ax,reg in zip(axes,['all','boundary']):
        x=np.arange(len(selected));v=[lookup[m,1,reg]['mse'] for m in selected]
        ax.bar(x,v,color=['#397fbc']+['#72a7cc']*3+['#e6b261']*2+['#c36d6d']*2)
        ax.set(xticks=x,xticklabels=labels,ylabel='Normalized outgoing vx MSE',title=f'Budget 1: {reg}')
        ax.tick_params(axis='x',rotation=35,labelsize=8)
    fig.tight_layout();fig.savefig(figdir/'baseline_comparison.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for b,ax in enumerate(axes,1):
        for m,label in zip(selected,labels):
            r=lookup[m,b,'all'];ax.scatter(r['cost'],r['mse'],s=55,label=label)
        ax.set(xlabel='Mean acquired observation cost',ylabel='Normalized outgoing vx MSE',title=f'Maximum budget {b}')
        ax.grid(alpha=.2)
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='lower center',ncol=4,fontsize=9)
    fig.tight_layout(rect=[0,.14,1,1]);fig.savefig(figdir/'cost_risk.png',dpi=180);plt.close(fig)

    chosen_family=next(f for f,a in sorted(family_actions.items()) if a>={-1,0,1})
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True,gridspec_kw={'height_ratios':[1,3]})
    rr=sorted([r for r in refs if r['family']==chosen_family],key=lambda r:r['residual_mean'])
    axes[0].scatter([r['residual_mean']*1000 for r in rr],[r['reference_action'] for r in rr],s=60)
    axes[0].set(yticks=[-1,0,1],yticklabels=['Stop','View','Probe'],title=f'Family {chosen_family}: numeric query value changes observation choice')
    for m in ['joint_evsi','grid33_evsi','query_type_rule','plugin_refit_evsi']:
        rr=sorted([r for r in rows if r['family']==chosen_family and r['method']==m and r['budget']==1],key=lambda r:r['residual_mean'])
        axes[1].plot([r['residual_mean']*1000 for r in rr],[r['mse'] for r in rr],marker='o',label=m)
    axes[1].set(xlabel='Prior mean wall offset after translation (mm)',ylabel='Normalized vx MSE');axes[1].legend(fontsize=8)
    for ax in axes:
        for x in [-50,50]:ax.axvline(x,color='black',ls='--',alpha=.3)
        ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(figdir/'same_family_queries.png',dpi=180);plt.close(fig)

    d=np.load('data/physics_table.npz');offset=d['offsets'];es=d['es'];v=d['values']
    fig,ax=plt.subplots(figsize=(9,4))
    for e in [.2,.5,.8]:
        j=int(np.argmin(abs(es-e)));ax.plot(offset*1000,v[:,j,0],label=f'Restitution = {es[j]:.2f}')
    ax.axvline(50,color='black',ls='--',alpha=.5,label='Nominal contact boundary')
    ax.set(xlim=(20,65),xlabel='Absolute wall y offset (mm)',ylabel='Normalized outgoing vx',title='PyBullet response near grazing contact')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(figdir/'contact_response.png',dpi=180);plt.close(fig)

    table=['# Continuous geometry 실험 결과표','',
        'Held-out 64 family. 모든 query는 translate_wall_y type이다. CI는 paired scene-family bootstrap이다.','',
        '| Method | B | MSE | Penalized MSE | Cost | Brier | CRPS | Design regret |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in ['joint_evsi','fine_evsi','grid33_evsi','quadrature256_evsi','particles512_evsi',
              'joint_variance_rule','query_type_rule','plugin_refit_evsi','plugin_explore25','plugin_static_vp','joint_parameter_eig','joint_random']:
        if m not in methods:continue
        for b in [1,2]:
            r=lookup[m,b,'all']
            table.append(f"| {m} | {b} | {r['mse']:.6f} | {r['penalized_mse']:.6f} | {r['cost']:.3f} | {r['brier']:.6f} | {r['crps']:.6f} | {r['design_regret']:.6f} |")
    (OUT/'tables.md').write_text('\n'.join(table))
    print(json.dumps(selection,indent=2));print(json.dumps(checks,indent=2))
    print('\n'.join(table))


if __name__=='__main__':main()
