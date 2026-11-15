import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from experiment import design_cache, gains, collapse

ROOT=Path('results/pilot')
DATA=Path('data/pilot')


def main():
    rows=list(csv.DictReader((ROOT/'family_metrics.csv').open()))
    nonnumeric={'method','split'}
    rows=[{k:(v if k in nonnumeric else float(v)) for k,v in r.items()} for r in rows]
    groups=defaultdict(list)
    for r in rows:
        if r['split']=='test':
            groups[(r['sigma'],int(r['q']),int(r['budget']),r['method'])].append(r)
    metrics=['mse','brier','crps','coverage90','width90','trajectory_mse','energy','posterior_risk',
             'cost','penalized_mse','first_view','first_probe','first_repeat','first_stop']
    summary=[]
    lookup={}
    for key,group in sorted(groups.items()):
        result=dict(sigma=key[0],q=key[1],budget=key[2],method=key[3],families=20)
        result.update({m:float(np.mean([r[m] for r in group])) for m in metrics})
        summary.append(result); lookup[key]=result
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2))

    rng=np.random.default_rng(161803)
    bootstrap=rng.integers(0,20,(10000,20))
    contrasts=[]
    for sigma in [.5,1,2]:
        for q in [1,2,3]:
            for budget in [0,1,2]:
                for baseline in ['plugin_refit_evsi','plugin_fixed_evsi','joint_query_rule',
                                 'joint_parameter_eig','joint_query_eig','plugin_refit_fixed_view']:
                    for metric in ['mse','penalized_mse','brier','crps']:
                        values=[]
                        for method in [baseline,'joint_evsi']:
                            grouped=defaultdict(list)
                            for r in groups[(sigma,q,budget,method)]:
                                grouped[int(r['family'])].append(r[metric])
                            values.append(np.array([np.mean(grouped[f]) for f in sorted(grouped)]))
                        delta=values[0]-values[1]
                        samples=delta[bootstrap].mean(axis=1)
                        lo,hi=np.quantile(samples,[.025,.975])
                        contrasts.append(dict(sigma=sigma,q=q,budget=budget,baseline=baseline,
                            metric=metric,mean=float(delta.mean()),ci95=[float(lo),float(hi)],
                            unit='paired scene family',independent_n=20))
    (ROOT/'paired_contrasts.json').write_text(json.dumps(contrasts,indent=2))

    arrays=np.load(DATA/'worlds.npz')
    validations=json.loads((DATA/'physics_validation.json').read_text())
    assert max(r['factual_max_rgb_difference'] for r in validations['checks'])==0
    assert all(r['mse'] < 1e-14 for r in summary if r['q']==3)
    for sigma in [.5,1,2]:
        for budget in range(3):
            base=lookup[(sigma,2,budget,'joint_evsi')]
            for method in ['plugin_fixed_evsi','plugin_refit_evsi']:
                assert abs(base['mse']-lookup[(sigma,2,budget,method)]['mse'])<1e-12
            assert lookup[(sigma,1,budget,'joint_known_geometry')]['brier']<1e-12
    assert all(r['crps']>=-1e-10 and r['energy']>=-1e-10 for r in summary)
    np.testing.assert_allclose(arrays['observation_mean'][:,0:2,1],arrays['observation_mean'][:,2:4,1])
    quadrature_errors=[]
    disagreements=0
    compared=0
    # Numerical validation only, on development/validation families, no tuning.
    for f in range(40):
        p=rng.dirichlet(np.ones(4),size=16)
        p=np.concatenate([p,np.full((1,4),.25),collapse(p,f%2)])
        for sigma in [.5,1,2]:
            sd=np.array([sigma,2**.5*sigma,sigma])
            c32=design_cache(arrays['observation_mean'][f],sd,32)
            c64=design_cache(arrays['observation_mean'][f],sd,64)
            for q in range(3):
                z=arrays['velocity'][f,:,q]
                low=gains(p,z,c32,'evsi'); high=gains(p,z,c64,'evsi')
                quadrature_errors.append(float(np.max(abs(low-high))))
                a=np.where(low.max(axis=1)>.005,low.argmax(axis=1),-1)
                b=np.where(high.max(axis=1)>.005,high.argmax(axis=1),-1)
                disagreements+=int(np.sum(a!=b)); compared+=len(p)
    checks=dict(q3_zero_error=True,q2_joint_plugin_equal=True,known_geometry_zero_brier=True,
                probe_does_not_reveal_original_geometry=True,proper_scores_nonnegative=True,
                quadrature32_vs64_max_gain_difference=max(quadrature_errors),
                quadrature32_vs64_action_disagreements=disagreements,action_comparisons=compared,
                max_restitution_error=max(r['max_restitution_error'] for r in validations['checks']))
    (ROOT/'analysis_checks.json').write_text(json.dumps(checks,indent=2))

    figures=ROOT/'figures'; figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    styles=[('joint_evsi','Joint + EVSI','o'),('plugin_refit_evsi','Plug-in refit + EVSI','s'),
            ('joint_fixed_probe','Joint + fixed probe','^'),('joint_query_rule','Joint + query rule','x')]
    for q,ax in enumerate(axes,1):
        for method,label,marker in styles:
            ax.plot(range(3),[lookup[(2,q,b,method)]['mse'] for b in range(3)],marker=marker,label=label)
        ax.set(title=f'Q{q}',xlabel='Maximum observation budget',xticks=[0,1,2])
        ax.set_ylabel('Normalized velocity MSE')
        ax.grid(alpha=.2)
        if q==3:
            ax.set_ylim(-.001,.01)
            ax.text(.05,.85,'All errors < 1e-14 (numerical zero)',transform=ax.transAxes,fontsize=9)
    axes[0].legend(fontsize=8)
    fig.suptitle('Held-out 20 scene families; centroid noise = 2 px')
    fig.tight_layout(); fig.savefig(figures/'risk_budget.png',dpi=180); plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    methods=['joint_evsi','plugin_refit_evsi','joint_parameter_eig','joint_query_eig','joint_query_rule']
    labels=['Joint EVSI','Plug-in EVSI','Parameter EIG','Query EIG','Query rule']
    for q,ax in enumerate(axes,1):
        left=np.zeros(len(methods))
        for key,color in [('view','#4386c6'),('probe','#ed9c42'),('repeat','#9a83b8'),('stop','#a6a6a6')]:
            v=np.array([lookup[(2,q,1,m)]['first_'+key] for m in methods])
            ax.barh(labels,v,left=left,label=key,color=color); left+=v
        ax.set(title=f'Q{q}: first action',xlabel='Fraction',xlim=(0,1))
        ax.invert_yaxis()
    axes[-1].legend(loc='lower right',fontsize=8)
    fig.tight_layout(); fig.savefig(figures/'selection.png',dpi=180); plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    mlist=['joint_evsi','plugin_refit_evsi','plugin_refit_fixed_view']
    labs=['Joint EVSI','Plug-in EVSI','Plug-in + fixed view']
    xx=np.arange(3)
    axes[0].bar(xx-.18,[lookup[(2,1,1,m)]['mse'] for m in mlist],width=.36,label='Realized MSE')
    axes[0].bar(xx+.18,[lookup[(2,1,1,m)]['posterior_risk'] for m in mlist],width=.36,label='Believed variance')
    axes[0].set(xticks=xx,xticklabels=labs,title='Q1, budget 1: overconfidence'); axes[0].legend()
    axes[1].bar(labs,[lookup[(2,1,1,m)]['coverage90'] for m in mlist])
    axes[1].axhline(.9,color='black',linestyle='--',label='Nominal 90%')
    axes[1].set(ylim=(0,1.05),title='90% central interval coverage'); axes[1].legend()
    for ax in axes:
        ax.tick_params(axis='x',labelsize=8)
    fig.tight_layout(); fig.savefig(figures/'uncertainty.png',dpi=180); plt.close(fig)

    fig,axes=plt.subplots(4,4,figsize=(8,7))
    for i,(g,e) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
        for j,frame in enumerate(['factual','top','probe0','probe1']):
            axes[i,j].imshow(Image.open(DATA/'examples'/f'g{g}_e{e}_{frame}.png'))
            axes[i,j].set_xticks([]); axes[i,j].set_yticks([])
            if i==0: axes[i,j].set_title(frame)
            if j==0: axes[i,j].set_ylabel(f'g={g}, e mode={e}')
    fig.suptitle('Actual orthographic RGB observations (family 0, before detector noise)')
    fig.tight_layout(); fig.savefig(figures/'observations.png',dpi=180); plt.close(fig)

    lines=['# Held-out test 결과표','', 'centroid noise = 2 px, 20 scene family, 32 noise realization/world.','']
    for q in [1,2,3]:
        lines += [f'## Q{q}', '', '| Method | Budget | MSE | Brier | CRPS | Cost | Coverage90 |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for method in ['joint_evsi','plugin_refit_evsi','plugin_fixed_evsi','joint_fixed_view',
                       'plugin_refit_fixed_view','joint_fixed_probe','joint_random','joint_parameter_eig',
                       'joint_query_eig','joint_query_rule','joint_known_geometry','joint_known_restitution']:
            for b in [1,2]:
                r=lookup[(2,q,b,method)]
                lines.append(f"| {method} | {b} | {r['mse']:.6f} | {r['brier']:.6f} | {r['crps']:.6f} | {r['cost']:.3f} | {r['coverage90']:.3f} |")
        lines.append('')
    (ROOT/'tables.md').write_text('\n'.join(lines))
    print(json.dumps(checks,indent=2))
    print('Q1 budget 1:',json.dumps(lookup[(2,1,1,'joint_evsi')]))
    print('Q1 plugin:',json.dumps(lookup[(2,1,1,'plugin_refit_evsi')]))


if __name__=='__main__': main()
