"""Held-out analysis; bootstrap families, never individual queries or seeds."""
import csv,json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('results/main');OUT=Path('results/analysis');OUT.mkdir(parents=True,exist_ok=True)
def read(pattern):
    result=[]
    for path in sorted(ROOT.glob(pattern)):
        for row in csv.DictReader(path.open()):
            for k,v in row.items():
                if k not in ['split','method'] and v!='':
                    try:row[k]=float(v)
                    except ValueError:pass
            result.append({k:v for k,v in row.items() if v!=''})
    return result

rows=read('metrics_*.csv');diag=read('diagnostics_*.csv');timings=read('timings_*.csv');checks=read('reference_checks_*.csv')
assert len(rows)==96*8*18*8,len(rows)
reference=[r for r in rows if r['method']=='reference']
rule={}
for signed in [0,1]:
    for sigma in [.03,.01,.003,.001]:
        val=[r for r in reference if r['split']=='validation' and r['signed']==signed and r['sigma']==sigma]
        risk={a:np.mean([r[a+'_mse'] for r in val])+(.0005 if a!='factual' else 0) for a in ['view','probe','factual']}
        rule[(signed,sigma)]=min(risk,key=risk.get)
        print('rule',signed,sigma,rule[(signed,sigma)],risk)
for row in reference:
    for name,a in [('query_type_rule',rule[(row['signed'],row['sigma'])]),('fixed_view','view'),('fixed_probe','probe'),('no_acquisition','factual')]:
        rr=dict(row);rr['method']=name
        for key in ['mse','crps','brier','coverage90','prediction']:rr[key]=row[a+'_'+key]
        rr['cost']=float(a!='factual');rr['penalized_mse']=rr['mse']+.0005*rr['cost']
        rr['action']={'view':0,'probe':1,'factual':2}[a];rr['agreement']=float(rr['action']==row['reference_action'])
        for key in ['regret','gain_error','prediction_error_vs_ref','width90','risk']:rr.pop(key,None)
        rows.append(rr)

test=[r for r in rows if r['split']=='test']
methods=list(dict.fromkeys(r['method'] for r in test))
metrics=['mse','penalized_mse','crps','brier','coverage90','cost','regret','agreement','gain_error','prediction_error_vs_ref','factual_mse','view_mse','probe_mse']
def means(data,keys):return {k:float(np.mean([r[k] for r in data if k in r])) for k in keys if any(k in r for r in data)}
summary=[]
for signed in [-1,0,1]:
    for sigma in [-1,.03,.01,.003,.001]:
        for method in methods:
            data=[r for r in test if r['method']==method and (signed<0 or r['signed']==signed) and (sigma<0 or r['sigma']==sigma)]
            summary.append(dict(signed=signed,sigma=sigma,method=method,**means(data,metrics)))

def families(data,metric):
    groups=defaultdict(list)
    for r in data:
        if metric in r:groups[int(r['family'])].append(r[metric])
    return {f:np.mean(v) for f,v in groups.items()}
rng=np.random.default_rng(991);boot=rng.integers(0,64,(10000,64));contrasts=[]
for sigma in [-1,.03,.01,.003,.001]:
    for signed in [-1,0,1]:
        subset=[r for r in test if (sigma<0 or r['sigma']==sigma) and (signed<0 or r['signed']==signed)]
        for method in methods:
            if method=='reference':continue
            for metric in ['mse','regret','view_mse','factual_mse']:
                a=families([r for r in subset if r['method']==method],metric)
                b=families([r for r in subset if r['method']=='reference'],metric)
                if not a:continue
                diff=np.array([a[f]-b[f] for f in sorted(a)])
                assert len(diff)==64
                low,high=np.quantile(diff[boot].mean(1),[.025,.975])
                contrasts.append(dict(sigma=sigma,signed=signed,method=method,metric=metric,difference=float(diff.mean()),ci95=[float(low),float(high)]))

dsummary=[]
for signed in [0,1]:
    for sigma in [.03,.01,.003,.001]:
        for method in methods:
            data=[r for r in diag if r['split']=='test' and r['method']==method and r['signed']==signed and r['sigma']==sigma]
            if not data:continue
            dsummary.append(dict(signed=signed,sigma=sigma,method=method,**means(data,['corr_g_e','corr_abs_g_e','mi_hist','ess','positive_mass','factual_rmse','in_three_sigma']),
                ess_median=float(np.median([r['ess'] for r in data])),ess_min=float(min(r['ess'] for r in data)),
                mode_loss_rate=float(np.mean([r['positive_mass']<.05 or r['positive_mass']>.95 for r in data])) if signed else 0))
runtime=[dict(method=m,**means([r for r in timings if r['method']==m and r['split']=='test'],['seconds','atoms','factual_evals','conditional_quantiles','stages','mh_accept'])) for m in methods if any(r['method']==m for r in timings)]
variety=[]
for signed in [0,1]:
    for sigma in [.03,.01,.003,.001]:
        groups=defaultdict(set);counts=np.zeros(3,dtype=int)
        for r in reference:
            if r['split']=='test' and r['signed']==signed and r['sigma']==sigma:
                groups[int(r['family'])].add(int(r['action']));counts[int(r['action'])]+=1
        variety.append(dict(signed=signed,sigma=sigma,counts=counts.tolist(),families_multi=sum(len(s)>1 for s in groups.values()),families_all3=sum(len(s)==3 for s in groups.values())))
verification=dict(expected_rows=96*8*18*8,observed_rows=len([r for r in rows if r['method'] in set(t['method'] for t in timings)]),test_families=64,
    reference_gain_max_error=max(r['gain_max_error'] for r in checks),reference_gain_mean_error=np.mean([r['gain_max_error'] for r in checks]),
    reference_action_agreement=np.mean([r['action_agreement'] for r in checks]),reference_prediction_max_error=max(r['mean_max_error'] for r in checks),
    reference_checks=len(checks),finite_metrics=all(np.isfinite(r['mse']) and np.isfinite(r['crps']) for r in test))
for name,obj in [('summary',summary),('contrasts',contrasts),('posterior_diagnostics',dsummary),('runtime',runtime),('within_type_actions',variety),('verification',verification),('query_type_rule',[dict(signed=k[0],sigma=k[1],action=v) for k,v in rule.items()])]:
    (OUT/f'{name}.json').write_text(json.dumps(obj,indent=2))

selected=['reference','grid33x17','grid65x33','prior512','prior2048','posterior512','smc512','conditional_quad512','factorized_marginals','query_type_rule']
lines=['| noise | method | MSE | regret | action agreement | coverage90 |','|---:|---|---:|---:|---:|---:|']
for sigma in [.03,.01,.003,.001]:
    for m in selected:
        r=next(r for r in summary if r['signed']==-1 and r['sigma']==sigma and r['method']==m)
        lines.append(f"| {sigma} | {m} | {r['mse']:.6f} | {r.get('regret',float('nan')):.6f} | {r['agreement']:.3f} | {r['coverage90']:.3f} |")
(OUT/'tables.md').write_text('\n'.join(lines))

fig,axs=plt.subplots(1,3,figsize=(14,4))
for method in selected[:8]:
    data=sorted([r for r in summary if r['signed']==-1 and r['sigma']>0 and r['method']==method],key=lambda r:r['sigma'])
    for ax,key in zip(axs,['mse','regret','agreement']):ax.plot([r['sigma'] for r in data],[r[key] for r in data],'-o',label=method,ms=4);ax.set_xscale('log');ax.set_xlabel('Factual velocity noise');ax.set_ylabel(key);ax.grid(alpha=.2)
axs[0].legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'noise_sweep.png',dpi=180);plt.close(fig)

names=['reference','grid33x17','prior512','smc512','conditional_quad512','factorized_marginals']
fig,axs=plt.subplots(2,6,figsize=(17,6),sharey=True)
for si in [0,1]:
    refpic=np.load(ROOT/f'posterior_f32_s{si}_n3_reference.npz')
    extent=refpic['g'][refpic['w']>max(refpic['w'])*1e-5]*1000
    gap=(max(extent)-min(extent))*.05
    for j,name in enumerate(names):
        d=np.load(ROOT/f'posterior_f32_s{si}_n3_{name}.npz');w=d['w'];mask=w>max(w)*1e-5
        axs[si,j].scatter(d['g'][mask]*1000,d['e'][mask],s=1+40*w[mask]/max(w),c=np.log10(np.maximum(w[mask],1e-12)),cmap='viridis',vmin=-5,vmax=-1,alpha=.65)
        axs[si,j].plot(float(d['true_g'])*1000,float(d['true_e']),'rx',ms=8)
        axs[si,j].set_title(name,fontsize=9);axs[si,j].set_xlabel('Geometry offset [mm]');axs[si,j].set_ylim(.08,.92)
        axs[si,j].set_xlim(min(extent)-gap,max(extent)+gap)
    axs[si,0].set_ylabel('Restitution / '+('two branches' if si else 'one branch'))
fig.suptitle('Factual grazing posterior, noise=0.001; first held-out family (red: truth)',fontsize=12)
fig.tight_layout();fig.savefig(OUT/'posterior_support.png',dpi=180);plt.close(fig)
print(json.dumps(verification,indent=2))
