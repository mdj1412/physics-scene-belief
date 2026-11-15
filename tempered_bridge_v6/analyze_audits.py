import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import load,diagnostics,stability

out=Path('results/analysis');out.mkdir(exist_ok=True);rows=[]
for case in [66,68]:
    old=json.loads((Path('../boundary_belief_v5/results/conditional_failure_audit')/f'case_{case:03d}.json').read_text())
    new=json.loads((Path('results/failure_audit')/f'case_{case:03d}.json').read_text())
    for i,r in enumerate(old['rows']):
        ref=np.mean([a['mean'] for a in r['references']],axis=0)
        row=dict(case=case,hypothesis=r['hypothesis'],old_ess=r['ess'],old_rmse=float(np.sqrt(np.mean((np.array(r['conditional_mean'])-ref)**2))),old_ref_between=r['between_smc_seeds_rmse'])
        for method in ['bridge','full']:
            rs=[a for a in new['records'] if a['method']==method];means=np.array([a['summary']['mean'][i] for a in rs])
            row[method+'_rmse']=float(np.sqrt(np.mean((means.mean(0)-ref)**2)))
            row[method+'_between']=float(np.sqrt(np.mean((means[0]-means[1])**2)))
            row[method+'_stage_ess_min']=min(a['stat']['min_stage_ess'][i] for a in rs)
            row[method+'_one_step_ess_min']=min(a['stat']['one_step_ess'][i] for a in rs)
        rows.append(row)
(out/'failure_audit.json').write_text(json.dumps(rows,indent=2))
series=[('old','Image-conditioned IS'),('bridge','Tempered bridge'),('full','Full SMC')]
if Path('results/one_step/failure.json').exists():
    control=json.loads(Path('results/one_step/failure.json').read_text())
    for row in rows:row['one_step_rmse']=next(r['rmse'] for r in control if (r['case'],r['hypothesis'])==(row['case'],row['hypothesis']))
    series.insert(1,('one_step','Factual-bank SNIS'))
fig,ax=plt.subplots(figsize=(10,4));xx=np.arange(len(rows));width=.8/len(series)
for i,(m,label) in enumerate(series):ax.bar(xx+(i-(len(series)-1)/2)*width,[r[m+'_rmse'] for r in rows],width,label=label)
ax.set_yscale('log');ax.set_xticks(xx,[f"{r['case']}/{r['hypothesis']}" for r in rows]);ax.set_ylabel('Predictive mean RMSE [m/s]');ax.set_xlabel('World / hypothetical observation');ax.legend(ncol=2,fontsize=9);fig.tight_layout();fig.savefig(out/'failure_recovery.png',dpi=180);plt.close(fig)

audit=[]
for case in [1,25,49,66,68]:
    folder=Path('results/budget_audit')/f'case_{case:03d}'
    if not (folder/'complete.json').exists():continue
    for quality in ['high','pixel_only']:
        b=load(folder,quality,'bridge');f=load(folder,quality,'full');dg=diagnostics(b,f)
        dg.update(case=case,quality=quality,bridge_stability=stability(b,90000+case),full_stability=stability(f,90000+case))
        for method,new in [('bridge',b),('full',f)]:
            old=load(Path('results/main')/f'case_{case:03d}',quality,method)
            # First 64 observations isolate increased inner budget from outer sampling.
            oldgain=old['samples'].mean(0);newgain=new['samples'][:64].mean(0)
            dg[method+'_inner_rank_agreement']=(oldgain.argmax(1)==newgain.argmax(1)).tolist()
            dg[method+'_inner_utility_mae']=float(np.mean(np.abs(oldgain-newgain)))
            dg[method+'_inner_prediction_rmse']=float(np.sqrt(np.mean((old['means'].mean(0)-new['means'].mean(0)[:64])**2)))
            dg[method+'_128_choice']=new['samples'].mean(0).argmax(1).tolist()
        audit.append(dg)
(out/'budget_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(dict(failure_rows=len(rows),budget_rows=len(audit)),indent=2))
