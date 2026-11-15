"""Post-hoc numerical diagnostic selected by ESS, without test target truth.

Refit the three lowest-ESS and one median-ESS hypothetical views with full SMC.
Do not replace main acquisition scores or treat these selected cases as a new test set.
"""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
from engine import tensor
from belief import Belief,predict,parameter_summary
from conditional_view import conditional,factual_evidence

p=argparse.ArgumentParser();p.add_argument('--case',type=int,required=True);a=p.parse_args();case=a.case
out=Path('results/conditional_failure_audit');out.mkdir(exist_ok=True)
d=json.loads((Path('results/main')/f'case_{case:03d}.json').read_text())
old=np.load(Path('results/main')/f'factual_{case:03d}.npz');atoms=tuple(tensor(old[k]) for k in ['theta','rounding','weight'])
model=Belief(d['family'],'high','boundary');yf=d['observations']['factual'][0]
record=conditional(model,atoms,yf,factual_evidence(d['family'],yf,44000+case),128,2048,45000+case,details=True)
assert record['ess_median']<50,record['ess_median']
ess=np.array(record['details']['ess']);order=np.argsort(ess);chosen=[int(x) for x in order[:3]]+[int(order[len(order)//2])]
rows=[];means=np.array(record['details']['prediction_mean'])
for h in chosen:
    yy=record['details']['observations'][h];refs=[]
    for repeat in [0,1]:
        samples,cost=model.smc([('factual',d['observations']['factual']),('view',yy)],n=4096,moves=18,seed=660000+case*1000+h*2+repeat)
        prediction=predict(samples,d['family'])['velocity'];mean=torch.einsum('qnd,n->qd',prediction,samples[2]).cpu().numpy()
        refs.append(dict(mean=mean.tolist(),parameters=parameter_summary(samples),cost=cost,disagreement_from_conditional=float(np.sqrt(np.mean((mean-means[:,h,:])**2)))))
    between=float(np.sqrt(np.mean((np.array(refs[0]['mean'])-np.array(refs[1]['mean']))**2)))
    rows.append(dict(hypothesis=h,ess=float(ess[h]),observation=yy,conditional_mean=means[:,h,:].tolist(),references=refs,between_smc_seeds_rmse=between))
    print('failure audit',case,'hypothesis',h,'ESS',float(ess[h]),'conditional vs SMC',[r['disagreement_from_conditional'] for r in refs],'between SMC',between,flush=True)
(out/f'case_{case:03d}.json').write_text(json.dumps(dict(case=case,family=d['family'],quality='high',mode='boundary',criterion='median ESS < 50; lowest three and median ESS hypothetical views',job=os.environ.get('SLURM_JOB_ID'),time=time.time(),conditional=record,rows=rows),indent=2))
