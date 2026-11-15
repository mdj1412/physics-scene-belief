"""Illustrative public-evidence cases, selected by predicted margin, not actual loss."""
import os,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import Bank,ACTIONS
from run import R,write
assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(1)
policies=[json.loads((R/'test/policies'/f'{i:04d}.json').read_text()) for i in range(32)]
chosen=[]
for action in [1,3]:
    candidates=[i for i,p in enumerate(policies) if p['actions']['joint']==action]
    i=max(candidates,key=lambda j:np.sort(policies[j]['joint_risk'])[1]-np.min(policies[j]['joint_risk']));chosen.append(i)
b=Bank(41);records=[];masses=[]
for i in chosen:
    p=json.loads((R/'test/public'/f'{i:04d}.json').read_text());w=b.posterior(p);wf=b.factorize(w);a=policies[i]['actions']['joint']
    truth=np.load(R/'test/private'/f'{i:04d}.npz');obs=truth['obs'+str(a)][0]
    post=b.update(w,a,obs)[0];fpost=b.update(wf,a,obs)[0]
    prob=lambda v:float(v.reshape(2,2,41,41,41).sum((0,2,3,4))[1])
    mass=w.reshape(2,2,41,41,41).sum((0,3)).cpu().numpy();masses.append(mass)
    records.append(dict(world=i,action=ACTIONS[a],predicted_risks=policies[i]['joint_risk'],p_friction_factual=prob(w),p_friction_joint_after_first_selected_obs=prob(post),p_friction_factorized_after_first_selected_obs=prob(fpost),diagnostics=policies[i]['diagnostics']))
fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained');s=np.unique(b.theta[:,2].cpu().numpy());e=np.unique(b.theta[:,4].cpu().numpy());vmax=max(m.max() for m in masses)
for row,(i,mass) in enumerate(zip(chosen,masses)):
    for c in range(2):
        ax=axs[row,c];im=ax.pcolormesh(e,s,mass[c],cmap='magma',vmin=0,vmax=vmax,shading='nearest')
        ax.set_xlabel('Restitution e');ax.set_ylabel('Factual slope s');ax.set_title(f'World {i:03d}: {ACTIONS[policies[i]["actions"]["joint"]]} | '+('frictionless' if c==0 else 'frictional'))
fig.colorbar(im,ax=axs.ravel().tolist(),label='Joint quadrature posterior mass (g, a marginalized)')
(R/'analysis').mkdir(exist_ok=True);fig.savefig(R/'analysis/posterior_examples.png',dpi=180);plt.close(fig)
write(R/'analysis/examples.json',records);print(json.dumps(records,indent=2))
