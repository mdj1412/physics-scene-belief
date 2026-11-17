"""Compare actual and fitted fast-condition release response in the existing test bank."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

path=Path(__file__).resolve().parent
v73=path.parents[1]/'racket_video_v73/gpu/control'
meta=json.loads((v73/'run.json').read_text())
bank=np.load(v73/'predictions.npz')
n=meta['target_held']
x=bank['incident'][:n]
y=bank['target'][:n]
fast=abs(x[:,1])>=meta['fast_cutoff_mps']
idx=np.flatnonzero(fast & (abs(y[:,0]-y[:,2])>.20))
s=x[idx,0]-x[idx,2]
idx=idx[np.argsort(abs(s))]
s=x[idx,0]-x[idx,2]
eff=(1-(y[idx,0]-y[idx,2])/s)/2.5
fig,ax=plt.subplots(figsize=(7,3.7))
for k,c in [(4,'#b76f26'),(8,'#276397')]:
    p=bank[f'elastic_impulse_k{k}'][:,idx,:]
    alpha=(1-(p[...,0]-p[...,2])/s)/2.5
    lo,mid,hi=np.percentile(alpha,[10,50,90],axis=0)
    ax.plot(abs(s),mid,color=c,lw=1.5,label=f'{k} paid impacts, median')
    ax.fill_between(abs(s),lo,hi,color=c,alpha=.12,lw=0)
ax.scatter(abs(s),eff,c='#272727',s=27,zorder=5,label='Observed effective response')
ax.axhline(.4,color='#636363',lw=1,ls='--')
ax.text(.98,.402,'Reversal threshold',transform=ax.get_yaxis_transform(),ha='right',va='bottom',fontsize=9,color='#3c3c3c')
ax.set_xlabel('Incident relative tangential speed |s| (m/s)')
ax.set_ylabel('Effective impulse coefficient')
ax.set_title('Fast A-B-S II soft impacts with a readable release sign (n = 22)',fontsize=11)
ax.grid(alpha=.13);ax.spines[['right','top']].set_visible(False)
ax.legend(fontsize=8,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.22),frameon=False)
fig.tight_layout()
fig.savefig(path/'fast_response.png',bbox_inches='tight',dpi=160)
