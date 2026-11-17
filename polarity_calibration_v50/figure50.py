"""Plots estimator shrinkage, decision risk by budget, and loss under session drift."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
s=json.loads((root/'score50.json').read_text()); cal=json.loads((root/'calibration50.json').read_text())
k=np.array(s['kappa']); budgets=np.array(s['budget']); mean=np.array(s['uniform_mean'])
fig,axs=plt.subplots(1,3,figsize=(12.1,4),layout='constrained')
a=axs[0]
a.plot(k,k,ls='--',lw=1,color='#555555',label='Perfect estimate')
for j,c in zip((1,2,3),('#ac5228','#135883','#277860')):
    a.plot(k,np.array(cal['mean'])[:,j],color=c,marker='o',label=f'{budgets[j]} pairs')
a.set(xlabel='True sign retention',ylabel='Mean estimated retention',title='A   Few-pair calibration',xticks=k,yticks=k)
a=axs[1]
for m,label,c in [(0,'Hard sign','#135883'),(6,'Soft mixture','#277860'),(3,'Adaptive fixed site','#ac5228'),(4,'Unsigned fit','#737373')]:
    a.plot(budgets,mean[:,m],marker='o' if m!=4 else None,ls='-' if m!=4 else ':',color=c,label=label)
a.set(xlabel='Calibration pairs per environment',ylabel='Held-out Brier loss',title='B   Uniform regime average',xticks=budgets)
for a in axs[:2]:
    a.spines[['top','right']].set_visible(False)
    a.grid(axis='y',alpha=.17)
    a.legend(fontsize=8,frameon=False)
a=axs[2]
z=np.array(s['cross_law_hard_minus_frozen'])
v=abs(z).max()
im=a.imshow(z,cmap='RdBu_r',vmin=-v,vmax=v,origin='lower')
a.set(xticks=range(5),yticks=range(5),xticklabels=k,yticklabels=k,xlabel='Calibration-session retention',
      ylabel='Deployment retention',title='C   Hard − frozen, 4 pairs')
for j in range(5): a.plot(j,j,'o',markersize=4,markerfacecolor='none',markeredgecolor='black',lw=.6)
fig.colorbar(im,ax=a,shrink=.65,label='Brier difference')
fig.savefig(root/'polarity_calibration_v50.png',dpi=180)
