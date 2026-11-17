"""Replicability and decision cost for a frozen policy, in Brier units."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
ROOT=Path(__file__).resolve().parent
s=json.loads((ROOT/'score56.json').read_text())
w=np.array(s['W'])
budgets=np.array(s['K'])
routes=np.array(s['switch_by_w_k'])
fig,axes=plt.subplots(1,2,figsize=(9.0,3.4),layout='constrained')
ax=axes[0]
for k,color in [(4,'#7884ab'),(8,'#177a72'),(16,'#5d5252')]:
    ax.plot(w,routes[:,list(budgets).index(k)],marker='o',ms=4,lw=1.6,color=color,label=f'K = {k}')
ax.axvspan(.35,.52,color='#65798c',alpha=.09)
ax.axvline(.4348074444131813,color='0.45',ls=':',lw=1.2)
ax.set(xlabel='High-mismatch session fraction (w)',ylabel='Probability of selecting fixed 3R+I',ylim=(0,1),title='A   Pilot decisions')
ax.legend(fontsize=8,frameon=False)
ax=axes[1]
x=np.arange(len(budgets),dtype=float)
for j,(name,col) in enumerate([('Frozen gate','#2d6887'),('Fixed 3R+I','#a65b3d')]):
    ci=np.asarray([v['mean_contrast'][j] for v in s['mean_by_k']])*1e6
    ax.errorbar(x+(.06 if j else -.06),ci[:,0],yerr=np.array([ci[:,0]-ci[:,1],ci[:,2]-ci[:,0]]).clip(0),fmt='o',color=col,ms=4,capsize=3,label=name)
ax.axhline(0,color='.35',lw=.9)
ax.set(xticks=x,xticklabels=budgets,xlabel='Paid pilot attempts per surface (K)',ylabel='Switch minus comparator Brier (×10⁻⁶)',title='B   Mean loss near the crossing')
ax.legend(frameon=False,fontsize=8)
for ax in axes:
    ax.grid(axis='y',alpha=.17)
    ax.spines[['top','right']].set_visible(False)
fig.savefig(ROOT/'pilot_switch56.png',dpi=185)
