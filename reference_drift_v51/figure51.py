"""Risk contrast and clustered failure under the same mean transfer."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
res=json.loads((root/'score51.json').read_text())
effects=np.array(res['effect_simultaneous95']); marks=np.array(res['both_upper_below_zero'])
grid=res['e']; n=len(grid)
fig,axes=plt.subplots(1,3,figsize=(12.2,4),layout='constrained')
for i,(label,factor) in enumerate([('A   Soft minus vote',100000),('B   Soft minus unsigned',1000)]):
    ax=axes[i]; vals=effects[...,i,0]*factor; limit=np.abs(vals).max()
    im=ax.imshow(vals,origin='lower',cmap='RdBu_r',vmin=-limit,vmax=limit)
    a,b=np.nonzero(marks)
    ax.scatter(b,a,s=22,facecolors='none',edgecolor='white',linewidth=.9)
    ax.set(xticks=range(n),xticklabels=[f'{v:g}' for v in grid],yticks=range(n),yticklabels=[f'{v:g}' for v in grid],
           title=label,xlabel='Session drift probability (g)',ylabel='Reference–impact mismatch (e)')
    fig.colorbar(im,ax=ax,shrink=.65,label=f'Brier difference × {factor:,}')
with np.load(root/'rollout_sessions51.npz') as d:
    risk=d['risk']
ax=axes[2]
for i,j,label,col in [(2,0,'Object error (e = .25)','#146283'),(0,2,'Session drift (g = .25)','#a14e24')]:
    delta=(risk[:,i,j,...,0]-risk[:,i,j,...,2]).flatten()
    delta.sort(); ax.plot(delta*1000,np.arange(1,len(delta)+1)/len(delta),label=label,color=col)
ax.axvline(0,color='#555',ls='--',lw=.8);ax.set(xlabel='Session Brier difference × 1,000',ylabel='Fraction of sessions',
 title='C   Same mean, different spread',ylim=(0,1.01))
ax.legend(loc='lower right',frameon=False,fontsize=8)
ax.spines[['top','right']].set_visible(False)
fig.savefig(root/'reference_drift_v51.png',dpi=180)
