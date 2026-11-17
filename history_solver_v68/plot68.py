"""Paired Brier contrasts; independently parameterized histories and terminal engines."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

H=Path(__file__).resolve().parent
d=json.loads((H/'crossover68.json').read_text()); x=np.array(d['by_catalog'])
# catalog, family, law, input/correction, label solver, decoder
groups=[(0,0,'Archived inputs / prescribed labels'),
        (1,0,'New prescribed / no calibration update'),
        (3,0,'New prescribed / four-pair update'),
        (2,1,'New compliant / no calibration update'),
        (4,1,'New compliant / four-pair update')]
color=('#757575','#2e8190','#125364','#b96949','#83391c')
fig,arr=plt.subplots(1,2,figsize=(10.3,4.5),sharey=True)
for l,ax in enumerate(arr):
    for g,(s,e,name) in enumerate(groups):
        risk=x[:,:,l,s,e]
        effects=np.column_stack([risk[:,0,1]-risk[:,0,0],risk[:,1,1]-risk[:,1,0],
                                 (risk[:,:,4]-risk[:,:,3]).mean(1)])*1000
        m=effects.mean(0)
        half=t.ppf(1-.025/6,11)*effects.std(0,ddof=1)/np.sqrt(12)
        ax.errorbar(m,np.arange(3)+(g-2)*.115,xerr=half,fmt=('x','o','s','o','s')[g],
                    markerfacecolor=('white' if g in (1,3) else color[g]),ms=4,capsize=2,
                    color=color[g],label=name)
    ax.axvline(0,lw=.9,color='#505050')
    ax.set_title(('Rate-strengthening','Rate-weakening')[l],loc='left',fontweight='bold')
    ax.grid(axis='y',color='.91');ax.spines[['top','right']].set_visible(False)
    ax.set_xlabel('Paired difference in held-out Brier (×10⁻³)')
    ax.set_yticks(range(3),['Wendland\nsoft − full','Dispersive\nsoft − full','Both families\nrestored − dropped'])
arr[0].set_ylim(2.45,-.45)
xmin=min(ax.get_xlim()[0] for ax in arr)
xmax=max(ax.get_xlim()[1] for ax in arr)
for ax in arr:ax.set_xlim(xmin,xmax)
handles,labels=arr[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,fontsize=8)
fig.tight_layout(rect=(0,.19,1,1))
for ext in ('pdf','png'):fig.savefig(H/f'v68_results.{ext}',dpi=180,bbox_inches='tight')
