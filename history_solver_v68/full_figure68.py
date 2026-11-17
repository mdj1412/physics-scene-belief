"""Compare law-dependent, full-history Brier contrasts by contact solver and resolution."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t
h=Path(__file__).resolve().parent
def read(name):
    return json.loads((h/name).read_text())
m=np.array(read('score_main68.json')['catalog_absolute'])
fine=np.array(read('fine_history/fine_scores.json')['catalogs'])
external=np.array(read('bullet_history/external_history_result.json')['catalog_mean'])[:,0]
sets=(m[:,0,:,:,2].mean(3),m[:,1,:,:,2].mean(3),fine[:,1,1].mean(3),external)
titles=('Prescribed pressure','Compliant · 9 sites','Compliant · 17 sites','PyBullet · rigid sphere')
colors=('#687484','#216d76','#1896a3','#a1511d')
fig,axes=plt.subplots(1,3,figsize=(11.4,3.7),sharey=True)
for i,ax in enumerate(axes):
    for r,(data,col) in enumerate(zip(sets,colors)):
        diff=(data[:,i,: ,1]-data[:,i,:,0]) if i<2 else (data[...,4]-data[...,3]).mean(1)
        for j,mark in enumerate(('o','s')):
            x=diff[:,j]*1000;m0=x.mean();hw=t.isf(.025/6,11)*x.std(ddof=1)/np.sqrt(12)
            ax.errorbar(m0,r+(-.115 if j==0 else .115),xerr=hw,color=col,fmt=mark,
                      markerfacecolor=('white' if j==0 else col),ms=5,capsize=2,linewidth=1.25)
    ax.set_title(('Wendland\nsoft − full','Dispersive\nsoft − full','Both families\nrestored − dropped')[i],loc='left',fontsize=11,fontweight='bold')
    ax.axvline(0,color='.27',lw=.8)
    ax.set_yticks(range(4),titles);ax.set_ylim(3.5,-.5);ax.grid(axis='y',color='.91')
    ax.spines[['top','right']].set_visible(False)
    ax.set_xlabel('Brier difference (×10⁻³)',fontsize=9)
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([0],[0],marker='o',linestyle='none',markerfacecolor='white',color='black',label='Rate-strengthening'),
                    Line2D([0],[0],marker='s',linestyle='none',color='black',label='Rate-weakening')],
           loc='lower center',ncol=2,frameon=False)
fig.tight_layout(rect=(0,.10,1,1),w_pad=1.0)
for ext in ('pdf','png'):fig.savefig(h/f'full_history_v68.{ext}',dpi=180,bbox_inches='tight')
