"""Outcome, nested information gain, and heterogeneity of the prespecified policy."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
policy=json.loads((root/'policy_stats38.json').read_text())
oracle=json.loads((root/'oracle_stats38.json').read_text())['primary']
fig,axes=plt.subplots(1,3,figsize=(13,4.1),layout='constrained')
colors=['#087c83','#95461f','#27729c','#267a54']
names=['Fixed 32.5%','Binary','Three actions','Continuous mean']
def dots(ax,names,means,intervals,palette):
    for row,(v,(lo,hi),color) in enumerate(zip(means,intervals,palette)):
        ax.errorbar(v,row,xerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='o',color=color,capsize=3,ms=6)
    ax.set_yticks(np.arange(len(names)),names)
    ax.invert_yaxis()
    ax.axvline(0,color='#777777',lw=.9)
    ax.grid(axis='x',alpha=.16)
    ax.spines[['right','top','left']].set_visible(False)
p=policy['primary']
ci=np.array(p['primary_sim95'])
dots(axes[0],['vs. fixed','vs. binary'],1000*np.array(p['diff'])[:2],1000*ci,colors[:2])
axes[0].set_title('A   Three-action contrasts',loc='left',fontweight='bold')
axes[0].set_xlabel('Brier difference × 1,000\nnegative favors three actions')
means=[oracle['ideal']['mean'],*p['gain']]
intervals=[oracle['ideal']['ci'],*p['gain_ci']]
dots(axes[1],['Bayes reference',*names],1000*np.array(means),1000*np.array(intervals),['#303030',*colors])
axes[1].set_title('B   Value of fourth probe',loc='left',fontweight='bold')
axes[1].set_xlabel('Brier reduction × 1,000\npositive favors four observations')
cs=np.array(policy['cell_loss'])
ys=np.arange(len(cs))
axes[2].barh(ys,1000*(cs[:,1,2]-cs[:,1,0]),color=[colors[1] if a>0 else colors[3] for a in cs[:,1,2]-cs[:,1,0]])
axes[2].set_yticks(ys,['Shoulder / weakening','Shoulder / strengthening','Bead / weakening','Bead / strengthening'],fontsize=8)
axes[2].invert_yaxis()
axes[2].set_title('C   Fixed target cells',loc='left',fontweight='bold')
axes[2].axvline(0,color='#777777',lw=.9)
axes[2].set_xlabel('Three-action minus fixed Brier × 1,000\ncell means')
axes[2].spines[['right','top','left']].set_visible(False)
fig.savefig(root/'v38.png',dpi=170,bbox_inches='tight')
